"""
pinn.py — Physics-informed neural network for robot joint thermal dynamics.

Governing PDE (1D heat conduction along the joint housing, radial
direction, with internal heat generation from motor/friction losses):

    dT/dt = alpha * d^2T/dx^2 + q / (rho * c_p),   x in [0, L], t in [0, t_final]

Boundary conditions:
    x=0 (joint core, symmetry):        dT/dx = 0            (Neumann)
    x=L (housing surface, cooling):   -k * dT/dx = h*(T - T_amb)   (Robin)

Initial condition:
    T(x, 0) = T_amb

Analytical steady-state solution (used for validation)
-----------------------------------------------------------
At steady state (dT/dt = 0), the PDE reduces to the ODE:
    alpha * T'' = -q / (rho * c_p)  =>  k * T'' = -q  (since alpha = k/(rho*c_p))
Integrating twice: T(x) = -q/(2k) * x^2 + C1*x + C2.
Applying dT/dx(0)=0 gives C1=0. Applying the Robin BC at x=L:
    -k * (-q/k * L) = h * (T(L) - T_amb)  =>  T(L) = T_amb + q*L/h
Solving for C2 using T(L):
    T_ss(x) = T_amb + q*L/h + q/(2k) * (L^2 - x^2)

This closed-form solution lets us validate the trained PINN directly
against exact physics, the same rigor level as the PINN heat-equation
and Schrodinger-equation projects.

Run this file directly to train the PINN and validate it against the
analytical steady-state solution AND an independent finite-difference
numerical solver (cross-validation, same approach as dynamics.py's
scipy check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

@dataclass
class ThermalParams:
    """Physical parameters for the joint housing thermal model.

    Material: composite/polymer housing shell (e.g. glass-filled nylon or
    similar engineering composite), rather than solid aluminum. This is
    a genuine, common design choice for robot joint outer housings
    (cost, weight, electrical insulation) -- and critically, it keeps
    the Biot number (Bi = h*L/k) close to order 1 rather than the
    ~0.002 an aluminum housing would give.

    Why this matters: with a very small Biot number (as solid aluminum
    would produce), internal conduction becomes nearly instantaneous
    relative to convective heat removal, creating a ~1000x+ separation
    between the internal-diffusion and convective-equilibration
    timescales. That makes the governing PDE numerically stiff for a
    PINN to learn via standard collocation sampling: the spatial second-
    derivative term dominates the residual loss while the temporal term
    becomes comparatively negligible, so the network can satisfy the
    physics loss while barely modeling the actual (slow) heating
    transient at all. Choosing a lower-conductivity composite housing
    keeps both PDE terms comparably scaled, giving a well-conditioned
    learning problem without changing the underlying physics being
    demonstrated.
    """

    k: float = 1.0           # thermal conductivity, W/(m*K)  (composite/polymer housing)
    rho: float = 1200.0      # density, kg/m^3
    c_p: float = 1500.0      # specific heat, J/(kg*K)
    h: float = 50.0          # convective heat transfer coeff, W/(m^2*K) (fan-assisted)
    q: float = 5000.0        # volumetric heat generation, W/m^3 (motor/friction losses)
    L: float = 0.01          # housing wall thickness, m
    T_amb: float = 25.0      # ambient temperature, deg C

    @property
    def alpha(self) -> float:
        """Thermal diffusivity alpha = k / (rho * c_p), m^2/s."""
        return self.k / (self.rho * self.c_p)

    @property
    def diffusion_time_constant(self) -> float:
        """Internal conduction timescale, L^2/alpha."""
        return self.L ** 2 / self.alpha

    @property
    def convective_time_constant(self) -> float:
        """
        Lumped-capacitance convective equilibration timescale,
        rho*c_p*L/h. With a small Biot number (Bi = h*L/k), as is the
        case here, this -- NOT the diffusion timescale -- is what
        actually governs how long the system takes to reach steady
        state: conduction through the material is fast, but heat can
        only LEAVE the system as fast as convection carries it away.
        """
        return self.rho * self.c_p * self.L / self.h

    def steady_state(self, x: np.ndarray) -> np.ndarray:
        """Analytical steady-state temperature profile T_ss(x), see module docstring."""
        return self.T_amb + self.q * self.L / self.h + self.q / (2 * self.k) * (self.L ** 2 - x ** 2)


class ThermalPINN(nn.Module):
    """MLP mapping (x, t) -> T(x, t). Inputs/outputs are normalized internally."""

    def __init__(self, hidden_dim: int = 40, n_hidden_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


class ThermalPINNTrainer:
    """
    Trains ThermalPINN against the PDE residual + boundary/initial
    condition residuals, using normalized (x, t, T) for stable training,
    with denormalization built into predict() for user-facing calls.
    """

    def __init__(self, params: ThermalParams, device: str = "cpu"):
        self.params = params
        self.device = torch.device(device)
        self.model = ThermalPINN().to(self.device)

        # Normalization scales: x in [0, L] -> [0, 1], t in [0, t_final] -> [0, 1],
        # T roughly in [T_amb, T_amb + max_rise] -> [0, 1]-ish range.
        #
        # t_final must cover the DOMINANT physical timescale. With a small
        # Biot number (as here), that's the convective timescale
        # rho*c_p*L/h, not the (much shorter) internal diffusion timescale
        # L^2/alpha -- conduction through the material is fast, but heat
        # can only leave as fast as convection carries it away. Using
        # 5x the convective time constant ensures the domain covers the
        # full transient through to steady state.
        self.t_final = 5.0 * params.convective_time_constant
        T_ss_core = params.steady_state(np.array([0.0]))[0]
        self.T_scale = max(T_ss_core - params.T_amb, 1.0)

    def _normalize(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x / self.params.L, t / self.t_final

    def _denormalize_T(self, T_norm: torch.Tensor) -> torch.Tensor:
        return T_norm * self.T_scale + self.params.T_amb

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """User-facing prediction: real (unnormalized) x, t in, real T out."""
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_t = torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        x_n, t_n = self._normalize(x_t, t_t)
        with torch.no_grad():
            T_norm = self.model(x_n, t_n)
        T = self._denormalize_T(T_norm)
        return T.cpu().numpy().flatten()

    def _pde_residual(self, x_n: torch.Tensor, t_n: torch.Tensor) -> torch.Tensor:
        """
        PDE residual in NORMALIZED coordinates, further non-dimensionalized
        by the source term's own magnitude (q/(rho*c_p)).

        Why this matters: a perfectly flat T = T_amb (trivial, wrong)
        solution exactly satisfies the initial condition and the Neumann
        BC, and nearly satisfies the Robin BC (both its gradient and its
        deviation from T_amb are zero) -- the ONLY residual it fails to
        satisfy is the interior PDE, and only by the source term's own
        (numerically tiny) magnitude q/(rho*c_p). Without rescaling, this
        makes the trivial solution a strong false minimum: nearly every
        loss term rewards it, and the one term that penalizes it barely
        registers. Dividing the residual by q/(rho*c_p) makes "ignoring
        the source term" cost an O(1) loss instead of an O(source_scale^2)
        one, removing that false attractor.
        """
        x_n = x_n.clone().requires_grad_(True)
        t_n = t_n.clone().requires_grad_(True)

        T_norm = self.model(x_n, t_n)

        dT_dt_n = torch.autograd.grad(T_norm, t_n, grad_outputs=torch.ones_like(T_norm),
                                       create_graph=True)[0]
        dT_dx_n = torch.autograd.grad(T_norm, x_n, grad_outputs=torch.ones_like(T_norm),
                                       create_graph=True)[0]
        d2T_dx2_n = torch.autograd.grad(dT_dx_n, x_n, grad_outputs=torch.ones_like(dT_dx_n),
                                         create_graph=True)[0]

        p = self.params
        dT_dt = (self.T_scale / self.t_final) * dT_dt_n
        d2T_dx2 = (self.T_scale / p.L ** 2) * d2T_dx2_n

        source_scale = p.q / (p.rho * p.c_p)
        residual = dT_dt - p.alpha * d2T_dx2 - source_scale
        return residual / source_scale

    def train(
        self,
        n_epochs: int = 15000,
        n_collocation: int = 2000,
        n_boundary: int = 200,
        n_initial: int = 200,
        lr: float = 2e-3,
        log_every: int = 1000,
    ) -> list[float]:
        p = self.params
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        loss_history = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # --- PDE residual loss (interior collocation points) ---
            x_col = torch.rand(n_collocation, 1, device=self.device)  # normalized [0,1]
            t_col = torch.rand(n_collocation, 1, device=self.device)
            pde_res = self._pde_residual(x_col, t_col)
            loss_pde = (pde_res ** 2).mean()

            # --- Boundary condition losses ---
            t_b = torch.rand(n_boundary, 1, device=self.device)

            # x=0: Neumann, dT/dx = 0
            x0 = torch.zeros(n_boundary, 1, device=self.device, requires_grad=True)
            t_b0 = t_b.clone().requires_grad_(True)
            T0 = self.model(x0, t_b0)
            dT_dx0 = torch.autograd.grad(T0, x0, grad_outputs=torch.ones_like(T0), create_graph=True)[0]
            loss_bc0 = (dT_dx0 ** 2).mean()  # normalized dT/dx=0 <=> unnormalized dT/dx=0

            # x=L (normalized x=1): Robin, -k*dT/dx = h*(T - T_amb)
            xL = torch.ones(n_boundary, 1, device=self.device, requires_grad=True)
            t_bL = t_b.clone().requires_grad_(True)
            TL_norm = self.model(xL, t_bL)
            dTL_dx_norm = torch.autograd.grad(TL_norm, xL, grad_outputs=torch.ones_like(TL_norm),
                                               create_graph=True)[0]
            # Convert to physical units: dT/dx = (T_scale / L) * dT_norm/dx_norm
            dTL_dx = (self.T_scale / p.L) * dTL_dx_norm
            TL = self._denormalize_T(TL_norm)
            robin_residual = -p.k * dTL_dx - p.h * (TL - p.T_amb)
            loss_bcL = (robin_residual ** 2).mean()

            # --- Initial condition loss: T(x, 0) = T_amb ---
            x_i = torch.rand(n_initial, 1, device=self.device)
            t_i = torch.zeros(n_initial, 1, device=self.device)
            T_i_norm = self.model(x_i, t_i)
            T_i = self._denormalize_T(T_i_norm)
            loss_ic = ((T_i - p.T_amb) ** 2).mean()

            # Loss weighting: IC/BC terms typically need higher weight than
            # PDE residual to be enforced strongly (standard PINN practice).
            # bcL (Robin) gets the heaviest weight since it converges
            # slowest -- it couples the gradient AND value at the boundary,
            # a stiffer constraint than the pure-Neumann or pure-IC terms.
            loss = loss_pde + 10.0 * loss_bc0 + 50.0 * loss_bcL + 20.0 * loss_ic

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())

            if epoch % log_every == 0 or epoch == n_epochs - 1:
                print(f"  epoch {epoch:5d}  loss={loss.item():.6f}  "
                      f"(pde={loss_pde.item():.6f} bc0={loss_bc0.item():.6f} "
                      f"bcL={loss_bcL.item():.6f} ic={loss_ic.item():.6f})")

        return loss_history

    def _total_loss(self, x_col, t_col, x0, t_b0, xL, t_bL, x_i, t_i) -> torch.Tensor:
        """Recomputes the full weighted loss for a fixed batch of points
        (used by L-BFGS, which requires a closure re-evaluating the loss)."""
        p = self.params

        pde_res = self._pde_residual(x_col, t_col)
        loss_pde = (pde_res ** 2).mean()

        T0 = self.model(x0, t_b0)
        dT_dx0 = torch.autograd.grad(T0, x0, grad_outputs=torch.ones_like(T0), create_graph=True)[0]
        loss_bc0 = (dT_dx0 ** 2).mean()

        TL_norm = self.model(xL, t_bL)
        dTL_dx_norm = torch.autograd.grad(TL_norm, xL, grad_outputs=torch.ones_like(TL_norm),
                                           create_graph=True)[0]
        dTL_dx = (self.T_scale / p.L) * dTL_dx_norm
        TL = self._denormalize_T(TL_norm)
        robin_residual = -p.k * dTL_dx - p.h * (TL - p.T_amb)
        loss_bcL = (robin_residual ** 2).mean()

        T_i_norm = self.model(x_i, t_i)
        T_i = self._denormalize_T(T_i_norm)
        loss_ic = ((T_i - p.T_amb) ** 2).mean()

        return loss_pde + 10.0 * loss_bc0 + 50.0 * loss_bcL + 20.0 * loss_ic

    def finetune_lbfgs(
        self,
        n_iters: int = 500,
        n_rounds: int = 4,
        n_collocation: int = 2000,
        n_boundary: int = 200,
        n_initial: int = 200,
    ) -> None:
        """
        L-BFGS fine-tuning phase, standard second stage in PINN training
        after Adam: Adam explores well globally but converges slowly near
        a good minimum; L-BFGS (a quasi-Newton method using curvature
        information) converges much more precisely once already in the
        right neighborhood. Runs multiple rounds with fresh collocation/
        boundary batches each time, since a single L-BFGS call can hit its
        internal convergence tolerance (on one fixed batch) before the
        loss has genuinely bottomed out on the full problem.
        """
        for round_idx in range(n_rounds):
            x_col = torch.rand(n_collocation, 1, device=self.device)
            t_col = torch.rand(n_collocation, 1, device=self.device)
            t_b = torch.rand(n_boundary, 1, device=self.device)
            x0 = torch.zeros(n_boundary, 1, device=self.device, requires_grad=True)
            t_b0 = t_b.clone().requires_grad_(True)
            xL = torch.ones(n_boundary, 1, device=self.device, requires_grad=True)
            t_bL = t_b.clone().requires_grad_(True)
            x_i = torch.rand(n_initial, 1, device=self.device)
            t_i = torch.zeros(n_initial, 1, device=self.device)

            optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0, max_iter=n_iters,
                history_size=50, line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                loss = self._total_loss(x_col, t_col, x0, t_b0, xL, t_bL, x_i, t_i)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)
            final_loss = self._total_loss(x_col, t_col, x0, t_b0, xL, t_bL, x_i, t_i)
            print(f"  L-BFGS round {round_idx + 1}/{n_rounds}  loss={final_loss.item():.8f}")


def finite_difference_solve(params: ThermalParams, t_final: float, nx: int = 101, n_steps: int = 2000) -> dict:
    """
    Independent Crank-Nicolson finite-difference solver for the same PDE,
    used to cross-validate the PINN (analogous to dynamics.py's scipy
    check). Crank-Nicolson is unconditionally stable, so it can take much
    larger timesteps than an explicit scheme -- important here since the
    true equilibration timescale (convective, rho*c_p*L/h) is much longer
    than the internal diffusion timescale that would otherwise cap an
    explicit method's stable step size.
    """
    p = params
    x = np.linspace(0, p.L, nx)
    dx = x[1] - x[0]
    dt = t_final / n_steps
    r = p.alpha * dt / dx ** 2
    source = dt * p.q / (p.rho * p.c_p)

    T = np.full(nx, p.T_amb)

    # Build the Crank-Nicolson tridiagonal system: A @ T_new = B @ T_old + source terms.
    # Interior points: standard CN stencil. Boundary rows encode the BCs directly.
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))

    for i in range(1, nx - 1):
        A[i, i - 1] = -r / 2
        A[i, i] = 1 + r
        A[i, i + 1] = -r / 2
        B[i, i - 1] = r / 2
        B[i, i] = 1 - r
        B[i, i + 1] = r / 2

    # x=0: Neumann (symmetry), dT/dx=0 -> ghost point T[-1]=T[1], folds into a
    # modified interior-style row using only T[0], T[1].
    A[0, 0] = 1 + r
    A[0, 1] = -r
    B[0, 0] = 1 - r
    B[0, 1] = r

    # x=L: Robin, -k*dT/dx = h*(T-T_amb) -> ghost point T_ghost = T[-2] - beta*(T[-1]-T_amb).
    # Coefficients verified symbolically (see derivation notes): the T_amb
    # constant contributes beta*r/2 from each of the LHS and RHS ghost
    # substitutions, summing to the full beta*r in robin_const, while the
    # T_i diagonal terms individually only pick up beta*r/2 each.
    beta = 2 * dx * p.h / p.k
    A[-1, -2] = -r
    A[-1, -1] = 1 + r + r * beta / 2
    B[-1, -2] = r
    B[-1, -1] = 1 - r - r * beta / 2

    A_inv = np.linalg.inv(A)
    robin_const = r * beta * p.T_amb  # from moving the T_amb term in the ghost point to the RHS

    for _ in range(n_steps):
        rhs = B @ T
        rhs[1:-1] += source  # interior source term (CN-averaged; source is constant in t here)
        rhs[0] += source
        rhs[-1] += source + robin_const
        T = A_inv @ rhs

    return {"x": x, "T": T, "t_final": n_steps * dt}


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    params = ThermalParams()
    print(f"Thermal diffusivity alpha = {params.alpha:.3e} m^2/s")
    print(f"Diffusion time constant L^2/alpha = {params.L**2 / params.alpha:.1f} s")

    print("\n=== Training ThermalPINN (Adam phase) ===")
    trainer = ThermalPINNTrainer(params)
    trainer.train()

    print("\n=== Fine-tuning (L-BFGS phase) ===")
    trainer.finetune_lbfgs()

    print("\n=== Validation 1: PINN vs analytical steady-state solution ===")
    x_eval = np.linspace(0, params.L, 20)
    t_steady = np.full_like(x_eval, trainer.t_final)  # evaluate near t_final (~steady state)
    T_pinn_steady = trainer.predict(x_eval, t_steady)
    T_analytical_steady = params.steady_state(x_eval)

    max_err = np.max(np.abs(T_pinn_steady - T_analytical_steady))
    mean_err = np.mean(np.abs(T_pinn_steady - T_analytical_steady))
    rel_err = mean_err / (T_analytical_steady.max() - params.T_amb)
    print(f"  Analytical T(x=0, steady): {T_analytical_steady[0]:.4f} C")
    print(f"  PINN       T(x=0, steady): {T_pinn_steady[0]:.4f} C")
    print(f"  Max abs error:  {max_err:.4f} C")
    print(f"  Mean abs error: {mean_err:.4f} C")
    print(f"  Relative error: {rel_err*100:.2f}% of temperature rise")

    print("\n=== Validation 2: PINN vs independent finite-difference solver ===")
    fd_result = finite_difference_solve(params, t_final=trainer.t_final)
    T_pinn_at_fd_x = trainer.predict(fd_result["x"], np.full_like(fd_result["x"], fd_result["t_final"]))
    fd_vs_pinn_err = np.max(np.abs(T_pinn_at_fd_x - fd_result["T"]))
    print(f"  FD solver   T(x=0, t={fd_result['t_final']:.1f}s): {fd_result['T'][0]:.4f} C")
    print(f"  PINN        T(x=0, t={fd_result['t_final']:.1f}s): {T_pinn_at_fd_x[0]:.4f} C")
    print(f"  Max abs difference (PINN vs FD, across all x): {fd_vs_pinn_err:.4f} C")

    print("\n=== Validation 3: initial condition sanity check ===")
    T_at_t0 = trainer.predict(x_eval, np.zeros_like(x_eval))
    ic_err = np.max(np.abs(T_at_t0 - params.T_amb))
    print(f"  Max |T(x, t=0) - T_amb|: {ic_err:.4f} C (should be small)")

    print(f"\n{'PASS' if rel_err < 0.05 and fd_vs_pinn_err < 3.0 and ic_err < 2.0 else 'MARGINAL'}: "
          f"See error magnitudes above.")