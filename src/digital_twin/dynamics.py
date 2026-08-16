"""
dynamics.py — Runge-Kutta 4th-order integration of robot joint dynamics.

Mathematical model
----------------------
The true manipulator equation for a coupled rigid-body arm is:

    M(theta) @ theta_ddot + C(theta, theta_dot) @ theta_dot + G(theta) = tau

where M is the (configuration-dependent) mass/inertia matrix, C captures
Coriolis/centrifugal coupling, and G is gravity loading. Deriving M and C
exactly (e.g. via recursive Newton-Euler) requires per-link mass,
inertia tensor, and center-of-mass parameters that aren't available
without a specific robot's spec sheet.

Simplification used here (standard for control-oriented digital twins
when full multi-body coupling isn't the focus): each joint is modeled as
an independent damped pendulum under gravity --

    I_i * theta_ddot_i + b_i * theta_dot_i + c_i * sin(theta_i) = tau_i(t)

    theta_ddot_i = (tau_i(t) - b_i * theta_dot_i - c_i * sin(theta_i)) / I_i

This is genuinely physically grounded (I_i = effective joint inertia,
b_i = viscous damping, c_i = effective gravity-torque coefficient
i.e. roughly mass * g * effective_lever_arm) and nonlinear (via sin),
which is enough to properly exercise RK4 on a real nonlinear ODE system
-- just decoupled across joints rather than fully coupled. The
per-joint state is (theta_i, theta_dot_i); the full system state is a
12-dimensional vector (6 joints x 2) for the 6-DOF arm from kinematics.py.

Runge-Kutta 4th order (RK4)
-------------------------------
For y' = f(t, y), RK4 estimates the next state with 4 slope evaluations:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

This is 4th-order accurate: local truncation error is O(dt^5), global
error O(dt^4) -- verified below both analytically (against a known
closed-form solution) and against scipy's independent RK45 solver.

Run this file directly to verify RK4 convergence order, energy
dissipation under damping, and cross-validate against scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rk4_step(f, t: float, y: np.ndarray, dt: float, *args) -> np.ndarray:
    """
    One generic RK4 integration step for y' = f(t, y, *args).

    Args:
        f: derivative function, signature f(t, y, *args) -> dy/dt.
        t: current time.
        y: current state vector.
        dt: time step.
        *args: extra arguments forwarded to f (e.g. system parameters).

    Returns:
        y at time t + dt.
    """
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + dt / 2 * k1, *args)
    k3 = f(t + dt / 2, y + dt / 2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(f, t0: float, y0: np.ndarray, dt: float, n_steps: int, *args) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates y' = f(t, y, *args) forward n_steps using fixed-step RK4.

    Returns:
        (t_array, y_array): t_array has shape (n_steps+1,), y_array has
        shape (n_steps+1, len(y0)).
    """
    t_array = t0 + dt * np.arange(n_steps + 1)
    y_array = np.zeros((n_steps + 1, len(y0)))
    y_array[0] = y0

    y = y0.copy()
    for i in range(n_steps):
        y = rk4_step(f, t_array[i], y, dt, *args)
        y_array[i + 1] = y

    return t_array, y_array


@dataclass
class JointDynamicsParams:
    """Per-joint effective dynamics parameters for the damped-pendulum model."""

    inertia: np.ndarray       # I_i, kg*m^2 (effective)
    damping: np.ndarray       # b_i, N*m*s/rad (viscous friction)
    gravity_coeff: np.ndarray  # c_i, N*m (effective mass*g*lever_arm)

    @classmethod
    def default_6dof(cls) -> "JointDynamicsParams":
        """Reasonable, illustrative parameters for a UR5-scale 6-DOF arm."""
        return cls(
            inertia=np.array([0.5, 0.8, 0.6, 0.15, 0.15, 0.05]),
            damping=np.array([0.8, 1.2, 0.9, 0.3, 0.3, 0.1]),
            gravity_coeff=np.array([0.0, 15.0, 8.0, 1.5, 1.0, 0.0]),
        )


def joint_dynamics_derivative(
    t: float, y: np.ndarray, params: JointDynamicsParams, torque_fn
) -> np.ndarray:
    """
    Derivative function for the per-joint damped-pendulum-under-gravity
    system. State y = [theta_1..n, theta_dot_1..n] (2n-dim).

    Args:
        t: current time.
        y: current state (2n,).
        params: JointDynamicsParams (per-joint I, b, c).
        torque_fn: callable t -> tau array (n,), the commanded joint torques.

    Returns:
        dy/dt (2n,): [theta_dot_1..n, theta_ddot_1..n]
    """
    n = len(params.inertia)
    theta = y[:n]
    theta_dot = y[n:]

    tau = torque_fn(t)
    theta_ddot = (tau - params.damping * theta_dot - params.gravity_coeff * np.sin(theta)) / params.inertia

    return np.concatenate([theta_dot, theta_ddot])


def simulate_joint_trajectory(
    theta0: np.ndarray,
    theta_dot0: np.ndarray,
    params: JointDynamicsParams,
    torque_fn,
    duration: float,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """
    Simulates joint trajectories forward in time via RK4.

    Returns dict with 't' (n_steps+1,), 'theta' (n_steps+1, n_joints),
    'theta_dot' (n_steps+1, n_joints).
    """
    y0 = np.concatenate([theta0, theta_dot0])
    n_steps = int(duration / dt)
    t_array, y_array = integrate_rk4(
        joint_dynamics_derivative, 0.0, y0, dt, n_steps, params, torque_fn
    )
    n = len(theta0)
    return {"t": t_array, "theta": y_array[:, :n], "theta_dot": y_array[:, n:]}


if __name__ == "__main__":
    print("=== RK4 convergence order verification (analytical solution) ===")
    # Simple undamped, ungravitied single-DOF oscillator: I*theta_ddot = -k*theta
    # (linear, i.e. small-angle SHM) has exact solution theta(t) = A*cos(omega*t),
    # omega = sqrt(k/I). We check that RK4's error shrinks as O(dt^4).
    I_test, k_test = 1.0, 4.0
    omega = np.sqrt(k_test / I_test)
    A = 1.0

    def shm_derivative(t, y):
        theta, theta_dot = y
        return np.array([theta_dot, -k_test / I_test * theta])

    t_final = 2.0
    errors = []
    dts = [0.1, 0.05, 0.025, 0.0125]
    for dt in dts:
        n_steps = int(t_final / dt)
        y0 = np.array([A, 0.0])
        t_arr, y_arr = integrate_rk4(shm_derivative, 0.0, y0, dt, n_steps)
        theta_numeric = y_arr[-1, 0]
        theta_exact = A * np.cos(omega * t_arr[-1])
        err = abs(theta_numeric - theta_exact)
        errors.append(err)
        print(f"  dt={dt:.4f}  error={err:.10f}")

    # Convergence order: log(error_ratio) / log(dt_ratio) should be ~4.
    orders = []
    for i in range(1, len(errors)):
        order = np.log(errors[i - 1] / errors[i]) / np.log(dts[i - 1] / dts[i])
        orders.append(order)
        print(f"  Observed order between dt={dts[i-1]} and dt={dts[i]}: {order:.3f}")
    assert all(3.5 < o < 4.5 for o in orders), "RK4 should show ~4th-order convergence."
    print("RK4 confirmed 4th-order accurate (observed order ~4.0).")

    print("\n=== Cross-validation against scipy.integrate.solve_ivp (RK45) ===")
    from scipy.integrate import solve_ivp

    params = JointDynamicsParams.default_6dof()
    theta0 = np.array([0.3, -0.5, 0.8, 0.1, -0.2, 0.0])
    theta_dot0 = np.zeros(6)

    def zero_torque(t):
        return np.zeros(6)

    duration, dt = 2.0, 0.001
    result_rk4 = simulate_joint_trajectory(theta0, theta_dot0, params, zero_torque, duration, dt)

    y0 = np.concatenate([theta0, theta_dot0])
    sol = solve_ivp(
        lambda t, y: joint_dynamics_derivative(t, y, params, zero_torque),
        [0, duration], y0, method="RK45", rtol=1e-10, atol=1e-12,
        t_eval=[duration],
    )
    theta_scipy_final = sol.y[:6, -1]
    theta_rk4_final = result_rk4["theta"][-1]

    max_diff = np.max(np.abs(theta_scipy_final - theta_rk4_final))
    print(f"  Our RK4 final theta:   {np.round(theta_rk4_final, 6)}")
    print(f"  scipy RK45 final theta: {np.round(theta_scipy_final, 6)}")
    print(f"  Max difference: {max_diff:.8f} rad")
    assert max_diff < 1e-4, "Our RK4 integrator disagrees with scipy's independent solver."
    print("Cross-validated against scipy: agreement within 1e-4 rad.")

    print("\n=== Physical sanity check: energy dissipation under damping ===")
    # With damping > 0 and zero commanded torque, total mechanical energy
    # (kinetic + potential) should monotonically decrease over time as the
    # system settles toward equilibrium.
    theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # displaced from rest
    result = simulate_joint_trajectory(theta0, np.zeros(6), params, zero_torque, duration=5.0, dt=0.001)

    kinetic = 0.5 * params.inertia * result["theta_dot"] ** 2
    potential = params.gravity_coeff * (1 - np.cos(result["theta"]))  # pendulum PE, per joint
    total_energy = (kinetic + potential).sum(axis=1)

    # Check energy is non-increasing (allow tiny numerical tolerance).
    energy_diffs = np.diff(total_energy)
    n_violations = np.sum(energy_diffs > 1e-6)
    print(f"  Energy at t=0:   {total_energy[0]:.6f} J")
    print(f"  Energy at t=end: {total_energy[-1]:.6f} J")
    print(f"  Timesteps where energy increased (should be ~0): {n_violations} / {len(energy_diffs)}")
    assert total_energy[-1] < total_energy[0], "Damped system should lose energy over time."
    assert n_violations < len(energy_diffs) * 0.01, "Energy should be essentially monotonically decreasing."
    print("Energy dissipation confirmed physically sensible under damping.")

    print("\nAll checks passed.")