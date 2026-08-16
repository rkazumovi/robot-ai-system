"""
test_digital_twin.py — Unit tests for Component 2: the robot digital twin.

Covers kinematics.py, dynamics.py, anomaly.py, and the analytical/
structural parts of pinn.py. Deliberately does NOT run full PINN
training here (that takes minutes and was already validated manually,
see the project's development history) -- these test the fast,
deterministic pieces: forward/inverse kinematics, the Jacobian, RK4
integration, the analytical thermal formula, and FFT-based anomaly
detection.
"""

import numpy as np
import pytest
import torch

from anomaly import compute_spectrum, detect_spectral_anomalies, find_peak_frequency
from dynamics import JointDynamicsParams, integrate_rk4, simulate_joint_trajectory
from kinematics import make_generic_6dof_arm
from pinn import ThermalParams, ThermalPINN

class TestKinematics:
    @pytest.fixture(scope="module")
    def arm(self):
        return make_generic_6dof_arm()

    def test_forward_kinematics_produces_valid_rotation(self, arm):
        pose = arm.forward_kinematics(np.zeros(6))
        R = pose[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_jacobian_matches_finite_differences(self, arm):
        np.random.seed(0)
        theta = np.random.uniform(-1, 1, 6)
        J_analytic = arm.jacobian(theta)

        eps = 1e-6
        pose0 = arm.forward_kinematics(theta)
        J_numeric = np.zeros((6, 6))
        for i in range(6):
            theta_pert = theta.copy()
            theta_pert[i] += eps
            pose1 = arm.forward_kinematics(theta_pert)
            J_numeric[:3, i] = (pose1[:3, 3] - pose0[:3, 3]) / eps

        assert np.max(np.abs(J_analytic[:3] - J_numeric[:3])) < 1e-3

    def test_inverse_kinematics_converges_to_correct_pose(self, arm):
        np.random.seed(1)
        theta_true = np.random.uniform(-1.0, 1.0, 6)
        target_pose = arm.forward_kinematics(theta_true)

        theta_solved, converged, _ = arm.inverse_kinematics(target_pose)
        achieved_pose = arm.forward_kinematics(theta_solved)

        pos_error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3])
        assert converged
        assert pos_error < 1e-3

    def test_ik_solutions_stay_within_joint_limits(self, arm):
        np.random.seed(2)
        target_pose = arm.forward_kinematics(np.random.uniform(-1, 1, 6))
        theta_solved, _, _ = arm.inverse_kinematics(target_pose)
        for angle, (lo, hi) in zip(theta_solved, arm.joint_limits):
            assert lo - 1e-6 <= angle <= hi + 1e-6


class TestDynamics:
    def test_rk4_achieves_fourth_order_convergence(self):
        """RK4 error should shrink as O(dt^4) against a known analytical solution."""
        I_test, k_test, A = 1.0, 4.0, 1.0
        omega = np.sqrt(k_test / I_test)

        def shm_derivative(t, y):
            theta, theta_dot = y
            return np.array([theta_dot, -k_test / I_test * theta])

        t_final = 2.0
        dts = [0.1, 0.05, 0.025]
        errors = []
        for dt in dts:
            n_steps = int(t_final / dt)
            y0 = np.array([A, 0.0])
            t_arr, y_arr = integrate_rk4(shm_derivative, 0.0, y0, dt, n_steps)
            exact = A * np.cos(omega * t_arr[-1])
            errors.append(abs(y_arr[-1, 0] - exact))

        order_1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
        order_2 = np.log(errors[1] / errors[2]) / np.log(dts[1] / dts[2])
        assert 3.5 < order_1 < 4.5
        assert 3.5 < order_2 < 4.5

    def test_damped_system_loses_energy_over_time(self):
        params = JointDynamicsParams.default_6dof()
        theta0 = np.ones(6)

        result = simulate_joint_trajectory(
            theta0=theta0, theta_dot0=np.zeros(6), params=params,
            torque_fn=lambda t: np.zeros(6), duration=3.0, dt=0.001,
        )

        kinetic = 0.5 * params.inertia * result["theta_dot"] ** 2
        potential = params.gravity_coeff * (1 - np.cos(result["theta"]))
        total_energy = (kinetic + potential).sum(axis=1)

        assert total_energy[-1] < total_energy[0]


class TestThermalPINN:
    def test_analytical_steady_state_is_physically_sensible(self):
        """Higher heat generation should predict a higher steady-state
        core temperature -- a basic physical sanity check on the
        closed-form formula independent of any trained network."""
        params_low_q = ThermalParams(q=1000.0)
        params_high_q = ThermalParams(q=10000.0)

        t_low = params_low_q.steady_state(np.array([0.0]))[0]
        t_high = params_high_q.steady_state(np.array([0.0]))[0]

        assert t_high > t_low
        assert t_low > params_low_q.T_amb  # any heat generation raises temp above ambient

    def test_steady_state_temperature_decreases_from_core_to_surface(self):
        """Core (x=0) should be hottest, surface (x=L) coolest -- heat
        flows outward, so no interior point should exceed the core."""
        params = ThermalParams()
        x = np.linspace(0, params.L, 10)
        temps = params.steady_state(x)
        assert np.all(np.diff(temps) <= 1e-9)  # non-increasing as x grows

    def test_pinn_forward_pass_produces_correct_shape(self):
        """Architecture-level check: an untrained PINN should still
        produce correctly-shaped output for valid (x, t) inputs."""
        model = ThermalPINN()
        x = torch.rand(10, 1)
        t = torch.rand(10, 1)
        out = model(x, t)
        assert out.shape == (10, 1)


class TestVibrationAnomalyDetection:
    def test_fft_detects_known_frequency_within_resolution(self):
        dt = 0.001
        n = 5000
        t = np.arange(n) * dt
        freq_resolution = 1.0 / (n * dt)

        f_true = 23.0
        signal = np.sin(2 * np.pi * f_true * t)
        freqs, mag = compute_spectrum(signal, dt)
        f_detected = find_peak_frequency(freqs, mag, f_min=0.5)

        assert abs(f_detected - f_true) <= freq_resolution

    def test_healthy_vs_healthy_signals_do_not_false_alarm(self):
        dt = 0.001
        healthy_ref = simulate_joint_trajectory(
            theta0=np.array([0.0]), theta_dot0=np.array([0.0]),
            params=JointDynamicsParams(inertia=np.array([0.05]), damping=np.array([0.05]),
                                        gravity_coeff=np.array([0.0])),
            torque_fn=lambda t: np.array([0.1 * np.sin(2 * np.pi * 8.0 * t)]),
            duration=2.0, dt=dt,
        )["theta_dot"][:, 0]
        healthy_test = healthy_ref.copy()

        result = detect_spectral_anomalies(healthy_ref, healthy_test, dt)
        assert not result["is_anomalous"]

    def test_injected_fault_frequency_is_detected(self):
        dt = 0.001
        params = JointDynamicsParams(inertia=np.array([0.05]), damping=np.array([0.05]),
                                      gravity_coeff=np.array([0.0]))

        healthy = simulate_joint_trajectory(
            theta0=np.array([0.0]), theta_dot0=np.array([0.0]), params=params,
            torque_fn=lambda t: np.array([0.1 * np.sin(2 * np.pi * 8.0 * t)]),
            duration=2.0, dt=dt,
        )["theta_dot"][:, 0]

        fault_freq = 37.0
        faulty = simulate_joint_trajectory(
            theta0=np.array([0.0]), theta_dot0=np.array([0.0]), params=params,
            torque_fn=lambda t: np.array([
                0.1 * np.sin(2 * np.pi * 8.0 * t) + 0.03 * np.sin(2 * np.pi * fault_freq * t)
            ]),
            duration=2.0, dt=dt,
        )["theta_dot"][:, 0]

        result = detect_spectral_anomalies(healthy, faulty, dt)
        assert result["is_anomalous"]
        assert any(abs(f - fault_freq) < 1.0 for f in result["anomaly_freqs"])