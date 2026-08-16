"""
kinematics.py — Forward/inverse kinematics for the robot digital twin.

Mathematical foundations
--------------------------
SO(3) rotations + homogeneous coordinates -> kinematics:
    A rigid body's orientation is a rotation matrix R in SO(3) (3x3,
    orthogonal, det=1). Combined with a translation p, a full rigid-body
    transform is a 4x4 homogeneous matrix:

        T = [ R  p ]
            [ 0  1 ]

    Chaining joints: T_end_effector = T_1 @ T_2 @ ... @ T_n, where each
    T_i is the transform from joint i-1's frame to joint i's frame.

Denavit-Hartenberg (DH) parameters:
    Each joint-to-joint transform is fully described by 4 numbers
    (theta, d, a, alpha) -- joint angle, link offset, link length, link
    twist -- the standard convention used across industrial robot arms
    (UR, KUKA, ABB, etc.), rather than a full unconstrained 6-DOF pose
    per joint.

Geometric Jacobian (used here, and by dynamics.py for velocity analysis):
    For a chain of revolute joints, the Jacobian relates joint velocities
    to end-effector linear+angular velocity: v_end = J(theta) @ theta_dot.
    Column i (revolute joint) = [ z_i x (p_end - p_i) ]
                                 [        z_i          ]
    where z_i is joint i's rotation axis (in world frame) and p_i is
    joint i's origin (in world frame), both taken from the partial
    forward-kinematics chain up to joint i.

Inverse kinematics:
    General 6-DOF arms rarely have closed-form IK. We use a numerical
    damped least-squares (Levenberg-Marquardt-style) iterative solver:
    at each step, compute the pose error, use the Jacobian's damped
    pseudo-inverse to get a joint-angle update that reduces that error,
    and repeat until convergence. The damping term keeps the solver
    stable near singularities (where J becomes rank-deficient).

Run this file directly to sanity-check forward kinematics, verify the
Jacobian against finite-difference numerical derivatives, and verify
inverse kinematics converges back to a known target pose.
"""

from __future__ import annotations

import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Standard (Denavit-Hartenberg) single-joint homogeneous transform.

    Args:
        theta: joint angle (radians) -- the variable for revolute joints.
        d: link offset along previous z-axis.
        a: link length along common normal.
        alpha: link twist (radians) about the common normal.

    Returns:
        4x4 homogeneous transform matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ])


class RobotArm:
    """
    A generic N-DOF revolute-joint robot arm, parametrized by DH parameters.

    Args:
        dh_params: list of (d, a, alpha) per joint -- the fixed link
            geometry. theta is NOT included here since it's the variable
            joint angle, supplied per-call to forward_kinematics/jacobian.
        joint_limits: optional list of (min, max) radians per joint, used
            to clip inverse_kinematics solutions to physically valid range.
    """

    def __init__(
        self,
        dh_params: list[tuple[float, float, float]],
        joint_limits: list[tuple[float, float]] | None = None,
    ):
        self.dh_params = dh_params
        self.n_joints = len(dh_params)
        self.joint_limits = joint_limits or [(-np.pi, np.pi)] * self.n_joints
        assert len(self.joint_limits) == self.n_joints

    def joint_transforms(self, theta: np.ndarray) -> list[np.ndarray]:
        """
        Returns the cumulative transform T_0_i (base frame -> joint i frame)
        for i = 1..n_joints, i.e. joint_transforms(theta)[i-1] = T_0_i.
        """
        assert len(theta) == self.n_joints
        transforms = []
        T = np.eye(4)
        for i, (d, a, alpha) in enumerate(self.dh_params):
            T_i = dh_transform(theta[i], d, a, alpha)
            T = T @ T_i
            transforms.append(T.copy())
        return transforms

    def forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        """Returns the 4x4 end-effector pose (base frame) for joint angles theta."""
        return self.joint_transforms(theta)[-1]

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Geometric Jacobian (6 x n_joints): rows 0-2 = linear velocity
        contribution, rows 3-5 = angular velocity contribution, for
        revolute joints.
        """
        transforms = self.joint_transforms(theta)
        p_end = transforms[-1][:3, 3]

        J = np.zeros((6, self.n_joints))
        z_prev = np.array([0.0, 0.0, 1.0])  # base frame z-axis
        p_prev = np.array([0.0, 0.0, 0.0])  # base frame origin

        for i in range(self.n_joints):
            J[:3, i] = np.cross(z_prev, p_end - p_prev)
            J[3:, i] = z_prev

            # Update z_prev, p_prev to joint i's frame for the NEXT column
            # (joint i's own axis/origin are defined by the transform up
            # to and including joint i, i.e. transforms[i]).
            z_prev = transforms[i][:3, 2]
            p_prev = transforms[i][:3, 3]

        return J

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        initial_theta: np.ndarray | None = None,
        max_iters: int = 200,
        tol: float = 1e-6,
        damping: float = 0.01,
        step_scale: float = 1.0,
        n_restarts: int = 4,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, bool, int]:
        """
        Damped least-squares (Levenberg-Marquardt-style) numerical IK, with
        multi-start retries for robustness.

        Numerical IK on a general 6-DOF arm is NOT guaranteed to converge
        from an arbitrary starting guess -- the solver can get stuck near
        a singularity or in a configuration on the wrong side of a local
        minimum. This is a standard, expected limitation of iterative IK
        (not specific to this implementation), typically handled by
        retrying from multiple random starting configurations and keeping
        the best result -- which is what n_restarts does here.

        Args:
            target_pose: desired 4x4 end-effector pose.
            initial_theta: starting joint angles for the FIRST attempt
                (defaults to zeros). Subsequent retries (if needed) use
                random restarts.
            max_iters: maximum solver iterations per attempt.
            tol: convergence threshold on combined position+orientation error norm.
            damping: Levenberg-Marquardt damping factor.
            step_scale: scales the per-iteration joint-angle update.
            n_restarts: number of additional random-restart attempts if
                the first attempt (from initial_theta) does not converge.
            rng: optional numpy random Generator for reproducible restarts.

        Returns:
            (theta, converged, n_iters) for the best attempt found (lowest
            final pose error), across the initial attempt + all restarts.
        """
        rng = rng or np.random.default_rng()

        starting_points = [
            np.zeros(self.n_joints) if initial_theta is None else initial_theta.copy()
        ]
        for _ in range(n_restarts):
            starting_points.append(rng.uniform(-np.pi, np.pi, self.n_joints))

        best_theta = starting_points[0]
        best_converged = False
        best_iters = 0
        best_error = float("inf")

        for start_theta in starting_points:
            theta, converged, n_iters, final_error = self._ik_single_attempt(
                target_pose, start_theta, max_iters, tol, damping, step_scale
            )
            if final_error < best_error:
                best_theta, best_converged, best_iters, best_error = theta, converged, n_iters, final_error
            if converged:
                break  # good enough, no need to burn more restarts

        return best_theta, best_converged, best_iters

    def _ik_single_attempt(
        self,
        target_pose: np.ndarray,
        initial_theta: np.ndarray,
        max_iters: int,
        tol: float,
        damping: float,
        step_scale: float,
    ) -> tuple[np.ndarray, bool, int, float]:
        """One damped-least-squares IK attempt from a single starting point.
        Returns (theta, converged, n_iters, final_error_norm)."""
        theta = initial_theta.copy()
        target_p = target_pose[:3, 3]
        target_R = target_pose[:3, :3]

        converged = False
        error_norm = float("inf")
        n_iters = 0
        for n_iters in range(1, max_iters + 1):
            current_pose = self.forward_kinematics(theta)
            current_p = current_pose[:3, 3]
            current_R = current_pose[:3, :3]

            pos_error = target_p - current_p
            R_error = target_R @ current_R.T
            ori_error = _rotation_matrix_to_vector(R_error)

            error = np.concatenate([pos_error, ori_error])
            error_norm = np.linalg.norm(error)
            if error_norm < tol:
                converged = True
                break

            J = self.jacobian(theta)
            JJt = J @ J.T
            damped_inv = J.T @ np.linalg.inv(JJt + (damping ** 2) * np.eye(6))
            delta_theta = step_scale * (damped_inv @ error)

            theta = theta + delta_theta
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

        for i, (lo, hi) in enumerate(self.joint_limits):
            theta[i] = np.clip(theta[i], lo, hi)

        # Re-measure error post-clip, since clipping (rare, only affects
        # solutions outside the physical joint range) can change the pose.
        final_pose = self.forward_kinematics(theta)
        final_pos_error = np.linalg.norm(final_pose[:3, 3] - target_p)
        final_R_error = target_R @ final_pose[:3, :3].T
        final_ori_error = np.linalg.norm(_rotation_matrix_to_vector(final_R_error))
        final_error = final_pos_error + final_ori_error

        return theta, converged and final_error < tol * 10, n_iters, final_error


def _rotation_matrix_to_vector(R: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector (axis * angle) from a rotation matrix,
    via the skew-symmetric part of R. Valid for small rotations, which is
    the regime IK operates in near convergence."""
    return 0.5 * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def make_generic_6dof_arm() -> RobotArm:
    """
    A generic, non-degenerate 6-DOF revolute arm (UR5-like proportions,
    in meters/radians) used for testing and by the rest of the digital
    twin. DH params: (d, a, alpha) per joint.
    """
    dh_params = [
        (0.0892, 0.0,     np.pi / 2),
        (0.0,   -0.425,   0.0),
        (0.0,   -0.392,   0.0),
        (0.1093, 0.0,     np.pi / 2),
        (0.0947, 0.0,    -np.pi / 2),
        (0.0825, 0.0,     0.0),
    ]
    joint_limits = [(-np.pi, np.pi)] * 6
    return RobotArm(dh_params, joint_limits)


if __name__ == "__main__":
    np.random.seed(0)
    arm = make_generic_6dof_arm()

    print("=== Forward kinematics sanity check ===")
    theta_zero = np.zeros(6)
    pose_zero = arm.forward_kinematics(theta_zero)
    print("End-effector pose at theta=0:\n", np.round(pose_zero, 4))
    R = pose_zero[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8), "Rotation part must be orthogonal (SO(3))."
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8), "Rotation part must have det=1."
    print("Rotation matrix verified as valid SO(3) element (orthogonal, det=1).")

    print("\n=== Jacobian verification against finite differences ===")
    theta_test = np.random.uniform(-1, 1, 6)
    J_analytic = arm.jacobian(theta_test)

    eps = 1e-6
    J_numeric = np.zeros((6, 6))
    pose0 = arm.forward_kinematics(theta_test)
    p0, R0 = pose0[:3, 3], pose0[:3, :3]
    for i in range(6):
        theta_pert = theta_test.copy()
        theta_pert[i] += eps
        pose1 = arm.forward_kinematics(theta_pert)
        p1, R1 = pose1[:3, 3], pose1[:3, :3]

        J_numeric[:3, i] = (p1 - p0) / eps
        dR = R1 @ R0.T
        J_numeric[3:, i] = _rotation_matrix_to_vector(dR) / eps

    max_diff = np.max(np.abs(J_analytic - J_numeric))
    print(f"Max |analytic Jacobian - finite-difference Jacobian|: {max_diff:.8f}")
    assert max_diff < 1e-3, "Analytic Jacobian does not match finite-difference approximation."
    print("Jacobian verified correct via finite differences.")

    print("\n=== Inverse kinematics convergence check ===")
    theta_true = np.random.uniform(-1.2, 1.2, 6)
    target_pose = arm.forward_kinematics(theta_true)
    print(f"True (hidden) joint angles: {np.round(theta_true, 4)}")

    initial_guess = np.random.uniform(-0.3, 0.3, 6)  # different starting point
    theta_solved, converged, n_iters = arm.inverse_kinematics(
        target_pose, initial_theta=initial_guess
    )
    print(f"IK converged: {converged} in {n_iters} iterations")
    print(f"Solved joint angles:        {np.round(theta_solved, 4)}")

    achieved_pose = arm.forward_kinematics(theta_solved)
    pos_error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3])
    rot_error = np.linalg.norm(
        _rotation_matrix_to_vector(target_pose[:3, :3] @ achieved_pose[:3, :3].T)
    )
    print(f"Final position error:    {pos_error:.8f} m")
    print(f"Final orientation error: {rot_error:.8f} rad")
    assert converged, "IK solver did not converge."
    assert pos_error < 1e-4, "IK position error too large."
    assert rot_error < 1e-4, "IK orientation error too large."
    print("IK verified: converges to the correct end-effector pose "
          "(joint angles may differ -- multiple valid solutions can exist).")

    print("\nAll checks passed.")