from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg

from monetary_rl.models import LQBenchmarkModel


@dataclass
class RiccatiSolution:
    P: np.ndarray
    F: np.ndarray
    K: np.ndarray
    closed_loop_A: np.ndarray
    closed_loop_eigenvalues: np.ndarray
    stationary_covariance: np.ndarray
    value_constant: float


def solve_discounted_lq_riccati(model: LQBenchmarkModel) -> RiccatiSolution:
    """Solve the infinite-horizon discounted LQ control problem with cross term N."""

    beta = model.config.discount_factor
    A = model.A
    B = model.B
    Sigma = model.Sigma
    Q, N, R = model.qnr_matrices()

    sqrt_beta = np.sqrt(beta)
    A_tilde = sqrt_beta * A
    B_tilde = sqrt_beta * B

    P = linalg.solve_discrete_are(A_tilde, B_tilde, Q, R, s=N)
    gain_core = R + beta * B.T @ P @ B
    F = np.linalg.solve(gain_core, N.T + beta * B.T @ P @ A)
    K = -F
    closed_loop_A = A + B @ K
    eigvals = np.linalg.eigvals(closed_loop_A)
    process_cov = Sigma @ Sigma.T
    stationary_covariance = linalg.solve_discrete_lyapunov(closed_loop_A, process_cov)
    value_constant = float(beta / (1.0 - beta) * np.trace(P @ process_cov))

    return RiccatiSolution(
        P=P,
        F=F,
        K=K,
        closed_loop_A=closed_loop_A,
        closed_loop_eigenvalues=eigvals,
        stationary_covariance=stationary_covariance,
        value_constant=value_constant,
    )


def build_optimal_linear_policy(solution: RiccatiSolution) -> Callable[[np.ndarray, int], float]:
    """Return a policy callable using u_t = K s_t = -F s_t."""

    def policy(state: np.ndarray, t: int) -> float:
        del t
        state_vec = np.asarray(state, dtype=float).reshape(-1, 1)
        action = float((solution.K @ state_vec).item())
        return action

    return policy
