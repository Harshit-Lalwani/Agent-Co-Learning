from __future__ import annotations

import numpy as np

from twmacl.portfolio import project_l1_leverage


def markowitz_weights(mu: np.ndarray, cov: np.ndarray, leverage_cap: float) -> np.ndarray:
    """Compute optimal mean-variance portfolio weights and project to leverage cap.

    The unconstrained Markowitz solution is w* = Σ⁻¹μ (ignoring the risk-aversion
    scalar since we scale by the leverage cap anyway). This is the theoretical optimum
    for maximizing the Sharpe ratio of a portfolio given known μ and Σ.
    """
    cov_inv = np.linalg.inv(cov)
    raw_weights = cov_inv @ mu
    return project_l1_leverage(raw_weights, leverage_cap)


def evaluate_markowitz(
    mu: np.ndarray,
    cov: np.ndarray,
    leverage_cap: float,
    num_steps: int,
    seed: int,
    steps_per_year: float = 1.0,
) -> dict[str, float]:
    """Evaluate the Markowitz portfolio over a simulated episode.

    Returns:
        Dict with keys: sharpe, annualized_sharpe, cumulative_return,
        log_cumulative_return, and the weights used.
    """
    weights = markowitz_weights(mu, cov, leverage_cap)
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(num_steps):
        r = rng.multivariate_normal(mu, cov)
        port_return = float(weights @ r)
        returns.append(port_return)

    returns_arr = np.array(returns)
    mean_r = returns_arr.mean()
    std_r = returns_arr.std()
    sharpe = mean_r / std_r if std_r > 1e-12 else 0.0
    annualized_sharpe = sharpe * np.sqrt(steps_per_year)
    cumulative = returns_arr.sum()
    log_cumulative = np.sum(np.log1p(np.clip(returns_arr, -0.9999, None)))

    return {
        "sharpe": float(sharpe),
        "annualized_sharpe": float(annualized_sharpe),
        "cumulative_return": float(cumulative),
        "log_cumulative_return": float(log_cumulative),
        "weights": weights,
    }


def markowitz_sharpe_analytical(mu: np.ndarray, cov: np.ndarray) -> float:
    """Compute the theoretical Sharpe of the Markowitz portfolio analytically.

    For w = Σ⁻¹μ, the portfolio Sharpe is sqrt(μᵀ Σ⁻¹ μ).
    This is the true population Sharpe (no sampling noise).
    """
    cov_inv = np.linalg.inv(cov)
    return float(np.sqrt(mu @ cov_inv @ mu))
