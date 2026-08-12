"""Walk-forward portfolio construction, simulation, costs, and fact-sheet metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHOD_LABELS = {
    "equal_weight": "Equal-Weight",
    "min_variance": "Minimum-Variance",
    "max_sharpe": "Maximum-Sharpe",
    "risk_parity": "Risk Parity",
}


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    targets: pd.DataFrame
    metrics_gross: dict
    metrics_net: dict


def _clean_training_window(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    keep = clean.columns[clean.notna().sum() >= max(60, int(0.8 * len(clean)))]
    clean = clean[keep].fillna(0.0)
    if clean.shape[1] < 2:
        raise ValueError("at least two assets with sufficient history are required")
    return clean


def estimate_weights(
    returns: pd.DataFrame,
    method: str,
    periods_per_year: int,
    max_weight: float | None = None,
) -> pd.Series:
    """Estimate long-only, fully invested weights from past returns only."""
    if method not in METHOD_LABELS:
        raise ValueError(f"unknown method: {method}")
    train = _clean_training_window(returns)
    assets = train.columns
    n_assets = len(assets)
    cap = max_weight if max_weight is not None else 1.0
    if cap * n_assets < 1.0 - 1e-10:
        raise ValueError("max_weight is infeasible for the number of assets")
    if method == "equal_weight":
        return pd.Series(np.repeat(1.0 / n_assets, n_assets), index=assets, name="weight")

    means = train.mean().to_numpy(dtype=float) * periods_per_year
    covariance = train.cov().to_numpy(dtype=float) * periods_per_year
    ridge = max(float(np.trace(covariance)) / max(n_assets, 1) * 1e-6, 1e-10)
    covariance = covariance + np.eye(n_assets) * ridge
    inverse_vol = 1.0 / np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    x0 = inverse_vol / inverse_vol.sum()
    x0 = np.minimum(x0, cap)
    x0 = x0 / x0.sum()
    bounds = [(0.0, cap) for _ in range(n_assets)]
    constraints = ({"type": "eq", "fun": lambda weights: weights.sum() - 1.0},)

    def variance_objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def sharpe_objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(variance_objective(weights), 1e-14))
        return -float(weights @ means) / volatility

    def risk_parity_objective(weights: np.ndarray) -> float:
        marginal = covariance @ weights
        portfolio_variance = max(float(weights @ marginal), 1e-14)
        contributions = weights * marginal / portfolio_variance
        return float(np.square(contributions - 1.0 / n_assets).sum())

    objective = {
        "min_variance": variance_objective,
        "max_sharpe": sharpe_objective,
        "risk_parity": risk_parity_objective,
    }[method]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"{method} optimisation failed: {result.message}")
    weights = np.clip(result.x, 0.0, cap)
    weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name="weight")


def monthly_rebalance_dates(index: pd.DatetimeIndex, live_start: str = "2021-01-01") -> pd.DatetimeIndex:
    """Return the first observed date in each month on or after ``live_start``."""
    dates = pd.DatetimeIndex(index).sort_values()
    dates = dates[dates >= pd.Timestamp(live_start)]
    if dates.empty:
        raise ValueError("no dates on or after live_start")
    frame = pd.DataFrame(index=dates)
    return pd.DatetimeIndex(frame.groupby(frame.index.to_period("M")).head(1).index)


def walk_forward_targets(
    returns: pd.DataFrame,
    method: str,
    periods_per_year: int,
    live_start: str = "2021-01-01",
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Estimate monthly targets on expanding data strictly before each rebalance."""
    matrix = returns.sort_index().copy()
    records: list[dict] = []
    for rebalance_date in monthly_rebalance_dates(matrix.index, live_start):
        training = matrix.loc[matrix.index < rebalance_date]
        if len(training) < 200:
            continue
        weights = estimate_weights(training, method, periods_per_year, max_weight=max_weight)
        records.extend(
            {"rebalance_date": rebalance_date, "ticker": ticker, "weight": float(weight)}
            for ticker, weight in weights.items()
        )
    targets = pd.DataFrame(records)
    if targets.empty:
        raise ValueError("walk-forward estimation produced no target weights")
    return targets.sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)


def simulate_targets(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Simulate monthly target weights with drift and one-way turnover costs.

    Turnover is 100% at initial formation and half the L1 change against the
    drifted pre-trade holdings thereafter.  Cost is deducted on rebalance days.
    """
    matrix = returns.sort_index().fillna(0.0)
    target_map = {
        pd.Timestamp(date): group.set_index("ticker")["weight"].reindex(matrix.columns).fillna(0.0)
        for date, group in targets.groupby("rebalance_date")
    }
    if not target_map:
        raise ValueError("targets are empty")
    first_live = min(target_map)
    current: pd.Series | None = None
    rows: list[dict] = []
    cost_rate = transaction_cost_bps / 10_000.0

    for date, asset_return in matrix.loc[matrix.index >= first_live].iterrows():
        turnover = 0.0
        if date in target_map:
            target = target_map[date]
            if current is None:
                turnover = float(target.abs().sum())
            else:
                turnover = float(0.5 * (target - current).abs().sum())
            current = target.copy()
        if current is None:
            continue
        gross = float((current * asset_return).sum())
        trading_cost = turnover * cost_rate
        net = (1.0 + gross) * (1.0 - trading_cost) - 1.0
        rows.append(
            {
                "date": date,
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "trading_cost": trading_cost,
            }
        )
        denominator = 1.0 + gross
        if denominator > 1e-12:
            current = current * (1.0 + asset_return) / denominator
            total = current.sum()
            current = current / total if abs(total) > 1e-12 else target_map[max(d for d in target_map if d <= date)].copy()

    result = pd.DataFrame(rows).set_index("date")
    for return_col, prefix in [("gross_return", "gross"), ("net_return", "net")]:
        result[f"growth_{prefix}"] = (1.0 + result[return_col]).cumprod()
        result[f"drawdown_{prefix}"] = result[f"growth_{prefix}"] / result[f"growth_{prefix}"].cummax() - 1.0
    return result


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Return investor fact-sheet metrics using a zero risk-free rate."""
    values = pd.Series(daily_returns).dropna().astype(float)
    if values.empty:
        return {
            "observations": 0,
            "annualized_return": np.nan,
            "arithmetic_annual_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "ending_growth": np.nan,
        }
    growth = (1.0 + values).cumprod()
    annualized_return = float(growth.iloc[-1] ** (periods_per_year / len(values)) - 1.0)
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year)) if values.std(ddof=1) > 0 else np.nan
    drawdown = growth / growth.cummax() - 1.0
    return {
        "observations": int(len(values)),
        "annualized_return": annualized_return,
        "arithmetic_annual_return": float(values.mean() * periods_per_year),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "ending_growth": float(growth.iloc[-1]),
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    periods_per_year: int = 252,
    live_start: str = "2021-01-01",
    max_weight: float | None = None,
    transaction_cost_bps: float = 10.0,
) -> BacktestResult:
    """Run an expanding-window monthly walk-forward out-of-sample backtest."""
    targets = walk_forward_targets(
        returns,
        method,
        periods_per_year,
        live_start=live_start,
        max_weight=max_weight,
    )
    simulated = simulate_targets(returns, targets, transaction_cost_bps=transaction_cost_bps)
    return BacktestResult(
        returns=simulated,
        targets=targets,
        metrics_gross=performance_metrics(simulated["gross_return"], periods_per_year),
        metrics_net=performance_metrics(simulated["net_return"], periods_per_year),
    )
