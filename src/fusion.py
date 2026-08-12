"""Look-ahead-safe, transaction-cost-aware sentiment fusion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolios import performance_metrics, simulate_targets


def apply_sentiment(
    base_targets: pd.DataFrame,
    sentiment: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    """Tilt monthly base targets using the already-lagged finance signal.

    The cross-section is winsorised at +/-2.5 standard deviations and applied
    multiplicatively with ``exp(strength * z)``.  This preserves long-only,
    fully invested weights without arbitrary clipping.
    """
    signal = sentiment.sort_index()
    records: list[dict] = []
    for date, group in base_targets.groupby("rebalance_date"):
        base = group.set_index("ticker")["weight"].sort_index()
        available = signal.reindex([pd.Timestamp(date)], method="ffill").iloc[0].reindex(base.index).fillna(0.0)
        standard_deviation = float(available.std(ddof=0))
        z = (available - available.mean()) / standard_deviation if standard_deviation > 1e-12 else available * 0.0
        z = z.clip(-2.5, 2.5)
        tilted = base * np.exp(strength * z)
        tilted = tilted / tilted.sum()
        records.extend(
            {
                "rebalance_date": pd.Timestamp(date),
                "ticker": ticker,
                "weight": float(weight),
                "sentiment_z": float(z.loc[ticker]),
                "strength": float(strength),
            }
            for ticker, weight in tilted.items()
        )
    return pd.DataFrame(records)


def tune_strength(
    returns: pd.DataFrame,
    base_targets: pd.DataFrame,
    sentiment: pd.DataFrame,
    strengths: list[float],
    periods_per_year: int,
    discovery_start: str = "2021-01-01",
    discovery_end: str = "2022-12-31",
    holdout_start: str = "2023-01-01",
    holdout_end: str = "2023-12-31",
    transaction_cost_bps: float = 10.0,
) -> tuple[float, pd.DataFrame]:
    """Tune on 2021-2022 net Sharpe and report the untouched 2023 holdout."""
    rows: list[dict] = []
    for strength in strengths:
        targets = apply_sentiment(base_targets, sentiment, strength)
        simulation = simulate_targets(returns, targets, transaction_cost_bps=transaction_cost_bps)
        discovery = simulation.loc[discovery_start:discovery_end, "net_return"]
        holdout = simulation.loc[holdout_start:holdout_end, "net_return"]
        discovery_metrics = performance_metrics(discovery, periods_per_year)
        holdout_metrics = performance_metrics(holdout, periods_per_year)
        rows.append(
            {
                "strength": strength,
                "discovery_net_sharpe": discovery_metrics["sharpe_ratio"],
                "discovery_net_annualized_return": discovery_metrics["annualized_return"],
                "holdout_net_sharpe": holdout_metrics["sharpe_ratio"],
                "holdout_net_annualized_return": holdout_metrics["annualized_return"],
                "total_turnover": simulation["turnover"].sum(),
            }
        )
    table = pd.DataFrame(rows).sort_values("strength").reset_index(drop=True)
    selected = float(table.loc[table["discovery_net_sharpe"].idxmax(), "strength"])
    table["selected_on_discovery"] = table["strength"].eq(selected)
    return selected, table
