"""Fast tests for imports, look-ahead controls, constraints, and artifacts."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access  # noqa: E402
from src.portfolios import performance_metrics, walk_forward_targets  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_imports():
    assert hasattr(data_access, "load_equity_prices")


def test_data_loads():
    try:
        eq = data_access.load_equity_prices()
    except RuntimeError as exc:
        pytest.skip(f"official hosted data unavailable in this environment: {exc}")
    assert eq.shape[0] > 0
    assert {"ticker", "date", "adjClose", "sector"}.issubset(eq.columns)


def test_walk_forward_is_unaffected_by_future_returns():
    dates = pd.bdate_range("2020-01-01", periods=520)
    rng = np.random.default_rng(5545)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(dates), 5)), index=dates, columns=list("ABCDE"))
    changed = returns.copy()
    cutoff = pd.Timestamp("2021-10-01")
    changed.loc[changed.index >= cutoff] = rng.normal(0.1, 0.2, changed.loc[changed.index >= cutoff].shape)
    original_targets = walk_forward_targets(returns, "min_variance", 252, live_start="2021-01-01", max_weight=0.4)
    changed_targets = walk_forward_targets(changed, "min_variance", 252, live_start="2021-01-01", max_weight=0.4)
    left = original_targets.loc[original_targets["rebalance_date"] <= cutoff].reset_index(drop=True)
    right = changed_targets.loc[changed_targets["rebalance_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, atol=1e-9, rtol=1e-9)


def test_required_artifacts_are_coherent():
    required = [
        ROOT / "results/data/fund_returns.csv",
        ROOT / "results/data/fund_weights.csv",
        ROOT / "results/data/sector_sentiment_index.csv",
        ROOT / "results/tables/performance_metrics.csv",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    weights = pd.read_csv(required[1])
    sums = weights.groupby(["fund", "rebalance_date"])["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-8)
    assert weights["weight"].min() >= -1e-10
    metrics = pd.read_csv(required[3])
    assert len(metrics) == 12
    assert metrics[["net_annualized_return", "net_annualized_volatility", "net_sharpe_ratio", "net_max_drawdown"]].notna().all().all()


def test_metrics_known_series():
    metrics = performance_metrics(pd.Series([0.01, -0.005, 0.002]), periods_per_year=252)
    assert metrics["observations"] == 3
    assert metrics["ending_growth"] == pytest.approx(1.01 * 0.995 * 1.002)


if __name__ == "__main__":
    test_imports()
    print("imports OK")
    try:
        test_data_loads()
        print("data load OK")
    except Exception as e:
        print("data load skipped/failed (need network or FINS_DATA_ZIP):", e)
