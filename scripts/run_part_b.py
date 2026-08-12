"""Reproduce FINS5545 Project B results from the official hosted data.

Run from the project root::

    python scripts/run_part_b.py

The script reuses the student's Project A cleaning and feature logic, runs all
walk-forward funds, builds the finance-aware sentiment index and disciplined
fusion extension, and writes only derived artifacts under ``results/``.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(__file__).resolve().parent.parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, features, fusion, portfolios, sentiment  # noqa: E402

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"
SOURCE = "Source: FINS5545 course project data; student calculations. OOS sample: 2021-2023."
COLORS = {
    "navy": "#12263A",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#E76F51",
    "blue": "#457B9D",
    "ink": "#24303F",
    "muted": "#6B7280",
    "paper": "#F7F4EF",
    "grid": "#D8D2C8",
}
METHOD_COLORS = {
    "equal_weight": COLORS["blue"],
    "min_variance": COLORS["teal"],
    "max_sharpe": COLORS["coral"],
    "risk_parity": COLORS["gold"],
}
FAMILY_LABELS = {"equity": "Equity", "crypto": "Crypto", "combined": "Combined"}


def _prepare_dirs() -> None:
    for folder in (TABLES, FIGURES, DATA, ROOT / ".mplconfig"):
        folder.mkdir(parents=True, exist_ok=True)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["navy"],
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "savefig.facecolor": COLORS["paper"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def _savefig(
    fig: plt.Figure,
    filename: str,
    source: str = SOURCE,
    rect: tuple[float, float, float, float] = (0, 0.04, 1, 0.97),
) -> None:
    fig.text(0.01, 0.01, source, ha="left", va="bottom", fontsize=7.5, color=COLORS["muted"])
    fig.tight_layout(rect=rect)
    fig.savefig(FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fund_name(family: str, method: str) -> str:
    return f"{FAMILY_LABELS[family]} {portfolios.METHOD_LABELS[method]}"


def _run_funds(
    equity: pd.DataFrame,
    crypto: pd.DataFrame,
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, portfolios.BacktestResult]]:
    universes = {"equity": equity, "crypto": crypto, "combined": combined}
    annualisation = {"equity": 252, "crypto": 365, "combined": 252}
    max_weight = {"equity": 0.20, "crypto": 0.25, "combined": 0.20}
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
    return_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    outputs: dict[str, portfolios.BacktestResult] = {}

    for family, matrix in universes.items():
        for method in methods:
            name = _fund_name(family, method)
            print(f"Backtesting {name} ...")
            result = portfolios.oos_backtest(
                matrix,
                method=method,
                periods_per_year=annualisation[family],
                live_start="2021-01-01",
                max_weight=max_weight[family],
                transaction_cost_bps=10.0,
            )
            outputs[name] = result
            daily = result.returns.reset_index().copy()
            daily.insert(1, "fund", name)
            daily.insert(2, "family", family)
            daily.insert(3, "method", method)
            daily.insert(4, "periods_per_year", annualisation[family])
            return_rows.append(daily)

            targets = result.targets.copy()
            targets.insert(0, "fund", name)
            targets.insert(1, "family", family)
            targets.insert(2, "method", method)
            targets["asset_class"] = np.where(targets["ticker"].str.endswith("-USD"), "Crypto", "Equity")
            weight_rows.append(targets)

            row = {
                "fund": name,
                "family": family,
                "method": method,
                "live_start": daily["date"].min(),
                "live_end": daily["date"].max(),
                "training_start": matrix.index.min(),
                "initial_training_observations": int((matrix.index < daily["date"].min()).sum()),
                "rebalances": targets["rebalance_date"].nunique(),
                "periods_per_year": annualisation[family],
                "risk_free_rate": 0.0,
                "transaction_cost_bps": 10.0,
                "annual_turnover": daily["turnover"].sum() / (len(daily) / annualisation[family]),
            }
            row.update({f"gross_{key}": value for key, value in result.metrics_gross.items()})
            row.update({f"net_{key}": value for key, value in result.metrics_net.items()})
            metric_rows.append(row)

    fund_returns = pd.concat(return_rows, ignore_index=True)
    fund_weights = pd.concat(weight_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["family", "method"]).reset_index(drop=True)
    return fund_returns, fund_weights, metrics, outputs


def _add_sector_labels(weights: pd.DataFrame, equity_universe: pd.DataFrame) -> pd.DataFrame:
    labelled = weights.merge(equity_universe, on="ticker", how="left", validate="many_to_one")
    labelled["allocation_group"] = np.where(labelled["asset_class"].eq("Crypto"), "Crypto", labelled["sector"])
    return labelled.drop(columns="sector")


def _sanity_checks(metrics: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict] = []
    for (fund, date), group in weights.groupby(["fund", "rebalance_date"]):
        checks.append(
            {
                "fund": fund,
                "rebalance_date": date,
                "weight_sum": group["weight"].sum(),
                "min_weight": group["weight"].min(),
                "max_weight": group["weight"].max(),
            }
        )
    table = pd.DataFrame(checks)
    if not np.allclose(table["weight_sum"], 1.0, atol=1e-8):
        raise AssertionError("portfolio weights do not sum to one")
    if (table["min_weight"] < -1e-10).any():
        raise AssertionError("long-only constraint was violated")
    pivot = metrics.pivot(index="family", columns="method", values="net_sharpe_ratio")
    if pivot.nunique(axis=1).eq(1).any():
        raise AssertionError("optimisation methods produced indistinguishable metrics")
    return table


def _sentiment_outputs(
    news: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Scoring distinct headlines with VADER and finVADER ...")
    scores = sentiment.score_headlines(news, trading_dates)
    ticker_day = sentiment.ticker_day_sentiment(scores)
    sector_index = sentiment.sector_sentiment_index(ticker_day, trading_dates, universe)
    fear_greed = sentiment.fear_greed_index(ticker_day, trading_dates)
    validation = sentiment.model_validation(scores)
    ticker_signal = sentiment.ticker_signal_matrix(ticker_day, trading_dates, universe)
    return scores, ticker_day, sector_index, fear_greed, validation, ticker_signal


def _run_fusion(
    equity_returns: pd.DataFrame,
    base_result: portfolios.BacktestResult,
    ticker_signal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strengths = [-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75]
    selected, tuning = fusion.tune_strength(
        equity_returns,
        base_result.targets,
        ticker_signal,
        strengths=strengths,
        periods_per_year=252,
        transaction_cost_bps=10.0,
    )
    # The required before/after attempt is a mild +0.25 momentum tilt specified
    # before reading the holdout.  The separate grid above is a robustness and
    # governance exercise; it may rationally select zero exposure.
    candidate_strength = 0.25
    tilted_targets = fusion.apply_sentiment(base_result.targets, ticker_signal, candidate_strength)
    tilted = portfolios.simulate_targets(equity_returns, tilted_targets, transaction_cost_bps=10.0)
    base = base_result.returns.copy()

    comparisons: list[dict] = []
    periods = {
        "Full OOS (descriptive; strength tuned within period)": ("2021-01-01", "2023-12-31"),
        "Discovery (strength selected here)": ("2021-01-01", "2022-12-31"),
        "Untouched holdout": ("2023-01-01", "2023-12-31"),
    }
    for period, (start, end) in periods.items():
        for model, frame in [("Base Equity Minimum-Variance", base), ("Finance-Sentiment Tilt", tilted)]:
            values = frame.loc[start:end, "net_return"]
            metrics = portfolios.performance_metrics(values, 252)
            comparisons.append(
                {
                    "period": period,
                    "model": model,
                    "candidate_strength": candidate_strength,
                    "discovery_selected_strength": selected,
                    "transaction_cost_bps": 10.0,
                    **metrics,
                    "turnover": frame.loc[start:end, "turnover"].sum(),
                }
            )

    robustness: list[dict] = []
    for cost_bps in [0, 5, 10, 25, 50]:
        base_cost = portfolios.simulate_targets(equity_returns, base_result.targets, cost_bps)
        tilt_cost = portfolios.simulate_targets(equity_returns, tilted_targets, cost_bps)
        for model, frame in [("Base", base_cost), ("Sentiment tilt", tilt_cost)]:
            for period, (start, end) in {
                "Full OOS": ("2021-01-01", "2023-12-31"),
                "2023 holdout": ("2023-01-01", "2023-12-31"),
            }.items():
                metrics = portfolios.performance_metrics(frame.loc[start:end, "net_return"], 252)
                robustness.append(
                    {
                        "model": model,
                        "period": period,
                        "transaction_cost_bps": cost_bps,
                        "candidate_strength": candidate_strength,
                        "discovery_selected_strength": selected,
                        **metrics,
                        "turnover": frame.loc[start:end, "turnover"].sum(),
                    }
                )

    fusion_returns = pd.concat(
        [
            base.reset_index().assign(model="Base Equity Minimum-Variance"),
            tilted.reset_index().assign(model="A-priori +0.25 Finance-Sentiment Tilt"),
        ],
        ignore_index=True,
    )
    return (
        fusion_returns,
        tilted_targets,
        tuning,
        pd.DataFrame(comparisons),
        pd.DataFrame(robustness),
    )


def _plot_growth(fund_returns: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=False)
    for ax, family in zip(axes, ["equity", "crypto", "combined"]):
        data = fund_returns.loc[fund_returns["family"].eq(family)]
        for method, group in data.groupby("method"):
            ax.plot(group["date"], group["growth_net"], label=portfolios.METHOD_LABELS[method], color=METHOD_COLORS[method], lw=1.55)
        ax.axhline(1.0, color=COLORS["muted"], lw=0.8, ls="--")
        ax.set_title(f"{FAMILY_LABELS[family]} funds")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1 after 10 bps costs")
        ax.grid(axis="y")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Walk-forward funds diverged materially across methods and universes", fontsize=15, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "growth_of_one_dollar.png")


def _plot_drawdown(fund_returns: pd.DataFrame) -> None:
    data = fund_returns.loc[fund_returns["family"].eq("combined")]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for method, group in data.groupby("method"):
        ax.fill_between(group["date"], group["drawdown_net"] * 100, 0, color=METHOD_COLORS[method], alpha=0.11)
        ax.plot(group["date"], group["drawdown_net"] * 100, label=portfolios.METHOD_LABELS[method], color=METHOD_COLORS[method], lw=1.4)
    ax.set_title("Combined funds experienced different paths below their prior peaks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown after 10 bps costs (%)")
    ax.grid(axis="y")
    ax.legend(ncol=2, fontsize=8)
    _savefig(fig, "combined_fund_drawdown.png")


def _plot_weights(weights: pd.DataFrame) -> None:
    data = weights.loc[weights["family"].eq("combined")].copy()
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
    groups = list(data["allocation_group"].dropna().unique())
    preferred = ["Healthcare", "Consumer", "Utilities", "Real Estate", "Tech", "Financials", "Industrials", "Energy", "Materials", "Comm/Telecom", "Crypto"]
    groups = [group for group in preferred if group in groups] + [group for group in groups if group not in preferred]
    palette = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), methods):
        subset = data.loc[data["method"].eq(method)]
        wide = subset.pivot_table(index="rebalance_date", columns="allocation_group", values="weight", aggfunc="sum").fillna(0.0)
        wide = wide.reindex(columns=groups, fill_value=0.0)
        ax.stackplot(wide.index, *(wide[column] * 100 for column in wide.columns), labels=wide.columns, colors=palette, alpha=0.88)
        ax.set_title(portfolios.METHOD_LABELS[method])
        ax.set_ylabel("Target allocation (%)")
        ax.set_ylim(0, 100)
        ax.grid(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=6, fontsize=7.2)
    fig.suptitle("Combined-fund target allocations changed at each monthly decision", fontsize=15, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "combined_weights_over_time.png", rect=(0, 0.04, 1, 0.86))


def _plot_sharpe(metrics: pd.DataFrame) -> None:
    order = ["equity", "crypto", "combined"]
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
    fig, ax = plt.subplots(figsize=(11, 5.7))
    x = np.arange(len(order))
    width = 0.19
    for position, method in enumerate(methods):
        values = [float(metrics.loc[(metrics["family"].eq(family)) & (metrics["method"].eq(method)), "net_sharpe_ratio"].iloc[0]) for family in order]
        bars = ax.bar(x + (position - 1.5) * width, values, width, label=portfolios.METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(0, color=COLORS["muted"], lw=0.8)
    ax.set_xticks(x, [FAMILY_LABELS[value] for value in order])
    ax.set_ylabel("Net out-of-sample Sharpe ratio")
    ax.set_title("Risk-adjusted performance depended on both the universe and the rule")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y")
    _savefig(fig, "sharpe_by_fund_and_method.png")


def _plot_risk_return(metrics: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9.5, 6))
    markers = {"equity": "o", "crypto": "^", "combined": "s"}
    for _, row in metrics.iterrows():
        ax.scatter(row["net_annualized_volatility"] * 100, row["net_annualized_return"] * 100, s=78, color=METHOD_COLORS[row["method"]], marker=markers[row["family"]], edgecolor="white", linewidth=0.7)
        if row["family"] == "combined":
            ax.annotate(
                portfolios.METHOD_LABELS[row["method"]],
                (row["net_annualized_volatility"] * 100, row["net_annualized_return"] * 100),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.2,
            )
    ax.set_xlabel("Annualised volatility after costs (%)")
    ax.set_ylabel("Annualised compound return after costs (%)")
    ax.set_title("Fund choice traded annual return against realised risk")
    ax.grid()
    method_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS[method], label=portfolios.METHOD_LABELS[method], markersize=7) for method in METHOD_COLORS]
    family_handles = [Line2D([0], [0], marker=markers[family], color=COLORS["ink"], linestyle="none", label=FAMILY_LABELS[family], markersize=7) for family in markers]
    first = ax.legend(handles=method_handles, loc="upper left", title="Method", fontsize=7.5, title_fontsize=8)
    ax.add_artist(first)
    ax.legend(handles=family_handles, loc="lower right", title="Universe", fontsize=7.5, title_fontsize=8)
    _savefig(fig, "risk_return_map.png")


def _plot_sentiment(sector_index: pd.DataFrame) -> None:
    sectors = list(sector_index["sector"].drop_duplicates())
    fig, axes = plt.subplots(5, 2, figsize=(12, 11), sharex=True, sharey=True)
    for ax, sector in zip(axes.ravel(), sectors):
        data = sector_index.loc[sector_index["sector"].eq(sector)]
        ax.axhline(0, color=COLORS["muted"], lw=0.7, ls="--")
        ax.plot(data["date"], data["finvader_21d"], color=COLORS["teal"], lw=1.2)
        ax.fill_between(data["date"], data["finvader_21d"], 0, where=data["finvader_21d"].ge(0), color=COLORS["teal"], alpha=0.18)
        ax.fill_between(data["date"], data["finvader_21d"], 0, where=data["finvader_21d"].lt(0), color=COLORS["coral"], alpha=0.18)
        ax.set_title(sector, fontsize=9)
        ax.set_ylabel("Lagged finVADER\ncompound")
        ax.grid(axis="y")
    fig.suptitle("Finance-aware sector sentiment moved through distinct cycles", fontsize=15, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "sector_sentiment_index.png", "Source: FINS5545 news headlines; finVADER; ticker-equal-weighted; one trading-day lag; 21-day mean. 2020-2023.")


def _plot_fear_greed(fear_greed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    axes[0].plot(fear_greed["date"], fear_greed["fear_greed_21d"], color=COLORS["teal"], lw=1.4)
    axes[0].axhline(50, color=COLORS["muted"], ls="--", lw=0.8)
    axes[0].set_ylabel("Fear/greed score (0-100)")
    axes[0].set_title("Raw levels have a positive baseline")
    axes[1].plot(fear_greed["date"], fear_greed["expanding_z"], color=COLORS["coral"], lw=1.2)
    axes[1].axhline(0, color=COLORS["muted"], ls="--", lw=0.8)
    axes[1].axhspan(-4, -1.5, color=COLORS["coral"], alpha=0.12)
    axes[1].axhspan(1.5, 4, color=COLORS["teal"], alpha=0.12)
    axes[1].set_ylabel("Expanding-window z-score")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Live-standardisation reveals unusually fearful and greedy regimes")
    fig.suptitle("MarketLens finance-news fear and greed index", fontsize=15, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "fear_greed_index.png", "Source: FINS5545 news headlines; finVADER; full market average. 2020-2023.")


def _plot_model_validation(validation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(validation["model"], validation["neutral_share"] * 100, color=[COLORS["blue"], COLORS["teal"]])
    axes[0].set_ylabel("Exactly neutral headlines (%)")
    axes[0].set_title("Finance vocabulary changes lexical coverage")
    axes[1].bar(validation["model"], validation["mean_compound"], color=[COLORS["blue"], COLORS["teal"]])
    axes[1].axhline(0, color=COLORS["muted"], lw=0.8)
    axes[1].set_ylabel("Mean compound score")
    axes[1].set_title("Model choice shifts the measured tone")
    fig.suptitle("Plain VADER and finVADER do not read the same headlines identically", fontsize=13.5, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "sentiment_model_validation.png", "Source: FINS5545 news headlines; distinct text scored by VADER and finVADER. 2020-2023.")


def _plot_fusion(fusion_returns: pd.DataFrame, candidate_strength: float, selected_strength: float) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    holdout = fusion_returns.loc[fusion_returns["date"].between("2023-01-01", "2023-12-31")].copy()
    for model, group in holdout.groupby("model"):
        growth = (1.0 + group.set_index("date")["net_return"]).cumprod()
        color = COLORS["navy"] if model.startswith("Base") else COLORS["coral"]
        ax.plot(growth.index, growth, label=model, color=color, lw=1.6)
    ax.axhline(1.0, color=COLORS["muted"], lw=0.8, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 after 10 bps costs")
    ax.set_title(
        f"A-priori sentiment fusion on the untouched 2023 holdout (candidate {candidate_strength:+.2f}; discovery selected {selected_strength:+.2f})"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    _savefig(fig, "fusion_holdout_comparison.png", "Source: FINS5545 equity returns and lagged finVADER; +0.25 candidate specified a priori; discovery grid uses 2021-2022. 2023 holdout.")


def _plot_cost_robustness(robustness: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, period in zip(axes, ["Full OOS", "2023 holdout"]):
        for model, group in robustness.loc[robustness["period"].eq(period)].groupby("model"):
            color = COLORS["navy"] if model == "Base" else COLORS["coral"]
            ax.plot(group["transaction_cost_bps"], group["sharpe_ratio"], marker="o", label=model, color=color)
        ax.set_title(period)
        ax.set_xlabel("One-way transaction cost (basis points)")
        ax.grid()
    axes[0].set_ylabel("Net Sharpe ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("The sentiment result must survive the cost and turnover it creates", fontsize=14, fontweight="bold", color=COLORS["navy"])
    _savefig(fig, "fusion_transaction_cost_robustness.png", "Source: FINS5545 equity returns and lagged finVADER; monthly rebalance; cost scenarios 0-50 bps.")


def main() -> None:
    _prepare_dirs()
    _set_style()

    eq, eq_checks, eq_extreme = etl.load_clean_equities()
    cr, cr_checks, cr_extreme = etl.load_clean_crypto()
    news, news_checks = etl.load_clean_news()
    eq_long = features.daily_returns(eq)
    cr_long = features.daily_returns(cr)
    eq_returns = features.return_matrix(eq_long)
    cr_returns = features.return_matrix(cr_long)
    combined_returns = features.align_returns_to_equity_calendar(eq_long, cr_long).dropna(how="all")

    fund_returns, fund_weights, metrics, backtests = _run_funds(eq_returns, cr_returns, combined_returns)
    universe = eq[["ticker", "sector"]].drop_duplicates().sort_values("ticker")
    fund_weights = _add_sector_labels(fund_weights, universe)
    weight_checks = _sanity_checks(metrics, fund_weights)

    scores, ticker_day, sector_index, fear_greed, validation, ticker_signal = _sentiment_outputs(
        news, eq_returns.index, universe
    )
    base_name = "Equity Minimum-Variance"
    fusion_returns, fusion_weights, tuning, fusion_performance, robustness = _run_fusion(
        eq_returns, backtests[base_name], ticker_signal
    )
    selected_strength = float(tuning.loc[tuning["selected_on_discovery"], "strength"].iloc[0])
    candidate_strength = 0.25

    integrity = pd.DataFrame(eq_checks + cr_checks + news_checks)
    extreme = pd.concat([eq_extreme, cr_extreme], ignore_index=True)
    design = pd.DataFrame(
        [
            {"family": "Equity", "assets": eq_returns.shape[1], "calendar": "Equity trading days", "periods_per_year": 252, "initial_window": "2020 expanding", "first_live_date": fund_returns.loc[fund_returns["family"].eq("equity"), "date"].min(), "rebalance": "First observed day monthly", "max_single_asset_weight": 0.20},
            {"family": "Crypto", "assets": cr_returns.shape[1], "calendar": "Calendar days", "periods_per_year": 365, "initial_window": "2020 expanding", "first_live_date": fund_returns.loc[fund_returns["family"].eq("crypto"), "date"].min(), "rebalance": "First observed day monthly", "max_single_asset_weight": 0.25},
            {"family": "Combined", "assets": combined_returns.shape[1], "calendar": "Equity trading days", "periods_per_year": 252, "initial_window": "2020 expanding", "first_live_date": fund_returns.loc[fund_returns["family"].eq("combined"), "date"].min(), "rebalance": "First observed day monthly", "max_single_asset_weight": 0.20},
        ]
    )
    current_holdings = fund_weights.merge(
        fund_weights.groupby("fund")["rebalance_date"].max().rename("rebalance_date"),
        on=["fund", "rebalance_date"],
        how="inner",
    ).sort_values(["fund", "weight"], ascending=[True, False])
    top_holdings = (
        current_holdings.groupby("fund", group_keys=False)
        .head(5)
        .groupby("fund")
        .apply(
            lambda group: "; ".join(
                f"{ticker} {weight:.1%}" for ticker, weight in zip(group["ticker"], group["weight"])
            ),
            include_groups=False,
        )
        .rename("top_five_current_holdings")
    )
    fact_sheets = metrics.merge(top_holdings, on="fund", how="left", validate="one_to_one")

    fund_returns.to_csv(DATA / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA / "fund_weights.csv", index=False)
    sector_index.to_csv(DATA / "sector_sentiment_index.csv", index=False)
    fear_greed.to_csv(DATA / "fear_greed_index.csv", index=False)
    fusion_returns.to_csv(DATA / "fusion_returns.csv", index=False)
    fusion_weights.to_csv(DATA / "fusion_weights.csv", index=False)
    current_holdings.to_csv(DATA / "current_holdings.csv", index=False)
    ticker_day.to_csv(DATA / "ticker_day_sentiment.csv", index=False)

    metrics.to_csv(TABLES / "performance_metrics.csv", index=False)
    fact_sheets.to_csv(TABLES / "fund_fact_sheets.csv", index=False)
    design.to_csv(TABLES / "backtest_design.csv", index=False)
    integrity.to_csv(TABLES / "reused_part_a_integrity_summary.csv", index=False)
    extreme.to_csv(TABLES / "reused_part_a_extreme_returns.csv", index=False)
    validation.to_csv(TABLES / "sentiment_model_validation.csv", index=False)
    tuning.to_csv(TABLES / "fusion_strength_tuning.csv", index=False)
    fusion_performance.to_csv(TABLES / "fusion_performance.csv", index=False)
    robustness.to_csv(TABLES / "fusion_cost_robustness.csv", index=False)
    weight_checks.to_csv(TABLES / "portfolio_constraint_checks.csv", index=False)

    _plot_growth(fund_returns)
    _plot_drawdown(fund_returns)
    _plot_weights(fund_weights)
    _plot_sharpe(metrics)
    _plot_risk_return(metrics)
    _plot_sentiment(sector_index)
    _plot_fear_greed(fear_greed)
    _plot_model_validation(validation)
    _plot_fusion(fusion_returns, candidate_strength, selected_strength)
    _plot_cost_robustness(robustness)

    print(f"Clean data: equities={eq.shape}, crypto={cr.shape}, headlines={news.shape}")
    print(f"Fund observations={len(fund_returns):,}; target weights={len(fund_weights):,}")
    print(f"Selected sentiment strength on 2021-2022: {selected_strength:+.2f}")
    print(f"A-priori mild momentum candidate evaluated before selection: {candidate_strength:+.2f}")
    print(f"Wrote reproducible artifacts to {ROOT / 'results'}")


if __name__ == "__main__":
    main()
