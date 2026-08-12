"""MarketLens Signal Studio - lightweight investor dashboard.

The deployed app reads precomputed CSV artifacts only. It neither downloads raw
data nor imports NLTK/finVADER or re-runs portfolio optimisation.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

st.set_page_config(page_title="MarketLens Signal Studio", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    :root { --navy:#12263A; --teal:#2A9D8F; --gold:#E9C46A; --coral:#E76F51; }
    .stApp { background:linear-gradient(180deg,#F7F4EF 0%,#FFFFFF 42%); }
    [data-testid="stSidebar"] { background:#12263A; }
    [data-testid="stSidebar"] * { color:#F7F4EF; }
    h1,h2,h3 { color:#12263A; letter-spacing:-.02em; }
    .eyebrow { color:#2A9D8F; font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .hero { border-left:6px solid #2A9D8F; padding:.3rem 0 .3rem 1.1rem; margin-bottom:1rem; }
    .hero p { max-width:900px; color:#536170; font-size:1.05rem; }
    [data-testid="stMetric"] { background:white; border:1px solid #E7E1D8; border-radius:14px; padding:14px 16px; box-shadow:0 4px 16px rgba(18,38,58,.05); }
    .method-card { background:white; border:1px solid #E7E1D8; border-radius:12px; padding:1rem; min-height:128px; }
    .note { background:#EEF7F5; border-left:4px solid #2A9D8F; padding:.8rem 1rem; border-radius:6px; color:#24303F; }
    .warning { background:#FFF4ED; border-left:4px solid #E76F51; padding:.8rem 1rem; border-radius:6px; color:#24303F; }
    div[data-testid="stTabs"] button { font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_artifacts() -> dict[str, pd.DataFrame]:
    files = {
        "fund_returns": DATA / "fund_returns.csv",
        "current_holdings": DATA / "current_holdings.csv",
        "sector_sentiment": DATA / "sector_sentiment_index.csv",
        "fear_greed": DATA / "fear_greed_index.csv",
        "metrics": TABLES / "performance_metrics.csv",
        "fusion_performance": TABLES / "fusion_performance.csv",
        "fusion_robustness": TABLES / "fusion_cost_robustness.csv",
        "sentiment_validation": TABLES / "sentiment_model_validation.csv",
        "backtest_design": TABLES / "backtest_design.csv",
    }
    missing = [str(path.relative_to(ROOT)) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing precomputed artifacts: {missing}. Run python scripts/run_part_b.py.")
    artifacts = {name: pd.read_csv(path) for name, path in files.items()}
    for name in ["fund_returns", "current_holdings", "sector_sentiment", "fear_greed"]:
        for column in ("date", "rebalance_date"):
            if column in artifacts[name]:
                artifacts[name][column] = pd.to_datetime(artifacts[name][column])
    return artifacts


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _metric_row(row: pd.Series) -> None:
    cols = st.columns(4)
    cols[0].metric("Annualised return", _pct(float(row["net_annualized_return"])))
    cols[1].metric("Annualised volatility", _pct(float(row["net_annualized_volatility"])))
    cols[2].metric("Sharpe ratio", f"{float(row['net_sharpe_ratio']):.2f}")
    cols[3].metric("Maximum drawdown", _pct(float(row["net_max_drawdown"])))


def _display_metrics(metrics: pd.DataFrame) -> None:
    shown = metrics[["fund", "net_annualized_return", "net_annualized_volatility", "net_sharpe_ratio", "net_max_drawdown", "annual_turnover"]].rename(
        columns={"fund": "Fund", "net_annualized_return": "Annual return", "net_annualized_volatility": "Volatility", "net_sharpe_ratio": "Sharpe", "net_max_drawdown": "Max drawdown", "annual_turnover": "Annual turnover"}
    )
    formatter = {"Annual return": lambda v: _pct(v), "Volatility": lambda v: _pct(v), "Sharpe": lambda v: f"{v:.2f}", "Max drawdown": lambda v: _pct(v), "Annual turnover": lambda v: f"{v:.2f}x"}
    st.dataframe(shown.style.format(formatter).background_gradient(subset=["Sharpe"], cmap="GnBu"), width="stretch", hide_index=True)


artifacts = _load_artifacts()
fund_returns = artifacts["fund_returns"]
metrics = artifacts["metrics"]

with st.sidebar:
    st.markdown("## ◈ MarketLens")
    st.markdown("**Signal Studio**")
    st.caption("Systematic multi-asset funds + finance-news analytics")
    st.divider()
    st.markdown("**Evidence contract**")
    st.caption("2021-2023 walk-forward OOS · monthly rebalance · 10 bps one-way costs · zero risk-free rate")
    st.divider()
    universe_filter = st.multiselect("Universes in comparison", ["equity", "crypto", "combined"], default=["equity", "crypto", "combined"], format_func=str.title)
    st.caption("Research prototype. Historical results are not investment advice.")

st.markdown(
    """<div class="hero"><div class="eyebrow">FINS5545 · Stations 3–4</div><h1>MarketLens Signal Studio</h1><p>Compare transparent systematic funds, inspect an out-of-sample fact sheet, build a personal fund mix, and read finance-aware news sentiment without recomputing models in the app.</p></div>""",
    unsafe_allow_html=True,
)

tab_compare, tab_fact, tab_allocate, tab_sentiment, tab_method = st.tabs(["Compare funds", "Fund fact sheet", "Allocation lab", "Sentiment & innovation", "Method & limits"])

with tab_compare:
    filtered = metrics.loc[metrics["family"].isin(universe_filter)].copy()
    if filtered.empty:
        st.info("Choose at least one universe in the sidebar.")
    else:
        best = filtered.loc[filtered["net_sharpe_ratio"].idxmax()]
        lowest_risk = filtered.loc[filtered["net_annualized_volatility"].idxmin()]
        cols = st.columns(3)
        cols[0].metric("Funds compared", f"{len(filtered)}")
        cols[1].metric("Highest net Sharpe", f"{best['net_sharpe_ratio']:.2f}", best["fund"])
        cols[2].metric("Lowest realised risk", _pct(lowest_risk["net_annualized_volatility"]), lowest_risk["fund"])
        st.subheader("Evidence table")
        _display_metrics(filtered.sort_values("net_sharpe_ratio", ascending=False))
        st.subheader("Growth after estimated trading costs")
        names = list(filtered["fund"])
        chosen = st.multiselect("Funds on chart", names, default=names[: min(5, len(names))], key="compare_chart")
        chart = fund_returns.loc[fund_returns["fund"].isin(chosen)].pivot(index="date", columns="fund", values="growth_net")
        st.line_chart(chart, height=420, x_label="Date", y_label="Growth of $1")
        st.markdown('<div class="note"><b>Read this as a trade-off.</b> The top-returning crypto funds also carry the deepest drawdowns. Net Sharpe, drawdown and universe should be considered together.</div>', unsafe_allow_html=True)

with tab_fact:
    fund_name = st.selectbox("Open a fund", sorted(metrics["fund"].unique()), index=0)
    selected_metric = metrics.loc[metrics["fund"].eq(fund_name)].iloc[0]
    _metric_row(selected_metric)
    left, right = st.columns([1.45, 1])
    daily = fund_returns.loc[fund_returns["fund"].eq(fund_name)].sort_values("date")
    with left:
        st.subheader("Growth of $1")
        st.line_chart(daily.set_index("date")[["growth_net"]], height=290, color="#2A9D8F", y_label="US dollars")
        st.subheader("Drawdown")
        st.area_chart(daily.set_index("date")[["drawdown_net"]] * 100, height=220, color="#E76F51", y_label="Percent below peak")
    with right:
        st.subheader("Current target holdings")
        holdings = artifacts["current_holdings"].loc[artifacts["current_holdings"]["fund"].eq(fund_name)].nlargest(15, "weight")
        st.bar_chart(holdings.set_index("ticker")[["weight"]] * 100, height=360, color="#E9C46A", horizontal=True, x_label="Target weight (%)")
        if not holdings.empty:
            st.caption(f"Most recent monthly target: {holdings['rebalance_date'].max():%d %b %Y}. Top 15 positions shown.")
        cap = artifacts["backtest_design"].loc[artifacts["backtest_design"]["family"].str.lower().eq(selected_metric["family"]), "max_single_asset_weight"].iloc[0]
        st.markdown(f'<div class="method-card"><b>{fund_name}</b><br><br>Monthly expanding-window walk-forward test. Long-only, fully invested. Single-asset cap: {cap * 100:.0f}%. Net series deducts 10 bps per unit of one-way turnover.</div>', unsafe_allow_html=True)

with tab_allocate:
    st.subheader("Build an illustrative fund-level allocation")
    st.caption("Select two to four funds. Slider amounts are normalised to 100%. This is an illustrative constant-mix estimator, not a recommendation.")
    defaults = [name for name in ["Equity Equal-Weight", "Combined Risk Parity", "Crypto Minimum-Variance"] if name in set(metrics["fund"])]
    allocation_funds = st.multiselect("Funds in allocation", sorted(metrics["fund"].unique()), default=defaults, max_selections=4)
    if not allocation_funds:
        st.info("Select at least one fund to build an allocation.")
    else:
        raw_weights: dict[str, float] = {}
        slider_cols = st.columns(min(4, len(allocation_funds)))
        for position, name in enumerate(allocation_funds):
            raw_weights[name] = slider_cols[position % len(slider_cols)].slider(name, 0, 100, 100 // len(allocation_funds), 5)
        total = sum(raw_weights.values())
        if total <= 0:
            st.warning("At least one slider must be above zero.")
        else:
            normalised = {name: value / total for name, value in raw_weights.items()}
            fee_bps = st.slider("Annual management fee scenario (basis points)", 0, 200, 50, 10)
            wide = fund_returns.loc[fund_returns["fund"].isin(allocation_funds)].pivot(index="date", columns="fund", values="net_return").fillna(0.0)
            mix = sum(wide[name] * weight for name, weight in normalised.items())
            periods = 365 if any("Crypto" in name for name in allocation_funds) else 252
            mix_after_fee = (1.0 + mix) * (1.0 - fee_bps / 10_000 / periods) - 1.0
            growth = (1.0 + mix_after_fee).cumprod()
            drawdown = growth / growth.cummax() - 1.0
            annual_return = growth.iloc[-1] ** (periods / len(growth)) - 1.0
            volatility = mix_after_fee.std() * np.sqrt(periods)
            sharpe = mix_after_fee.mean() / mix_after_fee.std() * np.sqrt(periods)
            cols = st.columns(4)
            cols[0].metric("Normalised allocation", _pct(sum(normalised.values()), 0))
            cols[1].metric("Annualised return", _pct(annual_return))
            cols[2].metric("Annualised volatility", _pct(volatility))
            cols[3].metric("Sharpe ratio", f"{sharpe:.2f}")
            st.line_chart(pd.DataFrame({"Allocation growth": growth}), height=350, y_label="Growth of $1")
            st.dataframe(pd.DataFrame({"Fund": normalised.keys(), "Allocation": normalised.values()}).style.format({"Allocation": "{:.1%}"}), width="stretch", hide_index=True)
            st.caption(f"Worst allocation drawdown: {_pct(drawdown.min())}. Management fee scenario: {fee_bps} bps annually.")

with tab_sentiment:
    st.subheader("Finance-aware headline analytics")
    fear = artifacts["fear_greed"].dropna(subset=["expanding_z"]).sort_values("date")
    latest = fear.iloc[-1]
    label = "Extreme fear" if latest["expanding_z"] < -1.5 else "Fear" if latest["expanding_z"] < -0.5 else "Neutral" if latest["expanding_z"] < 0.5 else "Greed" if latest["expanding_z"] < 1.5 else "Extreme greed"
    validation = artifacts["sentiment_validation"]
    neutral_plain = float(validation.loc[validation["model"].eq("Plain VADER"), "neutral_share"].iloc[0])
    neutral_fin = float(validation.loc[validation["model"].eq("finVADER"), "neutral_share"].iloc[0])
    cols = st.columns(3)
    cols[0].metric("Latest news score", f"{latest['fear_greed_100']:.1f}/100")
    cols[1].metric("Live-standardised reading", f"z = {latest['expanding_z']:.2f}", label)
    cols[2].metric("Exactly-neutral headlines", _pct(neutral_fin), f"Plain VADER: {_pct(neutral_plain)}")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Fear and greed through time")
        st.line_chart(fear.set_index("date")[["fear_greed_21d"]], height=310, color="#2A9D8F", y_label="21-day score (0-100)")
    with right:
        sector = st.selectbox("Sector", sorted(artifacts["sector_sentiment"]["sector"].dropna().unique()))
        sector_data = artifacts["sector_sentiment"].loc[artifacts["sector_sentiment"]["sector"].eq(sector)].set_index("date")
        st.markdown("#### Sector signal")
        st.line_chart(sector_data[["finvader_21d"]], height=310, color="#E76F51", y_label="Lagged compound")
    st.markdown("#### Fusion: evidence before productisation")
    holdout = artifacts["fusion_performance"].loc[artifacts["fusion_performance"]["period"].eq("Untouched holdout")].copy()
    shown = holdout[["model", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "turnover"]].rename(columns={"model": "Model", "annualized_return": "Annual return", "annualized_volatility": "Volatility", "sharpe_ratio": "Sharpe", "max_drawdown": "Max drawdown", "turnover": "Turnover"})
    st.dataframe(shown.style.format({"Annual return": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}", "Turnover": "{:.2f}x"}), width="stretch", hide_index=True)
    st.markdown('<div class="warning"><b>Governance conclusion:</b> the a-priori +0.25 momentum candidate slightly improved the 2023 holdout at 10 bps, but the 2021–2022 discovery grid selected zero tilt. At 25–50 bps the candidate advantage disappears. MarketLens therefore exposes sentiment as research analytics rather than claiming a proven investable edge.</div>', unsafe_allow_html=True)
    cost = st.select_slider("Inspect one-way cost assumption", [0, 5, 10, 25, 50], value=10, format_func=lambda value: f"{value} bps")
    robustness = artifacts["fusion_robustness"]
    cost_table = robustness.loc[(robustness["period"].eq("2023 holdout")) & (robustness["transaction_cost_bps"].eq(cost)), ["model", "annualized_return", "sharpe_ratio", "max_drawdown", "turnover"]]
    st.dataframe(cost_table.style.format({"annualized_return": "{:.1%}", "sharpe_ratio": "{:.2f}", "max_drawdown": "{:.1%}", "turnover": "{:.2f}x"}), width="stretch", hide_index=True)

with tab_method:
    st.subheader("How to read the evidence")
    columns = st.columns(4)
    descriptions = [("Equal-Weight", "One over N; no estimated moments."), ("Minimum-Variance", "Minimises variance from expanding covariance history."), ("Maximum-Sharpe", "Uses expanding means and covariance; estimation-error sensitive."), ("Risk Parity", "Targets equal volatility contribution across assets.")]
    for column, (title, text) in zip(columns, descriptions):
        column.markdown(f'<div class="method-card"><b>{title}</b><br><br>{text}</div>', unsafe_allow_html=True)
    st.markdown("#### Backtest contract")
    st.dataframe(artifacts["backtest_design"], width="stretch", hide_index=True)
    st.markdown('<div class="note"><b>No-look-ahead checks.</b> Every monthly target uses dates strictly before the rebalance date. Returns are computed on native calendars before alignment. Sentiment is mapped to the same/next equity day and shifted one trading day before it enters a decision.</div>', unsafe_allow_html=True)
    st.markdown("#### Limits")
    st.markdown("- Short, regime-specific sample (2020 training; 2021–2023 OOS).\n- Headlines are not full articles; lexical sentiment can misread context.\n- Crypto has no news signal; fusion applies to equities only.\n- Costs are scenarios, not broker-specific estimates.\n- Backtests are historical evidence, not forecasts or guarantees.")
