"""Return features and text assembly reused from Project A."""
from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple returns within ticker on each asset's native calendar."""
    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"daily_returns missing columns: {sorted(missing)}")
    out = prices.sort_values(["ticker", "date"]).copy()
    out["return"] = out.groupby("ticker", sort=False)[price_col].pct_change(fill_method=None)
    return out


def return_matrix(return_frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long return frame to date x ticker and remove the all-missing first row."""
    matrix = return_frame.pivot(index="date", columns="ticker", values="return").sort_index()
    return matrix.dropna(how="all")


def align_returns_to_equity_calendar(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Combine already-computed returns on the equity trading calendar."""
    equity_wide = return_matrix(equity_returns)
    crypto_wide = return_matrix(crypto_returns)
    crypto_aligned = crypto_wide.reindex(equity_wide.index)
    panel = pd.concat([equity_wide, crypto_aligned], axis=1).sort_index()
    panel.index.name = "date"
    return panel


def align_headlines_to_trading_days(
    headlines: pd.DataFrame, trading_dates: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Map every headline to the same or next equity trading day."""
    required = {"date", "ticker", "sector", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines missing columns: {sorted(missing)}")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("trading_dates is empty")
    out = headlines.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    lookup = pd.DataFrame({"trading_date": dates.astype("datetime64[ns]")})
    out = pd.merge_asof(
        out.sort_values("date"),
        lookup,
        left_on="date",
        right_on="trading_date",
        direction="forward",
        allow_exact_matches=True,
    )
    return out.dropna(subset=["trading_date"]).reset_index(drop=True)


def assemble_headline_panel(
    headlines: pd.DataFrame, trading_dates: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Assemble a ticker-sector-trading-day text panel without altering headline text."""
    aligned = align_headlines_to_trading_days(headlines, trading_dates)
    aligned["headline_count"] = 1
    return (
        aligned.groupby(["ticker", "sector", "trading_date"], as_index=False)
        .agg(
            headline_count=("headline_count", "sum"),
            headlines=("title", lambda values: " || ".join(values.astype(str))),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
