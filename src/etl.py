"""Stations 1-2 foundation reused from the student's Project A.

Raw data is loaded only through :mod:`src.data_access`.  The functions return
clean frames plus audit records so Part B can demonstrate that it is using the
same documented foundation rather than silently changing the sample.
"""
from __future__ import annotations

import pandas as pd

from src import data_access

PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]
PRICE_COLS = ["open", "high", "low", "close", "adjClose"]


def normalise_date(series: pd.Series) -> pd.Series:
    """Return timezone-naive, midnight-normalised pandas timestamps."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).dt.normalize()


def _price_integrity(
    df: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    raw_rows = len(df)
    out = df.copy()
    out["date"] = normalise_date(out["date"])
    duplicate_rows = int(out.duplicated(PRICE_KEY, keep=False).sum())
    out = out.drop_duplicates(PRICE_KEY, keep="first")

    nonpositive = int((out[PRICE_COLS] <= 0).any(axis=1).sum())
    negative_volume = int((out["volume"] < 0).sum())
    ohlc_bad = int(
        (
            (out["high"] < out[["open", "close", "low"]].max(axis=1))
            | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    required = PRICE_KEY + PRICE_COLS + ["volume"]
    missing_required = int(out[required].isna().any(axis=1).sum())
    out = out.dropna(subset=required)
    out = out.loc[
        (out[PRICE_COLS] > 0).all(axis=1)
        & (out["volume"] >= 0)
        & (out["high"] >= out[["open", "close", "low"]].max(axis=1))
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1))
    ].sort_values(PRICE_KEY).reset_index(drop=True)

    temporary = out[["ticker", "date", "adjClose"]].copy()
    temporary["return"] = temporary.groupby("ticker", sort=False)["adjClose"].pct_change(
        fill_method=None
    )
    extreme = temporary.loc[temporary["return"].abs() >= 0.20].dropna().copy()
    extreme.insert(0, "dataset", dataset)

    per_ticker = out.groupby("ticker")["date"].agg(["min", "max", "nunique"])
    expected = per_ticker.apply(
        lambda row: len(
            pd.date_range(row["min"], row["max"], freq="D" if dataset == "crypto" else "B")
        ),
        axis=1,
    )
    non_observed_days = int((expected - per_ticker["nunique"]).clip(lower=0).sum())

    summary = [
        {"dataset": dataset, "check": "raw_rows", "value": raw_rows, "resolution": "Loaded through src.data_access"},
        {"dataset": dataset, "check": "duplicate_key_rows", "value": duplicate_rows, "resolution": "Deduplicated on ticker-date"},
        {"dataset": dataset, "check": "missing_required_rows", "value": missing_required, "resolution": "Removed incomplete required records"},
        {"dataset": dataset, "check": "nonpositive_price_rows", "value": nonpositive, "resolution": "Removed invalid price records"},
        {"dataset": dataset, "check": "negative_volume_rows", "value": negative_volume, "resolution": "Removed invalid volume records"},
        {"dataset": dataset, "check": "ohlc_inconsistent_rows", "value": ohlc_bad, "resolution": "Removed internally inconsistent OHLC records"},
        {
            "dataset": dataset,
            "check": "non_observed_calendar_days",
            "value": non_observed_days,
            "resolution": "Audited only; equity weekdays include exchange holidays",
        },
        {
            "dataset": dataset,
            "check": "extreme_return_rows_abs_ge_20pct",
            "value": len(extreme),
            "resolution": "Retained as genuine market observations",
        },
        {"dataset": dataset, "check": "clean_rows", "value": len(out), "resolution": "Final clean panel"},
    ]
    return out, summary, extreme


def load_clean_equities():
    """Return cleaned equities, audit records, and retained extreme returns."""
    return _price_integrity(data_access.load_equity_prices(), "equity")


def load_clean_crypto():
    """Return cleaned crypto capped at 2023-12-31 and its audit records."""
    raw = data_access.load_crypto_prices().copy()
    raw["date"] = normalise_date(raw["date"])
    raw = raw.loc[raw["date"] <= pd.Timestamp("2023-12-31")].copy()
    return _price_integrity(raw, "crypto")


def load_clean_news():
    """Return headline news deduplicated on ticker, date, and title."""
    raw = data_access.load_news_headlines().copy()
    raw_rows = len(raw)
    raw["date"] = normalise_date(raw["date"])
    duplicate_rows = int(raw.duplicated(NEWS_KEY, keep=False).sum())
    missing_title = int(raw["title"].isna().sum())
    out = (
        raw.dropna(subset=["ticker", "date", "title"])
        .drop_duplicates(NEWS_KEY, keep="first")
        .sort_values(["ticker", "date", "title"])
        .reset_index(drop=True)
    )
    summary = [
        {"dataset": "news", "check": "raw_rows", "value": raw_rows, "resolution": "Loaded through src.data_access"},
        {"dataset": "news", "check": "duplicate_headline_rows", "value": duplicate_rows, "resolution": "Deduplicated on ticker-date-title"},
        {"dataset": "news", "check": "missing_title_rows", "value": missing_title, "resolution": "Removed records without a title"},
        {"dataset": "news", "check": "clean_rows", "value": len(out), "resolution": "Final headline panel"},
    ]
    return out, summary
