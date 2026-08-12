"""Finance-aware headline scoring and look-ahead-safe sentiment indices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import align_headlines_to_trading_days


def build_vader_models():
    """Build plain VADER and finVADER once for fast repeated scoring."""
    import nltk

    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from finvader.Henry import lexicon2
    from finvader.SentiBignomics import lexicon1

    plain = SentimentIntensityAnalyzer()
    finance = SentimentIntensityAnalyzer()
    sentibignomics = {term: value * 0.1 for term, value in lexicon1().items()}
    finance.lexicon.update({**sentibignomics, **lexicon2()})
    return plain, finance


def score_headlines(
    headlines: pd.DataFrame,
    trading_dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Score distinct raw headlines with VADER and finVADER, preserving text."""
    aligned = align_headlines_to_trading_days(headlines, trading_dates)
    unique_titles = pd.Series(pd.unique(aligned["title"].dropna()), name="title")
    plain, finance = build_vader_models()
    score_lookup = pd.DataFrame(
        {
            "title": unique_titles,
            "vader_compound": [plain.polarity_scores(str(text))["compound"] for text in unique_titles],
            "finvader_compound": [finance.polarity_scores(str(text))["compound"] for text in unique_titles],
        }
    )
    scored = aligned.merge(score_lookup, on="title", how="left", validate="many_to_one")
    scored["vader_score_100"] = (scored["vader_compound"] + 1.0) * 50.0
    scored["finvader_score_100"] = (scored["finvader_compound"] + 1.0) * 50.0
    return scored


def ticker_day_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Average headlines within ticker-day before any cross-ticker averaging."""
    return (
        scores.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            vader_compound=("vader_compound", "mean"),
            finvader_compound=("finvader_compound", "mean"),
            headline_count=("title", "size"),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )


def ticker_signal_matrix(
    ticker_day: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    ticker_sector: pd.DataFrame,
    carry_days: int = 5,
    smoothing_days: int = 21,
) -> pd.DataFrame:
    """Build the finance sentiment signal used by funds.

    A missing ticker-day carries the last score for at most ``carry_days`` and
    then becomes neutral.  The filled score is shifted one full trading day and
    only then smoothed, so a Monday-aligned headline is first usable Tuesday.
    """
    tickers = ticker_sector["ticker"].drop_duplicates().sort_values()
    raw = ticker_day.pivot(index="trading_date", columns="ticker", values="finvader_compound")
    raw = raw.reindex(index=pd.DatetimeIndex(trading_dates).sort_values(), columns=tickers)
    carried = raw.ffill(limit=carry_days).fillna(0.0)
    lagged = carried.shift(1).fillna(0.0)
    return lagged.rolling(smoothing_days, min_periods=5).mean().fillna(0.0)


def sector_sentiment_index(
    ticker_day: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    ticker_sector: pd.DataFrame,
    carry_days: int = 5,
) -> pd.DataFrame:
    """Build equal-weight sector indices and a one-day-lagged usable signal."""
    dates = pd.DatetimeIndex(trading_dates).sort_values()
    sectors = ticker_sector["sector"].drop_duplicates().sort_values()
    records: list[pd.DataFrame] = []
    for model in ["vader_compound", "finvader_compound"]:
        raw = ticker_day.pivot(index="trading_date", columns="ticker", values=model)
        raw = raw.reindex(index=dates, columns=ticker_sector["ticker"].drop_duplicates())
        carried = raw.ffill(limit=carry_days).fillna(0.0)
        lagged = carried.shift(1).fillna(0.0)
        long = lagged.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["date", "ticker", "value"]
        long = long.merge(ticker_sector, on="ticker", how="left", validate="many_to_one")
        sector = long.groupby(["date", "sector"], as_index=False)["value"].mean()
        sector = sector.rename(columns={"value": f"{model}_lagged"})
        records.append(sector)

    out = records[0].merge(records[1], on=["date", "sector"], how="outer")
    out["finvader_score_100_lagged"] = (out["finvader_compound_lagged"] + 1.0) * 50.0
    out["finvader_21d"] = out.groupby("sector")["finvader_compound_lagged"].transform(
        lambda series: series.rolling(21, min_periods=5).mean()
    )

    def expanding_z(series: pd.Series) -> pd.Series:
        history = series.expanding(min_periods=60)
        return (series - history.mean()) / history.std()

    out["finvader_expanding_z"] = out.groupby("sector")["finvader_compound_lagged"].transform(expanding_z)
    full_index = pd.MultiIndex.from_product([dates, sectors], names=["date", "sector"])
    return out.set_index(["date", "sector"]).reindex(full_index).reset_index()


def fear_greed_index(ticker_day: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate finance sentiment into historical and expanding fear/greed measures."""
    aggregate = ticker_day.groupby("trading_date", as_index=True)["finvader_compound"].mean()
    aggregate = aggregate.reindex(pd.DatetimeIndex(trading_dates).sort_values())
    aggregate = aggregate.ffill(limit=3).fillna(0.0)
    frame = aggregate.rename("finvader_compound").to_frame()
    frame["fear_greed_100"] = (frame["finvader_compound"] + 1.0) * 50.0
    frame["fear_greed_21d"] = frame["fear_greed_100"].rolling(21, min_periods=5).mean()
    expanding = frame["fear_greed_100"].expanding(min_periods=252)
    frame["expanding_z"] = (frame["fear_greed_100"] - expanding.mean()) / expanding.std()
    full_mean = frame["fear_greed_100"].mean()
    full_std = frame["fear_greed_100"].std()
    frame["historical_full_sample_z"] = (frame["fear_greed_100"] - full_mean) / full_std
    return frame.reset_index(names="date")


def model_validation(scores: pd.DataFrame) -> pd.DataFrame:
    """Quantify coverage and agreement between plain and finance VADER."""
    rows: list[dict] = []
    for label, column in [("Plain VADER", "vader_compound"), ("finVADER", "finvader_compound")]:
        series = scores[column].astype(float)
        rows.append(
            {
                "model": label,
                "headlines": len(series),
                "mean_compound": series.mean(),
                "std_compound": series.std(),
                "neutral_share": (series.abs() < 1e-12).mean(),
                "positive_share": (series > 0).mean(),
                "negative_share": (series < 0).mean(),
            }
        )
    validation = pd.DataFrame(rows)
    both_nonzero = (scores["vader_compound"] != 0) & (scores["finvader_compound"] != 0)
    disagreement = np.sign(scores.loc[both_nonzero, "vader_compound"]) != np.sign(
        scores.loc[both_nonzero, "finvader_compound"]
    )
    validation["sign_disagreement_share_when_both_nonzero"] = float(disagreement.mean())
    return validation
