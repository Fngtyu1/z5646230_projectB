# MarketLens Signal Studio — FINS5545 Project B

This is z5646230's reproducible Station 3–4 project: twelve walk-forward funds,
finance-specific news sentiment, a governed sentiment-fusion experiment, and a
precomputed Streamlit investor dashboard. The 2020 initial window is followed by
monthly out-of-sample evaluation from 2021 to 2023. Reported fund results are net
of a stated 10 bps one-way trading cost unless labelled otherwise.

## How to run

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
    .venv\Scripts\python scripts/run_part_b.py
    .venv\Scripts\python -m pytest -q
    .venv\Scripts\streamlit run streamlit_app.py

Load raw data through src/data_access.py (see context/DATA_GUIDE.md); never commit
raw data. The deployed app, by contrast, reads your precomputed artifacts from
results/ - those ARE committed.

The first build downloads the official hosted course ZIP through
`src/data_access.py` and downloads the NLTK VADER lexicon. Set `FINS_DATA_ZIP` to
the official ZIP URL if the course host changes. Raw data is never committed.
The deployed app does no downloads or modelling; it reads committed CSV files
under `results/`.

## Evidence map

- streamlit_app.py    the app entrypoint (repo root)
- .streamlit/         app config
- PROJECT_BRIEF.md    the full assignment brief for your course (read this first)
- src/                your code (data_access is provided; portfolios/sentiment/fusion are yours)
- scripts/            runnable scripts that reproduce your results
- results/            your outputs: figures in results/figures/, tables in results/tables/, app data artifacts in results/data/
- context/            provided data guide and project context (do not edit)
- report/             your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/                 your prompt logs and AI notes
- `results/tables/performance_metrics.csv`: four methods for equity, crypto, and combined families
- `results/data/fund_returns.csv` and `fund_weights.csv`: fact-sheet series and monthly targets
- `results/data/sector_sentiment_index.csv`: equal-ticker sector signal with a one-day lag
- `results/tables/fusion_strength_tuning.csv`: discovery/holdout separation
- `results/tables/fusion_cost_robustness.csv`: turnover and cost sensitivity
- `report/report.docx` and `report/report.pdf`: editable and submitted reports
- `ai/`: prompts, validation decisions, and authorship notes

## Key methods

Each family includes Equal-Weight, Minimum-Variance, Maximum-Sharpe, and Risk
Parity. Optimised weights are long-only, fully invested, and capped at 20% for
equity/combined or 25% for crypto. Sector sentiment averages headlines within
ticker-day before equally averaging tickers; missing scores carry for at most
five trading days, then return to neutral. finVADER augments VADER with financial
lexicons. A +0.25 multiplicative sentiment tilt is evaluated as an a-priori
candidate, while 2021–2022 is kept separate as a discovery window and 2023 as an
untouched holdout.

## Deploy + hand in

This folder is its own GitHub repository. Before submission:

    .venv\Scripts\python scripts/check_handin.py

The student must personally create/push the GitHub repository, connect
`streamlit_app.py` on Streamlit Community Cloud, make both public at hand-in,
verify them while logged out, and submit both URLs plus the supplied ZIP. Exact
steps are in `docs/FINAL_HANDOFF.md`.
