# AGENTS.md — Project B working contract

This repository is z5646230's FINS5545 Project B (Data Factory Floor Stations
3–4). Read `PROJECT_BRIEF.md`, `context/DATA_GUIDE.md`, and
`context/project_context.md` before editing. Reuse the data definitions and
cleaning logic from Project A; load hosted raw data only through
`src/data_access.py`. Never commit the raw ZIP, extracted raw data, credentials,
or local absolute paths.

## Evidence rules

- Do not invent numbers. Report only values in generated artifacts under
  `results/`, after running `scripts/run_part_b.py`.
- All portfolio weights must be estimated from dates strictly earlier than the
  rebalance date. Use an expanding 2020 initial window, monthly rebalancing,
  and a first live date in 2021.
- Calculate equity and crypto returns on their native calendars before aligning
  crypto to the equity calendar for the combined family.
- Annualise equity and combined results with 252 periods; use 365 for crypto.
- Portfolios are long-only and fully invested. Check weight sums, bounds, drift,
  turnover, and the stated one-way transaction-cost model.
- Align news to the same/next equity trading day, lag it by one complete trading
  day, then smooth. Never use a full-sample standardisation for a live signal.
- Separate discovery (2021–2022) from the untouched 2023 holdout. Preserve a
  null or negative innovation finding instead of selecting on holdout results.

## Product and code rules

- Keep modelling in `src/`; orchestration and chart generation belong in
  `scripts/`; tests belong in `tests/`.
- The deployed `streamlit_app.py` must read committed, precomputed `results/`
  CSV files only. It must not import NLTK/finVADER, download data, or optimise.
- Prefer small, testable functions, deterministic seeds, explicit assumptions,
  helpful errors, and portable paths relative to the repository root.
- Use the MarketLens visual system: navy, teal, warm ivory, gold, and coral.

## Verification and authorship

Run the pipeline, smoke tests, Streamlit app test, and
`scripts/check_handin.py`. Inspect every chart and every rendered report page.
Record prompts, risks, corrections, and rejected alternatives in `ai/`. AI may
draft code and prose, but the student must verify financial logic, rewrite the
interpretation in their own voice, and personally complete GitHub/Streamlit
login and final submission steps.
