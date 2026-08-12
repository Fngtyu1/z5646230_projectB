# Prompt log — finance sentiment, fusion, and robustness

## What I wanted

Implement Week 9 finance-aware sentiment and test whether it improves an equity
fund without look-ahead or a misleading “best result” claim.

## Prompt(s)

“Build a sector index by averaging within ticker-day before equal-weighting
tickers. Use finVADER, state missing-day treatment, lag at least one trading day,
compare plain VADER, separate discovery and holdout, and include transaction-cost
and turnover robustness.”

## What the assistant produced

Plain VADER and finVADER scores, ticker-day/sector/fear-and-greed indices, a
multiplicative long-only sentiment tilt, a 2021–2022 strength grid, an untouched
2023 table, and cost scenarios from 0 to 50 bps.

## What was wrong or risky

The +0.25 candidate slightly improved 2023 net Sharpe at 10 bps. Selecting it
after viewing 2023 would contaminate the holdout. Full-sample z-scores would also
use future distribution information.

## What I changed and why

I froze +0.25 as an a-priori mild candidate, required selection to use only
2021–2022, and preserved the discovery result of 0.00. The live signal uses a
one-day lag and expanding historical standardisation. I reported that the small
2023 gain disappears at higher costs instead of calling it a proven edge.
