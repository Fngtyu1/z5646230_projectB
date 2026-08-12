# Prompt log — brief, data foundation, and funds

## What I wanted

Complete Station 3 from the actual Project B brief and reuse my Project A data
factory without inventing performance results.

## Prompt(s)

“Read Project B brief, starter and Week 9/10 first. Reuse my own Project A data
foundation. Complete walk-forward out-of-sample funds and reproduce every
number from code.”

## What the assistant produced

ETL and feature modules based on Project A, four long-only methods across equity,
crypto and combined universes, expanding monthly target estimation, a drift-aware
simulation, one-way trading costs, fund returns/weights and fact-sheet metrics.

## What was wrong or risky

Optimised methods could silently use the rebalance-day return, and a combined
calendar merge could turn a weekend crypto move into a false one-day return.
Maximum-Sharpe estimates were also likely to be unstable.

## What I changed and why

I required training rows to satisfy `date < rebalance_date`, calculated returns
on native calendars before alignment, capped weights, reported all four methods
rather than cherry-picking one, and added a synthetic future-data perturbation
test. I kept the weak Combined Maximum-Sharpe result instead of deleting it.
