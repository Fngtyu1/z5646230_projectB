# Student review guide — preserve your own explanation space

The code and exhibits are complete, but the report is a draft you must understand
and own. Before submitting, read every paragraph and rewrite any wording that does
not sound like you. In particular, be ready to explain these five judgments:

1. **Why Equal-Weight is a credible benchmark.** It does not estimate expected
   returns and was competitive in the equity and combined universes.
2. **Why the highest return is not automatically the best product.** Crypto
   Minimum-Variance had the strongest return/Sharpe in this sample but also
   extreme volatility and drawdown.
3. **Why Combined Minimum-Variance resembles Equity Minimum-Variance.** The
   optimiser assigned negligible crypto weight because crypto risk dominated
   the covariance objective; this is an outcome, not a data error.
4. **Why finVADER is an improvement but not proof of alpha.** It greatly reduces
   exact-neutral classifications on financial headlines, yet forecast value must
   still be tested with lags, holdout data, turnover and costs.
5. **Why the innovation conclusion is conservative.** The +0.25 candidate has a
   small low-cost 2023 benefit, but the discovery window selected no tilt and
   higher costs erase the benefit. A responsible product keeps it as research.

Check the report against the CSVs rather than editing numbers manually. If you
change the method or code, rerun `scripts/run_part_b.py`, rebuild the report, and
repeat all checks. Add your own short reflection about what you learned from the
false-positive risk in the fusion experiment; that is the best place to make the
submission genuinely yours.
