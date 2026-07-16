# Business Memo: Credit Line Increase Program Targeting

**To:** VP, Card Portfolio Strategy
**From:** [Your Name]
**Re:** Recommendation on risk-based targeting for the proposed Credit Line Increase (CLI) program

## Recommendation

Launch the CLI program, but restrict targeting to the bottom 75% of the
proposed candidate pool by modeled default risk. This trades a 25% smaller
program for a **10.5% larger net expected value** ($130,476 vs. $118,081
annualized) and nearly **double the ROI** (54.8% vs. 31.5%).

## Background

The proposed CLI program targets existing cardholders who are current on
payments but running high balances, on the logic that they're both
low-risk (no delinquencies) and revenue-positive (they'll actually use
more credit if offered). Applying the straightforward business rule,
zero delinquencies in 24 months, utilization at or above 60%, tenure of
at least 12 months, identifies 5,175 eligible customers.

## What the data shows

A logistic regression model trained on the existing portfolio (AUC 0.72,
outperforming a gradient boosting benchmark at 0.71) shows that **utilization
rate is the single strongest predictor of default risk**, more predictive
than credit score itself. This matters here specifically: the CLI-eligible
population was selected *because* of high utilization, which means the
naive targeting rule is unintentionally selecting for a riskier-than-average
sub-population. The eligible pool's actual default rate (11.46%) runs
higher than the portfolio's overall "Low Risk" tier (3.0%), despite every
customer in it having a clean delinquency record.

## The business case

| | Naive SQL Rule | Model-Refined |
|---|---|---|
| Customers targeted | 5,175 | 3,881 |
| Incremental revenue | $492,513 | $368,764 |
| Incremental expected loss | $374,432 | $238,288 |
| **Net expected value** | **$118,081** | **$130,476** |
| ROI | 31.5% | 54.8% |

Removing the riskiest quartile of the eligible pool, identified by the
model, not just the business rule, cuts expected losses by 36% while
only giving up 25% of revenue. The net effect is a smaller, more
profitable program.

## Caveats and what I'd want to test before rollout

- **Behavioral assumption:** this assumes CLI recipients use 15% of their
  new limit as incremental balance. Real utilization response should be
  validated with a small pilot before scaling.
- **Loss severity:** I assumed 100% loss-given-default on incremental
  exposure, deliberately conservative. Real collections/recovery typically
  recovers 20-40% of defaulted balances, so actual losses are likely lower
  than modeled here, meaning this recommendation is a floor, not a ceiling,
  on the program's value.
- **Model scope:** this model was trained on this portfolio's own
  behavioral and demographic data only; a production version would
  incorporate bureau data, which likely improves the AUC meaningfully
  beyond 0.72.

## Bottom line

Model-based refinement isn't just "more sophisticated", it changes the
actual business decision. A team without a risk model would ship the
naive version and leave real money on the table by overexposing to the
riskiest quarter of an already risk-selected population.
