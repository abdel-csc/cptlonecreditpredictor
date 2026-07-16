"""
Business impact of a Credit Line Increase (CLI) program.

ASSUMPTIONS (stated explicitly, this is the part you must be able to
defend -- every number here is a judgment call, not a fact):

1. CLI amount: +30% of current credit limit.
2. Behavioral response: customers use 15% of the new incremental limit
   as additional revolving balance. This is a conservative assumption,
   real CLI programs see anywhere from 10-25% utilization of new limit
   in the first year; 15% sits in the middle.
3. Revenue: incremental balance * current APR (interest income proxy).
   Real card economics also include interchange revenue, but interest
   income is the dominant, most directly attributable piece for a CLI
   specifically, so it's the right thing to isolate.
4. Loss exposure: incremental balance * predicted 12mo default probability,
   assuming 100% loss-given-default on the incremental exposure. This is
   deliberately conservative (real LGD after collections/recovery is
   usually 60-80%, not 100%) -- if the program still looks good under
   a conservative loss assumption, that's a stronger recommendation.

TWO TARGETING STRATEGIES COMPARED:
A) Naive: the SQL rule from the segmentation step (0 delinquencies,
   utilization >= 60%, tenure >= 12mo). This is what a team WITHOUT a
   risk model would ship.
B) Model-refined: same pool, but excluding the top risk quartile by
   predicted_default_prob. This is what the risk model actually buys you.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/capone_project/data/portfolio_scored.csv")

CLI_PCT = 0.30
UTILIZATION_OF_NEW_LIMIT = 0.15
LGD = 1.00  # conservative; see assumptions above

candidates = df[
    (df["num_delinquencies_24mo"] == 0)
    & (df["utilization_rate"] >= 0.60)
    & (df["tenure_months"] >= 12)
].copy()


def evaluate_segment(segment: pd.DataFrame, label: str) -> dict:
    incremental_limit = segment["credit_limit"] * CLI_PCT
    incremental_balance = incremental_limit * UTILIZATION_OF_NEW_LIMIT
    incremental_revenue = incremental_balance * (segment["current_apr"] / 100)
    incremental_expected_loss = incremental_balance * segment["predicted_default_prob"] * LGD
    net_value = incremental_revenue - incremental_expected_loss

    result = {
        "strategy": label,
        "num_customers": len(segment),
        "total_incremental_balance": incremental_balance.sum(),
        "total_incremental_revenue": incremental_revenue.sum(),
        "total_incremental_expected_loss": incremental_expected_loss.sum(),
        "net_expected_value": net_value.sum(),
        "avg_predicted_default_prob": segment["predicted_default_prob"].mean(),
        "roi_pct": (net_value.sum() / incremental_expected_loss.sum() * 100)
                   if incremental_expected_loss.sum() > 0 else float("nan"),
    }
    return result


# Strategy A: naive SQL rule, no model refinement
result_naive = evaluate_segment(candidates, "A: Naive SQL rule")

# Strategy B: model-refined, drop the riskiest quartile within the candidate pool
risk_cutoff = candidates["predicted_default_prob"].quantile(0.75)
refined = candidates[candidates["predicted_default_prob"] <= risk_cutoff]
result_refined = evaluate_segment(refined, "B: Model-refined (drop riskiest 25%)")

comparison = pd.DataFrame([result_naive, result_refined])
comparison_display = comparison.copy()
for col in ["total_incremental_balance", "total_incremental_revenue",
            "total_incremental_expected_loss", "net_expected_value"]:
    comparison_display[col] = comparison_display[col].map(lambda x: f"${x:,.0f}")
comparison_display["avg_predicted_default_prob"] = comparison_display["avg_predicted_default_prob"].map(lambda x: f"{x:.2%}")
comparison_display["roi_pct"] = comparison_display["roi_pct"].map(lambda x: f"{x:.1f}%")

print("=" * 90)
print("CREDIT LINE INCREASE PROGRAM: STRATEGY COMPARISON (portfolio-wide, annualized)")
print("=" * 90)
print(comparison_display.to_string(index=False))

customers_excluded = len(candidates) - len(refined)
value_gain = result_refined["net_expected_value"] - result_naive["net_expected_value"]
print(f"\nExcluding {customers_excluded:,} highest-risk customers from the naive pool "
      f"({customers_excluded / len(candidates):.1%} of candidates)")
print(f"changes net expected value from ${result_naive['net_expected_value']:,.0f} "
      f"to ${result_refined['net_expected_value']:,.0f} "
      f"even though the program now touches fewer customers.")

comparison.to_csv("/home/claude/capone_project/output/business_case_comparison.csv", index=False)
