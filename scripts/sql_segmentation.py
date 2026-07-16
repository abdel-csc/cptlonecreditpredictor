"""
Loads the portfolio into SQLite and runs the segmentation analysis in SQL.

WHY SQL HERE AND NOT JUST PANDAS: the job listing explicitly calls out SQL
as a tool. In a real BA role, you're pulling from a warehouse, not a CSV,
so this step exists to demonstrate that muscle, not because SQLite is
necessary for 50k rows.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/claude/capone_project/data/portfolio.db")
df = pd.read_csv("/home/claude/capone_project/data/portfolio.csv")
df.to_sql("portfolio", conn, if_exists="replace", index=False)

queries = {
    "risk_tier_summary": """
        SELECT
            CASE
                WHEN utilization_rate < 0.30 AND num_delinquencies_24mo = 0 THEN 'Low Risk'
                WHEN utilization_rate < 0.60 AND num_delinquencies_24mo <= 1 THEN 'Medium Risk'
                ELSE 'High Risk'
            END AS risk_tier,
            COUNT(*) AS num_customers,
            ROUND(AVG(default_12mo) * 100, 2) AS default_rate_pct,
            ROUND(AVG(income), 0) AS avg_income,
            ROUND(AVG(credit_limit), 0) AS avg_credit_limit,
            ROUND(AVG(utilization_rate) * 100, 1) AS avg_utilization_pct,
            ROUND(AVG(current_apr), 2) AS avg_apr
        FROM portfolio
        GROUP BY risk_tier
        ORDER BY default_rate_pct;
    """,
    "credit_increase_candidates": """
        -- Candidates for a credit line increase: low risk (proven, 0 delinquencies)
        -- AND currently high utilization (signal they'd actually use more credit,
        -- which is what makes the program revenue-positive rather than a freebie).
        SELECT
            COUNT(*) AS num_candidates,
            ROUND(AVG(income), 0) AS avg_income,
            ROUND(AVG(credit_limit), 0) AS avg_current_limit,
            ROUND(AVG(utilization_rate) * 100, 1) AS avg_utilization_pct,
            ROUND(AVG(default_12mo) * 100, 2) AS actual_default_rate_pct
        FROM portfolio
        WHERE num_delinquencies_24mo = 0
          AND utilization_rate >= 0.60
          AND tenure_months >= 12;
    """,
}

print("=" * 70)
for name, q in queries.items():
    print(f"\n--- {name} ---")
    result = pd.read_sql(q, conn)
    print(result.to_string(index=False))
    result.to_csv(f"/home/claude/capone_project/output/{name}.csv", index=False)

conn.close()
