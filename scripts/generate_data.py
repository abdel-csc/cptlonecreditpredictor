"""
Generates a synthetic credit card portfolio.

DESIGN CHOICE: features are correlated with the default outcome via an
explicit logistic function, not left purely random. This matters because
it's the difference between "a model that happens to fit noise" and
"a model where I can explain WHY each feature matters." When you defend
this project, the story is: I built the ground truth, so I know exactly
what signal exists and why the model finds it.

Features chosen to mirror what a real card issuer actually has on file
(no external credit bureau data needed, all self-contained):
- income, credit_score_proxy: standard underwriting inputs
- tenure_months: how long they've held the card (longer = more trust)
- utilization_rate: balance / credit_limit (the single strongest real-world
  predictor of default risk in card lending)
- num_delinquencies_24mo: past behavior, strongest behavioral signal
- current_apr: pricing already assigned to this customer
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 50_000

income = np.random.lognormal(mean=10.9, sigma=0.4, size=N).clip(20000, 250000)
credit_score = np.random.normal(680, 70, N).clip(300, 850)
tenure_months = np.random.exponential(36, N).clip(1, 240)
credit_limit = (credit_score * 20 + income * 0.02).clip(500, 30000)
utilization_rate = np.random.beta(2, 3, N)  # skews toward lower utilization, long tail up
balance = utilization_rate * credit_limit
num_delinquencies_24mo = np.random.poisson(
    lam=np.clip((700 - credit_score) / 150, 0.02, 3), size=N
)
current_apr = (24.99 - (credit_score - 300) / 550 * 15).clip(9.99, 29.99)

# Ground-truth default probability: logistic function of standardized features.
# Coefficients reflect real underwriting intuition:
#   utilization and delinquency history are the dominant risk drivers;
#   income and tenure are protective; credit_score partially overlaps
#   with utilization/delinquency (multicollinearity on purpose, it's realistic).
z = (
    -3.2
    + 2.8 * utilization_rate
    + 0.55 * num_delinquencies_24mo
    - 0.000012 * income
    - 0.004 * tenure_months
    - 0.004 * (credit_score - 680)
)
default_prob_true = 1 / (1 + np.exp(-z))
default_12mo = np.random.binomial(1, default_prob_true)

df = pd.DataFrame({
    "customer_id": np.arange(1, N + 1),
    "income": income.round(0),
    "credit_score": credit_score.round(0),
    "tenure_months": tenure_months.round(0),
    "credit_limit": credit_limit.round(0),
    "balance": balance.round(2),
    "utilization_rate": utilization_rate.round(4),
    "num_delinquencies_24mo": num_delinquencies_24mo,
    "current_apr": current_apr.round(2),
    "default_12mo": default_12mo,
})

df.to_csv("/home/claude/capone_project/data/portfolio.csv", index=False)
print(f"Generated {len(df):,} customers")
print(f"Base default rate: {df['default_12mo'].mean():.2%}")
print(df.describe().round(2))
