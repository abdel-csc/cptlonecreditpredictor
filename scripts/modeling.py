"""
Two models, on purpose:

1. Logistic Regression -- this is the one that goes in the business memo.
   Its coefficients are directly interpretable ("each 10pt rise in
   utilization multiplies odds of default by X"), which is what a BA role
   actually needs: something you can explain to a non-technical VP in one
   sentence.

2. Gradient Boosting -- a benchmark to check we're not leaving real
   predictive power on the table by choosing the interpretable model.
   If GBM barely beats logistic regression, that's the justification for
   using the interpretable one in the actual business recommendation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/claude/capone_project/data/portfolio.csv")

features = [
    "income", "credit_score", "tenure_months",
    "utilization_rate", "num_delinquencies_24mo", "current_apr",
]
X = df[features]
y = df["default_12mo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Model 1: Logistic Regression ---
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_s, y_train)
logit_probs = logit.predict_proba(X_test_s)[:, 1]
logit_auc = roc_auc_score(y_test, logit_probs)
logit_brier = brier_score_loss(y_test, logit_probs)

coef_table = pd.DataFrame({
    "feature": features,
    "coefficient": logit.coef_[0].round(3),
    "odds_ratio_per_1sd": np.exp(logit.coef_[0]).round(3),
}).sort_values("coefficient", key=abs, ascending=False)

# --- Model 2: Gradient Boosting benchmark ---
gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)
gbm_probs = gbm.predict_proba(X_test)[:, 1]
gbm_auc = roc_auc_score(y_test, gbm_probs)
gbm_brier = brier_score_loss(y_test, gbm_probs)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(f"Logistic Regression -- AUC: {logit_auc:.4f}   Brier: {logit_brier:.4f}")
print(f"Gradient Boosting   -- AUC: {gbm_auc:.4f}   Brier: {gbm_brier:.4f}")
print(f"\nAUC gap: {gbm_auc - logit_auc:.4f} "
      f"({'small enough to prefer the interpretable model' if gbm_auc - logit_auc < 0.02 else 'GBM meaningfully better, worth the interpretability trade-off'})")

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION COEFFICIENTS (standardized, business narrative)")
print("=" * 70)
print(coef_table.to_string(index=False))

# Save the fitted logistic model's predicted probabilities back onto the
# full dataset for use in the business impact calculation.
full_scaled = scaler.transform(X)
df["predicted_default_prob"] = logit.predict_proba(full_scaled)[:, 1]
df.to_csv("/home/claude/capone_project/data/portfolio_scored.csv", index=False)

coef_table.to_csv("/home/claude/capone_project/output/model_coefficients.csv", index=False)
with open("/home/claude/capone_project/output/model_metrics.txt", "w") as f:
    f.write(f"Logistic Regression -- AUC: {logit_auc:.4f}   Brier: {logit_brier:.4f}\n")
    f.write(f"Gradient Boosting   -- AUC: {gbm_auc:.4f}   Brier: {gbm_brier:.4f}\n")

print("\nScored dataset saved to portfolio_scored.csv")
