import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

# --- Chart 1: Risk tier default rates ---
tier_df = pd.read_csv("/home/claude/capone_project/output/risk_tier_summary.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#2E8B57", "#DAA520", "#B22222"]
bars = ax.bar(tier_df["risk_tier"], tier_df["default_rate_pct"], color=colors)
for bar, val in zip(bars, tier_df["default_rate_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val}%", ha="center", fontweight="bold")
ax.set_ylabel("12-Month Default Rate (%)")
ax.set_title("Default Rate by Risk Tier")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("/home/claude/capone_project/output/chart_risk_tiers.png", dpi=150)
plt.close()

# --- Chart 2: Model coefficients (odds ratios) ---
coef_df = pd.read_csv("/home/claude/capone_project/output/model_coefficients.csv")
coef_df = coef_df.sort_values("odds_ratio_per_1sd")
fig, ax = plt.subplots(figsize=(7, 4.5))
colors2 = ["#B22222" if v > 1 else "#2E8B57" for v in coef_df["odds_ratio_per_1sd"]]
ax.barh(coef_df["feature"], coef_df["odds_ratio_per_1sd"], color=colors2)
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Odds Ratio per 1 Std. Dev. (>1 = increases risk, <1 = protective)")
ax.set_title("Default Risk Drivers (Logistic Regression)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("/home/claude/capone_project/output/chart_model_coefficients.png", dpi=150)
plt.close()

# --- Chart 3: Strategy comparison ---
comp_df = pd.read_csv("/home/claude/capone_project/output/business_case_comparison.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].bar(comp_df["strategy"], comp_df["net_expected_value"], color=["#8B8B8B", "#2E8B57"])
axes[0].set_ylabel("Net Expected Value ($)")
axes[0].set_title("Net Expected Value by Strategy")
axes[0].tick_params(axis="x", rotation=15)
axes[0].spines[["top", "right"]].set_visible(False)
for i, v in enumerate(comp_df["net_expected_value"]):
    axes[0].text(i, v + 1500, f"${v:,.0f}", ha="center", fontweight="bold", fontsize=9)

axes[1].bar(comp_df["strategy"], comp_df["roi_pct"], color=["#8B8B8B", "#2E8B57"])
axes[1].set_ylabel("ROI (%)")
axes[1].set_title("Program ROI by Strategy")
axes[1].tick_params(axis="x", rotation=15)
axes[1].spines[["top", "right"]].set_visible(False)
for i, v in enumerate(comp_df["roi_pct"]):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig("/home/claude/capone_project/output/chart_strategy_comparison.png", dpi=150)
plt.close()

print("3 charts saved to output/")
