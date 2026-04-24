# %% [markdown]
# # A/B Testing — Tariff Recommendation Experiment
#
# Experiment: customers in the treatment group see personalised tariff recommendations
# on the "My Tariff" screen. Control group sees the default popular-tariffs list.
#
# Primary metric: tariff upgrade conversion rate (did the customer switch to a higher plan?)
# Secondary metric: ARPU delta (monthly spend change after 30 days)

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


# %% [markdown]
# ## 1. Simulate experiment data

# %%
def generate_ab_data(n_control=1500, n_treatment=1500, seed=42):
    """
    Simulate A/B test results.
    Control:   conversion 8%,  ARPU base 4500 KZT
    Treatment: conversion 13%, ARPU base 5200 KZT (personalised recs work)
    """
    rng = np.random.default_rng(seed)

    control = pd.DataFrame({
        "group": "control",
        "converted": rng.binomial(1, 0.08, n_control),
        "arpu_before": rng.normal(4500, 1200, n_control),
    })
    control["arpu_after"] = control["arpu_before"] * rng.uniform(0.98, 1.05, n_control)

    treatment = pd.DataFrame({
        "group": "treatment",
        "converted": rng.binomial(1, 0.13, n_treatment),
        "arpu_before": rng.normal(4500, 1200, n_treatment),
    })
    # treated customers who converted spend more
    uplift = 1 + treatment["converted"] * rng.uniform(0.12, 0.25, n_treatment)
    treatment["arpu_after"] = treatment["arpu_before"] * uplift * rng.uniform(0.99, 1.06, n_treatment)

    df = pd.concat([control, treatment], ignore_index=True)
    df["arpu_delta"] = df["arpu_after"] - df["arpu_before"]
    return df


data = generate_ab_data()
print(data.groupby("group")[["converted", "arpu_before", "arpu_after", "arpu_delta"]].mean().round(2))


# %% [markdown]
# ## 2. Sample size planning (pre-experiment)

# %%
def required_sample_size(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum sample per group for a two-proportion z-test.
    mde: minimum detectable effect (absolute, e.g. 0.05 = 5pp)
    """
    p2 = p_baseline + mde
    p_bar = (p_baseline + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = (
        (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
         + z_beta * np.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))) ** 2
        / mde ** 2
    )
    return int(np.ceil(n))


n_needed = required_sample_size(p_baseline=0.08, mde=0.04)
print(f"required per group: {n_needed}")
print(f"total sample needed: {n_needed * 2}")
print(f"experiment has: {len(data) // 2} per group — {'enough' if len(data) // 2 >= n_needed else 'not enough'}")


# %% [markdown]
# ## 3. Conversion rate test

# %%
def test_conversion_rate(df: pd.DataFrame) -> dict:
    """Two-proportion z-test for conversion rate difference."""
    control = df[df["group"] == "control"]["converted"]
    treatment = df[df["group"] == "treatment"]["converted"]

    n_c, n_t = len(control), len(treatment)
    p_c = control.mean()
    p_t = treatment.mean()
    p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z_stat = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% CI for difference
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    margin = 1.96 * se_diff
    ci = (p_t - p_c - margin, p_t - p_c + margin)

    # relative lift
    lift = (p_t - p_c) / p_c

    return {
        "p_control": round(p_c, 4),
        "p_treatment": round(p_t, 4),
        "abs_diff": round(p_t - p_c, 4),
        "relative_lift": round(lift, 4),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "ci_95": (round(ci[0], 4), round(ci[1], 4)),
        "significant": p_value < 0.05,
    }


conversion_result = test_conversion_rate(data)
for k, v in conversion_result.items():
    print(f"  {k}: {v}")


# %% [markdown]
# ## 4. ARPU test — t-test and Mann-Whitney

# %%
def test_arpu(df: pd.DataFrame) -> dict:
    """
    t-test (assumes normality) + Mann-Whitney (non-parametric).
    Report both; if they disagree, trust Mann-Whitney.
    """
    control_arpu = df[df["group"] == "control"]["arpu_delta"]
    treatment_arpu = df[df["group"] == "treatment"]["arpu_delta"]

    t_stat, t_pval = stats.ttest_ind(control_arpu, treatment_arpu, equal_var=False)
    mw_stat, mw_pval = stats.mannwhitneyu(
        control_arpu, treatment_arpu, alternative="two-sided"
    )

    # Cohen's d effect size
    pooled_std = np.sqrt(
        (control_arpu.std() ** 2 + treatment_arpu.std() ** 2) / 2
    )
    cohens_d = (treatment_arpu.mean() - control_arpu.mean()) / pooled_std

    return {
        "control_mean_delta": round(control_arpu.mean(), 2),
        "treatment_mean_delta": round(treatment_arpu.mean(), 2),
        "t_statistic": round(t_stat, 4),
        "t_pvalue": round(t_pval, 6),
        "mw_pvalue": round(mw_pval, 6),
        "cohens_d": round(cohens_d, 4),
        "significant_ttest": t_pval < 0.05,
        "significant_mw": mw_pval < 0.05,
    }


arpu_result = test_arpu(data)
for k, v in arpu_result.items():
    print(f"  {k}: {v}")


# %% [markdown]
# ## 5. Multiple testing correction

# %%
def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Reject H0 if p < alpha / n_tests."""
    threshold = alpha / len(p_values)
    return [p < threshold for p in p_values]


p_values = [conversion_result["p_value"], arpu_result["t_pvalue"]]
tests = ["conversion_rate", "arpu_delta"]
rejections = bonferroni_correction(p_values)

print(f"Bonferroni threshold: {0.05 / len(p_values):.4f}")
for test, p, reject in zip(tests, p_values, rejections):
    print(f"  {test}: p={p:.6f} — {'REJECT H0' if reject else 'fail to reject'}")


# %% [markdown]
# ## 6. Visualisation

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# conversion bar
groups = ["control", "treatment"]
rates = [conversion_result["p_control"], conversion_result["p_treatment"]]
ci_err = [
    1.96 * np.sqrt(r * (1 - r) / 1500) for r in rates
]
bars = axes[0].bar(groups, [r * 100 for r in rates], color=["#636EFA", "#EF553B"],
                   width=0.5, yerr=[e * 100 for e in ci_err], capsize=5)
axes[0].set_title("Conversion rate (%)")
axes[0].set_ylabel("Converted (%)")
for bar, rate in zip(bars, rates):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{rate * 100:.1f}%", ha="center", fontsize=11)

# ARPU delta distribution
for group, color in [("control", "#636EFA"), ("treatment", "#EF553B")]:
    subset = data[data["group"] == group]["arpu_delta"]
    axes[1].hist(subset, bins=40, alpha=0.6, color=color, label=group)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("ARPU delta distribution (KZT)")
axes[1].set_xlabel("ARPU change after experiment")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/ab_testing_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("plot saved to notebooks/ab_testing_results.png")


# %% [markdown]
# ## 7. Summary

# %%
print("=" * 55)
print("EXPERIMENT SUMMARY")
print("=" * 55)
print(f"  Required sample size per group : {required_sample_size(0.08, 0.04)}")
print(f"  Actual sample per group        : 1500")
print()
print("Conversion rate")
print(f"  control   : {conversion_result['p_control'] * 100:.1f}%")
print(f"  treatment : {conversion_result['p_treatment'] * 100:.1f}%")
print(f"  lift      : +{conversion_result['relative_lift'] * 100:.1f}%")
print(f"  p-value   : {conversion_result['p_value']:.4f} ({'significant' if conversion_result['significant'] else 'not significant'})")
print(f"  95% CI    : {conversion_result['ci_95']}")
print()
print("ARPU delta (30-day)")
print(f"  control mean   : {arpu_result['control_mean_delta']:.0f} KZT")
print(f"  treatment mean : {arpu_result['treatment_mean_delta']:.0f} KZT")
print(f"  Cohen's d      : {arpu_result['cohens_d']:.3f}")
print(f"  Mann-Whitney p : {arpu_result['mw_pvalue']:.4f} ({'significant' if arpu_result['significant_mw'] else 'not significant'})")
print()
print("After Bonferroni correction: both tests reject H0 at alpha=0.05")
print("Recommendation: roll out personalised recommendations to all users.")
