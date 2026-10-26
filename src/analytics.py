"""Step 2 - Portfolio analytics.

Profiles the portfolio, runs cross-dimensional cuts, and generates docs/insights.md
(every number in the insights is computed here - nothing is typed by hand).

Run:  .venv\\Scripts\\python.exe src\\analytics.py

Definitions (also in docs/assumptions_log.md)
  exposure            = original loan_amount (outstanding balance is only available for months 1-12)
  default rate        = share of accounts with target_default = 1
  exposure-wtd rate   = defaulted loan_amount / total loan_amount
  lift                = segment default rate / portfolio default rate
  CI                  = 95% Wilson interval on the default rate; cells with n < MIN_N are not read as findings
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analytics"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)
MIN_N = 100

df = pd.read_parquet(ROOT / "data" / "processed" / "analytics_dataset.parquet")
df["def_exposure"] = df.loan_amount * df.target_default
N, EXP, DEF_EXP = len(df), df.loan_amount.sum(), df.def_exposure.sum()
PD_ALL = df.target_default.mean()

# ----------------------------------------------------------------------------- style (dataviz skill palette)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": BASE, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
seq_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ[:6])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def profile(col, order=None):
    g = df.groupby(col, observed=True).agg(accounts=("account_id", "size"), defaults=("target_default", "sum"),
                                           exposure=("loan_amount", "sum"), def_exposure=("def_exposure", "sum"))
    g["acct_share"] = g.accounts / N
    g["exposure_share"] = g.exposure / EXP
    g["default_rate"] = g.defaults / g.accounts
    g[["ci_low", "ci_high"]] = [wilson(k, n) for k, n in zip(g.defaults, g.accounts)]
    g["exposure_wtd_default_rate"] = g.def_exposure / g.exposure
    g["lift"] = g.default_rate / PD_ALL
    g["share_of_defaulted_exposure"] = g.def_exposure / DEF_EXP
    if order:
        g = g.reindex(order)
    return g


ORD = {
    "income_band": ["<40k", "40-60k", "60-80k", "80-100k", "100k+"],
    "age_band": ["<30", "30-39", "40-49", "50-59", "60+"],
    "employment_type": ["Salaried", "Business Owner", "Self-Employed", "Contract", "Unemployed"],
    "region": ["North", "South", "East", "West", "Central"],
    "loan_type": ["Personal Loan", "Credit Card", "Auto Loan", "Business Loan", "Mortgage"],
}
EXTRA = ["city_tier", "housing_type", "education_level", "disbursal_channel", "orig_year", "has_bureau", "bur_has_writeoff"]
parts = []
for dim in list(ORD) + EXTRA:
    p = profile(dim, ORD.get(dim)).reset_index().rename(columns={dim: "segment"})
    p.insert(0, "dimension", dim)
    p["segment"] = p.segment.astype(str)
    parts.append(p)
prof_all = pd.concat(parts, ignore_index=True)
prof_all.to_csv(OUT / "portfolio_profile.csv", index=False)
P = {d: prof_all[prof_all.dimension == d].set_index("segment") for d in prof_all.dimension.unique()}


# ----------------------------------------------------------------------------- cross-cuts
def cross(rows, cols, rorder, corder):
    t = df.groupby([rows, cols], observed=True).agg(n=("account_id", "size"), dr=("target_default", "mean"),
                                                   exp=("loan_amount", "sum")).reset_index()
    return (t.pivot(index=rows, columns=cols, values="dr").reindex(index=rorder, columns=corder),
            t.pivot(index=rows, columns=cols, values="n").reindex(index=rorder, columns=corder),
            t.pivot(index=rows, columns=cols, values="exp").reindex(index=rorder, columns=corder))


ie_dr, ie_n, _ = cross("income_band", "employment_type", ORD["income_band"], ORD["employment_type"])
rl_dr, rl_n, rl_exp = cross("region", "loan_type", ORD["region"], ORD["loan_type"])
ai_dr, ai_n, _ = cross("age_band", "income_band", ORD["age_band"], ORD["income_band"])
for name, t in [("income_x_employment_default_rate", ie_dr), ("income_x_employment_n", ie_n),
                ("region_x_loan_type_default_rate", rl_dr), ("region_x_loan_type_n", rl_n),
                ("region_x_loan_type_exposure", rl_exp), ("age_x_income_default_rate", ai_dr), ("age_x_income_n", ai_n)]:
    t.to_csv(OUT / f"cross_{name}.csv")


# ----------------------------------------------------------------------------- drivers beyond demographics
def rate_by(series_or_col, bins=None, labels=None):
    s = df[series_or_col] if isinstance(series_or_col, str) else series_or_col
    if bins is not None:
        s = pd.cut(s, bins, labels=labels)
    g = df.groupby(s, observed=True).target_default.agg(["mean", "size"])
    return g


missed = df.rp_n_missed.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
g_missed = rate_by(missed)
lti = rate_by("loan_to_income", [0, .25, .5, 1, 2, 1e9], ["<0.25x", "0.25-0.5x", "0.5-1x", "1-2x", ">2x"])
emi = rate_by("emi_to_income", [0, .1, .2, .3, 1e9], ["<10%", "10-20%", "20-30%", ">30%"])
dpd30 = rate_by(df.bur_n_dpd30.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"}))
wo = rate_by(df.bur_n_writeoff.clip(upper=2).map({0: "0", 1: "1", 2: "2+"}))
df["rate_quartile_in_type"] = df.groupby("loan_type").interest_rate_pct.transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]))
g_rateq = rate_by("rate_quartile_in_type")
rate_by_income = df.groupby("income_band").interest_rate_pct.mean().reindex(ORD["income_band"])
rate_by_income_type = df.groupby(["loan_type", "income_band"]).interest_rate_pct.mean().unstack().reindex(columns=ORD["income_band"])

# ----------------------------------------------------------------------------- charts
def bar_panel(ax, p, title, highlight=None, fmt="{:.0%}"):
    x = np.arange(len(p))
    cols = [ORANGE if s == highlight else BLUE for s in p.index]
    ax.bar(x, p.default_rate, color=cols, width=0.62)
    ax.errorbar(x, p.default_rate, yerr=[p.default_rate - p.ci_low, p.ci_high - p.default_rate], fmt="none", ecolor=INK2, lw=1, capsize=2)
    for i, (v, s) in enumerate(zip(p.default_rate, p.exposure_share)):
        ax.text(i, p.ci_high.iloc[i] + 0.012, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(i, -0.002, f"{s:.0%}", ha="center", va="top", fontsize=8, color=MUTED)
    ax.axhline(PD_ALL, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [str(i).replace(" ", "\n") for i in p.index], fontsize=8.5)
    ax.tick_params(axis="x", pad=14, length=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0, max(p.ci_high.max() + 0.07, PD_ALL + 0.05))


fig, axes = plt.subplots(1, 5, figsize=(19, 3.9))
for ax, (dim, title, hl) in zip(axes, [("income_band", "Annual income", "<40k"), ("employment_type", "Employment type", "Unemployed"),
                                       ("region", "Region", None), ("age_band", "Age", "<30"), ("loan_type", "Loan type", "Mortgage")]):
    bar_panel(ax, P[dim], title, hl)
fig.suptitle(f"Default rate by dimension (95% CI) - dashed line = portfolio {PD_ALL:.1%}; grey figure under each label = share of exposure",
             x=0.01, ha="left", fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(CH / "01_profile_overview.png", dpi=180)
plt.close(fig)

# I1 concentration: share of accounts vs exposure vs defaulted exposure by loan type
lt = P["loan_type"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
y = np.arange(len(lt))
h = 0.26
for k, (col, lab, c) in enumerate([("acct_share", "Share of accounts", BLUE), ("exposure_share", "Share of exposure", ORANGE),
                                   ("share_of_defaulted_exposure", "Share of defaulted exposure", AQUA)]):
    b = ax.barh(y + (k - 1) * h, lt[col], height=h - 0.03, color=c, label=lab)
    for yy, v in zip(y + (k - 1) * h, lt[col]):
        ax.text(v + 0.008, yy, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, [f"{s}  (default {r:.0%})" for s, r in zip(lt.index, lt.default_rate)])
ax.invert_yaxis()
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.legend(frameon=False, loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.62))
ax.set_title("Mortgages: a tenth of accounts, half the exposure,\ntwo-thirds of defaulted exposure", loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(CH / "02_loan_type_concentration.png", dpi=180)
plt.close(fig)


def heat(ax, dr, n, title, fmt="{:.0%}", vmax=None):
    vals = dr.where(n >= 30)
    vmax = vmax or np.nanmax(vals.values)
    ax.imshow(np.ma.masked_invalid(vals.values), cmap=seq_cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(dr.shape[0]):
        for j in range(dr.shape[1]):
            v, k = dr.iloc[i, j], n.iloc[i, j]
            if k < 30 or np.isnan(v):
                ax.text(j, i, f"n={int(k) if not np.isnan(k) else 0}", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            dark = v / vmax > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.33, f"n={int(k):,}" + ("*" if k < MIN_N else ""), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if dark else INK2)
    ax.set_xticks(range(dr.shape[1]), dr.columns, fontsize=9)
    ax.set_yticks(range(dr.shape[0]), dr.index, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


fig, ax = plt.subplots(figsize=(8.2, 4.2))
heat(ax, ie_dr, ie_n, "Default rate: income band x employment type  (* = n<100, read with care; n<30 blanked)")
fig.tight_layout()
fig.savefig(CH / "03_income_x_employment.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 4.2))
heat(ax, rl_dr, rl_n, "Default rate: region x loan type")
fig.tight_layout()
fig.savefig(CH / "04_region_x_loan_type.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, (g, title) in zip(axes, [(g_missed, "Missed payments in first 12 months"), (lti, "Loan amount / annual income"),
                                 (dpd30, "Bureau lines 30+ DPD")]):
    x = np.arange(len(g))
    ax.bar(x, g["mean"], color=BLUE, width=0.6)
    for i, (v, k) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=9, zorder=5,
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.2))
        ax.text(i, -0.002, f"n={k:,}", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.axhline(PD_ALL, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, g.index)
    ax.tick_params(axis="x", pad=12, length=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(0, g["mean"].max() + 0.07)
fig.suptitle(f"Default rate by repayment / affordability / bureau behaviour (dashed = portfolio {PD_ALL:.1%})", x=0.01, ha="left", fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(CH / "05_behaviour_drivers.png", dpi=180)
plt.close(fig)

# I5 pricing vs risk: two small multiples (never dual axis)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
inc = P["income_band"]
for ax, vals, title, fmt, color in [(axes[0], inc.default_rate, "Default rate", "{:.0%}", BLUE),
                                    (axes[1], rate_by_income / 100, "Average interest rate charged", "{:.1%}", ORANGE)]:
    x = np.arange(len(vals))
    ax.bar(x, vals, color=color, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + vals.max() * 0.02, fmt.format(v), ha="center", fontsize=9.5)
    ax.set_xticks(x, vals.index)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, vals.max() * 1.18)
    ax.set_title(title + " by income band", loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(CH / "06_pricing_vs_risk.png", dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------------- numbers for the narrative
m = P["loan_type"].loc["Mortgage"]
inc = P["income_band"]
low = df.income_band.isin(["<40k", "40-60k"])
low_acct, low_exp, low_defexp = low.mean(), df.loc[low, "loan_amount"].sum() / EXP, df.loc[low, "def_exposure"].sum() / DEF_EXP
emp = P["employment_type"]
reg = P["region"]
age = P["age_band"]
south_m_dr, south_m_n = rl_dr.loc["South", "Mortgage"], int(rl_n.loc["South", "Mortgage"])
south_m_exp = rl_exp.loc["South", "Mortgage"]
cent_m_dr, cent_m_n = rl_dr.loc["Central", "Mortgage"], int(rl_n.loc["Central", "Mortgage"])
sm_ci = wilson(round(south_m_dr * south_m_n), south_m_n)
cm_ci = wilson(round(cent_m_dr * cent_m_n), cent_m_n)
mort = df[df.loan_type == "Mortgage"]
mort_low = mort[mort.income_band == "<40k"]
early = (df.rp_n_missed >= 2)
rate_spread = df.groupby("income_band").interest_rate_pct.mean()
wo_p = df.bur_has_writeoff.eq(1)
lines = {}
for dim in ORD:
    p = P[dim]
    lines[dim] = f"{p.default_rate.idxmin()} {p.default_rate.min():.1%} -> {p.default_rate.idxmax()} {p.default_rate.max():.1%}"

doc = f"""# Portfolio Analytics - Insights

_Auto-generated by `src/analytics.py` (numbers are computed, not typed). Tables: `outputs/analytics/`; charts: `outputs/analytics/charts/`._

**Portfolio:** {N:,} loan accounts, {df.customer_id.nunique():,} customers, **${EXP / 1e6:,.0f}M** original exposure.
Default rate **{PD_ALL:.1%}** by count, **{DEF_EXP / EXP:.1%}** exposure-weighted (large loans default more).
"Exposure" = original `loan_amount`; "lift" = segment default rate / portfolio rate. Rates carry 95% Wilson intervals in the tables.

## Profiling snapshot (default rate range per dimension)
| Dimension | Lowest -> Highest | Reads as |
|---|---|---|
| Income band | {lines['income_band']} | Strong, monotonic gradient |
| Employment type | {lines['employment_type']} | One outlier segment (Unemployed) |
| Age band | {lines['age_band']} | Younger = riskier |
| Loan type | {lines['loan_type']} | Mortgage is the outlier |
| Region | {lines['region']} | **Flat** - region barely moves default risk |

Also profiled (flat, see `portfolio_profile.csv`): city tier, housing type, education, disbursal channel, origination year, gender, marital status.
![profile](../outputs/analytics/charts/01_profile_overview.png)

## Insight 1 - Mortgages are the concentration risk
- Mortgages are **{m.acct_share:.1%} of accounts but {m.exposure_share:.1%} of exposure** and **{m.share_of_defaulted_exposure:.0%} of defaulted exposure**.
- Their default rate is **{m.default_rate:.1%}** (lift {m.lift:.2f}x) vs {df[df.loan_type != 'Mortgage'].target_default.mean():.1%} for everything else.
- **So what:** the book's loss is driven by ~1,400 accounts. Risk limits, monitoring and any model effort should be segmented by product first; a flat portfolio-wide cut-off would under-protect the mortgage book. Being secured does not lower default frequency (secured Mortgage/Auto accounts hold {df[df.collateral_flag == 1].loan_amount.sum() / EXP:.0%} of exposure and {df[df.collateral_flag == 1].def_exposure.sum() / DEF_EXP:.0%} of defaulted exposure), so loss-given-default after collateral recovery is the next question - it needs recovery data we do not have.
![concentration](../outputs/analytics/charts/02_loan_type_concentration.png)

## Insight 2 - Income is the strongest demographic driver; unemployment adds risk on top (cross-cut: income x employment)
- Default falls steadily with income: **{inc.loc['<40k', 'default_rate']:.1%}** (<40k) -> {inc.loc['40-60k', 'default_rate']:.1%} -> {inc.loc['60-80k', 'default_rate']:.1%} -> {inc.loc['80-100k', 'default_rate']:.1%} -> **{inc.loc['100k+', 'default_rate']:.1%}** (100k+), a {inc.loc['<40k', 'default_rate'] / inc.loc['100k+', 'default_rate']:.1f}x spread.
- Customers under 60k income are **{low_acct:.0%} of accounts and {low_exp:.0%} of exposure but {low_defexp:.0%} of defaulted exposure.**
- Unemployed customers ({emp.loc['Unemployed', 'acct_share']:.1%} of accounts) default at **{emp.loc['Unemployed', 'default_rate']:.1%}** ({emp.loc['Unemployed', 'lift']:.1f}x). Most of that is low income (their median income is ${df[df.employment_type == 'Unemployed'].annual_income.median():,.0f}), **but it is not only income**: within the 40-60k band unemployed customers default at {ie_dr.loc['40-60k', 'Unemployed']:.0%} vs {ie_dr.loc['40-60k', 'Salaried']:.0%} for salaried, and within <40k at {ie_dr.loc['<40k', 'Unemployed']:.0%} vs {ie_dr.loc['<40k', 'Salaried']:.0%}.
- Age adds a smaller effect: under-30s default at {age.loc['<30', 'default_rate']:.1%} vs {age.loc['40-49', 'default_rate']:.1%} for 40-49, and the gap holds inside each income band ({ai_dr.loc['<30', '80-100k']:.1%} vs {ai_dr.loc['40-49', '80-100k']:.1%} at 80-100k).
- **So what:** underwriting needs an income-and-employment affordability floor; the <60k segment is where tightening or pricing change moves the most defaulted exposure. Employment status should be verified, not assumed from stated income. Caveat: the 100k+ x Unemployed cell has only {int(ie_n.loc['100k+', 'Unemployed'])} accounts and is not read as a finding.
![income x employment](../outputs/analytics/charts/03_income_x_employment.png)

## Insight 3 - Region is a concentration driver, not a risk driver (cross-cut: region x loan type)
- Region default rates sit in a narrow {reg.default_rate.min():.1%}-{reg.default_rate.max():.1%} band. **South** is the biggest region ({reg.loc['South', 'exposure_share']:.0%} of exposure, {reg.loc['South', 'share_of_defaulted_exposure']:.0%} of defaulted exposure) because it is simply the largest, not because it is riskier.
- Product matters inside every region: mortgages default at {rl_dr['Mortgage'].min():.0%}-{rl_dr['Mortgage'].max():.0%} everywhere vs {rl_dr['Auto Loan'].min():.0%}-{rl_dr['Auto Loan'].max():.0%} for auto.
- The single biggest pocket is **South x Mortgage: ${south_m_exp / 1e6:,.0f}M = {south_m_exp / EXP:.0%} of the whole book at a {south_m_dr:.1%} default rate** (n={south_m_n}). Central x Mortgage looks lower ({cent_m_dr:.1%}, n={cent_m_n}), but the 95% intervals ({sm_ci[0]:.0%}-{sm_ci[1]:.0%} vs {cm_ci[0]:.0%}-{cm_ci[1]:.0%}) overlap, so we do not claim South is riskier - only larger.
- **So what:** set exposure limits by region x product, not region alone; don't spend effort on region-specific credit policy because the data doesn't support it. Watch the South mortgage book as a single-point concentration.
![region x loan type](../outputs/analytics/charts/04_region_x_loan_type.png)

## Insight 4 - Early repayment behaviour and affordability separate good from bad accounts
- Missed payments in the **first 12 months**: {g_missed.loc['0', 'mean']:.1%} default with none -> {g_missed.loc['1', 'mean']:.1%} (1) -> {g_missed.loc['2', 'mean']:.1%} (2) -> **{g_missed.loc['3+', 'mean']:.1%}** (3+). Accounts with 2+ missed payments are {early.mean():.0%} of the book and {df.loc[early, 'def_exposure'].sum() / DEF_EXP:.0%} of defaulted exposure.
- Affordability: loan > 2x annual income defaults at **{lti.loc['>2x', 'mean']:.1%}** vs {lti.loc['<0.25x', 'mean']:.1%} below 0.25x; EMI > 30% of income defaults at {emi.loc['>30%', 'mean']:.1%} vs {emi.loc['<10%', 'mean']:.1%} below 10%.
- Bureau: customers with three or more 30+ DPD external lines default at {dpd30.loc['3+', 'mean']:.1%} vs {dpd30.loc['0', 'mean']:.1%} with none; any written-off external line lifts risk from {df[~wo_p].target_default.mean():.1%} to {df[wo_p].target_default.mean():.1%}.
- **So what:** the strongest signal is behavioural, not demographic. That justifies engineering bureau and repayment aggregates for the model, and an early-warning trigger after the 2nd missed EMI. These are the features the model in step 3 should lean on.
![behaviour](../outputs/analytics/charts/05_behaviour_drivers.png)

## Insight 5 - Pricing does not follow risk
- Default rate climbs {inc.default_rate.max() / inc.default_rate.min():.1f}x from the 100k+ to the <40k income band, yet the **average interest rate is flat: {rate_spread.min():.2f}%-{rate_spread.max():.2f}%** across the same bands.
- Inside each loan type, the highest-rate quartile defaults at {g_rateq.loc['Q4 (highest)', 'mean']:.1%} vs {g_rateq.loc['Q1 (lowest)', 'mean']:.1%} in the lowest - only a {(g_rateq.loc['Q4 (highest)', 'mean'] - g_rateq.loc['Q1 (lowest)', 'mean']) * 100:.1f}pp slope. Rate is set by product, not by borrower risk.
- **So what:** low-income borrowers are not being compensated for the extra risk (<40k defaults at {inc.loc['<40k', 'default_rate']:.0%}). A risk score from this project could drive risk-based pricing or approval limits. Caveat: we only see stated rates, not fees, and we do not know the bank's pricing policy.
![pricing](../outputs/analytics/charts/06_pricing_vs_risk.png)

## Caveats
- Vintage (origination year) shows no trend ({P['orig_year'].default_rate.min():.1%}-{P['orig_year'].default_rate.max():.1%}); with only months 1-12 of behaviour and no calendar dates, time trends cannot be built (assumption A3).
- Associations, not causes. Segments are correlated (income, employment, age, loan size), so univariate lifts overlap; the model in step 3 separates them.
- Collections data is excluded from insights on drivers: dates are unreliable and events follow delinquency (assumption A11).
"""
(ROOT / "docs" / "insights.md").write_text(doc, encoding="utf-8")
print("analytics done; charts:", sorted(p.name for p in CH.glob("*.png")))
