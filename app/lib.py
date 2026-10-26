"""Shared data loading, labels, colours and chart helpers for the dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL_OUT = ROOT / "outputs" / "model"

# ---- colours (dataviz skill reference palette) --------------------------------------------------
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_BLUE = [[0, "#cde2fb"], [0.35, "#6da7ec"], [0.7, "#256abf"], [1, "#0d366b"]]
BAND_ORDER = ["Low", "Medium", "High", "Very High"]
BAND_COLOR = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Very High": "#d03b3b"}  # status palette
BAND_RULE = {"Low": "PD < 10%", "Medium": "PD 10-20%", "High": "PD 20-35%", "Very High": "PD >= 35%"}

DIMENSIONS = {
    "Loan type": ("loan_type", ["Personal Loan", "Credit Card", "Auto Loan", "Business Loan", "Mortgage"]),
    "Income band": ("income_band", ["<40k", "40-60k", "60-80k", "80-100k", "100k+"]),
    "Employment type": ("employment_type", ["Salaried", "Business Owner", "Self-Employed", "Contract", "Unemployed"]),
    "Region": ("region", ["North", "South", "East", "West", "Central"]),
    "Age band": ("age_band", ["<30", "30-39", "40-49", "50-59", "60+"]),
    "City tier": ("city_tier", ["Tier 1", "Tier 2", "Tier 3"]),
    "Housing": ("housing_type", ["Owned", "Mortgaged", "Rented", "Living with Family"]),
    "Disbursal channel": ("disbursal_channel", ["Branch", "Online", "Mobile App", "Partner/DSA"]),
    "Origination year": ("orig_year", [2018, 2019, 2020, 2021, 2022, 2023]),
}

FEATURE_LABELS = {
    "age": "Age", "years_employed": "Years employed", "annual_income": "Annual income", "num_dependents": "Dependents",
    "years_at_residence": "Years at residence", "city_tier_num": "City tier", "years_employed_capped": "Years employed was capped",
    "years_employed_missing": "Years employed was missing", "num_dependents_missing": "Dependents was missing",
    "years_at_residence_missing": "Years at residence was missing", "log_loan_amount": "Loan amount (log)",
    "interest_rate_pct": "Interest rate", "tenure_months": "Tenure (months)", "collateral_flag": "Secured loan",
    "interest_rate_missing": "Interest rate was missing", "loan_to_income": "Loan / annual income",
    "emi_to_income": "Annual EMIs / income", "rp_emi": "Monthly EMI", "cust_n_accounts": "Accounts with the bank",
    "has_bureau": "Has bureau file", "bur_n_lines": "Bureau credit lines", "bur_n_active": "Active bureau lines",
    "bur_n_closed": "Closed bureau lines", "bur_n_writeoff": "Written-off bureau lines", "bur_has_writeoff": "Any bureau write-off",
    "bur_total_sanctioned": "Bureau sanctioned total", "bur_total_outstanding": "Bureau outstanding total",
    "bur_writeoff_outstanding": "Written-off outstanding", "bur_max_dpd": "Worst bureau DPD", "bur_avg_dpd": "Average bureau DPD",
    "bur_n_dpd30": "Bureau lines 30+ DPD", "bur_n_dpd_missing": "Bureau lines with missing DPD", "bur_pct_fintech": "Share of fintech lenders",
    "bur_n_credit_types": "Credit types held", "bur_oldest_line_yrs": "Oldest bureau line (yrs)",
    "bur_newest_line_yrs": "Newest bureau line (yrs)", "bur_n_opened_24m": "Lines opened in last 24m", "bur_utilization": "Bureau utilisation",
    "rp_n_late": "Late payments (months 1-12)", "rp_n_missed": "Missed payments (months 1-12)",
    "rp_pct_ontime": "On-time payment share (months 1-12)", "rp_max_dpd": "Worst DPD (months 1-12)",
    "rp_avg_dpd": "Average DPD (months 1-12)", "rp_dpd_first3": "Average DPD, first 3 months", "rp_dpd_last3": "Average DPD, last 3 months",
    "rp_dpd_trend": "DPD trend (last 3 - first 3)", "rp_max_consec_missed": "Longest missed streak",
    "rp_max_consec_late_or_missed": "Longest late/missed streak", "rp_paid_ratio": "Paid / due ratio",
    "rp_shortfall_total": "Total EMI shortfall", "rp_first_delinq_month": "First late/missed month",
    "rp_balance_ratio_m12": "Balance left at month 12 / loan", "cust_other_missed": "Missed payments on customer's other loans",
    "cust_other_late": "Late payments on customer's other loans", "cust_other_maxdpd": "Worst DPD on customer's other loans",
    "cust_max_missed": "Customer's max missed (any loan)", "cust_sum_missed": "Customer's total missed payments",
    "cust_emi_to_income": "All customer EMIs / income", "employment_type": "Employment type", "occupation_sector": "Occupation sector",
    "region": "Region", "housing_type": "Housing", "education_level": "Education", "loan_type": "Loan type",
    "loan_purpose": "Loan purpose", "disbursal_channel": "Disbursal channel",
}
_MONEY = {"annual_income", "rp_emi", "bur_total_sanctioned", "bur_total_outstanding", "bur_writeoff_outstanding", "rp_shortfall_total"}
_PCT = {"rp_pct_ontime", "rp_paid_ratio", "bur_utilization", "bur_pct_fintech", "rp_balance_ratio_m12", "interest_rate_pct"}
_X = {"loan_to_income", "emi_to_income", "cust_emi_to_income"}
_FLAG = {"collateral_flag", "has_bureau", "bur_has_writeoff", "years_employed_capped", "years_employed_missing",
         "num_dependents_missing", "years_at_residence_missing", "interest_rate_missing"}


def label(f):
    return FEATURE_LABELS.get(f, f.replace("_", " ").capitalize())


def fmt_value(f, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, str):
        return v
    if f in _MONEY:
        return f"${v:,.0f}"
    if f == "interest_rate_pct":
        return f"{v:.1f}%"
    if f in _PCT:
        return f"{v:.0%}"
    if f in _X:
        return f"{v:.2f}x"
    if f in _FLAG:
        return "Yes" if v >= 0.5 else "No"
    if f == "log_loan_amount":
        return f"${np.expm1(v):,.0f}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:,.1f}" if float(v) != int(v) else f"{int(v)}"


# ---- data ---------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_portfolio():
    d = pd.read_parquet(PROC / "analytics_dataset.parquet")
    s = pd.read_parquet(MODEL_OUT / "scored_accounts.parquet")[["account_id", "pd_xgb", "pd_lr", "risk_band", "score_rank_pct", "split"]]
    df = d.merge(s, on="account_id", validate="1:1")
    df["def_exposure"] = df.loan_amount * df.target_default
    df["exp_default"] = df.loan_amount * df.pd_xgb
    df["risk_band"] = pd.Categorical(df.risk_band, BAND_ORDER, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_shap():
    return pd.read_parquet(MODEL_OUT / "shap_values_oof.parquet").set_index("account_id")


@st.cache_data(show_spinner=False)
def load_repayment():
    return pd.read_parquet(PROC / "repayment_clean.parquet")


@st.cache_data(show_spinner=False)
def load_collections():
    return pd.read_parquet(PROC / "collections_clean.parquet")


@st.cache_data(show_spinner=False)
def load_signals():
    return pd.read_parquet(ROOT / "outputs" / "anomaly" / "signals.parquet")


@st.cache_data(show_spinner=False)
def load_bureau():
    return pd.read_csv(ROOT / "data" / "raw" / "bureau_credit.csv")


# ---- stats / chart helpers ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    d = 1 + z * z / np.maximum(n, 1)
    c = (p + z * z / (2 * np.maximum(n, 1))) / d
    h = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z * z / (4 * np.maximum(n, 1) ** 2)) / d
    return c - h, c + h


def style(fig, height=340, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color=INK2), showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", ticks="outside", tickcolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def segment_table(df, col, order=None):
    g = df.groupby(col, observed=True).agg(
        accounts=("account_id", "size"), defaults=("target_default", "sum"), exposure=("loan_amount", "sum"),
        def_exposure=("def_exposure", "sum"), exp_default=("exp_default", "sum"), avg_pd=("pd_xgb", "mean"))
    g["default_rate"] = g.defaults / g.accounts
    g["ci_low"], g["ci_high"] = wilson(g.defaults, g.accounts)
    g["exposure_share"] = g.exposure / df.loan_amount.sum()
    g["acct_share"] = g.accounts / len(df)
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    return g


def money(v, d=1):
    return f"${v / 1e6:,.{d}f}M" if abs(v) >= 1e6 else f"${v / 1e3:,.0f}K"
