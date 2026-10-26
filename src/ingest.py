"""Step 1 - Data ingestion.

Loads the 5 raw tables, validates keys, cleans (every fix is logged with a count and
a treatment), aggregates the one-to-many tables, and writes ONE account-level
analysis-ready dataset.

Run:  .venv\\Scripts\\python.exe src\\ingest.py

Outputs
  data/processed/analytics_dataset.csv|parquet   one row per loan account (main output)
  data/processed/repayment_clean.parquet         cleaned monthly rows (dashboard curves)
  data/processed/collections_clean.parquet       cleaned collections events (analytics only)
  data/processed/column_roles.csv                role of every column (feature / analytics_only / ...)
  docs/data_quality_notes.md                     auto-generated from the DQ log below
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(exist_ok=True)

# Bureau is treated as a snapshot taken at the end of the data window (max date_opened = 2025-12-29).
SNAPSHOT = pd.Timestamp("2025-12-31")
MIN_WORKING_AGE = 18

dq_log: list[dict] = []
checks: list[tuple[str, bool, str]] = []


def log(table, column, issue, n, total, treatment):
    dq_log.append(dict(table=table, column=column, issue=issue, n=int(n), pct=100 * n / total, treatment=treatment))


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# --------------------------------------------------------------------------- load
cust = pd.read_csv(RAW / "customers.csv", parse_dates=["customer_since"])
loan = pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"])
bur = pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"])
rep = pd.read_csv(RAW / "repayment_history.csv")
col = pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"])
raw_rows = {k: len(v) for k, v in dict(customers=cust, loan_accounts=loan, bureau_credit=bur,
                                       repayment_history=rep, collections_events=col).items()}

# --------------------------------------------------------------------------- key integrity
check("customers.customer_id unique", cust.customer_id.is_unique)
check("loan_accounts.account_id unique", loan.account_id.is_unique)
check("bureau_credit.bureau_id unique", bur.bureau_id.is_unique)
check("repayment_history (account_id, payment_month) unique", not rep.duplicated(["account_id", "payment_month"]).any())
check("collections_events.collection_event_id unique", col.collection_event_id.is_unique)
check("loan_accounts.customer_id all in customers", loan.customer_id.isin(cust.customer_id).all())
check("bureau_credit.customer_id all in customers", bur.customer_id.isin(cust.customer_id).all())
check("repayment_history.account_id all in loan_accounts", rep.account_id.isin(loan.account_id).all())
check("collections_events.account_id all in loan_accounts", col.account_id.isin(loan.account_id).all())
check("Every account has exactly 12 repayment months", (rep.groupby("account_id").size() == 12).all(),
      "repayment window is months 1-12 regardless of tenure")

# --------------------------------------------------------------------------- customers
n = len(cust)
bad_gender = cust.gender.isin(["M", "F"])
log("customers", "gender", "Inconsistent labels ('M'/'F' vs 'Male'/'Female')", bad_gender.sum(), n,
    "Mapped M->Male, F->Female (no information lost)")
cust["gender"] = cust.gender.replace({"M": "Male", "F": "Female"})

# Implausible tenure: years employed more than (age - 18)
too_long = cust.years_employed > (cust.age - MIN_WORKING_AGE)
log("customers", "years_employed", f"Impossible: years_employed > age - {MIN_WORKING_AGE}", too_long.sum(), n,
    f"Capped at age - {MIN_WORKING_AGE}; flagged in years_employed_capped")
cust["years_employed_capped"] = too_long.astype(int)
cust["years_employed"] = cust.years_employed.where(~too_long, cust.age - MIN_WORKING_AGE)

res_gt_age = cust.years_at_residence > cust.age
log("customers", "years_at_residence", "years_at_residence > age", res_gt_age.sum(), n,
    "Capped at age; negligible volume")
cust["years_at_residence"] = cust.years_at_residence.where(~res_gt_age, cust.age)

# Missing values: impute with group medians + keep a missingness flag (missing-at-random assumed)
for c_, grp in [("years_employed", "employment_type"), ("years_at_residence", "housing_type"), ("num_dependents", "marital_status")]:
    miss = cust[c_].isna()
    log("customers", c_, "Missing values", miss.sum(), n,
        f"Imputed with median by {grp}; flag column {c_}_missing added")
    cust[f"{c_}_missing"] = miss.astype(int)
    cust[c_] = cust[c_].fillna(cust.groupby(grp)[c_].transform("median"))
cust["num_dependents"] = cust.num_dependents.round().astype(int)

floor = cust.annual_income == cust.annual_income.min()
log("customers", "annual_income", f"Censored at floor value ({cust.annual_income.min():,.0f}) - looks like a floor/clip", floor.sum(), n,
    "Kept as is (true income is unknown, not wrong); noted as a limitation")

# --------------------------------------------------------------------------- loan accounts
n = len(loan)
miss = loan.interest_rate_pct.isna()
log("loan_accounts", "interest_rate_pct", "Missing values", miss.sum(), n,
    "Imputed with median by loan_type x tenure_months; flag column interest_rate_missing added")
loan["interest_rate_missing"] = miss.astype(int)
loan["interest_rate_pct"] = loan.interest_rate_pct.fillna(
    loan.groupby(["loan_type", "tenure_months"]).interest_rate_pct.transform("median"))
loan["interest_rate_pct"] = loan.interest_rate_pct.fillna(loan.groupby("loan_type").interest_rate_pct.transform("median"))

orig_before = loan.merge(cust[["customer_id", "customer_since"]], on="customer_id")
inc = (orig_before.origination_date < orig_before.customer_since)
log("loan_accounts", "origination_date", "Loan originated before customer_since (relationship start)", inc.sum(), n,
    "Not corrected (cannot tell which date is wrong); customer_since NOT used as a model feature")
log("loan_accounts", "loan_amount", "Right-skewed (mortgages up to ~445k vs median ~17k)", 0, n,
    "Kept; log_loan_amount derived for modelling; analytics segment by loan_type")

# --------------------------------------------------------------------------- bureau
n = len(bur)
closed = bur.account_status == "Closed"
miss_dpd = bur.days_past_due.isna()
log("bureau_credit", "days_past_due", "Missing values", miss_dpd.sum(), n,
    "Closed lines (DPD is always 0 on closed lines in this data): filled 0. Active/Written-Off: left NaN, "
    "aggregates ignore them and bur_n_dpd_missing counts them")
log("bureau_credit", "days_past_due", " - of which on Closed lines (filled with 0)", (miss_dpd & closed).sum(), n, "see above")
bur["days_past_due"] = bur.days_past_due.mask(miss_dpd & closed, 0)

fut = bur.date_closed > SNAPSHOT
log("bureau_credit", "date_closed", f"Closure date in the future (after snapshot {SNAPSHOT.date()})", fut.sum(), n,
    "Kept status as reported; date_closed not used in features")
log("bureau_credit", "date_closed", "Null date_closed - exactly the Active lines (consistent with account_status)",
    bur.date_closed.isna().sum(), n, "Expected, no action")
check("date_closed null <=> account_status == Active", bur.date_closed.isna().equals(bur.account_status == "Active"))
check("bureau outstanding_amount <= sanctioned_amount", (bur.outstanding_amount <= bur.sanctioned_amount).all())

no_bureau = ~cust.customer_id.isin(bur.customer_id)
log("customers x bureau", "customer_id", "Customers with NO bureau record (thin / no-file)", no_bureau.sum(), len(cust),
    "Left-join; has_bureau=0, counts/amounts=0, ratio features (utilisation, max/avg DPD, line ages) left NaN "
    "(NOT 0 - 'no history' is not 'clean history')")

# Bureau -> one row per customer
b = bur.copy()
b["is_active"] = (b.account_status == "Active").astype(int)
b["is_closed"] = closed.astype(int)
b["is_wo"] = (b.account_status == "Written-Off").astype(int)
b["dpd30"] = (b.days_past_due >= 30).astype(int)
b["dpd_missing"] = b.days_past_due.isna().astype(int)
b["fintech"] = (b.lender_category == "Fintech").astype(int)
b["age_yrs"] = (SNAPSHOT - b.date_opened).dt.days / 365.25
b["opened_24m"] = (b.date_opened > SNAPSHOT - pd.DateOffset(months=24)).astype(int)
b["active_out"] = b.outstanding_amount.where(b.is_active == 1, 0)
b["active_san"] = b.sanctioned_amount.where(b.is_active == 1, 0)
bg = b.groupby("customer_id")
bur_feat = pd.DataFrame({
    "bur_n_lines": bg.size(),
    "bur_n_active": bg.is_active.sum(),
    "bur_n_closed": bg.is_closed.sum(),
    "bur_n_writeoff": bg.is_wo.sum(),
    "bur_total_sanctioned": bg.sanctioned_amount.sum(),
    "bur_total_outstanding": bg.outstanding_amount.sum(),
    "bur_max_dpd": bg.days_past_due.max(),
    "bur_avg_dpd": bg.days_past_due.mean(),
    "bur_n_dpd30": bg.dpd30.sum(),
    "bur_n_dpd_missing": bg.dpd_missing.sum(),
    "bur_pct_fintech": bg.fintech.mean(),
    "bur_n_credit_types": bg.credit_type.nunique(),
    "bur_oldest_line_yrs": bg.age_yrs.max(),
    "bur_newest_line_yrs": bg.age_yrs.min(),
    "bur_n_opened_24m": bg.opened_24m.sum(),
})
bur_feat["bur_has_writeoff"] = (bur_feat.bur_n_writeoff > 0).astype(int)
bur_feat["bur_utilization"] = (bg.active_out.sum() / bg.active_san.sum().replace(0, np.nan))
bur_feat["bur_writeoff_outstanding"] = b[b.is_wo == 1].groupby("customer_id").outstanding_amount.sum()
bur_feat["bur_writeoff_outstanding"] = bur_feat.bur_writeoff_outstanding.fillna(0)
bur_feat = bur_feat.reset_index()

# --------------------------------------------------------------------------- repayment (12 months per account)
n = len(rep)
check("Missed => emi_paid == 0", (rep.loc[rep.payment_status == "Missed", "emi_paid"] == 0).all())
check("On-Time <=> days_past_due == 0", ((rep.payment_status == "On-Time") == (rep.days_past_due == 0)).all())
check("emi_paid <= emi_due", (rep.emi_paid <= rep.emi_due).all())
rep = rep.sort_values(["account_id", "payment_month"])
rep["late_or_missed"] = (rep.payment_status != "On-Time").astype(int)
rep["missed"] = (rep.payment_status == "Missed").astype(int)
rep["late"] = (rep.payment_status == "Late").astype(int)
rep["shortfall"] = rep.emi_due - rep.emi_paid


def max_run(s):
    best = cur = 0
    for v in s:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rg = rep.groupby("account_id")
first3 = rep[rep.payment_month <= 3].groupby("account_id").days_past_due.mean()
last3 = rep[rep.payment_month >= 10].groupby("account_id").days_past_due.mean()
first_bad = rep[rep.late_or_missed == 1].groupby("account_id").payment_month.min()
rep_feat = pd.DataFrame({
    "rp_emi": rg.emi_due.first(),
    "rp_n_late": rg.late.sum(),
    "rp_n_missed": rg.missed.sum(),
    "rp_pct_ontime": 1 - rg.late_or_missed.mean(),
    "rp_max_dpd": rg.days_past_due.max(),
    "rp_avg_dpd": rg.days_past_due.mean(),
    "rp_dpd_first3": first3,
    "rp_dpd_last3": last3,
    "rp_dpd_trend": last3 - first3,
    "rp_max_consec_missed": rg.missed.apply(max_run),
    "rp_max_consec_late_or_missed": rg.late_or_missed.apply(max_run),
    "rp_paid_ratio": rg.emi_paid.sum() / rg.emi_due.sum(),
    "rp_shortfall_total": rg.shortfall.sum(),
})
rep_feat["rp_first_delinq_month"] = first_bad.reindex(rep_feat.index).fillna(13)  # 13 = never delinquent in window
last_bal = rep[rep.payment_month == 12].set_index("account_id").outstanding_balance
rep_feat["rp_balance_m12"] = last_bal
rep_feat = rep_feat.reset_index()
log("repayment_history", "payment_month", "Window is always months 1-12 regardless of tenure/origination date", len(loan), len(loan),
    "Treated as 'first 12 months on book' behaviour; no calendar trend possible (see assumptions A3, A5)")

# --------------------------------------------------------------------------- collections (analytics only)
n = len(col)
col = col.merge(loan[["account_id", "origination_date", "target_default"]], on="account_id")
before = col.event_date < col.origination_date
log("collections_events", "event_date", "Event dated BEFORE the loan's origination_date", before.sum(), n,
    "Flagged (coll_n_before_orig); dates untrusted -> collections used for analytics only, never as model features")
share_def = col.target_default.mean()
log("collections_events", "account_id", f"Only {share_def:.0%} of events belong to defaulted accounts (dictionary says 'only delinquent')",
    (col.target_default == 0).sum(), n, "Kept; collections coverage is not a reliable proxy for default")
col["before_orig"] = before.astype(int)
col["legal_or_wo"] = col.action_type.isin(["Legal Notice", "Written-Off Referral"]).astype(int)
cg = col.groupby("account_id")
col_feat = pd.DataFrame({
    "coll_n_events": cg.size(),
    "coll_total_recovered": cg.amount_recovered.sum(),
    "coll_n_legal_or_wo": cg.legal_or_wo.sum(),
    "coll_n_before_orig": cg.before_orig.sum(),
}).reset_index()
col_clean = col.drop(columns=["target_default"])

# --------------------------------------------------------------------------- join -> account grain
df = (loan
      .merge(cust, on="customer_id", how="left", validate="m:1")
      .merge(bur_feat, on="customer_id", how="left", validate="m:1")
      .merge(rep_feat, on="account_id", how="left", validate="1:1")
      .merge(col_feat, on="account_id", how="left", validate="1:1"))
check("Join preserved row count (no fan-out): rows == loan_accounts", len(df) == len(loan), f"{len(df)} rows")
check("No account missing repayment features", df.rp_n_late.notna().all())

df["has_bureau"] = df.bur_n_lines.notna().astype(int)
zero_fill = ["bur_n_lines", "bur_n_active", "bur_n_closed", "bur_n_writeoff", "bur_total_sanctioned", "bur_total_outstanding",
             "bur_n_dpd30", "bur_n_dpd_missing", "bur_n_credit_types", "bur_n_opened_24m", "bur_has_writeoff",
             "bur_writeoff_outstanding"]
df[zero_fill] = df[zero_fill].fillna(0)
for c_ in ["coll_n_events", "coll_total_recovered", "coll_n_legal_or_wo", "coll_n_before_orig"]:
    df[c_] = df[c_].fillna(0)
df["coll_has_event"] = (df.coll_n_events > 0).astype(int)

# customer-level portfolio context + derived ratios / bands
df["cust_n_accounts"] = df.groupby("customer_id").account_id.transform("size")
df["cust_total_exposure"] = df.groupby("customer_id").loan_amount.transform("sum")
# cross-account behaviour: how the same customer behaves on their OTHER accounts (uses repayment only, never other accounts' target)
_gb, _others = df.groupby("customer_id"), (df.cust_n_accounts - 1).replace(0, np.nan)
for _col, _nm in [("rp_n_missed", "missed"), ("rp_n_late", "late"), ("rp_max_dpd", "maxdpd")]:
    df[f"cust_other_{_nm}"] = (_gb[_col].transform("sum") - df[_col]) / _others
df["cust_max_missed"] = _gb.rp_n_missed.transform("max")
df["cust_sum_missed"] = _gb.rp_n_missed.transform("sum")
df["cust_emi_to_income"] = _gb.rp_emi.transform("sum") * 12 / df.annual_income
df["log_loan_amount"] = np.log1p(df.loan_amount)
df["loan_to_income"] = df.loan_amount / df.annual_income
df["emi_to_income"] = df.rp_emi * 12 / df.annual_income
df["orig_year"] = df.origination_date.dt.year
df["orig_quarter"] = df.origination_date.dt.to_period("Q").astype(str)
df["age_band"] = pd.cut(df.age, [0, 29, 39, 49, 59, 200], labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
df["income_band"] = pd.cut(df.annual_income, [0, 40_000, 60_000, 80_000, 100_000, 1e9],
                           labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"]).astype(str)
df["rp_balance_ratio_m12"] = df.rp_balance_m12 / df.loan_amount
df = df.drop(columns=["rp_balance_m12"])

# --------------------------------------------------------------------------- column roles
roles = {}
for c_ in df.columns:
    if c_ in ("account_id", "customer_id"):
        roles[c_] = "id"
    elif c_ == "target_default":
        roles[c_] = "target"
    elif c_ in ("gender", "marital_status"):
        roles[c_] = "excluded (protected attribute - fairness-tested, not scored)"
    elif c_.startswith("coll_"):
        roles[c_] = "analytics_only (post-delinquency, leaks target)"
    elif c_ in ("customer_since", "origination_date", "orig_year", "orig_quarter", "cust_total_exposure"):
        roles[c_] = "analytics_only (dates inconsistent / portfolio context)"
    elif c_ in ("age_band", "income_band", "loan_amount"):
        roles[c_] = "analytics_only (binned / raw copy of model feature)"
    else:
        roles[c_] = "model_candidate"
pd.Series(roles, name="role").rename_axis("column").reset_index().to_csv(OUT / "column_roles.csv", index=False)

# --------------------------------------------------------------------------- write
df.to_csv(OUT / "analytics_dataset.csv", index=False)
df.to_parquet(OUT / "analytics_dataset.parquet", index=False)
rep.drop(columns=["late_or_missed", "missed", "late", "shortfall"]).to_parquet(OUT / "repayment_clean.parquet", index=False)
col_clean.to_parquet(OUT / "collections_clean.parquet", index=False)

# --------------------------------------------------------------------------- DQ notes (auto-generated)
rows = "\n".join(
    f"| {d['table']} | {d['column']} | {d['issue']} | {d['n']:,} | {d['pct']:.1f}% | {d['treatment']} |"
    for d in dq_log if d["n"] > 0 or "Window" in d["issue"] or "skewed" in d["issue"])
chk = "\n".join(f"| {'PASS' if ok else '**FAIL**'} | {name} | {detail} |" for name, ok, detail in checks)
miss_final = df.isna().sum()
miss_final = miss_final[miss_final > 0]
miss_tbl = "\n".join(f"| {k} | {v:,} | {100 * v / len(df):.1f}% |" for k, v in miss_final.items())
(DOCS / "data_quality_notes.md").write_text(f"""# Data Quality Notes

_Auto-generated by `src/ingest.py` - re-run the script to refresh._

## Join logic
| Step | Join | Cardinality | Note |
|---|---|---|---|
| base | `loan_accounts` | 1 row / account (**{len(loan):,}**) | Output grain = loan account, because `target_default` is account-level |
| + customers | `customer_id` | many accounts : 1 customer (validated `m:1`) | Demographics / income / region |
| + bureau | `customer_id` | bureau lines **aggregated to 1 row / customer first**, then `m:1` | Avoids fan-out ({raw_rows['bureau_credit']:,} lines -> {len(bur_feat):,} customers) |
| + repayment | `account_id` | 12 monthly rows **aggregated to 1 row / account first**, then `1:1` | Delinquency counts, DPD trend, paid ratio, balance run-off |
| + collections | `account_id` | events **aggregated to 1 row / account first**, then `1:1` | Analytics only - never model features |

Raw rows: {', '.join(f'{k} {v:,}' for k, v in raw_rows.items())}. Final dataset: **{len(df):,} rows x {df.shape[1]} columns**.

## Integrity checks
| Result | Check | Detail |
|---|---|---|
{chk}

## Issues found and how each was handled
| Table | Column | Issue | Rows | % | Treatment |
|---|---|---|---|---|---|
{rows}

## Remaining NaNs in the final dataset (intentional)
These are semantic "no history" values (no bureau file, or no active line to measure utilisation on), not errors. Tree models handle them natively; the logistic baseline will median-impute + use `has_bureau`.

| Column | NaN rows | % |
|---|---|---|
{miss_tbl}

## Principles used
1. **Never silently drop rows** - every account survives ingestion ({len(df):,} of {len(loan):,}).
2. **Impute only where the missing mechanism is plausibly benign**, with group medians, and always keep a `*_missing` flag so the model/analyst can see it.
3. **Don't fix what we can't verify** (e.g. conflicting dates are flagged and the columns excluded from modelling, rather than guessed).
4. **Absence != zero**: customers with no bureau file get `has_bureau=0`, and ratio features stay NaN.
5. **Leakage control**: columns are tagged in `data/processed/column_roles.csv`; collections columns are `analytics_only`.
""", encoding="utf-8")

print(f"OK: {len(df):,} rows x {df.shape[1]} cols -> {OUT}")
print("Checks failed:", [c_[0] for c_ in checks if not c_[1]])
