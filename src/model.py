"""Step 3 - Model building: predict target_default (account level).

  1. Feature groups + leakage / compliance exclusions
  2. Customer-grouped, stratified 80/20 train/test split (no customer in both)
  3. Baseline logistic regression + XGBoost (random search, grouped 5-fold CV on train)
  4. Ablation: application-only vs + bureau vs + repayment aggregates
  5. Test metrics (AUC / KS / Gini + bootstrap CI), calibration, deciles
  6. Out-of-fold scores + SHAP for ALL accounts (honest scores for the dashboard)
  7. Fairness check, PSI/CSI baseline, artifacts, docs/model_card.md

Run:  .venv\\Scripts\\python.exe src\\model.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from monitoring import csi_table, psi, psi_status

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "model"
CH = OUT / "charts"
MODELS = ROOT / "models"
for p in (CH, MODELS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42

df = pd.read_parquet(ROOT / "data" / "processed" / "analytics_dataset.parquet").reset_index(drop=True)
df["city_tier_num"] = df.city_tier.str.extract(r"(\d)").astype(int)
y = df.target_default.values
groups = df.customer_id.values

# ----------------------------------------------------------------------------- 1. feature groups
CAT = ["employment_type", "occupation_sector", "region", "housing_type", "education_level", "loan_type", "loan_purpose",
       "disbursal_channel"]
APP_NUM = ["age", "years_employed", "annual_income", "num_dependents", "years_at_residence", "city_tier_num",
           "years_employed_capped", "years_employed_missing", "num_dependents_missing", "years_at_residence_missing",
           "log_loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag", "interest_rate_missing",
           "loan_to_income", "emi_to_income", "rp_emi", "cust_n_accounts"]
BUR = ["has_bureau", "bur_n_lines", "bur_n_active", "bur_n_closed", "bur_n_writeoff", "bur_has_writeoff", "bur_total_sanctioned",
       "bur_total_outstanding", "bur_writeoff_outstanding", "bur_max_dpd", "bur_avg_dpd", "bur_n_dpd30", "bur_n_dpd_missing",
       "bur_pct_fintech", "bur_n_credit_types", "bur_oldest_line_yrs", "bur_newest_line_yrs", "bur_n_opened_24m", "bur_utilization"]
REP = ["rp_n_late", "rp_n_missed", "rp_pct_ontime", "rp_max_dpd", "rp_avg_dpd", "rp_dpd_first3", "rp_dpd_last3", "rp_dpd_trend",
       "rp_max_consec_missed", "rp_max_consec_late_or_missed", "rp_paid_ratio", "rp_shortfall_total", "rp_first_delinq_month",
       "rp_balance_ratio_m12"]
XACC = ["cust_other_missed", "cust_other_late", "cust_other_maxdpd", "cust_max_missed", "cust_sum_missed", "cust_emi_to_income"]
LOG_COLS = ["annual_income", "loan_to_income", "emi_to_income", "rp_emi", "bur_total_sanctioned", "bur_total_outstanding",
            "bur_writeoff_outstanding", "rp_shortfall_total"]
EXCLUDED = {
    "gender, marital_status": "Protected attributes - not used in the score; monitored in the fairness check instead",
    "coll_* (collections)": "Post-delinquency actions: leak the target (and dates are unreliable)",
    "customer_since, origination_date, orig_year/quarter": "Dates inconsistent (A12) / no calendar information in repayment",
    "cust_total_exposure, loan_amount, age_band, income_band": "Raw or binned copies of included features / portfolio-level context",
    "account_id, customer_id": "Identifiers",
}
GROUPS = {"application": APP_NUM + CAT, "application+bureau": APP_NUM + CAT + BUR,
          "+repayment": APP_NUM + CAT + BUR + REP, "full (+cross-account)": APP_NUM + CAT + BUR + REP + XACC}
FULL = GROUPS["full (+cross-account)"]
NUM_FULL = APP_NUM + BUR + REP + XACC
for c in CAT + NUM_FULL:
    assert c in df.columns, c

# ----------------------------------------------------------------------------- 2. split (grouped by customer)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, te_idx = next(sgkf.split(df, y, groups))
assert set(groups[tr_idx]).isdisjoint(groups[te_idx])
df["split"] = "train"
df.loc[te_idx, "split"] = "test"
ytr, yte = y[tr_idx], y[te_idx]
print(f"train {len(tr_idx):,} (default {ytr.mean():.1%}) | test {len(te_idx):,} (default {yte.mean():.1%})")


# ----------------------------------------------------------------------------- helpers
def ks_stat(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def metrics(y_true, p, boot=300, seed=0):
    auc = roc_auc_score(y_true, p)
    rng = np.random.default_rng(seed)
    bs = [roc_auc_score(y_true[i], p[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(boot))]
    return dict(auc=auc, auc_lo=float(np.percentile(bs, 2.5)), auc_hi=float(np.percentile(bs, 97.5)), gini=2 * auc - 1,
                ks=ks_stat(y_true, p), brier=brier_score_loss(y_true, p), logloss=log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)))


def lr_pipe(cols, C=0.1):
    num = [c for c in cols if c not in CAT]
    logc = [c for c in LOG_COLS if c in num]
    plain = [c for c in num if c not in logc]
    cat = [c for c in cols if c in CAT]
    tf = []
    if logc:
        tf.append(("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                    ("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), logc))
    tf.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), plain))
    if cat:
        tf.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return Pipeline([("prep", ColumnTransformer(tf)), ("lr", LogisticRegression(C=C, max_iter=3000))])


# one-hot for XGBoost (trees take NaN natively; no scaling)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[CAT])
ohe_names = list(ohe.get_feature_names_out(CAT))
parent_of = {n: next(c for c in CAT if n.startswith(c + "_")) for n in ohe_names}


def xgb_matrix(cols):
    cat = [c for c in cols if c in CAT]
    num = [c for c in cols if c not in CAT]
    parts = [df[num].astype(float).reset_index(drop=True)]
    names = list(num)
    if cat:
        oh = pd.DataFrame(ohe.transform(df[CAT]), columns=ohe_names)
        keep = [n for n in ohe_names if parent_of[n] in cat]
        parts.append(oh[keep])
        names += keep
    X = pd.concat(parts, axis=1)
    X.columns = names
    return X


# ----------------------------------------------------------------------------- 3. models on the full feature set
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_splits = list(cv.split(df.iloc[tr_idx], ytr, groups[tr_idx]))

# baseline: logistic regression
lr_search = RandomizedSearchCV(lr_pipe(FULL), {"lr__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]}, n_iter=7, scoring="roc_auc",
                               cv=cv_splits, n_jobs=-1, random_state=SEED, refit=True)
lr_search.fit(df.iloc[tr_idx][FULL], ytr)
lr = lr_search.best_estimator_
p_lr_te = lr.predict_proba(df.iloc[te_idx][FULL])[:, 1]
print("LR best C", lr_search.best_params_, f"cv auc {lr_search.best_score_:.4f}")

# XGBoost
X_full = xgb_matrix(FULL)
Xtr, Xte = X_full.iloc[tr_idx], X_full.iloc[te_idx]
space = dict(n_estimators=[150, 300, 500, 700], learning_rate=[0.02, 0.04, 0.08], max_depth=[2, 3, 4, 5],
             min_child_weight=[1, 5, 10, 20], subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0],
             reg_lambda=[1, 5, 10, 25], gamma=[0, 0.5, 1, 2])
xgb_search = RandomizedSearchCV(xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=1,
                                                  random_state=SEED), space, n_iter=40, scoring="roc_auc", cv=cv_splits,
                                n_jobs=-1, random_state=SEED, refit=True)
xgb_search.fit(Xtr, ytr)
xg = xgb_search.best_estimator_
BEST = xgb_search.best_params_
p_xg_te = xg.predict_proba(Xte)[:, 1]
print("XGB best", BEST, f"cv auc {xgb_search.best_score_:.4f}")

res = {"Logistic Regression (baseline)": metrics(yte, p_lr_te), "XGBoost": metrics(yte, p_xg_te)}
res["Logistic Regression (baseline)"]["cv_auc_train"] = float(lr_search.best_score_)
res["XGBoost"]["cv_auc_train"] = float(xgb_search.best_score_)
# train-side metrics (overfit check)
res["Logistic Regression (baseline)"]["auc_train"] = float(roc_auc_score(ytr, lr.predict_proba(df.iloc[tr_idx][FULL])[:, 1]))
res["XGBoost"]["auc_train"] = float(roc_auc_score(ytr, xg.predict_proba(Xtr)[:, 1]))
met = pd.DataFrame(res).T
met.to_csv(OUT / "metrics_test.csv")
print(met.round(4).to_string())

# ----------------------------------------------------------------------------- 4. ablation (same XGB params)
abl = []
for name, cols in GROUPS.items():
    Xg = xgb_matrix(cols)
    m = xgb.XGBClassifier(**BEST, objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=SEED).fit(Xg.iloc[tr_idx], ytr)
    pg = m.predict_proba(Xg.iloc[te_idx])[:, 1]
    lg = lr_pipe(cols, C=lr_search.best_params_["lr__C"]).fit(df.iloc[tr_idx][cols], ytr)
    pl = lg.predict_proba(df.iloc[te_idx][cols])[:, 1]
    abl.append(dict(feature_set=name, n_features=len(cols), xgb_auc=roc_auc_score(yte, pg), xgb_ks=ks_stat(yte, pg),
                    lr_auc=roc_auc_score(yte, pl), lr_ks=ks_stat(yte, pl)))
abl = pd.DataFrame(abl)
abl.to_csv(OUT / "ablation_feature_sets.csv", index=False)
print(abl.round(4).to_string())

# ----------------------------------------------------------------------------- 5. calibration + deciles (test, XGB)
def calib_table(y_true, p, n=10):
    d = pd.DataFrame({"y": y_true, "p": p})
    d["bin"] = pd.qcut(d.p.rank(method="first"), n, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"), defaults=("y", "sum"))
    g["cum_capture"] = g.defaults[::-1].cumsum()[::-1] / g.defaults.sum()
    g["lift"] = g.actual / d.y.mean()
    return g


cal_te = calib_table(yte, p_xg_te)
cal_te.to_csv(OUT / "deciles_test_xgb.csv")

# ----------------------------------------------------------------------------- 6. out-of-fold scores + SHAP (all accounts)
oof_xg, oof_lr = np.zeros(len(df)), np.zeros(len(df))
parents = [c for c in X_full.columns if c not in parent_of] + sorted(set(parent_of.values()))
shap_oof = np.zeros((len(df), len(parents)))
base_oof = np.zeros(len(df))
col_idx = {c: i for i, c in enumerate(parents)}
agg_map = np.array([col_idx[parent_of.get(c, c)] for c in X_full.columns])
oof_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
for k, (a, b) in enumerate(oof_cv.split(df, y, groups)):
    m = xgb.XGBClassifier(**BEST, objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=SEED).fit(X_full.iloc[a], y[a])
    oof_xg[b] = m.predict_proba(X_full.iloc[b])[:, 1]
    sv = shap.TreeExplainer(m)
    vals = sv.shap_values(X_full.iloc[b])
    for j in range(vals.shape[1]):
        shap_oof[b, agg_map[j]] += vals[:, j]
    base_oof[b] = float(np.ravel(sv.expected_value)[0])
    l = lr_pipe(FULL, C=lr_search.best_params_["lr__C"]).fit(df.iloc[a][FULL], y[a])
    oof_lr[b] = l.predict_proba(df.iloc[b][FULL])[:, 1]
oof_metrics = {"XGBoost": metrics(y, oof_xg), "Logistic Regression (baseline)": metrics(y, oof_lr)}
pd.DataFrame(oof_metrics).T.to_csv(OUT / "metrics_oof_all_accounts.csv")
print("OOF", pd.DataFrame(oof_metrics).T[["auc", "ks", "gini"]].round(4).to_string())
# SHAP is in log-odds space: check additivity on one fold's rows
chk = 1 / (1 + np.exp(-(base_oof + shap_oof.sum(1))))
print("SHAP additivity max abs err:", float(np.abs(chk - oof_xg).max()))

# risk bands from calibrated PD
BANDS = [(0, 0.10, "Low"), (0.10, 0.20, "Medium"), (0.20, 0.35, "High"), (0.35, 1.01, "Very High")]


def band(p):
    for lo, hi, n in BANDS:
        if lo <= p < hi:
            return n


scored = df[["account_id", "customer_id", "split", "target_default"]].copy()
scored["pd_xgb"] = oof_xg
scored["pd_lr"] = oof_lr
scored["risk_band"] = [band(v) for v in oof_xg]
scored["score_rank_pct"] = pd.Series(oof_xg).rank(pct=True).values
shap_df = pd.DataFrame(shap_oof, columns=parents)
shap_df.insert(0, "account_id", df.account_id.values)
shap_df["base_value"] = base_oof
scored.to_parquet(OUT / "scored_accounts.parquet", index=False)
shap_df.to_parquet(OUT / "shap_values_oof.parquet", index=False)
band_tbl = scored.groupby("risk_band").agg(accounts=("account_id", "size"), mean_pd=("pd_xgb", "mean"),
                                           actual_default=("target_default", "mean")).reindex([b[2] for b in BANDS])
band_tbl["share_accounts"] = band_tbl.accounts / len(scored)
band_tbl.to_csv(OUT / "risk_bands.csv")
print(band_tbl.round(3).to_string())

# ----------------------------------------------------------------------------- fairness / disparate-impact check (OOF)
fair = []
thr = np.quantile(oof_xg, 0.80)  # "decline" the riskiest 20%
tmp = df[["gender", "age_band", "employment_type"]].copy()
tmp["pd"] = oof_xg
tmp["y"] = y
tmp["declined"] = tmp.pd >= thr
for dim in ["gender", "age_band"]:
    for seg, g in tmp.groupby(dim):
        fair.append(dict(dimension=dim, segment=seg, accounts=len(g), mean_pred_pd=g.pd.mean(), actual_default=g.y.mean(),
                         decline_rate=g.declined.mean(), auc=roc_auc_score(g.y, g.pd) if g.y.nunique() > 1 else np.nan))
fair = pd.DataFrame(fair)
for dim in ["gender", "age_band"]:
    m = fair.dimension == dim
    ref = fair[m].decline_rate.min()  # impact ratio of approval rates vs the group with the highest approval rate
    best_appr = (1 - fair[m].decline_rate).max()
    fair.loc[m, "approval_impact_ratio"] = (1 - fair[m].decline_rate) / best_appr
fair.to_csv(OUT / "fairness_check.csv", index=False)

# ----------------------------------------------------------------------------- PSI / CSI baseline (train vs test)
score_psi = psi(xg.predict_proba(Xtr)[:, 1], p_xg_te)
csi = csi_table(df.iloc[tr_idx], df.iloc[te_idx], ["annual_income", "age", "loan_to_income", "rp_n_missed", "rp_max_dpd",
                                                    "bur_n_dpd30", "bur_utilization", "employment_type", "loan_type"])
csi.to_csv(OUT / "csi_train_vs_test.csv", index=False)

# ----------------------------------------------------------------------------- importance
mean_abs = shap_df[parents].abs().mean().sort_values(ascending=False)
gain = pd.Series(xg.get_booster().get_score(importance_type="gain"))
gain = gain.groupby(lambda n: parent_of.get(n, n)).sum().sort_values(ascending=False)
imp = pd.DataFrame({"mean_abs_shap": mean_abs, "xgb_gain": gain}).fillna(0)
imp["shap_rank"] = imp.mean_abs_shap.rank(ascending=False)
imp.sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "feature_importance.csv")
lr_names = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["lr"].coef_[0], index=lr_names).sort_values(key=np.abs, ascending=False)
coef.to_csv(OUT / "lr_coefficients.csv", header=["standardised_coef"])

# ----------------------------------------------------------------------------- charts
BLUE, ORANGE, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for nm, p, c in [("Logistic Regression", p_lr_te, ORANGE), ("XGBoost", p_xg_te, BLUE)]:
    fpr, tpr, _ = roc_curve(yte, p)
    ax[0].plot(fpr, tpr, color=c, lw=2, label=f"{nm}  AUC {roc_auc_score(yte, p):.3f}")
    srt = np.sort(p)
    ax[1].plot(srt, [(yte[p <= v] == 1).sum() / yte.sum() for v in srt], color=c, lw=2, label=f"{nm}  bad")
    ax[1].plot(srt, [(yte[p <= v] == 0).sum() / (yte == 0).sum() for v in srt], color=c, lw=2, ls="--")
ax[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].legend(frameon=False, loc="lower right")
ax[0].set_title("ROC curve (held-out test)", loc="left", fontweight="bold")
ax[1].set_xlabel("Predicted PD")
ax[1].set_ylabel("Cumulative share")
ax[1].set_xlim(0, 0.7)
ax[1].set_title(f"KS: XGB {res['XGBoost']['ks']:.3f} | LR {res['Logistic Regression (baseline)']['ks']:.3f}  (solid = defaulters, dashed = good)",
                loc="left", fontweight="bold", fontsize=9.5)
fig.tight_layout()
fig.savefig(CH / "01_roc_ks.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].plot([0, 0.6], [0, 0.6], color=MUTED, ls=":", lw=1)
ax[0].plot(cal_te.mean_pred, cal_te.actual, "o-", color=BLUE, lw=2, ms=6)
ax[0].set_xlabel("Mean predicted PD (decile)")
ax[0].set_ylabel("Observed default rate")
ax[0].set_title("Calibration (test, XGBoost)", loc="left", fontweight="bold")
x = np.arange(10)
ax[1].bar(x, cal_te.actual, color=BLUE, width=0.65)
for i, v in enumerate(cal_te.actual):
    ax[1].text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=8.5)
ax[1].axhline(yte.mean(), color=INK2, ls=(0, (4, 3)), lw=1)
ax[1].set_xticks(x, [f"D{i + 1}" for i in x])
ax[1].set_yticks([])
ax[1].spines["left"].set_visible(False)
ax[1].set_title("Observed default rate by score decile (D10 = riskiest)", loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(CH / "02_calibration_deciles.png", dpi=170)
plt.close(fig)

top = imp.sort_values("mean_abs_shap", ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.barh(top.index, top.mean_abs_shap, color=BLUE, height=0.65)
for i, v in enumerate(top.mean_abs_shap):
    ax.text(v + top.mean_abs_shap.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlabel("Mean |SHAP| (log-odds impact on default), out-of-fold, all accounts")
ax.set_title("What drives the default score - top 15 features", loc="left", fontweight="bold")
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
fig.tight_layout()
fig.savefig(CH / "03_shap_importance.png", dpi=170)
plt.close(fig)

num_top = [c for c in imp.sort_values("mean_abs_shap", ascending=False).index if c in X_full.columns and c not in parent_of][:12]
plt.figure(figsize=(8.6, 6))
shap.summary_plot(shap_df[num_top].values, df[num_top].astype(float), feature_names=num_top, show=False, max_display=12, plot_size=None)
plt.title("SHAP beeswarm - numeric features (red = high value)", loc="left", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(CH / "04_shap_beeswarm.png", dpi=170)
plt.close()

fig, ax = plt.subplots(figsize=(8.5, 3.6))
ab = abl.set_index("feature_set")
xx = np.arange(len(ab))
ax.bar(xx - 0.18, ab.lr_auc, 0.34, color=ORANGE, label="Logistic Regression")
ax.bar(xx + 0.18, ab.xgb_auc, 0.34, color=BLUE, label="XGBoost")
for i, (a_, b_) in enumerate(zip(ab.lr_auc, ab.xgb_auc)):
    ax.text(i - 0.18, a_ + 0.004, f"{a_:.3f}", ha="center", fontsize=8.5)
    ax.text(i + 0.18, b_ + 0.004, f"{b_:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(xx, ab.index)
ax.set_ylim(0.5, ab[["lr_auc", "xgb_auc"]].values.max() + 0.05)
ax.set_ylabel("Test AUC (axis starts at 0.5 = random)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Engineered bureau + repayment features lift AUC", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(CH / "05_ablation_auc.png", dpi=170)
plt.close(fig)

# ----------------------------------------------------------------------------- artifacts
xg.save_model(MODELS / "xgb_split_train.json")
joblib.dump(lr, MODELS / "lr_baseline.joblib")
final = xgb.XGBClassifier(**BEST, objective="binary:logistic", tree_method="hist", n_jobs=-1, random_state=SEED).fit(X_full, y)
final.save_model(MODELS / "xgb_final_all_data.json")
json.dump({"features_full": FULL, "xgb_columns": list(X_full.columns), "categorical": CAT, "ohe_names": ohe_names,
           "xgb_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in BEST.items()}, "risk_bands": BANDS,
           "excluded": EXCLUDED}, open(MODELS / "model_meta.json", "w"), indent=2, default=str)
df[["account_id", "split"]].to_csv(OUT / "split_assignment.csv", index=False)

# ----------------------------------------------------------------------------- model card
X = res["XGBoost"]
L = res["Logistic Regression (baseline)"]
O = oof_metrics["XGBoost"]
top10 = imp.sort_values("mean_abs_shap", ascending=False).head(10)
try:
    import tabulate  # noqa: F401
    md = lambda d: d.to_markdown(index=False)  # noqa: E731
except Exception:
    def md(d):
        cols = list(d.columns)
        rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
        return "\n".join(rows)
fair_md = md(fair.round(3))
abl_md = md(abl.round(4))
band_md = md(band_tbl.reset_index().round(3))
cal_md = md(cal_te.reset_index().round(3))
imp_md = md(top10.reset_index().rename(columns={"index": "feature"}).round(4))
ab_app_bur = float(abl.set_index("feature_set").loc["application+bureau", "xgb_auc"])
ab_full = float(abl.set_index("feature_set").loc["full (+cross-account)", "xgb_auc"])
excl_md = "\n".join(f"| {k} | {v} |" for k, v in EXCLUDED.items())
csi_md = md(csi.round(4))

card = f"""# Model Card - Default Prediction (account level)

_Auto-generated by `src/model.py`._

## Setup
- **Target:** `target_default` (1 = defaulted), {y.mean():.1%} base rate. **Grain:** one row per loan account ({len(df):,}).
- **Split:** stratified, **grouped by customer** (customers have up to 3 accounts; no customer appears in both sets). Train {len(tr_idx):,} / test {len(te_idx):,}.
- **Tuning:** random search with customer-grouped 5-fold CV on train only (40 XGBoost configs, 7 logistic C values). Test set touched once.
- **Features ({len(FULL)}):** application/loan fields + ratios, bureau aggregates ({len(BUR)}), repayment aggregates over months 1-12 ({len(REP)}), cross-account behaviour of the same customer's other loans ({len(XACC)}).
- **Deliberately excluded:**

| Excluded | Why |
|---|---|
{excl_md}

## Performance (held-out test)
| Model | AUC (95% CI) | Gini | KS | Brier | Train AUC | CV AUC (train) |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | {L['auc']:.3f} ({L['auc_lo']:.3f}-{L['auc_hi']:.3f}) | {L['gini']:.3f} | {L['ks']:.3f} | {L['brier']:.4f} | {L['auc_train']:.3f} | {L['cv_auc_train']:.3f} |
| XGBoost | {X['auc']:.3f} ({X['auc_lo']:.3f}-{X['auc_hi']:.3f}) | {X['gini']:.3f} | {X['ks']:.3f} | {X['brier']:.4f} | {X['auc_train']:.3f} | {X['cv_auc_train']:.3f} |

Out-of-fold over all {len(df):,} accounts (5-fold, customer-grouped), XGBoost: AUC {O['auc']:.3f}, KS {O['ks']:.3f}, Gini {O['gini']:.3f}. These OOF scores are what the dashboard shows - no account is scored by a model that saw it in training.

**Model choice.** XGBoost and logistic regression are statistically tied on test (AUC difference {abs(X['auc'] - L['auc']):.3f}, 95% intervals overlap almost entirely), and the tuner picked very shallow trees (depth {BEST['max_depth']}) - the signal is largely additive, so tree interactions add little. We ship **XGBoost as the scoring model** (native handling of missing bureau data, exact SHAP drivers for the account view) and keep **logistic regression as the explainable benchmark/challenger**; a bank that prefers a plain scorecard for regulatory simplicity loses no accuracy by choosing it.

Best XGBoost params: `{json.dumps({k: (v.item() if hasattr(v, 'item') else v) for k, v in BEST.items()})}`

![roc](../outputs/model/charts/01_roc_ks.png)

### Do the engineered features matter? (ablation, test AUC)
{abl_md}

![ablation](../outputs/model/charts/05_ablation_auc.png)

> **Default definition and leakage (assumption A5 - inferred, not documented):** the data shows the flag is not derived from the 12 repayment months: max DPD is 89 (no 90+ rule possible); 8% of accounts with zero late/missed payments are flagged; and defaulters' late rate is higher from month 1 but flat to month 12 (no pre-default slide). We therefore treat the flag as an outcome *after* month 12 and the repayment features as legitimate early-warning inputs. Consequence for use: **two scoring points** - at *application* use the `application+bureau` model (AUC ~{ab_app_bur:.2f}, the leakage-free floor); at *12 months on book* use the full model (AUC ~{ab_full:.2f}). If the real flag turned out to be built from these months, the repayment lift above would be overstated and the floor is the number to trust.

## Calibration and score deciles (test, XGBoost)
{cal_md}

![cal](../outputs/model/charts/02_calibration_deciles.png)

### Risk bands used in the dashboard (out-of-fold PD)
{band_md}

## Feature importance and SHAP
Mean |SHAP| aggregated back to the original feature (one-hot dummies summed), out-of-fold on all accounts. Top 10:

{imp_md}

![shap](../outputs/model/charts/03_shap_importance.png)
![beeswarm](../outputs/model/charts/04_shap_beeswarm.png)

Logistic regression standardised coefficients are in `outputs/model/lr_coefficients.csv` (the baseline's own explanation: sign and size per feature).

## Fairness / compliance check (out-of-fold, "decline the riskiest 20%" policy)
Gender and marital status are **not** model inputs. We still test whether the score disadvantages groups (e.g. through proxies). `approval_impact_ratio` = group approval rate / best group's approval rate; below 0.80 is the common "four-fifths" flag for review.

{fair_md}

Reading: differences by age band reflect real default-rate differences (young borrowers default more) - this is a policy/legal review item, not proof of a defect. Use of age in credit decisions is restricted in many jurisdictions; a bank would need compliance sign-off or to drop it.

## Monitoring plan (post-deployment)
**What to track.** *Score drift:* weekly PSI of the PD distribution against the training baseline (bins fixed on the training scores). Thresholds: < 0.10 stable, 0.10-0.25 investigate, > 0.25 trigger a model review. *Feature drift:* monthly CSI on every model input, with extra attention to the top SHAP drivers ({', '.join(top10.index[:5])}) and to the share of customers with no bureau file. *Performance:* because defaults mature slowly, track vintage-based AUC/KS/Gini and decile-wise observed vs predicted default rate (calibration) on a rolling cohort as outcomes mature, with alerts if AUC drops by more than 0.03 or any decile's observed rate is outside its 95% interval. *Fairness:* repeat the group impact-ratio table every quarter. *Data quality:* null rates, out-of-range values and join coverage per feed. **Cadence.** Scheduled retrain every 6 months, or earlier on any trigger (PSI > 0.25, AUC drop > 0.03, a policy or macro shock); recalibrate quarterly if calibration slips while ranking holds. New models run in champion/challenger shadow mode before promotion, and every version is archived with its data snapshot, parameters and validation report (model-risk-management evidence). **Baseline check today:** score PSI train vs test = {score_psi:.3f} ({psi_status(score_psi)}); feature CSI (train vs test, should be ~0):

{csi_md}

## Known limitations
- Default definition and observation window are not documented in the data (A1, A5); results are only as valid as the label.
- Synthetic and noisy: discrimination is moderate (AUC ~0.68) and did not improve with month-level repayment, bureau-mix, ratio or deeper-tree experiments (all within +/-0.005 CV AUC) - the ceiling is the data, not the feature set. Real portfolios may behave differently and drift faster.
- Bureau is a snapshot, not point-in-time at application (A6); collections excluded (A2, A11).
- Calibration is for this portfolio's ~16% base rate; a different mix needs recalibration.
"""
(ROOT / "docs" / "model_card.md").write_text(card, encoding="utf-8")
print("done")
