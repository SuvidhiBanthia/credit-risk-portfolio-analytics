"""Step 5 - Anomaly detection (the embedded AI feature).

Three unsupervised detectors (the default flag is NEVER used to build them - only to evaluate them afterwards):

  1. OWN HISTORY  - "sudden DPD spike": worst DPD in the latest 3 months (10-12) is >= `spike_dpd` while the
                    account's own earlier months (1-9) never exceeded `base_max` days past due.
  2. PEER SEGMENT - multivariate outlier: IsolationForest fitted within each loan type on repayment + bureau +
                    affordability features; flag the top (1 - `if_cut`) share of that loan type. The reasons are
                    the features furthest from the peer median (robust z-score).
  3. DIVERGENCE   - repayment and bureau disagree: (a) bureau shows stress but the customer is clean with us;
                    (b) >= 3 missed payments with us but a clean bureau record.

`apply_thresholds` is a pure function so the dashboard can re-apply thresholds live.

Run:  .venv\\Scripts\\python.exe src\\anomaly.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "anomaly"

DEFAULTS = dict(spike_dpd=60, base_max=14, if_cut=0.97)
IF_FEATURES = ["rp_avg_dpd", "rp_n_missed", "rp_n_late", "rp_max_dpd", "rp_dpd_trend", "bur_avg_dpd", "bur_max_dpd", "bur_utilization",
               "bur_n_writeoff", "bur_n_dpd30", "bur_n_opened_24m", "bur_n_lines", "loan_to_income", "emi_to_income", "rp_paid_ratio"]
LOWER_IS_WORSE = {"rp_paid_ratio"}  # for all other features, a high value is the "worse" direction
FEATURE_TEXT = {
    "rp_avg_dpd": "average DPD (months 1-12)", "rp_n_missed": "missed payments", "rp_n_late": "late payments", "rp_max_dpd": "worst DPD (months 1-12)",
    "rp_dpd_trend": "DPD trend", "bur_avg_dpd": "average bureau DPD", "bur_max_dpd": "worst bureau DPD", "bur_utilization": "bureau utilisation",
    "bur_n_writeoff": "bureau write-offs", "bur_n_dpd30": "bureau lines 30+ DPD", "bur_n_opened_24m": "new bureau lines (24m)",
    "bur_n_lines": "bureau credit lines", "loan_to_income": "loan / income", "emi_to_income": "EMIs / income", "rp_paid_ratio": "paid / due ratio",
}


# ----------------------------------------------------------------------------- signals (computed once)
def compute_signals(d: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest

    P = rep.pivot(index="account_id", columns="payment_month", values="days_past_due")
    d = d.set_index("account_id").loc[P.index]
    A = P.values
    base, rec = A[:, :9], A[:, 9:]
    sig = pd.DataFrame(index=P.index)
    sig["base_worst_dpd"] = base.max(1)
    sig["base_mean_dpd"] = base.mean(1)
    sig["base_ontime_months"] = (base == 0).sum(1)
    sig["rec_worst_dpd"] = rec.max(1)
    sig["spike_month"] = 10 + rec.argmax(1)
    sig["own_z"] = (sig.rec_worst_dpd - sig.base_mean_dpd) / np.maximum(base.std(1), 10)
    sig["escalating"] = ((A[:, 9] < A[:, 10]) & (A[:, 10] < A[:, 11]) & (A[:, 11] >= 45)).astype(int)
    for m in range(12):
        sig[f"dpd_m{m + 1}"] = A[:, m]

    # peer outliers: IsolationForest within each loan type (unsupervised)
    X = d[IF_FEATURES].copy()
    X = X.fillna(X.median())
    sc = np.zeros(len(d))
    top = [None] * len(d)
    z_all = np.zeros_like(X.values, dtype=float)
    for lt, idx in d.groupby("loan_type").indices.items():
        sub = X.iloc[idx]
        m = IsolationForest(n_estimators=300, random_state=0).fit(sub)
        sc[idx] = -m.score_samples(sub)
        med = sub.median()
        mad = (sub - med).abs().median() * 1.4826
        scale = np.maximum.reduce([mad.values, 0.5 * sub.std().values, np.full(sub.shape[1], 1e-6)])
        z = (sub - med).values / scale
        for j, f in enumerate(IF_FEATURES):
            if f in LOWER_IS_WORSE:
                z[:, j] = -z[:, j]
        z_all[idx] = z
    sig["if_score"] = sc
    sig["if_pct"] = pd.Series(sc, index=sig.index).groupby(d.loan_type).rank(pct=True)
    order = np.argsort(-z_all, axis=1)[:, :3]
    med_all = X.groupby(d.loan_type).transform("median")
    sig["peer_top"] = [
        json.dumps([[IF_FEATURES[j], float(X.iloc[i, j]), float(med_all.iloc[i, j]), float(z_all[i, j])] for j in order[i] if z_all[i, j] > 0.5])
        for i in range(len(d))]

    # divergence between bureau and repayment
    local_bad = d.rp_n_late + d.rp_n_missed
    sig["div_hidden_stress"] = (((d.bur_n_dpd30 >= 2) | (d.bur_n_writeoff >= 2)) & (local_bad == 0)).astype(int)
    sig["div_local_stress"] = (((d.bur_max_dpd.fillna(0) == 0) & (d.bur_n_lines > 0) & (d.rp_n_missed >= 3))).astype(int)
    return sig


# ----------------------------------------------------------------------------- thresholds -> flags (pure, reused by dashboard)
def apply_thresholds(sig, spike_dpd=60, base_max=14, if_cut=0.97, use_spike=True, use_peer=True, use_div=True):
    t_spike = use_spike & (sig.rec_worst_dpd >= spike_dpd) & (sig.base_worst_dpd <= base_max)
    t_peer = use_peer & (sig.if_pct >= if_cut)
    t_hid = use_div & (sig.div_hidden_stress == 1)
    t_loc = use_div & (sig.div_local_stress == 1)
    s_spike = (0.4 + 0.6 * ((sig.rec_worst_dpd - spike_dpd) / max(89 - spike_dpd, 1)).clip(0, 1) + 0.1 * sig.escalating).clip(upper=1.0)
    s_peer = 0.5 + 0.5 * ((sig.if_pct - if_cut) / max(1 - if_cut, 1e-9)).clip(0, 1)
    comp = pd.DataFrame({"spike": s_spike.where(t_spike, 0), "peer": s_peer.where(t_peer, 0),
                         "diverge_local": pd.Series(0.7, index=sig.index).where(t_loc, 0),
                         "diverge_hidden": pd.Series(0.4, index=sig.index).where(t_hid, 0)})
    fired = (comp > 0)
    out = pd.DataFrame(index=sig.index)
    out["t_spike"], out["t_peer"], out["t_hidden"], out["t_local"] = t_spike, t_peer, t_hid, t_loc
    out["n_triggers"] = fired.sum(axis=1)
    out["flagged"] = out.n_triggers > 0
    out["severity"] = (100 * comp.max(axis=1) + 10 * (out.n_triggers - 1).clip(lower=0)).clip(upper=100).round(0)
    out.loc[~out.flagged, "severity"] = 0
    out["primary"] = comp.idxmax(axis=1).where(out.flagged, "")
    return out


def reasons(row, sig_row, loan_type, labels=None):
    """Human-readable reasons for one flagged account."""
    r = []
    if row.t_spike:
        r.append(f"Own history: month {int(sig_row.spike_month)} DPD {int(sig_row.rec_worst_dpd)} days vs worst {int(sig_row.base_worst_dpd)} in months 1-9")
    if row.t_peer:
        feats = json.loads(sig_row.peer_top)
        txt = ", ".join(f"{FEATURE_TEXT[f]} {v:.2g} (peer median {m:.2g})" for f, v, m, z in feats) or "overall profile"
        r.append(f"Peer outlier (top {100 * (1 - sig_row.if_pct):.1f}% of {loan_type}): {txt}")
    if row.t_local:
        r.append("Missed 3+ payments with us but bureau shows no delinquency (bank-specific stress, dispute or data issue?)")
    if row.t_hidden:
        r.append("Bureau shows stress elsewhere (2+ lines 30+ DPD or 2+ write-offs) but no late/missed payments with us")
    return r


# ----------------------------------------------------------------------------- pipeline
def _md(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ROOT / "data" / "processed" / "analytics_dataset.parquet")
    rep = pd.read_parquet(ROOT / "data" / "processed" / "repayment_clean.parquet")
    scored = pd.read_parquet(ROOT / "outputs" / "model" / "scored_accounts.parquet").set_index("account_id")
    sig = compute_signals(d, rep)
    sig.to_parquet(OUT / "signals.parquet")
    dd = d.set_index("account_id").loc[sig.index].join(scored[["pd_xgb", "risk_band"]])
    y = dd.target_default.values
    base_rate = y.mean()
    n_def = y.sum()

    # ---- evaluation at default thresholds
    fl = apply_thresholds(sig, **DEFAULTS)
    rows = []
    for name, m in [("Own-history DPD spike", fl.t_spike), ("Peer-segment outlier", fl.t_peer), ("Divergence: clean bureau, 3+ missed with us", fl.t_local),
                    ("Divergence: bureau stress, clean with us", fl.t_hidden), ("ANY detector", fl.flagged)]:
        m = m.values
        rows.append(dict(detector=name, alerts=int(m.sum()), pct_of_accounts=f"{m.mean():.1%}", default_rate_flagged=f"{y[m].mean():.1%}",
                         default_rate_rest=f"{y[~m].mean():.1%}", lift=f"{y[m].mean() / y[~m].mean():.2f}x",
                         share_of_all_defaults_caught=f"{y[m].sum() / n_def:.1%}",
                         flagged_but_performing_FP_proxy=int((m & (y == 0)).sum()), missed_defaults_FN_proxy=int((~m & (y == 1)).sum())))
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "detector_eval.csv", index=False)

    # ---- sensitivity: how threshold choices trade volume vs precision proxy
    sens = []
    for c in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]:
        f_ = apply_thresholds(sig, use_spike=False, use_div=False, if_cut=c).flagged.values
        sens.append(dict(detector="peer", threshold=f"top {1 - c:.0%}", alerts=int(f_.sum()), default_rate=round(y[f_].mean(), 3), recall=round(y[f_].sum() / n_def, 3)))
    for s in [30, 45, 60, 75, 85]:
        f_ = apply_thresholds(sig, use_peer=False, use_div=False, spike_dpd=s).flagged.values
        sens.append(dict(detector="spike", threshold=f"DPD>={s}", alerts=int(f_.sum()), default_rate=round(y[f_].mean(), 3), recall=round(y[f_].sum() / n_def, 3)))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity.csv", index=False)

    # ---- injection test for the spike detector (known ground truth)
    rng = np.random.default_rng(0)
    A = sig[[f"dpd_m{m}" for m in range(1, 13)]].values.copy()
    clean = np.where(sig.base_worst_dpd.values <= DEFAULTS["base_max"])[0]
    pick = rng.choice(clean, 1500, replace=False)
    untouched = np.setdiff1d(clean, pick)
    inj = []
    for size in [30, 45, 60, 75, 89]:
        B = A.copy()
        B[pick, 11] = np.maximum(B[pick, 11], size)
        s2 = sig.copy()
        s2["rec_worst_dpd"] = B[:, 9:].max(1)
        f2 = apply_thresholds(s2, use_peer=False, use_div=False, **{k: v for k, v in DEFAULTS.items() if k != "if_cut"}).t_spike.values
        inj.append(dict(injected_spike_days=size, recall_on_injected=f"{f2[pick].mean():.0%}", natural_alert_rate_untouched_clean_accounts=f"{f2[untouched].mean():.1%}"))
    inj = pd.DataFrame(inj)
    inj.to_csv(OUT / "injection_test.csv", index=False)

    # ---- flags + priority at default thresholds
    flags = fl.join(dd[["customer_id", "loan_type", "loan_amount", "pd_xgb", "risk_band", "target_default"]])
    flags["priority"] = flags.severity * flags.loan_amount / 1e5
    flags = flags[flags.flagged].sort_values("priority", ascending=False)
    flags["reasons"] = [" | ".join(reasons(r, sig.loc[i], r.loan_type)) for i, r in flags.iterrows()]
    flags.reset_index().to_parquet(OUT / "anomaly_flags_default.parquet", index=False)

    # ---- model blind spots: flagged but rated Low/Medium by the PD model
    low_mid = flags.risk_band.astype(str).isin(["Low", "Medium"])
    blind = flags[low_mid]

    # ---- illustrative FP / FN examples (real accounts)
    ex = {}
    fp_s = flags[(flags.t_spike) & (~flags.t_peer) & (flags.target_default == 0) & low_mid.reindex(flags.index)].sort_values("severity", ascending=False)
    if len(fp_s):
        a = fp_s.index[0]
        ex["false_positive_spike"] = dict(account=a, **{k: (float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in fp_s.loc[a, ["loan_type", "loan_amount", "pd_xgb", "reasons"]].items()},
                                         months=[int(v) for v in sig.loc[a, [f"dpd_m{m}" for m in range(1, 13)]]])
    fp_p = flags[(flags.t_peer) & (flags.target_default == 0)].sort_values("severity", ascending=False)
    if len(fp_p):
        a = fp_p.index[0]
        ex["false_positive_peer"] = dict(account=a, **{k: (float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in fp_p.loc[a, ["loan_type", "loan_amount", "pd_xgb", "reasons"]].items()})
    fn = dd[(dd.target_default == 1) & (~fl.flagged.reindex(dd.index)) & (dd.risk_band.astype(str) == "Low") & ((dd.rp_n_late + dd.rp_n_missed) <= 1) & (dd.bur_n_dpd30 == 0)]
    if len(fn):
        a = fn.sort_values("loan_amount", ascending=False).index[0]
        ex["false_negative"] = dict(account=a, loan_type=str(dd.loc[a, "loan_type"]), loan_amount=float(dd.loc[a, "loan_amount"]), pd_xgb=float(dd.loc[a, "pd_xgb"]),
                                    months=[int(v) for v in sig.loc[a, [f"dpd_m{m}" for m in range(1, 13)]]],
                                    late_missed=int(dd.loc[a, "rp_n_late"] + dd.loc[a, "rp_n_missed"]))
    json.dump(ex, open(OUT / "examples.json", "w"), indent=2)

    # ---- documentation (numbers computed)
    spike_ev = ev.iloc[0]
    peer_ev = ev.iloc[1]
    any_ev = ev.iloc[4]
    doc = f"""# AI Feature - Anomaly Detection (embedded in the dashboard)

_Auto-generated by `src/anomaly.py`; the live tab is **Anomaly watch** in `app/dashboard.py`._

## What it does
Flags loan accounts whose repayment or bureau behaviour is unusual **against their own history** or **against their peer segment**, ranks them by `severity x exposure`, and puts them in a review queue for a credit analyst. It is **unsupervised**: the default flag is not used to build it, only to evaluate it afterwards.

## What triggers an alert (default thresholds, all adjustable in the dashboard)
| Detector | Trigger | Why |
|---|---|---|
| **Own-history spike** | Worst DPD in months 10-12 is **>= 60 days** while the account's own months 1-9 never exceeded **14 days** (a "sudden DPD spike", e.g. a clean payer who suddenly misses). Escalating DPD over the last 3 months raises severity | Catches a break from the account's own pattern that a static score averages away |
| **Peer-segment outlier** | Isolation Forest fitted **within each loan type** on 15 repayment, bureau and affordability features; flags the **top 3%** of each loan type. Reasons = features furthest from the peer median (robust z-score) | Finds unusual *combinations* (e.g. high missed payments + bureau write-offs + heavy loan/income) no single rule would |
| **Divergence** | (a) 3+ missed payments with us but a clean bureau; (b) bureau shows stress (2+ lines 30+ DPD or 2+ write-offs) while the customer is clean with us | Disagreement between sources signals disputes, data issues or hidden stress elsewhere |

Severity (0-100) = strongest trigger's strength (0.4-1.0), +10 per extra trigger. Priority = severity x loan amount.

## How well it works (evaluation at default thresholds)
{_md(ev)}

*Default is only a proxy for "worth a look": a detector built for operational review can legitimately flag accounts that never default, so "flagged but performing" is a **false-positive proxy**, and "defaulted but not flagged" a **false-negative proxy**.*

- Overall **{any_ev.alerts} alerts ({any_ev.pct_of_accounts} of accounts)**; flagged accounts default at **{any_ev.default_rate_flagged}** vs {any_ev.default_rate_rest} for the rest, and catch {any_ev.share_of_all_defaults_caught} of all defaults.
- The **peer-segment detector is the strong one**: {peer_ev.default_rate_flagged} default rate among flagged vs {peer_ev.default_rate_rest}.
- The **own-history spike is a weak risk signal on this data** ({spike_ev.default_rate_flagged} vs {spike_ev.default_rate_rest}): in this dataset months look independent given an account's own tendency, so a single bad month says little about default. We keep it because it is the operational "early-warning" use case and it is cheap, but it should not be used as a default predictor.
- The two **divergence** checks are small. "Clean bureau but 3+ missed with us" is informative ({ev.iloc[2].default_rate_flagged} default). "Bureau stress but clean with us" is **informational only** ({ev.iloc[3].default_rate_flagged} default, lift {ev.iloc[3].lift} - below average), so it gets the lowest severity (40) and sits at the bottom of the queue; it can be switched off.
- **{len(blind)} of {len(flags)} alerts ({len(blind) / max(len(flags), 1):.0%}) are accounts the PD model rates Low or Medium** - the places where the anomaly layer adds information beyond the score.

### Spike detector with known ground truth (synthetic injection into 1,500 clean accounts)
{_md(inj)}

The spike threshold (60 days) sits above the "Late" range (1-28 days) and inside the "Missed" range (30-89 days), so a 30-45 day injection is **deliberately not flagged** (a single standard miss is common). `natural_alert_rate` is how often clean-baseline accounts spike by chance - the irreducible false-positive floor.

### Threshold trade-off (how to choose sensitivity)
{_md(sens)}

## False positives and false negatives - what they look like
- **False positive (spike):** a mortgage customer pays on time for nine months, misses one payment (e.g. a bank holiday or a bounced debit) and pays on time again. The detector fires; the account never defaults.{" Real example: `" + ex['false_positive_spike']['account'] + "` (" + ex['false_positive_spike']['loan_type'] + ", PD " + format(ex['false_positive_spike']['pd_xgb'], '.0%') + ") - monthly DPD " + str(ex['false_positive_spike']['months']) + "." if 'false_positive_spike' in ex else ''}
- **False positive (peer):** an account that looks extreme on paper (heavy loan/income, thin bureau, high utilisation) but pays reliably.{" Real example: `" + ex['false_positive_peer']['account'] + "` - " + ex['false_positive_peer']['reasons'] + "." if 'false_positive_peer' in ex else ''}
- **False negative:** a defaulted account with a clean-looking record and no bureau stress, rated Low by the model - nothing in the data warned.{" Real example: `" + ex['false_negative']['account'] + "` (" + ex['false_negative']['loan_type'] + ", $" + format(ex['false_negative']['loan_amount'], ',.0f') + ", PD " + format(ex['false_negative']['pd_xgb'], '.0%') + ", only " + str(ex['false_negative']['late_missed']) + " late/missed in 12 months)." if 'false_negative' in ex else ''} Anomaly detection cannot catch what the data does not show; external signals (job loss, fraud intelligence) would be needed.

## Human-in-the-loop
1. **No automated action.** An alert never changes a PD, declines credit or contacts a customer. It only creates a task in the **Anomaly watch** review queue.
2. **Analyst review.** The analyst sees the reasons, the account's own 12-month DPD timeline, the peer comparison and the PD score, then records a decision: *Confirm concern*, *Dismiss - false positive*, *Need more information* or *Escalate to collections early-warning*, with a reason code and a note. Every decision is appended to `outputs/anomaly/review_log.csv` with a timestamp, the thresholds in force and the detectors that fired (audit trail).
3. **Feedback loop.** Dismissals by detector give a live false-positive rate. If a detector's dismiss rate stays above ~70% for a review cycle, tighten its threshold; if analysts keep confirming accounts below the threshold, loosen it. Threshold changes are logged and approved by the model owner.
4. **Capacity control.** The queue is ranked by severity x exposure so analysts work the highest-impact alerts first; the sensitivity slider shows the alert volume each setting would create.
5. **Governance.** Detectors are documented, versioned and monitored like any model (score PSI, alert volume per week, dismiss rate); changes need model-risk sign-off.

## Limitations
- No ground-truth anomaly labels exist; "default" is a proxy, and the injection test only covers the spike detector.
- Repayment is limited to months 1-12 with no calendar dates, and the bureau is a snapshot, so "sudden change" in the bureau cannot be measured over time.
- Months look independent in this synthetic data, so own-history spikes carry little extra default information; real behaviour is usually more autocorrelated.
- Peer groups are loan types only; a bank would add region, vintage and product-tier peers.
"""
    (ROOT / "docs" / "anomaly_feature.md").write_text(doc, encoding="utf-8")
    print(ev.to_string())
    print(inj.to_string())
    print("blind spots", len(blind), "of", len(flags))


if __name__ == "__main__":
    main()
