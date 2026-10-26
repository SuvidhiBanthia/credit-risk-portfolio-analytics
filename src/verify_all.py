"""End-to-end verification: pipeline integrity, document claims vs data, artifacts, app smoke test.

Run:  .venv\\Scripts\\python.exe src\\verify_all.py        (after the pipeline has been run)
Prints a report of every check and exits non-zero if any check fails.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")  # the report contains arrows and other non-ASCII text

ROOT = Path(__file__).resolve().parents[1]
R = []  # (section, name, ok, detail)


def check(section, name, ok, detail=""):
    R.append((section, name, bool(ok), str(detail)))


# =============================================================================== A. pipeline integrity
d = pd.read_parquet(ROOT / "data" / "processed" / "analytics_dataset.parquet")
roles = pd.read_csv(ROOT / "data" / "processed" / "column_roles.csv")
sc = pd.read_parquet(ROOT / "outputs" / "model" / "scored_accounts.parquet")
shap = pd.read_parquet(ROOT / "outputs" / "model" / "shap_values_oof.parquet")
meta = json.load(open(ROOT / "models" / "model_meta.json"))
split = pd.read_csv(ROOT / "outputs" / "model" / "split_assignment.csv")
met = pd.read_csv(ROOT / "outputs" / "model" / "metrics_test.csv", index_col=0)
oof = pd.read_csv(ROOT / "outputs" / "model" / "metrics_oof_all_accounts.csv", index_col=0)
bands = pd.read_csv(ROOT / "outputs" / "model" / "risk_bands.csv")
abl = pd.read_csv(ROOT / "outputs" / "model" / "ablation_feature_sets.csv")
fair = pd.read_csv(ROOT / "outputs" / "model" / "fairness_check.csv")
ev = pd.read_csv(ROOT / "outputs" / "anomaly" / "detector_eval.csv")
sig = pd.read_parquet(ROOT / "outputs" / "anomaly" / "signals.parquet")
flags = pd.read_parquet(ROOT / "outputs" / "anomaly" / "anomaly_flags_default.parquet")

S = "A. Pipeline integrity"
check(S, "dataset has 14,685 unique accounts, 84 columns", len(d) == 14685 and d.account_id.is_unique and d.shape[1] == 84, d.shape)
check(S, "target default rate = 16.1%", round(d.target_default.mean() * 100, 1) == 16.1, d.target_default.mean())
rc = roles.role.str.split(" ").str[0].value_counts().to_dict()
check(S, "column roles: 66 model / 13 analytics-only / 2 excluded / 2 id / 1 target", rc == {"model_candidate": 66, "analytics_only": 13, "id": 2, "excluded": 2, "target": 1}, rc)
feats = meta["features_full"]
check(S, "model uses 66 features", len(feats) == 66, len(feats))
check(S, "no collections (coll_*) feature in the model", not any(f.startswith("coll_") for f in feats))
check(S, "gender / marital_status not in the model", not ({"gender", "marital_status"} & set(feats)))
bad_roles = set(roles[~roles.role.str.startswith("model_candidate")].column)
check(S, "no id / target / analytics-only / excluded column in the model", not (bad_roles & set(feats)), bad_roles & set(feats))
dd = d.merge(split, on="account_id")
tr, te = set(dd[dd.split == "train"].customer_id), set(dd[dd.split == "test"].customer_id)
check(S, "train and test share no customer (grouped split)", len(tr & te) == 0 and len(dd) == 14685, f"train {len(tr)} / test {len(te)} customers")
check(S, "scored_accounts: 14,685 unique accounts, PD in (0,1)", len(sc) == 14685 and sc.account_id.is_unique and sc.pd_xgb.between(0, 1).all())
m = sc.merge(d[["account_id", "target_default"]], on="account_id", suffixes=("", "_d"))
oof_auc = roc_auc_score(m.target_default, m.pd_xgb)
check(S, "out-of-fold AUC recomputed from scores matches the saved metric (<1e-6)", abs(oof_auc - oof.loc["XGBoost", "auc"]) < 1e-6, f"{oof_auc:.6f}")
s2 = shap.set_index("account_id").loc[sc.account_id]
recon = 1 / (1 + np.exp(-(s2.base_value + s2.drop(columns="base_value").sum(axis=1))))
check(S, "SHAP is additive: sigmoid(base + sum) reproduces the PD (max err < 1e-4)", np.abs(recon.values - sc.pd_xgb.values).max() < 1e-4, f"{np.abs(recon.values - sc.pd_xgb.values).max():.2e}")
check(S, "risk bands calibrated: |mean PD - observed| < 2pp in every band", ((bands.mean_pd - bands.actual_default).abs() < 0.02).all(),
      ((bands.mean_pd - bands.actual_default).abs().max()))
check(S, "bands are monotonic (observed default rises Low -> Very High)", bands.set_index("risk_band").loc[["Low", "Medium", "High", "Very High"], "actual_default"].is_monotonic_increasing)
dq = (ROOT / "docs" / "data_quality_notes.md").read_text(encoding="utf-8")
n_pass = len(re.findall(r"^\| PASS \|", dq, re.M))
check(S, "data-quality notes: 17 integrity checks, 0 failed", n_pass == 17 and "**FAIL**" not in dq, n_pass)
check(S, "anomaly signals cover all accounts; flag count matches evaluation", len(sig) == 14685 and len(flags) == int(ev.iloc[4].alerts), f"{len(flags)} vs {int(ev.iloc[4].alerts)}")
check(S, "test AUC > 0.5 for both models; XGB and LR within 0.02 (tied)", (met.auc > 0.6).all() and abs(met.auc.diff().iloc[-1]) < 0.02, met.auc.round(3).to_dict())

# =============================================================================== B. claims typed in documents vs data
tot = d.loan_amount.sum()
d2 = d.merge(sc[["account_id", "risk_band", "pd_xgb"]], on="account_id")
d2["de"] = d2.loan_amount * d2.target_default
mort = d2.loan_type == "Mortgage"
low = d2.income_band.isin(["<40k", "40-60k"])
hi = d2.risk_band.isin(["High", "Very High"])
rate_by_inc = d2.groupby("income_band").interest_rate_pct.mean()
dr_by_inc = d2.groupby("income_band").target_default.mean() * 100
dr_region = d2.groupby("region").target_default.mean() * 100
a = abl.set_index("feature_set").xgb_auc
Mx, Ml = met.loc["XGBoost"], met.loc["Logistic Regression (baseline)"]
f = fair.set_index("segment").approval_impact_ratio
div_share = ((flags.t_local | flags.t_hidden).sum()) / len(d) * 100
spk, peer = ev.iloc[0], ev.iloc[1]
pct = lambda s: float(str(s).strip("%x"))  # noqa: E731
rp = lambda x: f"{round(x):.0f}"  # noqa: E731

CLAIMS = [
    ("accounts and customers", "14,685", ["brief", "deck", "diagrams"]),
    ("exposure", f"${round(tot / 1e6):.0f}M", ["brief", "deck"]),
    ("default rate", f"{d.target_default.mean() * 100:.1f}%", ["deck"]),
    ("mortgage concentration (brief)", f"Mortgages are {rp(mort.mean() * 100)}% of accounts but {rp(d2.loan_amount[mort].sum() / tot * 100)}% of exposure and {rp(d2.de[mort].sum() / d2.de.sum() * 100)}% of defaulted exposure", ["brief", "deck"]),
    ("mortgage default rate", f"{d2.target_default[mort].mean() * 100:.1f}%", ["deck"]),
    ("mortgage count", f"{int(mort.sum()):,} mortgages", ["deck"]),
    ("low-income concentration", f"{rp(d2.loan_amount[low].sum() / tot * 100)}% of exposure but {rp(d2.de[low].sum() / d2.de.sum() * 100)}% of defaulted exposure", ["brief", "deck"]),
    ("flat interest rates", f"{rate_by_inc.min():.1f}-{rate_by_inc.max():.1f}%", ["brief"]),
    ("default range across income bands", f"{dr_by_inc.min():.1f}% to {rp(dr_by_inc.max())}%", ["brief", "deck"]),
    ("region range", f"{rp(dr_region.min())}% to {rp(dr_region.max())}%", ["deck"]),
    ("model AUC / KS (brief)", f"Test AUC {Mx.auc:.2f}, KS {Mx.ks:.2f}", ["brief"]),
    ("model AUC in deck", f"AUC {Mx.auc:.3f}", ["deck"]),
    ("model figures in diagram", f"XGB test: {Mx.auc:.2f} · {Mx.ks:.2f} · {Mx.gini:.2f}", ["diagrams"]),
    ("logistic AUC in deck", f"{Ml.auc:.3f}", ["deck"]),
    ("application+bureau AUC", f"AUC {a['application+bureau']:.2f}", ["deck"]),
    ("ablation chain in diagram", " → ".join(f"{x:.2f}" for x in a.values), ["diagrams"]),
    ("risk bands (brief)", f"{rp(hi.mean() * 100)}% of accounts, {rp(d2.loan_amount[hi].sum() / tot * 100)}% of exposure and {rp(d2.de[hi].sum() / d2.de.sum() * 100)}% of defaulted exposure ({rp(d2.target_default[hi].mean() * 100)}% default vs {rp(d2.target_default[d2.risk_band == 'Low'].mean() * 100)}% in the Low band)", ["brief"]),
    ("alert share of accounts", f"{pct(ev.iloc[4].pct_of_accounts):.1f}%", ["deck"]),
    ("peer-outlier default vs rest", f"{rp(pct(peer.default_rate_flagged))}% vs {rp(pct(peer.default_rate_rest))}%", ["deck"]),
    ("diagram: spike detector stats", f"alerts {pct(spk.pct_of_accounts):.1f}% · default {rp(pct(spk.default_rate_flagged))}% vs {rp(pct(spk.default_rate_rest))}%", ["diagrams"]),
    ("diagram: peer detector stats", f"alerts {pct(peer.pct_of_accounts):.1f}% · default {rp(pct(peer.default_rate_flagged))}% vs {rp(pct(peer.default_rate_rest))}%", ["diagrams"]),
    ("diagram: divergence share", f"alerts {div_share:.1f}% · informational", ["diagrams"]),
    ("fairness: under-30 impact ratio", f"{f['<30']:.2f}", ["deck"]),
    ("fairness: gender impact ratio", f"{f['Female']:.2f}", ["deck"]),
    ("integrity-check count", f"{n_pass} integrity checks", ["diagrams"]),
    ("feature count in diagram", "66 engineered features", ["diagrams"]),
    ("dataset shape in diagram", "14,685 × 84", ["diagrams"]),
]
n_assump = sum(1 for ln in (ROOT / "docs" / "assumptions_log.md").read_text(encoding="utf-8").splitlines() if ln.startswith("| A"))
CLAIMS += [("assumption count", f"{n_assump} assumptions", ["deck"]), ("assumption count in diagram", f"Assumptions log ({n_assump})", ["diagrams"])]


def norm(t):
    return re.sub(r"\s+", " ", t.replace("\u00a0", " ")).strip()


def doc_text():
    from docx import Document
    from pptx import Presentation
    docs = {}
    dx = Document(ROOT / "docs" / "product_brief.docx")
    parts = [p.text for p in dx.paragraphs]
    for t in dx.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    docs["brief"] = norm(" ".join(parts))
    pr = Presentation(ROOT / "docs" / "CRO_pitch_deck.pptx")
    parts = []
    notes = []
    for s in pr.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for c in row.cells:
                        parts.append(c.text)
        if s.has_notes_slide:
            notes.append(s.notes_slide.notes_text_frame.text)
    docs["deck"] = norm(" ".join(parts))
    docs["deck_notes"] = norm(" ".join(notes))
    html = (ROOT / "docs" / "architecture-diagrams.html").read_text(encoding="utf-8")
    docs["diagrams"] = norm(re.sub(r"<[^>]+>", " ", html))
    docs["readme"] = norm((ROOT / "README.md").read_text(encoding="utf-8"))
    return docs, len(pr.slides), notes


DOCS, n_slides, notes = doc_text()
S = "B. Numbers typed in documents match the data"
for name, expect, where in CLAIMS:
    expect_n = norm(expect)
    miss = [w for w in where if expect_n not in DOCS[w]]
    check(S, f"{name}: '{expect}'", not miss, "missing in: " + ", ".join(miss) if miss else "found in " + ", ".join(where))

# =============================================================================== C. artifacts
S = "C. Artifacts"
check(S, "shared deck has 12 slides and no speaker notes", n_slides == 12 and not any(n.strip() for n in notes), f"{n_slides} slides, {sum(1 for n in notes if n.strip())} with notes")
import pymupdf  # noqa: E402
pdf = pymupdf.open(ROOT / "docs" / "product_brief.pdf")
check(S, "product brief PDF is exactly 1 page", len(pdf) == 1, f"{len(pdf)} pages")
pdf_txt = norm(pdf[0].get_text())
check(S, "product brief PDF text matches the .docx (same key sentences)", all(norm(s) in pdf_txt and norm(s) in DOCS["brief"] for s in ["Alerts never act", "Risk is seen late and in pieces.", "OUT - not built, not claimed"]))
html = (ROOT / "docs" / "architecture-diagrams.html").read_text(encoding="utf-8")
check(S, "architecture HTML has 5 diagrams and 5 PNG exports", html.count("<svg ") == 5 and len(list((ROOT / "outputs" / "diagrams").glob("*.png"))) == 5)
missing = []
for md in ["insights.md", "model_card.md", "anomaly_feature.md", "data_quality_notes.md", "assumptions_log.md"]:
    txt = (ROOT / "docs" / md).read_text(encoding="utf-8")
    for link in re.findall(r"\]\((\.\./[^)]+)\)", txt):
        if not (ROOT / "docs" / link).resolve().exists():
            missing.append(f"{md}: {link}")
check(S, "every image link in the generated docs resolves", not missing, missing)
rd = (ROOT / "README.md").read_text(encoding="utf-8")
paths = set(re.findall(r"`((?:src|docs|app|data|outputs|models)/[\w./\-]+)`", rd))
missing = [p for p in paths if not any((ROOT / p).exists() for _ in [0]) and "*" not in p and not (ROOT / p.rstrip("/")).exists()]
check(S, "every path named in the README exists", not missing, missing)
al = (ROOT / "docs" / "assumptions_log.md").read_text(encoding="utf-8")
ids = [int(x) for x in re.findall(r"^\| A(\d+) \|", al, re.M)]
check(S, "assumption IDs A1..A37 are unique and complete", sorted(ids) == list(range(1, 38)), len(ids))
cited = set(int(x) for md in ["model_card.md", "insights.md", "anomaly_feature.md", "data_quality_notes.md"] for x in re.findall(r"\bA(\d+)\b", (ROOT / "docs" / md).read_text(encoding="utf-8")))
check(S, "every assumption ID cited in the generated docs exists in the log", cited <= set(ids), cited - set(ids))

# =============================================================================== D. app smoke test
S = "D. Dashboard"
try:
    from streamlit.testing.v1 import AppTest
    import os
    os.environ["ANOMALY_REVIEW_LOG"] = str(ROOT / "outputs" / "_verify_tmp_log.csv")
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    mv = {}
    for m_ in at.metric:  # first occurrence wins (the Portfolio tab's KPI, not the same label further down)
        mv.setdefault(m_.label, m_.value)
    check(S, "app starts with no exception", len(at.exception) == 0, [e.value[:80] for e in at.exception])
    check(S, "KPIs match the data (accounts, exposure, default rate)", mv.get("Accounts") == "14,685" and mv.get("Exposure") == f"${tot / 1e6:,.1f}M" and mv.get("Observed default") == f"{d.target_default.mean() * 100:.1f}%", mv)
    check(S, "anomaly tab alert count matches the evaluation", mv.get("Alerts") == f"{len(flags):,}", mv.get("Alerts"))
    steps = [("drill into a segment", lambda: at.selectbox(key="drill_seg").set_value("40-60k")),
             ("filter to Mortgage", lambda: at.multiselect(key="flt_Loan type").set_value(["Mortgage"])),
             ("reset filters", lambda: at.button[0].click()),
             ("look up an account", lambda: at.text_input(key="acct_lookup").set_value(str(flags.account_id.iloc[0]))),
             ("tighten anomaly thresholds", lambda: at.slider(key="an_peer").set_value(1.0)),
             ("switch anomaly detectors off", lambda: at.checkbox(key="an_use_div").set_value(False))]
    for nm, fn in steps:
        fn().run()
        check(S, f"interaction OK: {nm}", len(at.exception) == 0, [e.value[:80] for e in at.exception])
    tmp = ROOT / "outputs" / "_verify_tmp_log.csv"
    if tmp.exists():
        tmp.unlink()
except Exception as e:  # noqa: BLE001
    check(S, "dashboard smoke test ran", False, repr(e)[:200])

# =============================================================================== report
n_ok = sum(1 for _, _, ok, _ in R if ok)
lines = [f"# Final end-to-end check\n", f"**{n_ok}/{len(R)} checks passed.**\n"]
for sec in dict.fromkeys(s for s, *_ in R):
    lines.append(f"\n## {sec}\n\n| Result | Check | Detail |\n|---|---|---|")
    for s, name, ok, det in R:
        if s == sec:
            lines.append(f"| {'PASS' if ok else '**FAIL**'} | {name} | {det[:140]} |")
print("\n".join(lines))
for s, name, ok, det in R:
    if not ok:
        print("FAIL:", s, "|", name, "|", det)
print(f"\n{n_ok}/{len(R)} checks passed")
sys.exit(0 if n_ok == len(R) else 1)
