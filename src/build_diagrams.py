"""Builds docs/architecture-diagrams.html (same visual language as the dsider architecture-diagrams.html)
and exports one PNG per diagram to outputs/diagrams/ for use in the pitch deck.

Run:  .venv\\Scripts\\python.exe src\\build_diagrams.py
"""
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_HTML = ROOT / "docs" / "architecture-diagrams.html"
OUT_PNG = ROOT / "outputs" / "diagrams"
import pandas as pd  # noqa: E402

ANOMALY_EVAL = pd.read_csv(ROOT / "outputs" / "anomaly" / "detector_eval.csv")
_fl = pd.read_parquet(ROOT / "outputs" / "anomaly" / "anomaly_flags_default.parquet")
DIVERGENCE_PCT = (_fl.t_local | _fl.t_hidden).sum() / len(pd.read_parquet(ROOT / "outputs" / "anomaly" / "signals.parquet")) * 100
N_CHECKS = (ROOT / "docs" / "data_quality_notes.md").read_text(encoding="utf-8").count("| PASS |")
N_ASSUMPTIONS = sum(1 for ln in (ROOT / "docs" / "assumptions_log.md").read_text(encoding="utf-8").splitlines() if ln.startswith("| A"))

KINDS = {  # fill, stroke, title colour
    "blue": ("var(--primary-light)", "var(--primary)", "var(--primary)"),
    "gray": ("#f1f5f9", "var(--border)", "var(--text)"),
    "yellow": ("#fef3c7", "var(--yellow)", "#92400e"),
    "green": ("#e6f4ea", "var(--mongo)", "var(--mongo)"),
    "purple": ("#f3e8ff", "var(--opus)", "var(--opus)"),
    "teal": ("#e3f8f1", "var(--teal-dark)", "var(--teal-dark)"),
    "cyan": ("#e0f2fe", "var(--zilliz)", "var(--zilliz)"),
    "indigo": ("#eef2ff", "#6366f1", "#6366f1"),
    "orange": ("#fff4e5", "var(--cognito)", "var(--cognito)"),
    "red": ("#fee2e2", "var(--red)", "#991b1b"),
    "white": ("#ffffff", "var(--border)", "var(--slate)"),
}
ARROW = {"gray": "arrowGray", "blue": "arrowBlue", "teal": "arrowTeal", "red": "arrowRed", "purple": "arrowPurple", "yellow": "arrowYellow"}
STROKE = {"gray": "var(--text-muted)", "blue": "var(--primary)", "teal": "var(--teal-dark)", "red": "var(--red)", "purple": "var(--opus)", "yellow": "var(--yellow)"}

DEFS = """<defs>
  <marker id="arrowGray" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--text-muted)"/></marker>
  <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--primary)"/></marker>
  <marker id="arrowTeal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--teal-dark)"/></marker>
  <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--red)"/></marker>
  <marker id="arrowPurple" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--opus)"/></marker>
  <marker id="arrowYellow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--yellow)"/></marker>
</defs>"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(s, width_px, size):
    n = max(8, int(width_px / (size * 0.53)))
    return textwrap.wrap(s, n) if s else []


def text(x, y, s, size=10, weight=400, fill="var(--text)", anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}" {extra}>{esc(s)}</text>'


def box(x, y, w, h, title, sub=None, kind="blue", dashed=False, tsize=11, ssize=9, stat=None, rx=8, sw=1.2):
    fill, stroke, tc = KINDS[kind]
    t_lines = wrap(title, w - 14, tsize)
    s_lines = wrap(sub, w - 14, ssize) if sub else []
    st_lines = wrap(stat, w - 14, ssize) if stat else []
    block = len(t_lines) * (tsize + 3) + (len(s_lines) * (ssize + 3) + 3 if s_lines else 0) + (len(st_lines) * (ssize + 3) + 3 if st_lines else 0)
    yy = y + (h - block) / 2 + tsize - 1
    d = ' stroke-dasharray="5,3"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>']
    for ln in t_lines:
        out.append(text(x + w / 2, yy, ln, tsize, 600, tc, "middle"))
        yy += tsize + 3
    if s_lines:
        yy += 0
        for ln in s_lines:
            out.append(text(x + w / 2, yy, ln, ssize, 400, "var(--text-muted)", "middle"))
            yy += ssize + 3
    if st_lines:
        yy += 3
        for ln in st_lines:
            out.append(text(x + w / 2, yy, ln, ssize, 700, tc, "middle"))
            yy += ssize + 3
    return "\n".join(out)


def frame(x, y, w, h, label, tools=None, stroke="var(--border)", tint="var(--primary)", lc="var(--text-muted)", dashed=False, sw=1.5):
    d = ' stroke-dasharray="6,4"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#fff" stroke="{stroke}" stroke-width="{sw}"{d}/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{tint}" opacity="0.04"/>',
           text(x + w / 2, y + 22, label, 11, 700, lc, "middle", 'letter-spacing="0.4"')]
    if tools:
        out.append(text(x + w / 2, y + 37, tools, 9.5, 500, "var(--primary)", "middle"))
    return "\n".join(out)


def arrow(pts, kind="gray", dashed=False, width=1.5, label=None, lx=None, ly=None, lc=None, lsize=9, lweight=500, anchor="middle"):
    d = "M" + " L".join(f"{px},{py}" for px, py in pts)
    da = ' stroke-dasharray="6,3"' if dashed else ""
    out = [f'<path d="{d}" fill="none" stroke="{STROKE[kind]}" stroke-width="{width}"{da} marker-end="url(#{ARROW[kind]})"/>']
    if label:
        out.append(text(lx, ly, label, lsize, lweight, lc or STROKE[kind], anchor))
    return "\n".join(out)


def band(y, label, items):
    """Red dashed governance band with one-line chips; items = [(title, kind)]."""
    out = [frame(10, y, 1080, 76, label, None, stroke="var(--red)", tint="var(--red)", lc="var(--red)", dashed=True)]
    for i, (t, k) in enumerate(items):
        out.append(box(32 + i * 210, y + 34, 196, 30, t, None, k, tsize=10))
    return "\n".join(out)


def svg(w, h, body):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Urbanist, sans-serif">\n'
            + DEFS + "\n" + body + "\n</svg>")


# =============================================================================== D1 end-to-end
def d1():
    b = []
    X = [176, 413, 650, 887]  # stage frames, 201 wide
    b.append(frame(10, 20, 130, 340, "0 · SOURCE DATA", "5 CSV tables", stroke="var(--border)"))
    for lab, tools, x in [("1 · INGESTION", "Python · pandas", X[0]), ("2 · ANALYTICS", "pandas · matplotlib", X[1]),
                          ("3 · MODEL", "scikit-learn · XGBoost · SHAP", X[2]), ("4 · DASHBOARD", "Streamlit · Plotly", X[3])]:
        b.append(frame(x, 20, 201, 340, lab, tools, stroke="var(--primary)", sw=2))
    for i, t in enumerate(["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"]):
        b.append(box(22, 74 + i * 46, 106, 38, t, None, "yellow" if t == "loan_accounts" else "gray", tsize=9))
    stages = [
        [("Validate & clean", "keys checked · issues flagged"), ("Aggregate & join", "one row per loan account, no fan-out"),
         ("Data-quality report", "every issue: count + treatment")],
        [("Profile & cross-cuts", "5 dimensions · cross-cuts"), ("5 insights + so-what", "with 95% confidence intervals"),
         ("Auto-generated narrative", "numbers computed, never typed")],
        [("66 features · grouped split", "no customer in both train and test"), ("Logistic regression + XGBoost", "AUC · KS · Gini", "XGB test: 0.68 · 0.28 · 0.37"),
         ("SHAP + out-of-fold PD", "an explained score for every account")],
    ]
    for x, items in zip(X, stages):
        for i, it in enumerate(items):
            b.append(box(x + 10, 74 + i * 76, 181, 66, it[0], it[1], "blue", tsize=10.5, ssize=8.5, stat=it[2] if len(it) > 2 else None))
    b.append(box(X[3] + 10, 74, 181, 66, "Portfolio view", "risk bands · exposure · trends", "blue", tsize=10.5, ssize=8.5))
    b.append(box(X[3] + 10, 150, 181, 66, "Account view", "PD · SHAP drivers · 12-month history", "blue", tsize=10.5, ssize=8.5))
    b.append(box(X[3] + 10, 226, 181, 66, "AI feature · Anomaly watch", "3 detectors · analyst review", "yellow", tsize=10.5, ssize=8.5, sw=2))
    # hand-off artefacts
    for x, w, t, s in [(22, 106, "Raw files", "data/raw/"), (X[0] + 10, 181, "analytics_dataset.parquet", "14,685 × 84"), (X[1] + 10, 181, "insights.md + 6 charts", "docs/ · outputs/"),
                       (X[2] + 10, 181, "PD + SHAP per account", "scored_accounts.parquet"), (X[3] + 10, 181, "Interactive web app", "app/dashboard.py")]:
        b.append(box(x, 308, w, 40, t, s, "teal", tsize=10, ssize=8.3))
    for x1, x2 in [(140, X[0]), (X[0] + 201, X[1]), (X[1] + 201, X[2]), (X[2] + 201, X[3])]:
        b.append(arrow([(x1, 328), (x2, 328)], "teal", width=2))
    b.append(band(380, "GOVERNANCE, RISK & COMPLIANCE — applies to every stage",
                  [("Data-quality notes", "red"), (f"Assumptions log ({N_ASSUMPTIONS})", "red"), ("Leakage & fairness controls", "red"),
                   ("Model card & monitoring", "red"), ("Audit trail of decisions", "red")]))
    return svg(1100, 466, "\n".join(b))


# =============================================================================== D2 lineage
def d2():
    b = []
    for x, lab in [(110, "SOURCE TABLE"), (385, "TRANSFORM BEFORE JOINING"), (680, "JOIN")]:
        b.append(text(x, 22, lab, 10, 700, "var(--text-muted)", "middle", 'letter-spacing="0.5"'))
    rows = [
        ("customers", "10,000 rows", "gray", "Clean demographics", "labels fixed · impute + flags", "m:1 on customer_id", "blue"),
        ("loan_accounts", "14,685 rows · target_default", "yellow", "Base table", "grain = one loan account", "base (left-join anchor)", "yellow"),
        ("bureau_credit", "22,979 external lines", "gray", "Aggregate → 1 row / customer", "19 bureau features", "m:1 on customer_id", "blue"),
        ("repayment_history", "176,220 rows · 12 per account", "gray", "Aggregate → 1 row / account", "14 repayment + 6 cross-account features", "1:1 on account_id", "blue"),
        ("collections_events", "8,635 events", "gray", "Aggregate → 1 row / account", "5 columns · follows delinquency", "1:1 · analytics only, not in model", "red"),
    ]
    for i, (t, s_, k, tt, ts, lab, ak) in enumerate(rows):
        y = 40 + i * 72
        b.append(box(10, y, 200, 56, t, s_, k, tsize=11, ssize=8.8))
        b.append(arrow([(210, y + 28), (250, y + 28)], "gray"))
        b.append(box(250, y, 270, 56, tt, ts, "blue", tsize=10.5, ssize=8.8))
        b.append(arrow([(520, y + 28), (840, y + 28)], ak, dashed=(ak == "red"), width=1.8,
                       label=lab, lx=680, ly=y + 21, lc="#92400e" if ak == "yellow" else None))
    # output dataset
    b.append('<rect x="840" y="30" width="250" height="356" rx="14" fill="#fff" stroke="var(--primary)" stroke-width="2"/>')
    b.append('<rect x="840" y="30" width="250" height="356" rx="14" fill="var(--primary)" opacity="0.05"/>')
    b.append(text(965, 56, "analytics_dataset", 13, 700, "var(--primary)", "middle"))
    b.append(text(965, 72, "14,685 rows × 84 columns · 1 row per account", 9, 500, "var(--text-muted)", "middle"))
    for i, (t, k) in enumerate([("66 model features", "blue"), ("1 target · target_default", "yellow"), ("2 ids", "gray"),
                                ("2 excluded: gender, marital status", "red"), ("13 analytics only", "red")]):
        b.append(box(856, 88 + i * 56, 218, 46, t, None, k, tsize=10.5))
    b.append(box(10, 406, 1080, 40, f"{N_CHECKS} integrity checks pass: unique keys · no orphan IDs · 12 months per account · no fan-out after joins",
                 None, "teal", tsize=10.5))
    return svg(1100, 456, "\n".join(b))


# =============================================================================== D3 model
def d3():
    b = []
    cols = [10, 235, 460, 685, 910]
    r1 = [("66 engineered features", "application 27 · bureau 19 · repayment 14 · cross-account 6", "blue"),
          ("Customer-grouped split", "stratified 80/20 · no customer in both sets · test set touched once", "blue"),
          ("Tuning on train only", "customer-grouped 5-fold CV · random search (40 XGBoost, 7 logistic)", "blue")]
    for x, (t, s, k) in zip(cols, r1):
        b.append(box(x, 40, 180, 90, t, s, k))
    b.append(box(685, 28, 180, 56, "Logistic regression", "baseline · explainable benchmark", "gray", tsize=10.5))
    b.append(box(685, 92, 180, 56, "XGBoost", "scoring model · native SHAP · handles missing", "blue", tsize=10.5))
    b.append(box(910, 40, 180, 90, "Held-out evaluation", "AUC · Gini · KS · Brier · calibration · decile table", "teal", stat="XGB test: 0.68 · 0.37 · 0.28"))
    for x in [190, 415, 640]:
        b.append(arrow([(x, 85), (x + 45, 85)], "blue"))
    b.append(arrow([(865, 85), (910, 85)], "blue"))
    # row 2
    b.append(box(10, 210, 180, 90, "Ablation study", "does each feature group add lift?", "gray", stat="0.65 → 0.66 → 0.68 → 0.68"))
    b.append(box(235, 210, 180, 90, "Out-of-fold scoring", "5-fold, customer-grouped, all 14,685 accounts · no account scored by a model that saw it", "blue"))
    b.append(box(460, 210, 180, 90, "SHAP explanations", "per-account drivers · one-hot summed per feature · additive check ≈ 5e-7", "blue"))
    b.append(box(685, 210, 180, 90, "Risk bands", "Low <10% · Medium 10–20% · High 20–35% · Very High ≥35%", "yellow", stat="observed ≈ predicted per band"))
    b.append(box(910, 210, 180, 90, "Outputs → dashboard & anomaly layer", "scored_accounts · shap_values_oof", "teal"))
    b.append(arrow([(775, 148), (775, 178), (325, 178), (325, 210)], "blue", label="same tuned parameters", lx=550, ly=172))
    for x in [415, 640, 865]:
        b.append(arrow([(x, 255), (x + 45, 255)], "blue"))
    b.append(arrow([(190, 255), (235, 255)], "gray", dashed=True))
    # row 3
    b.append(band(335, "GOVERNANCE & MONITORING",
                  [("Versioned model artefacts", "gray"), ("Leakage & fairness tests", "red"), ("Two scoring points (0.66 / 0.68)", "yellow"),
                   ("Monitoring: PSI · CSI · AUC", "red"), ("Retrain: 6 months or PSI > 0.25", "red")]))
    return svg(1100, 421, "\n".join(b))


# =============================================================================== D4 anomaly
def d4():
    b = []
    # inputs -> detectors
    b.append(frame(10, 20, 190, 310, "INPUTS"))
    for i, (t, s) in enumerate([("Repayment history", "12 months of DPD"), ("Bureau aggregates", "lines · DPD · write-offs"),
                                ("Affordability & loan type", "loan and EMI vs income")]):
        b.append(box(20, 56 + i * 90, 170, 74, t, s, "gray"))
    b.append(frame(230, 20, 290, 310, "THREE UNSUPERVISED DETECTORS", stroke="var(--primary)", sw=2))
    sp, pe = ANOMALY_EVAL.iloc[0], ANOMALY_EVAL.iloc[1]
    pc = lambda s: float(str(s).strip("%x"))  # noqa: E731
    stat = lambda r: f"alerts {pc(r.pct_of_accounts):.1f}% · default {round(pc(r.default_rate_flagged))}% vs {round(pc(r.default_rate_rest))}%"  # noqa: E731
    dets = [("1 · Own-history spike", "DPD ≥ 60 after 9 clean months", stat(sp)),
            ("2 · Peer-segment outlier", "Isolation Forest vs same loan type · top 3%", stat(pe)),
            ("3 · Source divergence", "bureau and our repayment record disagree", f"alerts {DIVERGENCE_PCT:.1f}% · informational")]
    for i, (t, s, st) in enumerate(dets):
        b.append(box(240, 52 + i * 90, 270, 80, t, s, "blue", stat=st))
    b.append(arrow([(200, 175), (230, 175)], "gray", width=2))
    # ranked queue
    b.append(box(560, 125, 190, 80, "Ranked alert queue", "severity × exposure", "blue", stat="≈ 6% of accounts"))
    b.append(arrow([(520, 165), (560, 165)], "blue", width=2))
    # analyst review
    b.append('<rect x="790" y="20" width="300" height="218" rx="14" fill="#fffbeb" stroke="var(--yellow)" stroke-width="2"/>')
    b.append(text(940, 42, "ANALYST REVIEW — HUMAN-IN-THE-LOOP", 10.5, 700, "#92400e", "middle", 'letter-spacing="0.3"'))
    b.append(box(802, 54, 276, 58, "Evidence panel", "why flagged · DPD timeline · peers · PD", "white"))
    for i, (t, k) in enumerate([("Confirm concern", "teal"), ("Dismiss — false positive", "gray"), ("Need more information", "gray"), ("Escalate to early-warning", "teal")]):
        b.append(box(802 + (i % 2) * 141, 122 + (i // 2) * 44, 135, 38, t, None, k, tsize=9.5))
    b.append(text(940, 228, "every decision needs a reason code + note", 9, 500, "#92400e", "middle"))
    b.append(arrow([(750, 165), (790, 165)], "blue", width=2))
    b.append(box(790, 270, 300, 56, "Outcomes", "Confirm / Escalate → early-warning task", "teal"))
    b.append(arrow([(940, 238), (940, 270)], "teal"))
    # audit + feedback loop
    b.append(box(560, 262, 190, 64, "Audit log", "decision · reason · thresholds", "red"))
    b.append(arrow([(790, 226), (655, 226), (655, 262)], "red"))
    b.append(box(230, 362, 290, 52, "Threshold review", "> 70% dismissed → model owner tightens", "purple"))
    b.append(arrow([(655, 326), (655, 388), (520, 388)], "purple"))
    b.append(arrow([(375, 362), (375, 332)], "purple", dashed=True, label="re-tune thresholds", lx=383, ly=351, anchor="start"))
    # guardrail
    b.append(box(10, 438, 1080, 36, "Guardrail: an alert never changes a score, declines credit or contacts a customer. It only creates a review task.", None, "red", tsize=10.5))
    return svg(1100, 484, "\n".join(b))


# =============================================================================== D5 production target
def d5():
    b = []
    cols = [("SOURCE SYSTEMS", 10, 150), ("INGESTION & QUALITY", 180, 170), ("DATA PLATFORM", 370, 160), ("MODEL LIFECYCLE", 550, 180),
            ("SERVING", 750, 170), ("CONSUMERS", 940, 150)]
    for lab, x, w in cols:
        b.append(frame(x, 20, w, 330, lab, None, stroke="var(--border)", dashed=False))
    D = True
    src = ["Core banking & loan origination", "Customer master / CRM", "Credit bureau (point-in-time)", "Collections & recoveries"]
    for i, s in enumerate(src):
        b.append(box(20, 50 + i * 72, 130, 62, s, None, "orange", dashed=D, tsize=10))
    # (title, sub, kind, x, w) per column; 3 boxes each except the data platform (2)
    stacks = [
        (190, 150, [("Batch ETL / CDC", "nightly, incremental", "gray"), ("Data contracts & DQ gates", "prototype: src/ingest.py", "blue"), ("PII masking & access control", None, "gray")]),
        (380, 140, [("Lake / warehouse", "versioned snapshots", "gray"), ("Feature store", "point-in-time features", "gray")]),
        (560, 160, [("Training pipeline", "prototype: src/model.py", "blue"), ("Registry & versioning", "data + params + report", "gray"), ("Validation · champion / challenger", "sign-off before promotion", "gray")]),
        (760, 150, [("Batch scoring + API", "PD + reasons per account", "gray"), ("Risk dashboard", "prototype: Streamlit", "blue"), ("Anomaly watch + case management", "prototype: review queue", "yellow")]),
    ]
    for x, w, items in stacks:
        y0 = 50 + (4 - len(items)) * 36  # centre shorter stacks against the 4-box columns
        for i, (t, s, k) in enumerate(items):
            b.append(box(x, y0 + i * 72, w, 62, t, s, k, dashed=(k == "gray"), tsize=10, ssize=8.5))
    con = ["Credit risk / CRO", "Collections early-warning", "Pricing & limits", "Provisioning & regulatory reporting"]
    for i, s in enumerate(con):
        b.append(box(950, 50 + i * 72, 130, 62, s, None, "gray", dashed=D, tsize=10))
    for x1, x2 in [(160, 180), (350, 370), (530, 550), (730, 750), (920, 940)]:
        b.append(arrow([(x1, 189), (x2, 189)], "gray", width=2))
    b.append(band(370, "CROSS-CUTTING CONTROLS A BANK NEEDS",
                  [("Drift & performance monitoring", "red"), ("Lineage & audit trail", "red"), ("Security: SSO · role access", "red"),
                   ("Explainability · reason codes", "red"), ("Privacy & retention", "red")]))
    return svg(1100, 456, "\n".join(b))


# =============================================================================== page
SECTIONS = [
    ("d1", "End-to-End Architecture", "Ingestion → analytics → model → dashboard with the embedded AI feature, tools at each stage, and governance across all of it.", d1,
     [("var(--border)", "Source data"), ("var(--primary)", "Pipeline component"), ("var(--teal-dark)", "Output artefact"), ("var(--yellow)", "AI feature / decision point"),
      ("var(--red)", "Governance & compliance")],
     None),
    ("d2", "Data Lineage & Join Logic", "Five tables, three grains. Every one-to-many table is aggregated to its parent grain <em>before</em> joining, so the output stays at one row per loan account.", d2,
     [("var(--yellow)", "Base table"), ("var(--primary)", "Transform / join"), ("var(--red)", "Excluded from model (leakage)"), ("var(--teal-dark)", "Integrity check")],
     "<strong>Why account grain:</strong> <code>target_default</code> lives on <code>loan_accounts</code>. Collections events follow delinquency, so they feed analytics only, never the model."),
    ("d3", "Model Pipeline & Governance", "Customer-grouped split, tuned on train only, evaluated once on the held-out set; out-of-fold scores feed the dashboard.", d3,
     [("var(--primary)", "Model step"), ("var(--yellow)", "Decision / design choice"), ("var(--teal-dark)", "Result / output"), ("var(--red)", "Compliance & monitoring")],
     "<strong>Logistic regression and XGBoost are statistically tied</strong> on this data (AUC 0.689 vs 0.684). XGBoost is the scoring model for native SHAP and missing-value handling; logistic regression is the explainable benchmark."),
    ("d4", "AI Feature — Anomaly Detection with Human-in-the-Loop", "Three unsupervised detectors feed a ranked review queue; analyst decisions are logged and retune the thresholds.", d4,
     [("var(--primary)", "Detector / queue"), ("var(--yellow)", "Human review"), ("var(--teal-dark)", "Outcome"), ("var(--red)", "Audit & guardrail"), ("var(--opus)", "Feedback loop")],
     "<strong>Trigger:</strong> spike ≥ 60 days after a clean-ish baseline · top 3% of a loan type by Isolation Forest · bureau/repayment disagreement. Default is used only to evaluate the flags, never to build them."),
    ("d5", "Production Target Architecture (proposed)", "What this prototype would become inside a bank. Solid blue = built in this project; dashed = what the bank's IT / data team must provide.", d5,
     [("var(--primary)", "Built in this prototype"), ("var(--cognito)", "Bank source systems (needed)"), ("var(--text-muted)", "Proposed component (needed)"), ("var(--red)", "Cross-cutting controls")],
     "<strong>Needed from the bank to ship:</strong> (1) the default definition and dated outcomes; (2) point-in-time bureau pulls at application; (3) dated monthly repayments and balances; (4) recovery / LGD data; "
     "(5) SSO, role-based access and a case-management API; (6) a compliance ruling on permitted attributes (age) and adverse-action reason mapping."),
]

CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
  :root { --primary:#0172b6; --primary-dark:#015a91; --primary-light:#e8f4fc; --teal:#39cfa5; --teal-dark:#2ba882; --slate:#2c3e50; --surface:#f7f9fb;
    --text:#334155; --text-muted:#64748b; --border:#e2e8f0; --opus:#7c3aed; --sonnet:#2563eb; --openai:#10a37f; --mongo:#00684a; --zilliz:#00a1ea;
    --cognito:#dd6b20; --red:#dc3545; --yellow:#f59e0b; --green:#22c55e; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Urbanist',-apple-system,sans-serif; background:var(--surface); color:var(--text); line-height:1.6; }
  .page-header { background:linear-gradient(135deg,var(--slate) 0%,var(--primary-dark) 100%); color:#fff; padding:48px 0 40px; text-align:center; }
  .page-header h1 { font-size:2.2rem; font-weight:800; letter-spacing:-0.5px; }
  .page-header p { color:rgba(255,255,255,0.7); font-size:1rem; margin-top:6px; }
  .page-header .badge { display:inline-block; background:var(--teal); color:#fff; padding:4px 14px; border-radius:20px; font-size:0.75rem; font-weight:700; margin-top:12px; letter-spacing:0.5px; }
  .container { max-width:1200px; margin:0 auto; padding:0 24px; }
  .toc { background:#fff; border-radius:12px; padding:28px 32px; margin:-24px auto 40px; max-width:700px; box-shadow:0 4px 24px rgba(0,0,0,0.06); position:relative; z-index:2; }
  .toc h3 { font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted); margin-bottom:12px; }
  .toc a { display:block; padding:6px 0; color:var(--primary); text-decoration:none; font-weight:500; font-size:0.95rem; border-bottom:1px solid var(--border); }
  .toc a:last-child { border-bottom:none; } .toc a:hover { color:var(--teal-dark); }
  .toc a span { color:var(--text-muted); font-weight:400; margin-right:8px; }
  .diagram-section { background:#fff; border-radius:16px; padding:40px; margin-bottom:40px; box-shadow:0 2px 16px rgba(0,0,0,0.04); }
  .diagram-section h2 { font-size:1.4rem; font-weight:700; color:var(--slate); margin-bottom:6px; display:flex; align-items:center; gap:10px; }
  .diagram-section h2 .num { background:var(--primary); color:#fff; width:32px; height:32px; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; font-size:0.85rem; font-weight:700; }
  .diagram-section .subtitle { color:var(--text-muted); font-size:0.9rem; margin-bottom:28px; }
  .diagram-wrap { overflow-x:auto; padding:8px 0; } .diagram-wrap svg { display:block; margin:0 auto; max-width:100%; height:auto; }
  .legend { display:flex; flex-wrap:wrap; gap:16px; margin-top:24px; padding-top:20px; border-top:1px solid var(--border); }
  .legend-item { display:flex; align-items:center; gap:6px; font-size:0.8rem; color:var(--text-muted); }
  .legend-dot { width:12px; height:12px; border-radius:3px; }
  .note-box { background:var(--primary-light); border-left:4px solid var(--primary); padding:14px 18px; border-radius:0 8px 8px 0; margin-top:20px; font-size:0.88rem; color:var(--primary-dark); }
  .note-box strong { font-weight:700; } code { font-family:'JetBrains Mono',monospace; font-size:0.82em; background:#fff; padding:1px 5px; border-radius:4px; }
  @media print { .page-header { padding:24px 0; } .diagram-section { break-inside:avoid; box-shadow:none; border:1px solid var(--border); } }
"""


def build():
    toc = "\n".join(f'  <a href="#{i}"><span>{n + 1:02d}</span> {t}</a>' for n, (i, t, *_r) in enumerate(SECTIONS))
    secs = []
    for n, (i, t, sub, fn, legend, note) in enumerate(SECTIONS):
        leg = "\n".join(f'    <div class="legend-item"><div class="legend-dot" style="background:{c}"></div>{l}</div>' for c, l in legend)
        secs.append(f"""<div class="diagram-section" id="{i}">
  <h2><span class="num">{n + 1}</span> {t}</h2>
  <p class="subtitle">{sub}</p>
  <div class="diagram-wrap">
{fn()}
  </div>
  <div class="legend">
{leg}
  </div>
{f'  <div class="note-box">{note}</div>' + chr(10) if note else ""}</div>""")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Credit Risk Portfolio Analytics — Architecture Diagrams</title>
<style>{CSS}</style>
</head>
<body>
<div class="page-header"><div class="container">
  <h1>Credit Risk Portfolio Analytics</h1>
  <p>Architecture Diagrams</p>
  <div class="badge">ASSIGNMENT DELIVERABLE &mdash; ARCHITECTURE</div>
</div></div>
<div class="container">
<div class="toc">
  <h3>Contents</h3>
{toc}
</div>
{chr(10).join(secs)}
</div>
</body>
</html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print("wrote", OUT_HTML)


def export_png():
    from playwright.sync_api import sync_playwright
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    names = ["01_end_to_end_architecture", "02_data_lineage", "03_model_pipeline", "04_anomaly_detection", "05_production_target"]
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_context(viewport={"width": 1300, "height": 900}, device_scale_factor=2.5).new_page()
        pg.goto(OUT_HTML.as_uri())
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(2500)
        for i, nm in enumerate(names):
            pg.locator(".diagram-wrap svg").nth(i).screenshot(path=str(OUT_PNG / f"{nm}.png"))
        b.close()
    print("PNGs ->", OUT_PNG)


if __name__ == "__main__":
    build()
    export_png()
