"""Builds docs/product_brief.docx (1 page, A4).  Run: .venv\\Scripts\\python.exe src\\build_brief.py"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "product_brief.docx"
BLUE, SLATE, MUTED = RGBColor(0x01, 0x72, 0xB6), RGBColor(0x2C, 0x3E, 0x50), RGBColor(0x64, 0x74, 0x8B)
BODY = 9.6

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.6)
sec.top_margin, sec.bottom_margin = Cm(1.3), Cm(1.1)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(BODY)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.space_before = Pt(0)
st.paragraph_format.line_spacing = 1.0


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:color"), "auto")
    s.set(qn("w:fill"), hex_)
    tcPr.append(s)


def cell_margins(table, top=50, bottom=50, left=90, right=90):
    tblPr = table._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        m.append(e)
    tblPr.append(m)


def borders(table, color="E2E8F0", sz=4):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for k in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def set_widths(table, widths_cm):
    table.autofit = False
    tblPr = table._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    for row in table.rows:
        for c, w in zip(row.cells, widths_cm):
            c.width = Cm(w)


def bottom_rule(p, color="0172B6", sz=12):
    pPr = p._p.get_or_add_pPr()
    pb = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "1")
    b.set(qn("w:color"), color)
    pb.append(b)
    pPr.append(pb)


def run(p, text, bold=False, italic=False, size=BODY, color=None):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if color is not None:
        r.font.color.rgb = color
    return r


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run(p, text.upper(), True, size=9.5, color=BLUE)
    bottom_rule(p, "E2E8F0", 4)
    return p


def bullet(container, lead, rest, size=BODY, indent=0.38):
    p = container.add_paragraph()
    p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.first_line_indent = Cm(-0.3)
    p.paragraph_format.space_after = Pt(1.5)
    run(p, "•  ", size=size, color=BLUE)
    if lead:
        run(p, lead + " ", True, size=size, color=SLATE)
    run(p, rest, size=size)
    return p


# ------------------------------------------------------------------ title
p = doc.add_paragraph()
run(p, "Credit Risk Portfolio Monitor", True, size=19, color=SLATE)
run(p, "   Product Brief", False, size=12, color=BLUE)
bottom_rule(p)
p.paragraph_format.space_after = Pt(2)
p = doc.add_paragraph()
run(p, "See where portfolio risk sits, score every loan account with reasons, and surface accounts that behave unusually - in one dashboard.", False, True, 10, MUTED)
p.paragraph_format.space_after = Pt(1)
p = doc.add_paragraph()
run(p, "Written for: bank credit-risk leadership (CRO, Head of Portfolio Risk, Head of Collections) deciding whether to pilot this. Figures are from the synthetic 14,685-account, $601M portfolio.", False, True, 8, MUTED)

# ------------------------------------------------------------------ problem
heading("The problem")
bullet(doc, "Risk is seen late and in pieces.",
       "Customer, loan, bureau, repayment and collections data live in five tables. Joining them is manual, so the book is reviewed in static reports - often after accounts are already delinquent.")
bullet(doc, "Concentrations are hidden.",
       "Mortgages are 10% of accounts but 52% of exposure and 65% of defaulted exposure. Customers earning under $60k are 22% of exposure but 34% of defaulted exposure. Interest rates are flat (11.8-12.0%) across income bands whose default rates range from 9.5% to 32%.")
bullet(doc, "Scores without reasons, alerts without priority.",
       "Reviewers cannot see why an account is risky, and early-warning signals arrive unranked - there is no work queue sized to analyst capacity.")

# ------------------------------------------------------------------ users
heading("Target users")
t = doc.add_table(rows=5, cols=3)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
borders(t)
cell_margins(t)
rows = [("User", "Job to be done", "Uses"),
        ("Portfolio risk analyst  (primary, daily)", "Find risky segments, explain one account's score, work the alert queue", "Portfolio and Account views, Anomaly watch"),
        ("CRO / Head of Credit Risk  (decision-maker, monthly)", "Know concentrations, risk-band mix and model health; set appetite and limits", "Portfolio view, model card"),
        ("Collections early-warning lead", "Receive confirmed escalations and act on them", "Anomaly review outcomes"),
        ("Model risk and compliance", "Check leakage, fairness and the audit trail before sign-off", "Model card, assumptions log, audit log")]
for i, r_ in enumerate(rows):
    for j, txt in enumerate(r_):
        c = t.cell(i, j)
        c.paragraphs[0].text = ""
        run(c.paragraphs[0], txt, i == 0 or j == 0, size=9, color=SLATE if i == 0 or j == 0 else None)
        if i == 0:
            shade(c, "E8F4FC")
set_widths(t, [5.2, 8.0, 4.6])

# ------------------------------------------------------------------ what it does
heading("What it does")
bullet(doc, "One clean dataset.", "Five tables joined to one row per loan account; every data-quality fix is logged with its count and treatment.")
bullet(doc, "A risk score with reasons.",
       "Probability of default (XGBoost, with a logistic-regression benchmark) and SHAP drivers per account. Test AUC 0.68, KS 0.28 - moderate, but well calibrated: the High and Very High bands hold 25% of accounts, "
       "41% of exposure and 65% of defaulted exposure (30% default vs 8% in the Low band).")
bullet(doc, "An interactive dashboard.", "Portfolio view (risk bands, exposure, filters, drill-down, cross-cuts, vintage trends) and Account view (score, drivers, 12-month repayment, bureau lines).")
bullet(doc, "An embedded AI feature - Anomaly watch.",
       "Three unsupervised detectors (own-history DPD spike, peer-segment outlier, bureau-vs-repayment divergence) feed a ranked review queue. An analyst confirms or dismisses each alert and every decision is logged. Nothing is automated.")

# ------------------------------------------------------------------ scope
heading("Scope")
s = doc.add_table(rows=2, cols=2)
s.alignment = WD_TABLE_ALIGNMENT.CENTER
borders(s)
cell_margins(s, 60, 60, 110, 110)
shade(s.cell(0, 0), "E3F8F1")
shade(s.cell(0, 1), "FEE2E2")
for j, (txt, col) in enumerate([("IN - built in this project", RGBColor(0x2B, 0xA8, 0x82)), ("OUT - not built, not claimed", RGBColor(0x99, 0x1B, 0x1B))]):
    c = s.cell(0, j)
    c.paragraphs[0].text = ""
    run(c.paragraphs[0], txt, True, size=9, color=col)
IN = ["Ingestion, validation and data-quality handling for all 5 tables",
      "Portfolio analytics: income, employment, region, age, loan type, cross-cuts, 5 insights",
      "PD model: baseline + XGBoost, AUC / KS / Gini, SHAP, calibration, fairness check, PSI / CSI monitoring plan",
      "Streamlit dashboard with filters, drill-down and account view",
      "Anomaly watch with human review and audit log",
      "Model card, assumptions log, architecture diagrams, pitch deck"]
OUTS = ["Automated credit decisions (auto-decline, repricing, limit changes)",
        "Loss given default, expected-loss or provisioning numbers (no recovery data)",
        "Calendar-time forecasting (repayment months carry no dates)",
        "Production integration: core-banking feeds, SSO, case management, real-time scoring",
        "LLM \"Ask the Portfolio\" chat (anomaly detection chosen instead)",
        "Regulatory model validation and sign-off"]
for j, items in enumerate([IN, OUTS]):
    c = s.cell(1, j)
    c.paragraphs[0].text = ""
    first = True
    for it in items:
        if first:
            p = c.paragraphs[0]
            first = False
        else:
            p = c.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.3)
        p.paragraph_format.first_line_indent = Cm(-0.3)
        p.paragraph_format.space_after = Pt(1.5)
        run(p, "•  ", size=9, color=BLUE)
        run(p, it, size=9)
for j in range(2):
    shade(s.cell(1, j), "F8FAFC")
set_widths(s, [8.9, 8.9])

# ------------------------------------------------------------------ success + guardrails
heading("Pilot success measures and guardrails")
g = doc.add_table(rows=1, cols=2)
g.alignment = WD_TABLE_ALIGNMENT.CENTER
cell_margins(g, 20, 20, 0, 110)
set_widths(g, [8.9, 8.9])
left, right = g.cell(0, 0), g.cell(0, 1)
left.paragraphs[0].text = ""
run(left.paragraphs[0], "How we would know it works", True, size=9, color=SLATE)
for lead, rest in [("Time to answer \"where is the risk and why?\"", "- baseline vs pilot"),
                   ("Alert confirm rate by detector", "- retune any detector dismissed > 70% of the time"),
                   ("Score stability", "- PSI < 0.10; an AUC drop > 0.03 triggers review"),
                   ("Defaulted exposure in High + Very High bands", "- 65% in this back-test")]:
    bullet(left, lead, rest, size=9, indent=0.3)
right.paragraphs[0].text = ""
run(right.paragraphs[0], "Risks and guardrails", True, size=9, color=SLATE)
for lead, rest in [("Moderate discrimination", "(AUC 0.68 on noisy synthetic data) - validate on real data before relying on it."),
                   ("Default definition is assumed", "(flag treated as an outcome after month 12) - the bank must confirm."),
                   ("Age is a model input", "- needs a compliance ruling; gender and marital status are excluded and fairness-tested."),
                   ("Alerts never act", "- they cannot change a score, decline credit or contact a customer.")]:
    bullet(right, lead, rest, size=9, indent=0.3)

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
print("saved", OUT)
