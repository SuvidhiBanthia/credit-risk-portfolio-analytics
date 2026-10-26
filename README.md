# Credit Risk Portfolio Analytics

Assignment deliverables for the credit-risk portfolio exercise (5 synthetic tables, 10,000 customers, 14,685 loan accounts).

## Deliverables

| Assignment item | Where |
|---|---|
| 1. Data ingestion (+ data-quality notes) | `src/ingest.py` -> `data/processed/analytics_dataset.parquet`, `docs/data_quality_notes.md` |
| 2. Data analytics (profiling, cross-cuts, 5 insights) | `src/analytics.py` -> `docs/insights.md`, `outputs/analytics/` |
| 3. Model (LR baseline + XGBoost, AUC/KS/Gini, SHAP, monitoring) | `src/model.py`, `src/monitoring.py` -> `docs/model_card.md`, `outputs/model/`, `models/` |
| 4. Dashboard (portfolio + account view, filters, drill-down) | `app/dashboard.py` |
| 5. AI feature: anomaly detection, embedded in the dashboard | `src/anomaly.py`, "Anomaly watch" tab -> `docs/anomaly_feature.md` |
| 1-page product brief | `docs/product_brief.docx` / `.pdf` |
| Assignment brief (the task as given) | `docs/assignment/assignment_deliverables_one_pager.pdf`, `data/raw/DATA_DICTIONARY.md` |
| Architecture diagrams | `docs/architecture-diagrams.html`, PNGs in `outputs/diagrams/` |
| 10-minute CRO pitch deck | `docs/CRO_pitch_deck.pptx` |
| Assumptions log (+ what a bank must provide) | `docs/assumptions_log.md` |

## Setup

Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The 5 raw tables and their data dictionary are in `data/raw/`. Generated data, models and outputs are committed, so the dashboard runs without re-running the pipeline.

## Run order

```powershell
.venv\Scripts\python.exe src\ingest.py        # clean + join -> analytics dataset
.venv\Scripts\python.exe src\analytics.py     # profiling, insights, charts
.venv\Scripts\python.exe src\model.py         # train, evaluate, SHAP, out-of-fold scores (a few minutes)
.venv\Scripts\python.exe src\anomaly.py       # anomaly signals + evaluation
.venv\Scripts\streamlit run app\dashboard.py  # dashboard
```

Verification: `src/verify_all.py` re-checks integrity, every number typed in the brief / deck / diagrams against the data, the artifacts and the app.

Documents: `src/build_diagrams.py` (architecture diagrams) and `src/build_brief.py` (product brief).

## Read first
This is a synthetic dataset, the default definition is inferred (assumption A5), and discrimination is moderate (test AUC ~0.68). See the top of `docs/assumptions_log.md`.
