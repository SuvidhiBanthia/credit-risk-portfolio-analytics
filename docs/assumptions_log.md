# Assumptions Log

Everything we invented, approximated or inferred, and what a real bank's IT / data team would need to provide to ship this. IDs are stable - the model card, dashboard, diagrams and deck cite them (e.g. A5, A15), so **add new rows at the end and never renumber**.

**37 assumptions:** 6 invented · 7 approximated · 1 inferred · 15 design choice · 8 data limitation  |  impact if wrong: 6 high · 17 medium · 14 low

| Type | Meaning |
|---|---|
| **Invented** | A value, rule or plan we chose with no source in the data (e.g. alert thresholds, pilot plan) |
| **Approximated** | A proxy used for something the data does not contain (e.g. original loan amount as exposure) |
| **Inferred** | Deduced from the data but not documented, so uncertain (the default definition) |
| **Design choice** | A deliberate, defensible decision (e.g. excluding collections from the model) |
| **Data limitation** | A fact about this dataset that we work around rather than fix |

## The six that matter most
1. **A30 - the data is synthetic.** Treat every number as illustrative until re-run on real data.
2. **A5 - the default definition is inferred, not documented.** It decides whether repayment features are valid and at which point the score applies.
3. **A6 - the bureau is a snapshot, not point-in-time.** Application-stage performance (AUC ~0.66) is the leakage-free floor.
4. **A23 - discrimination is moderate (AUC ~0.68)** and did not improve with more features: the ceiling is the data.
5. **A18 - no recovery data**, so no loss-given-default or expected-loss view (A32 is only a proxy).
6. **A19 - protected attributes are excluded, but age is still an input** and fails the four-fifths heuristic for under-30s: it needs a compliance ruling.


## 1. Overarching
Read these first - they qualify everything else.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A30 | Data limitation | High | **The dataset is synthetic.** Patterns in it (months look independent given an account's own tendency, no vintage trend, collections dated before origination, income floored at 12,000) reflect how it was generated, not real banking behaviour. Every figure in the dashboard, brief and deck is a property of this dataset | Stated in the data dictionary | A real portfolio extract - expect different signal strength, drift and segment patterns |
| A5 | Inferred | High | **Default flag is a separate outcome observed after the 12-month repayment window** (not computed from those months), so repayment aggregates are valid predictors - but only at the "12 months on book" scoring point. At application, only the application+bureau model applies (AUC ~0.66). **Uncertain - inferred from the data, not documented.** Evidence: (1) max DPD in any month is 89, so no 90+ DPD rule can generate the flag; (2) 8% of accounts with zero late/missed payments are flagged, and only 34% of accounts with 3+ missed; (3) defaulters' late rate is already higher in month 1 (21% vs 15%) and stays flat through month 12 (no pre-default deterioration); (4) outstanding-balance run-off is the same for defaulters (0.74 vs 0.72), so payments did not stop in-window; (5) default rate is the same for 2018 and 2023 originations, so it is a static label, not "default within N months". Residual risk: the synthetic generator may have created repayment behaviour *from* the flag; the application+bureau ablation is the leakage-free floor | Default definition and observation window are not documented in the dictionary | Exact default definition (e.g. 90+ DPD, 12 months after the performance window) and the date the label was struck |
| A23 | Data limitation | High | Discrimination is moderate (test AUC ~0.68, KS ~0.28). Extra features (monthly DPD, bureau mix, ratios) and deeper trees did not help, so we stop at this feature set rather than overfit | Tested in grouped CV | Richer data (transactions, income verification, real bureau score) |
| A18 | Data limitation | High | Loss given default cannot be assessed; secured accounts are not shown to be safer on default frequency, but recoveries after default are not in the data | `amount_recovered` exists only in unreliable collections events | Recovery and write-off data per account |

## 2. Target, definitions and units
What 'default' and 'exposure' mean here.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A1 | Design choice | Medium | Model grain is one row per loan account (target is account-level); customer/bureau features are joined onto each account | `target_default` lives on `loan_accounts` | Confirm default definition (e.g. 90+ DPD) and observation window |
| A14 | Design choice | Low | Amounts are in USD as per the dictionary; no currency conversion or inflation adjustment | Dictionary says USD | — |
| A15 | Approximated | Medium | "Exposure" = original `loan_amount` (not current outstanding), because balances exist only for months 1-12 | No as-of outstanding balance | Current outstanding balance / EAD per account |
| A32 | Approximated | Medium | "Expected defaulted exposure" on the dashboard = predicted PD x original loan amount. It is a loss **proxy**, not expected credit loss: no LGD, no discounting, no staging | No recovery or balance data to do more | LGD, EAD and the bank's provisioning methodology |

## 3. Data ingestion and quality
How data issues were handled (each fix is also logged with counts in `data_quality_notes.md`).

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A6 | Approximated | High | Bureau is a point-in-time snapshot at 2025-12-31 and all lines are used for every account of the customer | Bureau `date_opened` is before the loan origination for only 37% of customers, so a point-in-time bureau cannot be rebuilt | Bureau pull at application date (point-in-time) |
| A7 | Approximated | Low | Missing `years_employed`, `years_at_residence`, `num_dependents`, `interest_rate_pct` are missing at random; imputed with group medians and flagged | Missingness rates are flat (~5% / 2%) and show no relation to target | Reason codes for missing fields from the source system |
| A8 | Approximated | Low | `years_employed` > age-18 treated as impossible and capped (10% of customers); minimum working age 18 assumed | Data is synthetic and contains such values | Source definition of the field (may include prior employment) |
| A9 | Approximated | Low | Bureau `days_past_due` missing on Closed lines filled with 0; missing on Active/Written-Off lines left NaN | DPD is always 0 on closed lines in the data | Reporting rules of the bureau |
| A10 | Design choice | Medium | Customers with no bureau file (10%) are "thin-file", not "clean": counts/amounts 0, ratios NaN, `has_bureau` flag | No history is not good history | Policy for thin-file customers |
| A12 | Data limitation | Low | `customer_since` is later than loan origination for ~50% of loans; not corrected and not used as a feature | Cannot tell which date is wrong | Customer master data lineage |
| A13 | Data limitation | Low | Annual income appears floored at 12,000 (76 customers); kept as is | Looks like a clip | Income verification method (stated vs documented) |
| A11 | Data limitation | Medium | Collections event dates are unreliable (45% precede origination; only 21% of events are on defaulted accounts), so collections are used for descriptive analytics only | Observed in the data | Clean, timestamped collections system extract |
| A2 | Design choice | Medium | Collections data is excluded from model features (post-delinquency, leaks the target); used only for analytics/dashboard | Collections actions are triggered by delinquency | Event timestamps reliable enough to build point-in-time features |

## 4. Analytics
Profiling, cross-cuts and insights.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A3 | Data limitation | Medium | No calendar date on repayment rows, so "trends" use origination-year vintage and repayment month 1–12 curves | Only `payment_month` index is provided | Dated monthly performance snapshots |
| A16 | Design choice | Low | Segment differences are read only where the 95% Wilson interval and cell size (n>=100) support them; cells with n<30 are blanked (e.g. 100k+ x Unemployed, n=5) | Avoid over-reading noise | — |
| A17 | Data limitation | Medium | "Pricing does not follow risk" uses stated interest rate only; fees, insurance and the bank's pricing policy are unknown | Rate is the only price field | Pricing grid / fee schedule |
| A31 | Design choice | Low | Band cut-points are our choices: income (<40k, 40-60k, 60-80k, 80-100k, 100k+), age (<30, 30-39, 40-49, 50-59, 60+), loan/income and EMI/income bins. Segment findings shift somewhat with other cut-points | Readable segments with enough accounts per cell | The bank's own segment definitions |

## 5. Modelling and compliance
Scoring, validation, fairness and monitoring.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A19 | Design choice | High | Gender and marital status are excluded from the model (protected attributes); age is kept but flagged for compliance review. Fairness is tested on gender and age with an approval impact ratio under a "decline riskiest 20%" policy | Credit-decisioning law in many jurisdictions restricts these attributes; jurisdiction unknown | Legal/compliance ruling on permitted attributes; real decline policy |
| A20 | Design choice | Medium | Train/test split is stratified and grouped by customer (80/20, seed 42); tuning uses grouped 5-fold CV on train only; dashboard scores are out-of-fold | Customers have up to 3 accounts; avoids optimistic scores | Out-of-time validation sample (needs dated outcomes) |
| A21 | Invented | Medium | Risk bands are fixed PD cut-offs: Low <10%, Medium 10-20%, High 20-35%, Very High >=35% | Round numbers that track observed default (8% / 14% / 26% / 42%) | Bank's own rating-grade / appetite thresholds |
| A22 | Design choice | Medium | Cross-account features (a customer's other loans' repayment behaviour) are legitimate predictors; they use repayment only, never other accounts' target | Defaults cluster within customers (5% vs 2.6% expected) | Confirm all of a customer's accounts are known at scoring time |
| A33 | Invented | Medium | Monitoring thresholds are industry rules of thumb, not tuned to this portfolio: score PSI < 0.10 stable / > 0.25 act; AUC drop > 0.03; retrain every 6 months; a detector dismissed > 70% of the time is tightened. The only PSI/CSI measured is train vs test (0.003) - there is no real drift history | No post-deployment data exists | The bank's model-risk monitoring standard and tolerances |
| A37 | Design choice | Low | Pipelines use fixed seeds and are re-runnable, but refitting XGBoost moved test AUC by ~0.001 between runs (0.6837 vs 0.6845). The deck uses the headline test value for the full model so charts and cards agree | Multi-threaded tree building is not bit-identical | A model registry with frozen artefacts and data snapshots |

## 6. AI feature - anomaly detection
The embedded AI feature and its human-in-the-loop controls.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A4 | Design choice | Low | Anomaly detection chosen over LLM chat (deterministic, no hallucinated numbers) | User decision | — |
| A26 | Approximated | Medium | Anomaly detection (not LLM chat) is the chosen AI feature. Detectors are unsupervised; default is used only to evaluate them, as a **proxy** for "worth a look" (a flagged account that never defaults is counted as a false-positive proxy, not necessarily a wrong alert) | No labelled anomalies exist | Analyst-labelled alert history to replace the proxy |
| A27 | Invented | Medium | Alert thresholds are judgement calls, set to give a ~6% alert volume: spike = DPD >= 60 in months 10-12 vs own months 1-9 never above 14 days; peer outlier = top 3% of each loan type (Isolation Forest, 15 features); divergence = bureau vs repayment disagreement. "Latest 3 months" = months 10-12 of the 12-month window. Own-history spikes are a weak default signal here (19% vs 16%) because months look independent given an account's tendency | Review capacity unknown; synthetic behaviour | Real analyst capacity, calendar-dated repayments and bureau time series for true 'sudden change' detection |
| A28 | Design choice | Low | The review log is a local CSV (`outputs/anomaly/review_log.csv`) with timestamp, analyst name (free text), thresholds in force and decision. In production this would be a database with authenticated users and role-based access | Prototype only | SSO / user identity, case-management system integration, retention policy |
| A29 | Design choice | Medium | Alerts never change a score, decline credit or contact a customer; they only create review tasks (human-in-the-loop). Peer groups are loan types only | Avoids automated adverse action; simplicity | Bank policy on early-warning actions; richer peer definitions (region, vintage, product tier) |

## 7. Dashboard, deck and documents
Prototype choices and the proposals made in the deliverables.

| # | Type | Impact | Assumption | Why | Needed from a real bank |
|---|---|---|---|---|---|
| A24 | Design choice | Low | Account scores shown are out-of-fold PDs from the XGBoost model; the "observed outcome" is displayed as a back-test reveal (only possible because data is historical). In production the score would come from the final model (`models/xgb_final_all_data.json`) on new accounts | Honest historical scores | Real-time scoring service and feature store feeding the dashboard |
| A25 | Design choice | Low | Risk-band colours reuse the status palette (green/amber/orange/red) always paired with the band name; drivers are SHAP in log-odds, aggregated per original feature | Accessibility: colour never carries meaning alone | Bank's own risk-grade colour conventions |
| A34 | Invented | Medium | The 90-day pilot plan (three phases), its success measures and the "decision requested" are proposals by us, not derived from the data | A pilot is the sensible next step for a moderate-AUC prototype | The bank's pilot governance, sponsor and analyst capacity |
| A35 | Invented | Low | The personas (portfolio analyst, CRO, collections lead, model risk) and the in/out scope in the product brief are assumed; no stakeholder interviews were held | The brief requires users and scope | Validation with real users and their workflows |
| A36 | Invented | Medium | The production target architecture (diagram 5) and its tools (e.g. MLflow, feature store, SSO, case management) are illustrative: not designed with the bank's IT team, sized or priced | Shows what must exist around the prototype | The bank's target platform, vendors and security standards |

## What a real bank's IT / data team must provide to ship this
Grouped by owner. **Blocks go-live** = the product cannot responsibly run on live data without it; otherwise it improves quality or unlocks a feature.

| # | Need | Why it matters | Blocks go-live? | Assumptions | Suggested owner |
|---|---|---|---|---|---|
| N1 | The default definition and the date the label is struck (e.g. 90+ DPD, observation and performance windows) | The target *is* the product; today it is inferred | **Yes** | A1, A5, A20 | Credit risk policy + data engineering |
| N2 | Dated monthly performance: repayment and balance snapshots per account | Calendar trends, out-of-time validation, real 'sudden change' detection | **Yes** (validation) | A3, A20, A27 | Data engineering |
| N3 | Point-in-time bureau pulls at application, plus bureau history over time | Removes look-ahead; thin-file policy; bureau trend detection | **Yes** | A6, A10, A27 | Bureau relations + data engineering |
| N4 | Current outstanding balance (EAD) and recovery data (LGD) | Exposure, expected loss, collateral value, provisioning inputs | Yes for limits / provisioning; no for monitoring | A15, A18, A32 | Finance + collections operations |
| N5 | Clean, timestamped collections events | Post-delinquency analytics and recovery curves | No | A2, A11 | Collections operations |
| N6 | Source-field definitions and reasons for missing values (years employed, interest rate, DPD, customer_since, income verification, pricing grid) | Replaces our imputation rules and date-conflict guesses | No (quality) | A7, A8, A9, A12, A13, A17 | Data owner of each source system |
| N7 | Compliance ruling on permitted attributes (age), fairness policy and adverse-action reason mapping | Legal exposure of the score and of any decision it informs | **Yes** | A19, A29 | Compliance and legal |
| N8 | Independent model validation, rating grades and appetite thresholds, the bank's monitoring standard | Governance sign-off; replaces our rule-of-thumb thresholds | **Yes** | A21, A23, A33 | Model risk management |
| N9 | Platform: scheduled feeds with data contracts and DQ gates, point-in-time feature store, scoring service, dashboard behind SSO and role-based access, case-management integration, audit database and retention policy | Productionisation of the prototype | **Yes** | A24, A28, A36 | IT, security and architecture |
| N10 | Analyst capacity and a labelled alert history | Tune alert volume; replace default-as-proxy with real precision | No (needed to tune) | A26, A27, A29, A34 | Credit risk operations |
