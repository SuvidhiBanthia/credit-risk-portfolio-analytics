"""Credit Risk Portfolio Monitor - Streamlit dashboard.

Run:  .venv\\Scripts\\streamlit run app\\dashboard.py
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly import DEFAULTS, FEATURE_TEXT, apply_thresholds, reasons  # noqa: E402
from lib import (AQUA, BAND_COLOR, BAND_ORDER, BAND_RULE, BLUE, DIMENSIONS, GRID, INK2, MUTED, ORANGE, RED, SEQ_BLUE,  # noqa: E402
                 fmt_value, label, load_bureau, load_collections, load_portfolio, load_repayment, load_shap, load_signals, money,
                 segment_table, style, wilson)

ROOT = Path(__file__).resolve().parents[1]
REVIEW_LOG = Path(os.environ.get("ANOMALY_REVIEW_LOG", ROOT / "outputs" / "anomaly" / "review_log.csv"))
LOG_COLS = ["timestamp", "account_id", "decision", "reason_code", "note", "analyst", "severity", "triggers", "thresholds"]
DECISIONS = ["Confirm concern", "Dismiss - false positive", "Need more information", "Escalate to collections early-warning"]
REASON_CODES = ["Deteriorating repayment", "Hidden stress elsewhere (bureau)", "One-off payment issue (holiday / bounced debit)",
                "Customer contacted - resolved", "Data error", "Expected behaviour for this segment", "Suspected fraud / dispute", "Other"]

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon=":bar_chart:", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem; max-width: 1500px;}
    h1 {font-size: 1.7rem !important; margin-bottom: 0.1rem;}
    div[data-testid="stMetric"] {background: #f4f4f1; border-radius: 10px; padding: 10px 14px;}
    .band-chip {display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;}
    .small {color:#52514e; font-size: 0.85rem;}
    </style>""", unsafe_allow_html=True)

full = load_portfolio()
PORT_RATE = full.target_default.mean()


# ---- anomaly helpers (settings live in session_state so every tab sees the same thresholds) ----------
def an_settings():
    ss = st.session_state
    return dict(spike_dpd=int(ss.get("an_spike", DEFAULTS["spike_dpd"])), base_max=int(ss.get("an_base", DEFAULTS["base_max"])),
                if_cut=1 - float(ss.get("an_peer", 100 * (1 - DEFAULTS["if_cut"]))) / 100, use_spike=bool(ss.get("an_use_spike", True)),
                use_peer=bool(ss.get("an_use_peer", True)), use_div=bool(ss.get("an_use_div", True)))


@st.cache_data(show_spinner=False)
def compute_flags(**settings):
    return apply_thresholds(load_signals(), **settings)


def read_review_log():
    if REVIEW_LOG.exists():
        return pd.read_csv(REVIEW_LOG)
    return pd.DataFrame(columns=LOG_COLS)


def save_review():
    ss = st.session_state
    ctx = ss.get("rv_ctx")
    if not ctx:
        return
    rec = dict(timestamp=datetime.now().isoformat(timespec="seconds"), account_id=ctx["account_id"], decision=ss["rv_dec"],
               reason_code=ss["rv_reason"], note=ss.get("rv_note", ""), analyst=ss.get("rv_analyst", "analyst"), severity=ctx["severity"],
               triggers=ctx["triggers"], thresholds=json.dumps(ctx["settings"]))
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec], columns=LOG_COLS).to_csv(REVIEW_LOG, mode="a", header=not REVIEW_LOG.exists(), index=False)
    ss["rv_saved"] = ctx["account_id"]

# =============================================================================== sidebar filters
FILTERS = {"Loan type": "loan_type", "Region": "region", "Employment type": "employment_type",
           "Income band": "income_band", "Age band": "age_band", "Risk band": "risk_band"}
OPTIONS = {k: [o for o in (DIMENSIONS[k][1] if k in DIMENSIONS else BAND_ORDER)] for k in FILTERS}
for k in FILTERS:
    st.session_state.setdefault(f"flt_{k}", OPTIONS[k])


def reset_filters():
    for k in FILTERS:
        st.session_state[f"flt_{k}"] = OPTIONS[k]


with st.sidebar:
    st.header("Filters")
    st.caption("Apply to every tab.")
    for k in FILTERS:
        st.multiselect(k, OPTIONS[k], key=f"flt_{k}")
    yr = st.slider("Origination year", 2018, 2023, (2018, 2023), key="flt_year")
    st.button("Reset filters", on_click=reset_filters, use_container_width=True)

mask = pd.Series(True, index=full.index)
for k, col in FILTERS.items():
    mask &= full[col].isin(st.session_state[f"flt_{k}"])
mask &= full.orig_year.between(*yr)
f = full[mask].copy()

st.title("Credit Risk Portfolio Monitor")
st.caption("14,685 loan accounts - 10,000 customers - synthetic data. PD = predicted probability of default (out-of-fold XGBoost, "
           "scored at 12 months on book).")
if f.empty:
    st.warning("No accounts match the current filters. Widen the filters in the sidebar.")
    st.stop()
with st.sidebar:
    st.divider()
    st.metric("Accounts in view", f"{len(f):,}", f"{len(f) / len(full):.0%} of portfolio", delta_color="off")

tab_port, tab_acct, tab_anom, tab_method = st.tabs(["Portfolio", "Account", "Anomaly watch", "Method & data"])


# =============================================================================== portfolio
def heat(piv, n, text, title, colorbar):
    z = piv.where(n >= 30)
    fig = go.Figure(go.Heatmap(z=z.values, x=list(piv.columns), y=list(piv.index), text=text.values, texttemplate="%{text}",
                               colorscale=SEQ_BLUE, hoverongaps=False, colorbar=dict(title=colorbar, thickness=12),
                               customdata=n.values, hovertemplate="%{y} x %{x}<br>%{text}<br>accounts: %{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, 380, legend=False)


with tab_port:
    k = st.columns(6)
    d_rate = (f.target_default.mean() - PORT_RATE) * 100
    k[0].metric("Accounts", f"{len(f):,}")
    k[1].metric("Exposure", money(f.loan_amount.sum()), help="Original loan amount (assumption A15).")
    k[2].metric("Observed default", f"{f.target_default.mean():.1%}", f"{d_rate:+.1f} pp" if abs(d_rate) >= 0.05 else "= portfolio",
                delta_color="inverse" if abs(d_rate) >= 0.05 else "off", help="Delta is versus the whole portfolio (16.1%).")
    k[3].metric("Average PD", f"{f.pd_xgb.mean():.1%}", help="Mean predicted probability of default (out-of-fold).")
    k[4].metric("Exp. default $", money(f.exp_default.sum()), help="Expected defaulted exposure: sum of PD x loan amount. A loss proxy before recoveries (no LGD data).")
    hi = f[f.risk_band.isin(["High", "Very High"])]
    k[5].metric("High-risk exposure", f"{hi.loan_amount.sum() / f.loan_amount.sum():.0%}", help="Share of exposure in the High and Very High risk bands.")

    # ---- risk segments
    st.subheader("Risk segments")
    bt = f.groupby("risk_band", observed=False).agg(accounts=("account_id", "size"), exposure=("loan_amount", "sum"),
                                                    avg_pd=("pd_xgb", "mean"), observed=("target_default", "mean"),
                                                    exp_default=("exp_default", "sum")).reindex(BAND_ORDER)
    c1, c2 = st.columns([5, 4])
    with c1:
        fig = go.Figure()
        for b in BAND_ORDER:
            fig.add_bar(x=[b], y=[bt.loc[b, "exposure"] / 1e6], marker_color=BAND_COLOR[b], name=b, width=0.55,
                        text=[f"${bt.loc[b, 'exposure'] / 1e6:,.0f}M<br>{bt.loc[b, 'exposure'] / bt.exposure.sum():.0%}"],
                        textposition="outside", hovertemplate=f"{b}: %{{y:,.1f}}M<extra></extra>")
        fig.update_yaxes(title="Exposure ($M)")
        fig.update_layout(bargap=0.3)
        fig.update_xaxes(categoryorder="array", categoryarray=BAND_ORDER)
        st.plotly_chart(style(fig, 330, legend=False), key="band_bar")
    with c2:
        t = pd.DataFrame({"Band": [f"{b} ({BAND_RULE[b]})" for b in bt.index], "Accounts": bt.accounts.values,
                          "% accts": bt.accounts.values / max(bt.accounts.sum(), 1) * 100, "Avg PD": bt.avg_pd.values * 100,
                          "Observed": bt.observed.values * 100, "Exp. default $M": bt.exp_default.values / 1e6})
        st.dataframe(t, hide_index=True, column_config={
            "Accounts": st.column_config.NumberColumn(format="%d"), "% accts": st.column_config.NumberColumn(format="%.0f%%"),
            "Avg PD": st.column_config.NumberColumn(format="%.1f%%"), "Observed": st.column_config.NumberColumn(format="%.1f%%", help="Observed default rate"),
            "Exp. default $M": st.column_config.NumberColumn(format="$%.1fM")})
        st.caption("Predicted PD tracks observed default in every band (calibration check on the Method tab). "
                   "Colours are risk status, always paired with the band name.")

    # ---- dimension explorer + drill-down
    st.subheader("Explore by dimension")
    e1, e2, e3 = st.columns([1.2, 1.4, 1.4])
    dim_name = e1.selectbox("Break down by", list(DIMENSIONS), index=1, key="dim_a")
    metric = e2.radio("Show", ["Observed default rate", "Average predicted PD", "Share of exposure"], horizontal=True, key="metric_a")
    dcol, dorder = DIMENSIONS[dim_name]
    seg = segment_table(f, dcol, dorder)
    drill_opts = ["(none)"] + [str(i) for i in seg.index]
    drill = e3.selectbox("Drill into segment", drill_opts, key="drill_seg",
                         help="Zooms the breakdown below into one segment, split by a second dimension.")
    ycol = {"Observed default rate": "default_rate", "Average predicted PD": "avg_pd", "Share of exposure": "exposure_share"}[metric]
    xs = [str(i) for i in seg.index]
    fig = go.Figure(go.Bar(
        x=xs, y=seg[ycol], marker_color=[ORANGE if x == drill else BLUE for x in xs], width=0.6,
        text=[f"{v:.1%}" for v in seg[ycol]], textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=13), cliponaxis=False,
        error_y=dict(type="data", symmetric=False, array=(seg.ci_high - seg.default_rate) if ycol == "default_rate" else None,
                     arrayminus=(seg.default_rate - seg.ci_low) if ycol == "default_rate" else None, color=INK2, thickness=1, width=3)
        if ycol == "default_rate" else None,
        customdata=np.c_[seg.accounts, seg.exposure / 1e6, seg.exposure_share, seg.default_rate, seg.avg_pd],
        hovertemplate="<b>%{x}</b><br>accounts: %{customdata[0]:,.0f}<br>exposure: $%{customdata[1]:,.1f}M (%{customdata[2]:.1%})"
                      "<br>observed default: %{customdata[3]:.1%}<br>avg PD: %{customdata[4]:.1%}<extra></extra>"))
    if ycol != "exposure_share":
        fig.add_hline(y=PORT_RATE if ycol == "default_rate" else f.pd_xgb.mean(), line_dash="dash", line_color=INK2,
                      annotation_text="portfolio" if ycol == "default_rate" else "view avg", annotation_position="top right")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(style(fig, 340, legend=False), key="dim_bar")
    st.caption("Whiskers = 95% confidence interval. Hover for accounts and exposure. Orange = drilled segment.")

    if drill != "(none)":
        sub = f[f[dcol].astype(str) == drill]
        with st.container(border=True):
            st.markdown(f"**Drill-down: {dim_name} = {drill}**")
            m = st.columns(5)
            m[0].metric("Accounts", f"{len(sub):,}", f"{len(sub) / len(f):.0%} of view", delta_color="off")
            m[1].metric("Exposure", money(sub.loan_amount.sum()), f"{sub.loan_amount.sum() / f.loan_amount.sum():.0%} of view", delta_color="off")
            m[2].metric("Observed default", f"{sub.target_default.mean():.1%}", f"{(sub.target_default.mean() - f.target_default.mean()) * 100:+.1f} pp vs view", delta_color="inverse")
            m[3].metric("Avg PD", f"{sub.pd_xgb.mean():.1%}")
            m[4].metric("Share of defaulted exposure", f"{sub.def_exposure.sum() / max(f.def_exposure.sum(), 1):.0%}")
            other = [d for d in DIMENSIONS if d != dim_name]
            split = st.selectbox("Then split this segment by", other, index=other.index("Loan type") if "Loan type" in other else 0, key="dim_b")
            scol, sorder = DIMENSIONS[split]
            s2 = segment_table(sub, scol, sorder)
            a, b_ = st.columns(2)
            xs2 = [str(i) for i in s2.index]
            fig2 = go.Figure(go.Bar(x=xs2, y=s2.default_rate, marker_color=BLUE, width=0.6, text=[f"{v:.0%}" for v in s2.default_rate],
                                    textposition="outside", cliponaxis=False,
                                    customdata=np.c_[s2.accounts, s2.exposure / 1e6],
                                    hovertemplate="<b>%{x}</b><br>observed default %{y:.1%}<br>accounts %{customdata[0]:,.0f}<br>exposure $%{customdata[1]:,.1f}M<extra></extra>"))
            fig2.add_hline(y=sub.target_default.mean(), line_dash="dash", line_color=INK2)
            fig2.update_yaxes(tickformat=".0%", rangemode="tozero", title="Observed default rate")
            a.plotly_chart(style(fig2, 300, legend=False), key="drill_rate")
            fig3 = go.Figure(go.Bar(x=xs2, y=s2.exposure / 1e6, marker_color=AQUA, width=0.6, text=[f"${v / 1e6:,.1f}M" for v in s2.exposure],
                                    textposition="outside", cliponaxis=False, hovertemplate="<b>%{x}</b><br>$%{y:,.1f}M<extra></extra>"))
            fig3.update_yaxes(title="Exposure ($M)", rangemode="tozero")
            b_.plotly_chart(style(fig3, 300, legend=False), key="drill_exp")
            if (s2.accounts < 100).any():
                st.caption("Segments with fewer than 100 accounts: " + ", ".join(f"{i} (n={int(n)})" for i, n in s2.accounts[s2.accounts < 100].items())
                           + " - read with care.")

    # ---- cross-cut
    st.subheader("Cross-dimension view")
    x1, x2, x3 = st.columns(3)
    row_dim = x1.selectbox("Rows", list(DIMENSIONS), index=3, key="x_rows")
    col_opts = [d for d in DIMENSIONS if d != row_dim]
    col_dim = x2.selectbox("Columns", col_opts, index=col_opts.index("Loan type"), key="x_cols")
    cm = x3.radio("Cell value", ["Observed default rate", "Exposure ($M)", "Average predicted PD"], horizontal=True, key="x_metric")
    rcol, rord = DIMENSIONS[row_dim]
    ccol, cord = DIMENSIONS[col_dim]
    g = f.groupby([rcol, ccol], observed=True).agg(n=("account_id", "size"), dr=("target_default", "mean"), exp=("loan_amount", "sum"),
                                                   pd_=("pd_xgb", "mean"), expd=("exp_default", "sum")).reset_index()
    piv_n = g.pivot(index=rcol, columns=ccol, values="n").reindex(index=[r for r in rord if r in set(g[rcol])], columns=[c for c in cord if c in set(g[ccol])])
    key_ = {"Observed default rate": "dr", "Exposure ($M)": "exp", "Average predicted PD": "pd_"}[cm]
    piv = g.pivot(index=rcol, columns=ccol, values=key_).reindex_like(piv_n)
    if key_ == "exp":
        piv = piv / 1e6
        txt = piv.map(lambda v: "" if pd.isna(v) else f"${v:,.0f}M")
    else:
        txt = piv.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    st.plotly_chart(heat(piv.astype(float), piv_n.fillna(0), txt, "", {"dr": "Default", "exp": "$M", "pd_": "PD"}[key_]), key="xheat")
    st.caption("Cells with fewer than 30 accounts are blank; hover shows account counts.")
    top = g.assign(share=g.exp / f.loan_amount.sum()).sort_values("expd", ascending=False).head(8)
    top_t = pd.DataFrame({row_dim: top[rcol].astype(str).values, col_dim: top[ccol].astype(str).values, "Accounts": top.n.values,
                          "Exposure $M": top.exp.values / 1e6, "% of exposure": top.share.values * 100,
                          "Observed default": top.dr.values * 100, "Avg PD": top.pd_.values * 100,
                          "Exp. default $M": top.expd.values / 1e6})
    st.markdown("**Largest concentrations of expected defaulted exposure** (cells ranked by PD x amount)")
    st.dataframe(top_t, hide_index=True, column_config={
        "Accounts": st.column_config.NumberColumn(format="%d"), "Exposure $M": st.column_config.NumberColumn(format="$%.1fM"),
        "% of exposure": st.column_config.NumberColumn(format="%.1f%%"), "Observed default": st.column_config.NumberColumn(format="%.1f%%"),
        "Avg PD": st.column_config.NumberColumn(format="%.1f%%"), "Exp. default $M": st.column_config.NumberColumn(format="$%.1fM")})

    # ---- trends
    st.subheader("Trends")
    t1, t2 = st.columns(2)
    with t1:
        v = f.groupby("orig_year").agg(accounts=("account_id", "size"), exposure=("loan_amount", "sum"), defaults=("target_default", "sum"),
                                       avg_pd=("pd_xgb", "mean"))
        v["rate"] = v.defaults / v.accounts
        v["lo"], v["hi"] = wilson(v.defaults, v.accounts)
        fig = go.Figure()
        fig.add_scatter(x=v.index, y=v.rate, mode="lines+markers", name="Observed default", line=dict(color=BLUE, width=2), marker=dict(size=8),
                        error_y=dict(type="data", symmetric=False, array=v.hi - v.rate, arrayminus=v.rate - v.lo, color="#9ec5f4", thickness=1))
        fig.add_scatter(x=v.index, y=v.avg_pd, mode="lines+markers", name="Average predicted PD", line=dict(color=ORANGE, width=2, dash="dot"),
                        marker=dict(size=8, symbol="diamond"))
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title="Default rate")
        fig.update_xaxes(dtick=1, title="Origination year (vintage)")
        st.markdown("**Default rate by origination vintage**")
        st.plotly_chart(style(fig, 330), key="vintage")
        st.caption("Flat across vintages: no deterioration or improvement trend (calendar-dated trends are not possible, assumption A3).")
    with t2:
        rep = load_repayment()
        r = rep[rep.account_id.isin(f.account_id)].merge(f[["account_id", "risk_band"]], on="account_id")
        r["bad"] = (r.payment_status != "On-Time").astype(int)
        cur = r.groupby(["payment_month", "risk_band"], observed=True).bad.mean().unstack()
        fig = go.Figure()
        for b in BAND_ORDER:
            if b in cur:
                fig.add_scatter(x=cur.index, y=cur[b], mode="lines+markers", name=b, line=dict(color=BAND_COLOR[b], width=2), marker=dict(size=6))
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title="Late or missed")
        fig.update_xaxes(dtick=1, title="Month on book")
        st.markdown("**Late/missed payment rate by month, by risk band**")
        st.plotly_chart(style(fig, 330), key="repcurve")
        st.caption("Riskier bands are late more often from month 1 and stay flat - behaviour is a persistent trait, not a late slide into default (assumption A5).")

    st.download_button("Download filtered accounts (CSV)", f[["account_id", "customer_id", "loan_type", "region", "employment_type", "income_band",
                                                              "age_band", "orig_year", "loan_amount", "pd_xgb", "risk_band", "target_default"]]
                       .to_csv(index=False).encode(), "filtered_accounts.csv", "text/csv")


# =============================================================================== account
with tab_acct:
    shap_df = load_shap()
    st.subheader("Account risk view")
    s1, s2_ = st.columns([1, 2])
    typed = s1.text_input("Look up an account ID", placeholder="e.g. ACC200001", key="acct_lookup").strip().upper()
    ranked = f.sort_values("pd_xgb", ascending=False).head(2000)
    sel = None
    if typed:
        if typed in set(full.account_id):
            sel = typed
            if typed not in set(f.account_id):
                s1.info("That account is outside the current sidebar filters - shown anyway.")
        else:
            s1.error("Account ID not found.")
    opt = s2_.selectbox("...or pick from the riskiest accounts in the current view", ranked.account_id.tolist(), key="acct_pick",
                        format_func=lambda a: (lambda r: f"{a}  -  {r.loan_type}, ${r.loan_amount:,.0f}  -  PD {r.pd_xgb:.0%}")(ranked.set_index("account_id").loc[a]),
                        help="Sorted by predicted PD, riskiest first (top 2,000 in view).")
    aid = sel or opt
    a = full.set_index("account_id").loc[aid]
    cust_id = a.customer_id
    band = str(a.risk_band)

    h1, h2, h3, h4 = st.columns([1.1, 1.1, 1.1, 1.3])
    h1.metric("Predicted PD", f"{a.pd_xgb:.1%}", f"{(a.pd_xgb - PORT_RATE) * 100:+.1f} pp vs portfolio avg", delta_color="inverse")
    h2.markdown(f"<div class='small'>Risk band</div><span class='band-chip' style='background:{BAND_COLOR[band]};color:{'#0b0b0b' if band == 'Medium' else '#fff'}'>{band}</span>"
                f"<div class='small'>{BAND_RULE[band]}</div>", unsafe_allow_html=True)
    h3.metric("Riskier than", "99%+" if a.score_rank_pct >= 0.995 else f"{a.score_rank_pct:.0%}", "of all accounts", delta_color="off")
    h4.metric("Observed outcome (back-test)", "Defaulted" if a.target_default == 1 else "Performing",
              help="Known only because this is historical data - shown to let you sanity-check the score.")

    # ---- anomaly banner (same thresholds as the Anomaly watch tab)
    _fl = compute_flags(**an_settings())
    if aid in _fl.index and bool(_fl.loc[aid, "flagged"]):
        _sig = load_signals().loc[aid]
        _why = reasons(_fl.loc[aid], _sig, a.loan_type)
        st.warning(f"**Anomaly alert - severity {int(_fl.loc[aid, 'severity'])}/100.** " + " | ".join(_why) +
                   "  \nReview it in the **Anomaly watch** tab (alerts never change the score).", icon="⚠️")

    # ---- drivers
    sv = shap_df.loc[aid]
    base = sv["base_value"]
    contrib = sv.drop("base_value")
    rowvals = full.set_index("account_id").loc[aid]
    feat_val = lambda fe: (int(str(rowvals["city_tier"]).split()[-1]) if fe == "city_tier_num" else rowvals.get(fe, np.nan))  # noqa: E731
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    other_sum = contrib.drop(top.index).sum()
    labels = [f"{label(i)} = {fmt_value(i, feat_val(i))}" for i in top.index] + [f"All other features ({len(contrib) - len(top)})"]
    vals = list(top.values) + [other_sum]
    dc1, dc2 = st.columns([3, 2])
    with dc1:
        st.markdown("**Key drivers of this account's score**")
        fig = go.Figure(go.Bar(y=labels[::-1], x=vals[::-1], orientation="h", marker_color=[RED if v > 0 else BLUE for v in vals[::-1]],
                               text=[f"{v:+.2f}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>impact on log-odds: %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color="#c3c2b7")
        lim = max(abs(min(vals)), abs(max(vals))) * 1.25
        fig.update_xaxes(title="Impact on default log-odds (right = raises risk)", zeroline=False,
                         range=[min(-lim * 0.15, min(vals) * 1.25), lim])
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style(fig, 400, legend=False), key="drivers")
        st.caption(f"Starts from the portfolio baseline PD of {1 / (1 + np.exp(-base)):.1%} and adds each driver's impact to reach "
                   f"{a.pd_xgb:.1%} (SHAP, exact for the model). Red raises risk, blue lowers it.")
    with dc2:
        up = [f"**{label(i)}** ({fmt_value(i, feat_val(i))})" for i, v in top.items() if v > 0][:3]
        dn = [f"**{label(i)}** ({fmt_value(i, feat_val(i))})" for i, v in top.items() if v < 0][:3]
        st.markdown("**In plain words**")
        st.markdown(f"- **Raises risk:** {', '.join(up) if up else 'nothing material'}.")
        st.markdown(f"- **Lowers risk:** {', '.join(dn) if dn else 'nothing material'}.")
        peers = full[full.loan_type == a.loan_type]
        fig = go.Figure(go.Histogram(x=peers.pd_xgb, nbinsx=40, marker_color="#9ec5f4", hovertemplate="PD %{x:.0%}: %{y} accounts<extra></extra>"))
        fig.add_vline(x=a.pd_xgb, line_color=ORANGE, line_width=3, annotation_text="this account", annotation_font_color=ORANGE)
        fig.update_xaxes(tickformat=".0%", title=f"Predicted PD - all {a.loan_type} accounts")
        fig.update_yaxes(title="Accounts")
        pct = (peers.pd_xgb < a.pd_xgb).mean()
        st.plotly_chart(style(fig, 230, legend=False), key="peerhist")
        st.caption(f"Riskier than {'99%+' if pct >= 0.995 else format(pct, '.0%')} of {a.loan_type} accounts.")

    # ---- repayment behaviour
    rep = load_repayment()
    ra = rep[rep.account_id == aid].sort_values("payment_month")
    st.markdown("**Repayment behaviour, first 12 months**")
    stat_col = {"On-Time": "#0ca30c", "Late": "#fab219", "Missed": "#d03b3b"}
    fig = go.Figure()
    for s_, c_ in stat_col.items():
        r_ = ra[ra.payment_status == s_]
        fig.add_bar(x=r_.payment_month, y=r_.days_past_due.where(r_.days_past_due > 0, 1.5), name=s_, marker_color=c_, width=0.6,
                    customdata=np.c_[r_.emi_due, r_.emi_paid, r_.days_past_due, r_.outstanding_balance],
                    hovertemplate=f"{s_} - month %{{x}}<br>days past due: %{{customdata[2]}}<br>EMI due $%{{customdata[0]:,.0f}}, paid $%{{customdata[1]:,.0f}}"
                                  "<br>balance after $%{customdata[3]:,.0f}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick=1, title="Month on book")
    fig.update_yaxes(title="Days past due")
    st.plotly_chart(style(fig, 260), key="repay")
    st.caption("On-time months are shown as a short green stub. Hover for EMI due/paid and balance.")

    # ---- profile
    p1, p2, p3 = st.columns(3)
    cs = a
    def table(rows):
        return pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field")
    p1.markdown("**Customer**")
    p1.table(table([("Customer ID", cust_id), ("Age", f"{cs.age}"), ("Employment", cs.employment_type), ("Years employed", fmt_value("x", cs.years_employed)),
                    ("Annual income", fmt_value("annual_income", cs.annual_income)), ("Region / tier", f"{cs.region} / {cs.city_tier}"),
                    ("Housing", cs.housing_type), ("Dependents", f"{cs.num_dependents}")]))
    p2.markdown("**Loan**")
    p2.table(table([("Type / purpose", f"{cs.loan_type} / {cs.loan_purpose}"), ("Amount", f"${cs.loan_amount:,.0f}"),
                    ("Rate / tenure", f"{cs.interest_rate_pct:.1f}% / {cs.tenure_months}m"), ("Monthly EMI", f"${cs.rp_emi:,.0f}"),
                    ("Loan / income", fmt_value("loan_to_income", cs.loan_to_income)), ("Originated", str(pd.Timestamp(cs.origination_date).date())),
                    ("Channel", cs.disbursal_channel), ("Secured", "Yes" if cs.collateral_flag else "No")]))
    p3.markdown("**Bureau and behaviour**")
    p3.table(table([("Bureau file", "Yes" if cs.has_bureau else "No (thin file)"), ("Credit lines (active)", f"{int(cs.bur_n_lines)} ({int(cs.bur_n_active)})"),
                    ("Written-off lines", f"{int(cs.bur_n_writeoff)}"), ("Worst bureau DPD", fmt_value("x", cs.bur_max_dpd)),
                    ("Late / missed (m1-12)", f"{int(cs.rp_n_late)} / {int(cs.rp_n_missed)}"), ("On-time share", fmt_value("rp_pct_ontime", cs.rp_pct_ontime)),
                    ("Customer's other accounts", f"{int(cs.cust_n_accounts) - 1}"), ("Customer's total missed", f"{int(cs.cust_sum_missed)}")]))

    with st.expander("Customer's other accounts, bureau lines and collections history"):
        oth = full[(full.customer_id == cust_id)][["account_id", "loan_type", "loan_amount", "pd_xgb", "risk_band", "target_default"]].copy()
        oth["pd_xgb"] = oth.pd_xgb * 100
        oth["risk_band"] = oth.risk_band.astype(str)
        oth["target_default"] = oth.target_default.map({1: "Defaulted", 0: "Performing"})
        st.markdown("**Accounts of this customer**")
        st.dataframe(oth.rename(columns={"account_id": "Account", "loan_type": "Loan type", "loan_amount": "Amount", "pd_xgb": "PD",
                                         "risk_band": "Band", "target_default": "Outcome (back-test)"}), hide_index=True,
                     column_config={"Amount": st.column_config.NumberColumn(format="$%.0f"), "PD": st.column_config.NumberColumn(format="%.1f%%")})
        bur = load_bureau()
        bl = bur[bur.customer_id == cust_id].drop(columns=["customer_id"])
        st.markdown("**External credit lines (bureau snapshot)**")
        if len(bl):
            st.dataframe(bl, hide_index=True)
        else:
            st.caption("No bureau record for this customer.")
        col = load_collections()
        ce = col[col.account_id == aid].copy()
        ce["event_date"] = pd.to_datetime(ce.event_date).dt.date.astype(str)
        ce["Dated before origination?"] = ce.before_orig.map({1: "Yes (data issue)", 0: "No"})
        ce = ce[["collection_event_id", "event_date", "action_type", "outcome", "amount_recovered", "Dated before origination?"]]
        st.markdown("**Collections events on this account**")
        if len(ce):
            st.dataframe(ce, hide_index=True)
        else:
            st.caption("No collections events.")
        st.caption("Collections are shown for context only; they follow delinquency and are not used in the score (assumptions A2, A11).")


# =============================================================================== anomaly watch (AI feature)
with tab_anom:
    st.subheader("Anomaly watch")
    st.caption("AI-assisted early warning: flags accounts whose repayment or bureau behaviour is unusual against **their own history** or "
               "**their peer segment** (unsupervised - the default flag is not used to detect). An alert never changes a score or triggers "
               "action: an analyst reviews it and records a decision.")
    with st.expander("Alert sensitivity - what triggers an alert", expanded=False):
        t1, t2, t3 = st.columns(3)
        t1.markdown("**Own-history spike**")
        t1.slider("DPD in months 10-12 at least (days)", 30, 89, DEFAULTS["spike_dpd"], key="an_spike",
                  help="'Missed' months are 30-89 days; 'Late' months are 1-28.")
        t1.slider("...while own months 1-9 never exceeded (days)", 0, 28, DEFAULTS["base_max"], key="an_base")
        t2.markdown("**Peer-segment outlier**")
        t2.slider("Flag the top % of each loan type", 1.0, 10.0, 100 * (1 - DEFAULTS["if_cut"]), 0.5, key="an_peer",
                  help="Isolation Forest fitted within each loan type; higher % = more alerts, lower precision.")
        t3.markdown("**Detectors on**")
        t3.checkbox("Own-history spike", True, key="an_use_spike")
        t3.checkbox("Peer-segment outlier", True, key="an_use_peer")
        t3.checkbox("Source divergence (bureau vs repayment)", True, key="an_use_div")
        st.caption("Defaults: spike >= 60 days after a clean-ish baseline; top 3% of each loan type; both divergence checks on. "
                   "Full definitions and evaluation: `docs/anomaly_feature.md`.")

    settings = an_settings()
    sig = load_signals()
    fl = compute_flags(**settings)
    view = f.set_index("account_id")[["customer_id", "loan_type", "loan_amount", "pd_xgb", "risk_band", "target_default"]].join(fl)
    view["priority"] = view.severity * view.loan_amount / 1e5
    al = view[view.flagged].copy()
    rlog = read_review_log()
    last_dec = rlog.sort_values("timestamp").groupby("account_id").decision.last() if len(rlog) else pd.Series(dtype=str)
    al["status"] = al.index.to_series().map(last_dec).astype(object).fillna("Open")
    rest = view[~view.flagged]

    k = st.columns(5)
    k[0].metric("Alerts", f"{len(al):,}", f"{len(al) / len(view):.1%} of accounts in view", delta_color="off")
    k[1].metric("Exposure flagged", money(al.loan_amount.sum()), f"{al.loan_amount.sum() / max(view.loan_amount.sum(), 1):.0%} of view", delta_color="off")
    dr_al = al.target_default.mean() if len(al) else np.nan
    k[2].metric("Default rate: alerts", "n/a" if np.isnan(dr_al) else f"{dr_al:.1%}",
                f"vs {rest.target_default.mean():.1%} not flagged" if len(rest) else None, delta_color="off",
                help="Back-test proxy only: default is not used to detect anomalies.")
    lm = al.risk_band.astype(str).isin(["Low", "Medium"]).mean() if len(al) else np.nan
    k[3].metric("Alerts the PD model rates Low/Medium", "n/a" if np.isnan(lm) else f"{lm:.0%}",
                help="Where the anomaly layer adds information beyond the score.")
    k[4].metric("Open / reviewed", f"{int((al.status == 'Open').sum()):,} / {int((al.status != 'Open').sum()):,}")

    if al.empty:
        st.info("No alerts under the current thresholds and filters. Loosen the thresholds above or widen the sidebar filters.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Alerts by detector**")
            cnt = {"Own-history spike": int(al.t_spike.sum()), "Peer outlier": int(al.t_peer.sum()),
                   "Clean bureau, missed here": int(al.t_local.sum()), "Bureau stress, clean here": int(al.t_hidden.sum())}
            cols = [BLUE, ORANGE, AQUA, "#eda100"]
            fig = go.Figure(go.Bar(y=list(cnt)[::-1], x=list(cnt.values())[::-1], orientation="h", marker_color=cols[::-1], width=0.6,
                                   text=list(cnt.values())[::-1], textposition="outside", cliponaxis=False))
            fig.update_xaxes(title="Accounts (may trigger several)", range=[0, max(max(cnt.values()), 1) * 1.25])
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(style(fig, 280, legend=False), key="an_det")
        with c2:
            st.markdown("**Alert rate by loan type**")
            lt = view.groupby("loan_type").flagged.mean().reindex(DIMENSIONS["Loan type"][1]).dropna()
            fig = go.Figure(go.Bar(x=list(lt.index), y=lt.values, marker_color=BLUE, width=0.6, text=[f"{v:.1%}" for v in lt.values],
                                   textposition="outside", cliponaxis=False))
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            st.plotly_chart(style(fig, 280, legend=False), key="an_lt")
        with c3:
            st.markdown("**Peer detector: volume vs hit-rate**")
            sub = sig.loc[view.index]
            yv = view.target_default.values
            pts = []
            for pct in np.arange(1.0, 10.01, 0.5):
                m = (sub.if_pct >= 1 - pct / 100).values
                pts.append((m.sum(), yv[m].mean() if m.any() else np.nan, pct))
            pts = pd.DataFrame(pts, columns=["alerts", "default_rate", "pct"])
            fig = go.Figure(go.Scatter(x=pts.alerts, y=pts.default_rate, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=5),
                                       customdata=pts.pct, hovertemplate="top %{customdata:.1f}% -> %{x} alerts, default %{y:.0%}<extra></extra>"))
            cur = pts.iloc[(pts.pct - 100 * (1 - settings["if_cut"])).abs().argmin()]
            fig.add_scatter(x=[cur.alerts], y=[cur.default_rate], mode="markers", marker=dict(size=13, color=ORANGE, line=dict(color="#fff", width=2)),
                            hovertemplate="current setting<extra></extra>")
            fig.add_hline(y=view.target_default.mean(), line_dash="dash", line_color=INK2, annotation_text="view average")
            fig.update_yaxes(tickformat=".0%", title="Default rate among alerts")
            fig.update_xaxes(title="Alerts (more alerts -> lower hit-rate)")
            st.plotly_chart(style(fig, 280, legend=False), key="an_curve")

        # ---- review queue
        st.markdown("### Review queue")
        q1, q2 = st.columns([1, 3])
        show = q1.radio("Show", ["Open only", "All"], horizontal=True, key="an_show")
        q = al if show == "All" else al[al.status == "Open"]
        q = q.sort_values("priority", ascending=False)
        q2.caption("Ranked by severity x exposure so the highest-impact alerts come first.")
        if q.empty:
            st.success("Queue is empty - every alert in view has been reviewed.")
        else:
            def trig(r):
                return ", ".join(n for n, c in [("spike", r.t_spike), ("peer", r.t_peer), ("divergence", r.t_local or r.t_hidden)] if c)
            qt = pd.DataFrame({"Account": q.index, "Loan type": q.loan_type.values, "Exposure": q.loan_amount.values,
                               "PD": q.pd_xgb.values * 100, "Band": q.risk_band.astype(str).values, "Severity": q.severity.values,
                               "Triggers": [trig(r) for r in q.itertuples()], "Status": q.status.values})
            st.dataframe(qt.head(40), hide_index=True, height=290, column_config={
                "Exposure": st.column_config.NumberColumn(format="$%.0f"), "PD": st.column_config.NumberColumn(format="%.1f%%"),
                "Severity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})
            st.caption(f"Showing the top 40 of {len(q):,} alerts.")
            pick = st.selectbox("Review an alert", q.head(100).index.tolist(), key="an_pick",
                                format_func=lambda a_: f"{a_}  -  severity {int(q.loc[a_, 'severity'])}  -  {q.loc[a_, 'loan_type']}, ${q.loc[a_, 'loan_amount']:,.0f}")
            r = view.loc[pick]
            s = sig.loc[pick]
            rs = reasons(r, s, r.loan_type)
            with st.container(border=True):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Severity", f"{int(r.severity)}/100")
                d2.metric("Predicted PD", f"{r.pd_xgb:.1%}", str(r.risk_band), delta_color="off")
                d3.metric("Exposure", f"${r.loan_amount:,.0f}")
                d4.metric("Observed outcome (back-test)", "Defaulted" if r.target_default == 1 else "Performing")
                e1, e2 = st.columns([5, 5])
                with e1:
                    st.markdown("**Why it was flagged**")
                    for line in rs:
                        st.markdown(f"- {line}")
                    if r.t_peer:
                        feats = json.loads(s.peer_top)
                        if feats:
                            pt = pd.DataFrame([(FEATURE_TEXT[f_], v, m_, z) for f_, v, m_, z in feats],
                                              columns=["Feature", "This account", "Peer median", "Robust z"])
                            st.dataframe(pt, hide_index=True, column_config={
                                "This account": st.column_config.NumberColumn(format="%.2f"), "Peer median": st.column_config.NumberColumn(format="%.2f"),
                                "Robust z": st.column_config.NumberColumn(format="%.1f")})
                    st.markdown("**Bureau vs repayment with us**")
                    ar = full.set_index("account_id").loc[pick]
                    st.markdown(f"- Bureau: {int(ar.bur_n_lines)} lines, {int(ar.bur_n_dpd30)} at 30+ DPD, {int(ar.bur_n_writeoff)} written off, worst DPD {fmt_value('x', ar.bur_max_dpd)}")
                    st.markdown(f"- With us: {int(ar.rp_n_late)} late, {int(ar.rp_n_missed)} missed, worst DPD {int(ar.rp_max_dpd)}")
                with e2:
                    dp = [int(s[f"dpd_m{m}"]) for m in range(1, 13)]
                    colr = [ORANGE if (r.t_spike and m >= 10 and v >= settings["spike_dpd"]) else "#9ec5f4" for m, v in zip(range(1, 13), dp)]
                    fig = go.Figure(go.Bar(x=list(range(1, 13)), y=dp, marker_color=colr, width=0.6, text=dp, textposition="outside",
                                           hovertemplate="month %{x}: %{y} days past due<extra></extra>"))
                    fig.add_hline(y=int(s.base_worst_dpd), line_dash="dot", line_color=INK2, annotation_text="own worst, months 1-9", annotation_position="top left")
                    fig.add_hline(y=settings["spike_dpd"], line_dash="dash", line_color=RED, annotation_text="spike threshold", annotation_position="top left")
                    fig.add_vrect(x0=9.5, x1=12.5, fillcolor="#f4f4f1", opacity=0.6, line_width=0, layer="below")
                    fig.update_xaxes(dtick=1, title="Month on book (shaded = latest 3 months)")
                    fig.update_yaxes(title="Days past due", range=[0, 100])
                    st.markdown("**Account's own repayment history**")
                    st.plotly_chart(style(fig, 300, legend=False), key="an_hist")

                # ---- human-in-the-loop decision
                st.markdown("**Analyst decision** (recorded to the audit log - nothing is changed automatically)")
                st.session_state["rv_ctx"] = dict(account_id=pick, severity=int(r.severity), settings=settings,
                                                  triggers="+".join(n for n, c in [("spike", r.t_spike), ("peer", r.t_peer), ("divergence", r.t_local or r.t_hidden)] if c))
                with st.form("review_form", clear_on_submit=True):
                    f1, f2 = st.columns(2)
                    f1.radio("Decision", DECISIONS, key="rv_dec")
                    f2.selectbox("Reason code", REASON_CODES, key="rv_reason")
                    f2.text_input("Note (optional)", key="rv_note")
                    f2.text_input("Analyst", value=st.session_state.get("rv_analyst", "analyst"), key="rv_analyst")
                    st.form_submit_button("Save decision", type="primary", on_click=save_review)
                if st.session_state.get("rv_saved"):
                    st.success(f"Decision saved for {st.session_state.pop('rv_saved')}. The alert leaves the open queue.")

    # ---- feedback loop
    st.markdown("### Feedback loop - what analysts are telling us")
    rlog = read_review_log()
    if rlog.empty:
        st.caption("No decisions recorded yet. Each saved decision updates the false-positive rate per detector below; "
                   "if a detector's dismiss rate stays high the threshold should be tightened.")
    else:
        rl = rlog.copy()
        rl["confirmed"] = rl.decision.isin(["Confirm concern", "Escalate to collections early-warning"])
        rl["dismissed"] = rl.decision.eq("Dismiss - false positive")
        rows = []
        for det in ["spike", "peer", "divergence"]:
            m = rl.triggers.fillna("").str.contains(det)
            if m.any():
                rows.append(dict(Detector=det, Reviewed=int(m.sum()), Confirmed=int(rl[m].confirmed.sum()), Dismissed=int(rl[m].dismissed.sum()),
                                 **{"Dismiss rate (false-positive rate)": rl[m].dismissed.mean() * 100}))
        fb = pd.DataFrame(rows)
        st.dataframe(fb, hide_index=True, column_config={"Dismiss rate (false-positive rate)": st.column_config.NumberColumn(format="%.0f%%")})
        for r_ in rows:
            if r_["Reviewed"] >= 10 and r_["Dismiss rate (false-positive rate)"] > 70:
                st.warning(f"The **{r_['Detector']}** detector is being dismissed {r_['Dismiss rate (false-positive rate)']:.0f}% of the time - consider tightening its threshold.")
        st.dataframe(rl.sort_values("timestamp", ascending=False).head(10)[["timestamp", "account_id", "decision", "reason_code", "note", "analyst"]], hide_index=True)
        st.download_button("Download full review log (CSV)", rlog.to_csv(index=False).encode(), "anomaly_review_log.csv", "text/csv")

    with st.expander("How this AI feature works - trigger, false positives / negatives, human-in-the-loop"):
        ex = {}
        exf = ROOT / "outputs" / "anomaly" / "examples.json"
        if exf.exists():
            ex = json.loads(exf.read_text())
        st.markdown(f"""
**Trigger.** An alert fires when (1) the latest 3 months show DPD >= {settings['spike_dpd']} days against an own baseline never above {settings['base_max']} days,
(2) the account is in the top {100 * (1 - settings['if_cut']):.1f}% of its loan type on an Isolation Forest over 15 repayment, bureau and affordability features, or
(3) bureau and repayment disagree. Computed from the data already in the portfolio - no default label is used.

**False positive** - the alert fires but nothing is wrong: {('e.g. `' + ex['false_positive_spike']['account'] + '` paid on time for 11 months, then one 89-day miss - a one-off (bounced debit, holiday) the analyst dismisses.') if 'false_positive_spike' in ex else 'a one-off missed payment on an otherwise clean account.'}
**False negative** - something is wrong but no alert fires: {('e.g. `' + ex['false_negative']['account'] + '`, a $' + format(ex['false_negative']['loan_amount'], ',.0f') + ' mortgage that defaulted with a clean record and no bureau stress; the data held no warning.') if 'false_negative' in ex else 'a default with a clean-looking record.'}

**Human-in-the-loop.** Alerts only create review tasks. The analyst confirms, dismisses, requests information or escalates, with a reason code and note; every decision is logged with the thresholds in force.
Dismiss rates per detector drive threshold changes, approved by the model owner. See `docs/anomaly_feature.md` for evaluation numbers (including the synthetic spike-injection test).
""")


# =============================================================================== method
with tab_method:
    st.subheader("What the score is - and is not")
    st.markdown("""
- **Target:** `target_default` per loan account (16.1% base rate). **Model:** XGBoost on 66 features: application fields, bureau aggregates,
  months 1-12 repayment aggregates and the customer's other-account behaviour. Logistic regression is the benchmark.
- **Scoring point:** the default flag is inferred to be an outcome *after* the 12-month repayment window (assumption A5), so this PD is a
  **12-months-on-book behavioural score**. At application time only the application+bureau features exist (test AUC ~0.66 vs ~0.68).
- **Scores shown are out-of-fold:** no account is scored by a model that saw it in training.
- **Excluded on purpose:** gender and marital status (protected), collections (leak the outcome), inconsistent dates.
""")
    mt = pd.read_csv(Path(__file__).resolve().parents[1] / "outputs" / "model" / "metrics_test.csv", index_col=0)[["auc", "gini", "ks", "brier"]]
    st.markdown("**Held-out test performance**")
    st.dataframe(mt.rename(columns=str.upper), column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC", "GINI", "KS", "BRIER"]})
    cal = pd.read_csv(Path(__file__).resolve().parents[1] / "outputs" / "model" / "deciles_test_xgb.csv")
    fig = go.Figure()
    fig.add_scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", line=dict(color=MUTED, dash="dot"), name="perfect calibration")
    fig.add_scatter(x=cal.mean_pred, y=cal.actual, mode="lines+markers", name="score deciles (test)", line=dict(color=BLUE, width=2))
    fig.update_xaxes(tickformat=".0%", title="Mean predicted PD")
    fig.update_yaxes(tickformat=".0%", title="Observed default rate")
    st.plotly_chart(style(fig, 320), key="calib")
    st.markdown("""
**Data caveats:** synthetic and noisy (AUC ceiling ~0.68); exposure = original loan amount; no recovery data so no loss-given-default;
repayment months carry no calendar dates, so trends are by origination vintage. Full details: `docs/data_quality_notes.md`,
`docs/model_card.md`, `docs/assumptions_log.md`.
""")
