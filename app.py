"""
AgriTrust AI – Agricultural Credit Risk Intelligence Platform
=============================================================
Main Streamlit application.

Architecture Note:
  This single-file Streamlit app is the presentation layer only.
  All business logic lives in the supporting modules:
    auth.py           → authentication & RBAC
    database.py       → SQLite persistence
    api_simulation.py → ML inference contract (swap for FastAPI in prod)

Cloud Deployment:
  • Containerise with Docker (streamlit run app.py --server.port 8501)
  • Deploy on GCP Cloud Run / AWS ECS Fargate (auto-scaling)
  • Static assets served via CDN
  • Secrets (DB URL, API keys) injected via env vars / Secret Manager

# TODO: Replace simulation with real FastAPI microservice deployment
"""

import io
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth
import database
from api_simulation import PredictRequest, predict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AgriTrust AI",
    page_icon  = "🌾",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: #050e1a;
  color: #e8f0fe;
}

/* ── Header ── */
.at-header {
  background: linear-gradient(135deg, #061a2e 0%, #072a1e 60%, #061a2e 100%);
  border-bottom: 1px solid rgba(74,222,128,0.15);
  padding: 28px 40px 24px;
  margin-bottom: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.at-logo { font-family:'DM Serif Display',serif; font-size:1.7rem; color:#4ade80; }
.at-sub  { font-size:0.78rem; color:rgba(255,255,255,0.45); letter-spacing:1.2px; text-transform:uppercase; margin-top:4px; }
.at-user-pill {
  background: rgba(74,222,128,0.1);
  border: 1px solid rgba(74,222,128,0.25);
  border-radius: 50px;
  padding: 6px 16px;
  font-size: 0.82rem;
  color: #4ade80;
}

/* ── Cards ── */
.card {
  background: linear-gradient(145deg, #0b1f35 0%, #0a2b20 100%);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 20px;
}
.card-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.05rem;
  color: #a3e4b5;
  margin-bottom: 18px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── KPI Metrics Override ── */
[data-testid="metric-container"] {
  background: linear-gradient(145deg, #0e2640, #0c3325);
  border: 1px solid rgba(74,222,128,0.15);
  border-radius: 14px;
  padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.55) !important; font-size:0.78rem !important; }
[data-testid="stMetricValue"] { color: #4ade80 !important; font-size: 2rem !important; font-weight:600 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03);
  border-radius: 12px 12px 0 0;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  gap: 4px;
  padding: 0 12px;
}
[data-baseweb="tab"] {
  color: rgba(255,255,255,0.45) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 12px 20px !important;
}
[aria-selected="true"] {
  color: #4ade80 !important;
  background: rgba(74,222,128,0.08) !important;
  border-bottom: 2px solid #4ade80 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #16a34a, #15803d);
  color: white;
  border: none;
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: 0.3px;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #22c55e, #16a34a);
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(74,222,128,0.3);
}

/* ── Risk Badges ── */
.risk-low      { background:#14532d; color:#4ade80; border:1px solid #4ade80; border-radius:8px; padding:6px 18px; font-weight:600; display:inline-block; }
.risk-moderate { background:#713f12; color:#fbbf24; border:1px solid #fbbf24; border-radius:8px; padding:6px 18px; font-weight:600; display:inline-block; }
.risk-high     { background:#7c2d12; color:#fb923c; border:1px solid #fb923c; border-radius:8px; padding:6px 18px; font-weight:600; display:inline-block; }
.risk-very     { background:#450a0a; color:#f87171; border:1px solid #f87171; border-radius:8px; padding:6px 18px; font-weight:600; display:inline-block; }

/* ── Inputs ── */
[data-baseweb="input"] input, [data-baseweb="select"] { background: #0b1e33 !important; border-color: rgba(255,255,255,0.1) !important; color:#e8f0fe !important; }
.stSlider [data-baseweb="slider"] { margin-top: 4px; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Footer ── */
.at-footer {
  text-align: center;
  color: rgba(255,255,255,0.2);
  font-size: 0.75rem;
  padding: 28px 0 12px;
  border-top: 1px solid rgba(255,255,255,0.05);
  margin-top: 40px;
}
.at-footer span { color: #4ade80; }
</style>
""", unsafe_allow_html=True)


# ── Bootstrap ──────────────────────────────────────────────────────────────────
database.init_db()
os.makedirs("reports", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ── Auth Gate ─────────────────────────────────────────────────────────────────
user = auth.require_auth()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_badge(category: str) -> str:
    cls = {
        "Low Risk":      "risk-low",
        "Moderate Risk": "risk-moderate",
        "High Risk":     "risk-high",
        "Very High Risk":"risk-very",
    }.get(category, "risk-moderate")
    return f'<span class="{cls}">{category}</span>'


def _gauge(score: float) -> go.Figure:
    """Plotly gauge chart for Trust Score."""
    color = (
        "#4ade80" if score >= 70 else
        "#fbbf24" if score >= 50 else
        "#fb923c" if score >= 30 else
        "#f87171"
    )
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number = {"font": {"size": 52, "color": color, "family": "DM Serif Display"}},
        title = {"text": "Trust Score", "font": {"size": 14, "color": "rgba(255,255,255,0.5)"}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 11}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(248,113,113,0.12)"},
                {"range": [30, 50], "color": "rgba(251,146,60,0.12)"},
                {"range": [50, 70], "color": "rgba(251,191,36,0.12)"},
                {"range": [70,100], "color": "rgba(74,222,128,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = {"t": 40, "b": 10, "l": 20, "r": 20},
        height        = 260,
        font          = {"color": "rgba(255,255,255,0.7)"},
    )
    return fig


def _feature_importance_chart() -> go.Figure | None:
    """Bar chart of XGBoost feature importances."""
    import joblib
    try:
        model = joblib.load(os.path.join("model", "credit_model.pkl"))
    except FileNotFoundError:
        return None

    feats = ["farm_size","soil_score","rainfall","previous_loans","yield_amount",]#crop_diversity
    imps  = model.feature_importances_.tolist()
    pairs = sorted(zip(feats, imps), key=lambda x: x[1])
    labels, values = zip(*pairs)

    fig = go.Figure(go.Bar(
        x           = list(values),
        y           = list(labels),
        orientation = "h",
        marker_color= ["#4ade80" if v == max(values) else "#2d6a4f" for v in values],
        marker_line_width = 0,
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = {"t": 10, "b": 10, "l": 10, "r": 10},
        height        = 240,
        xaxis = {"showgrid": False, "color": "rgba(255,255,255,0.3)", "tickfont": {"size": 10}},
        yaxis = {"showgrid": False, "color": "rgba(255,255,255,0.5)", "tickfont": {"size": 11}},
        bargap = 0.35,
    )
    return fig


def _explain(farm_size, soil_score, rainfall, previous_loans, yield_amount, trust_score) -> list[str]:
    """Generate dynamic, human-readable risk explanations."""
    notes = []
    if rainfall < 400:
        notes.append("⚠️ **Very low rainfall** (< 400 mm) severely limits crop yield potential and elevates seasonal default risk.")
    elif rainfall < 700:
        notes.append("🌧️ **Below-average rainfall** may constrain yields in drought-sensitive seasons.")
    else:
        notes.append("✅ **Adequate rainfall** supports stable agricultural output.")

    if yield_amount < 2.0:
        notes.append("⚠️ **Low historical yield** (< 2 t) suggests productivity challenges or land degradation.")
    elif yield_amount >= 5.0:
        notes.append("✅ **Strong yield performance** indicates productive land use and farming capability.")

    if soil_score < 40:
        notes.append("⚠️ **Poor soil quality** (score < 40) is a significant production risk factor.")
    elif soil_score >= 75:
        notes.append("✅ **High soil fertility score** supports consistent crop cycles.")

    if previous_loans >= 5:
        notes.append("⚠️ **High prior loan count** (≥ 5) may indicate financial overextension.")
    elif previous_loans == 0:
        notes.append("ℹ️ **No prior loan history** — limited credit data; assess collateral carefully.")
    else:
        notes.append("✅ **Manageable prior loan exposure** within acceptable range.")

    if farm_size < 2:
        notes.append("ℹ️ **Smallholder farm** (< 2 acres) — verify crop insurance coverage and cooperative membership.")
    elif farm_size >= 20:
        notes.append("✅ **Large farm acreage** diversifies risk and supports higher loan ceilings.")

    if trust_score >= 70:
        notes.append("🟢 **Overall profile is strong.** Recommend approval subject to standard documentation.")
    elif trust_score >= 50:
        notes.append("🟡 **Borderline creditworthiness.** Consider conditional approval with reduced loan amount.")
    else:
        notes.append("🔴 **Risk profile is elevated.** Recommend rejection or collateral-backed loan with enhanced monitoring.")

    return notes


def _generate_pdf(name, farm_size, soil_score, rainfall, previous_loans,
                  yield_amount, trust_score, risk_category, explanations) -> bytes:
    """Build a PDF lender report and return raw bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2.2*cm, rightMargin=2.2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Heading1"],
                                 fontSize=20, textColor=colors.HexColor("#14532d"),
                                 spaceAfter=4)
    sub_style   = ParagraphStyle("Sub", parent=styles["Normal"],
                                 fontSize=9, textColor=colors.HexColor("#6b7280"),
                                 spaceAfter=16)
    body_style  = ParagraphStyle("Body2", parent=styles["Normal"],
                                 fontSize=10, leading=16,
                                 textColor=colors.HexColor("#1f2937"))
    head_style  = ParagraphStyle("Head2", parent=styles["Heading2"],
                                 fontSize=12, textColor=colors.HexColor("#166534"),
                                 spaceBefore=12, spaceAfter=6)

    # TODO: Add bank branding and official watermark here
    story = [
        Paragraph("AgriTrust AI", title_style),
        Paragraph("Agricultural Loan Credit Risk Report — CONFIDENTIAL", sub_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d1fae5")),
        Spacer(1, 14),
        Paragraph("Applicant Details", head_style),
    ]

    applicant_data = [
        ["Applicant Name", name or "—"],
        ["Report Generated", datetime.now().strftime("%d %b %Y, %H:%M")],
        ["Assessed By",      "AgriTrust AI v1.3"],
    ]
    tbl = Table(applicant_data, colWidths=[5*cm, 10*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0fdf4")),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#166534")),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING",(0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    story += [tbl, Spacer(1, 14), Paragraph("Agricultural Data", head_style)]

    farm_data = [
        ["Parameter",        "Value",                         "Benchmark"],
        ["Farm Size",        f"{farm_size} acres",            "> 5 acres preferred"],
        ["Soil Score",       f"{soil_score} / 100",           "> 60 recommended"],
        ["Annual Rainfall",  f"{rainfall} mm",                "> 700 mm optimal"],
        ["Previous Loans",   str(previous_loans),             "< 3 preferred"],
        ["Yield Amount",     f"{yield_amount} tonnes",        "> 3 t preferred"],
    ]
    tbl2 = Table(farm_data, colWidths=[5*cm, 5*cm, 5*cm])
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#166534")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    story += [tbl2, Spacer(1, 18), Paragraph("Credit Assessment Result", head_style)]

    score_color = (
        colors.HexColor("#14532d") if trust_score >= 70 else
        colors.HexColor("#78350f") if trust_score >= 50 else
        colors.HexColor("#7c2d12")
    )
    score_data = [
        ["Trust Score", f"{trust_score} / 100"],
        ["Risk Category", risk_category],
        ["Recommendation", "Approve" if trust_score >= 50 else "Review / Reject"],
    ]
    tbl3 = Table(score_data, colWidths=[5*cm, 10*cm])
    tbl3.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#f0fdf4")),
        ("TEXTCOLOR",   (0,0), (0,-1), colors.HexColor("#166534")),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (1,0), (1,0), score_color),
        ("FONTNAME",    (1,0), (1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story += [tbl3, Spacer(1, 18), Paragraph("Risk Analysis & Explainability", head_style)]

    for exp in explanations:
        clean = exp.replace("**", "").replace("⚠️","[!]").replace("✅","[OK]") \
                   .replace("🟢","[✓]").replace("🟡","[~]").replace("🔴","[✗]") \
                   .replace("ℹ️","[i]").replace("🌧️","")
        story.append(Paragraph(f"• {clean}", body_style))
        story.append(Spacer(1, 4))

    story += [
        Spacer(1, 24),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d1fae5")),
        Spacer(1, 8),
        Paragraph(
            "© 2026 AgriTrust AI — This report is generated by an AI system and "
            "must be reviewed by a qualified credit officer before final decision.",
            ParagraphStyle("Disc", parent=styles["Normal"],
                           fontSize=7.5, textColor=colors.HexColor("#9ca3af"), leading=11)
        ),
    ]

    doc.build(story)
    return buf.getvalue()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="at-header">
  <div>
    <div class="at-logo">🌾 AgriTrust AI</div>
    <div class="at-sub">Rural Credit Intelligence Platform &nbsp;|&nbsp; Agricultural Loan Risk Assessment</div>
  </div>
  <div class="at-user-pill">
    {user['display']} &nbsp;·&nbsp; {user['role']} &nbsp;·&nbsp; {user['branch']}
  </div>
</div>
""", unsafe_allow_html=True)

# Logout in a tiny sidebar
with st.sidebar:
    st.markdown(f"### 👤 {user['display']}")
    st.caption(f"Role: **{user['role']}**")
    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True):
        auth.logout()

# ── Tabs ───────────────────────────────────────────────────────────────────────
if user["role"] == "Admin":
    tabs = st.tabs(["🎯 Credit Scoring", "🛡️ Admin Panel", "📋 Application History"])
    tab_score, tab_admin, tab_hist = tabs
else:
    tab_score = st.tabs(["🎯 Credit Scoring"])[0]
    tab_admin = tab_hist = None


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CREDIT SCORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">📝 Loan Application Details</div>', unsafe_allow_html=True)

        applicant_name = st.text_input("Applicant Name", placeholder="e.g. Ramesh Kumar Patel")

        c1, c2 = st.columns(2)
        with c1:
            farm_size = st.number_input("Farm Size (acres)", min_value=0.1, max_value=200.0,
                                        value=5.0, step=0.5, format="%.1f")
            rainfall  = st.number_input("Annual Rainfall (mm)", min_value=50.0, max_value=3000.0,
                                        value=750.0, step=10.0)
            previous_loans = st.number_input("Previous Loans (count)", min_value=0,
                                              max_value=20, value=1, step=1)
        with c2:
            soil_score   = st.slider("Soil Fertility Score (0–100)", 0, 100, 65)
            yield_amount = st.number_input("Historical Yield (tonnes)", min_value=0.1,
                                           max_value=50.0, value=3.5, step=0.1, format="%.1f")
           # crop_diversity = st.slider("Crop Diversity (types)", 1, 5, 2)

        st.markdown("</div>", unsafe_allow_html=True)

        evaluate = st.button("🔍 Evaluate Loan Application", use_container_width=True, type="primary")

    with right:
        if evaluate:
            req = PredictRequest(
            farm_size=farm_size,
            soil_score=soil_score,
            rainfall=rainfall,
            previous_loans=previous_loans,
            yield_amount=yield_amount,
            )
            
            result = predict(req)

            # Persist to DB
            database.insert_application(
                applicant_name = applicant_name or "Anonymous",
                farm_size      = farm_size,
                soil_score     = soil_score,
                rainfall       = rainfall,
                previous_loans = previous_loans,
                yield_amount   = yield_amount,
                trust_score    = result.trust_score,
                risk_category  = result.risk_category,
                officer_id     = st.session_state.get("username", "system"),
            )

            # KPI Row
            k1, k2, k3 = st.columns(3)
            k1.metric("Trust Score",  f"{result.trust_score}")
            k2.metric("Risk Level",   result.risk_category)
            k3.metric("Latency",      f"{result.latency_ms} ms")

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(_gauge(result.trust_score), use_container_width=True, config={"displayModeBar": False})

            # Risk Badge
            st.markdown(f"<div style='text-align:center;margin-bottom:12px;'>{_risk_badge(result.risk_category)}</div>",
                        unsafe_allow_html=True)

            st.caption(f"Model: {result.model_version} &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y %H:%M')}")

            # Store in session for export
            st.session_state["last_result"] = {
                "name": applicant_name, "farm_size": farm_size, "soil_score": soil_score,
                "rainfall": rainfall, "previous_loans": previous_loans,
                "yield_amount": yield_amount, "trust_score": result.trust_score,
                "risk_category": result.risk_category,
            }
        else:
            st.markdown('<div class="card" style="text-align:center;padding:60px 20px;">'
                        '<div style="font-size:3rem;margin-bottom:16px;">🌾</div>'
                        '<div style="color:rgba(255,255,255,0.4);font-size:0.9rem;">'
                        'Enter applicant data and click<br><strong style="color:#4ade80;">Evaluate Loan Application</strong>'
                        '</div></div>', unsafe_allow_html=True)

    # ── Explainability ─────────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">'
                    '<div class="card-title">🔍 AI Risk Explainability</div>', unsafe_allow_html=True)

        exp_col, imp_col = st.columns([1.2, 1], gap="large")

        with exp_col:
            explanations = _explain(r["farm_size"], r["soil_score"], r["rainfall"],
                                     r["previous_loans"], r["yield_amount"], r["trust_score"])
            for e in explanations:
                st.markdown(f"<div style='padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.88rem;'>{e}</div>",
                            unsafe_allow_html=True)

        with imp_col:
            st.markdown("**Feature Importance (XGBoost)**")
            fig_imp = _feature_importance_chart()
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Train the model first: `python train_model.py`")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── PDF Export ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        dl1, dl2, _ = st.columns([1, 1, 2])
        with dl1:
            if st.button("📄 Download Lender Report (PDF)", use_container_width=True):
                exps = _explain(r["farm_size"], r["soil_score"], r["rainfall"],
                                r["previous_loans"], r["yield_amount"], r["trust_score"])
                pdf_bytes = _generate_pdf(
                    r["name"], r["farm_size"], r["soil_score"], r["rainfall"],
                    r["previous_loans"], r["yield_amount"], r["trust_score"],
                    r["risk_category"], exps,
                )
                fname = f"agritrust_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                fpath = os.path.join("reports", fname)
                with open(fpath, "wb") as f:
                    f.write(pdf_bytes)
                st.download_button("⬇️ Save PDF", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf",
                                   use_container_width=True)

        with dl2:
            apps = database.fetch_all_applications()
            if apps:
                csv = pd.DataFrame(apps).to_csv(index=False).encode()
                st.download_button("📊 Download Application Summary (CSV)",
                                   data=csv, file_name="agritrust_summary.csv",
                                   mime="text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
if tab_admin:
    with tab_admin:
        st.markdown("<br>", unsafe_allow_html=True)
        stats = database.get_summary_stats()

        # KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Applications", stats["total"])
        m2.metric("Approved",           stats["approved"])
        m3.metric("Rejected",           stats["rejected"])
        m4.metric("Avg Trust Score",    f"{stats['avg_score']}")

        st.markdown("<br>", unsafe_allow_html=True)
        ch_col, tbl_col = st.columns([1, 1.2], gap="large")

        with ch_col:
            st.markdown('<div class="card"><div class="card-title">📊 Risk Distribution</div>', unsafe_allow_html=True)
            dist = stats["distribution"]
            if dist:
                risk_order = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
                labels = [r for r in risk_order if r in dist]
                values = [dist[r] for r in labels]
                bar_colors = ["#4ade80", "#fbbf24", "#fb923c", "#f87171"][:len(labels)]

                fig_bar = go.Figure(go.Bar(
                    x=labels, y=values,
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=values, textposition="outside",
                    textfont={"color": "rgba(255,255,255,0.7)", "size": 12},
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin={"t":20,"b":10,"l":10,"r":10},
                    xaxis={"showgrid":False,"color":"rgba(255,255,255,0.4)","tickfont":{"size":10}},
                    yaxis={"showgrid":False,"color":"rgba(255,255,255,0.3)"},
                    bargap=0.35,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("No applications yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tbl_col:
            st.markdown('<div class="card"><div class="card-title">🗄️ Recent Applications</div>', unsafe_allow_html=True)
            apps = database.fetch_all_applications()
            if apps:
                df_admin = pd.DataFrame(apps)[["applicant_name","trust_score","risk_category","timestamp","officer_id"]]
                df_admin.columns = ["Applicant","Score","Risk","Date","Officer"]
                st.dataframe(df_admin.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No records found.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ================= FUTURE FEATURE =================
        # TODO: Integrate advanced analytics dashboard here
        # • Geographic heatmap of loan approvals by district
        # • Seasonal trend analysis (monsoon vs. dry season)
        # • Officer-level performance benchmarking
        # • Automated anomaly detection alerts
        # ===================================================

        st.markdown('<div class="card" style="border:1px dashed rgba(74,222,128,0.2);">'
                    '<div class="card-title" style="color:rgba(74,222,128,0.5);">🔮 Advanced Analytics — Coming Soon</div>'
                    '<p style="color:rgba(255,255,255,0.3);font-size:0.85rem;">'
                    'Geographic heatmaps · Seasonal trends · Officer benchmarks · Anomaly detection'
                    '</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – APPLICATION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
if tab_hist:
    with tab_hist:
        st.markdown("<br>", unsafe_allow_html=True)

        # Filter bar
        f1, f2, _ = st.columns([1, 1, 3])
        with f1:
            risk_filter = st.selectbox("Filter by Risk Level",
                                       ["All","Low Risk","Moderate Risk","High Risk","Very High Risk"])
        with f2:
            sort_col = st.selectbox("Sort by", ["timestamp","trust_score","farm_size"])

        apps = database.fetch_all_applications(risk_filter)

        # TODO: Add advanced filtering and search functionality here
        # (date range picker, applicant name search, trust score range slider)

        if apps:
            df_hist = pd.DataFrame(apps)
            df_hist = df_hist.sort_values(sort_col, ascending=(sort_col != "timestamp"))
            st.dataframe(df_hist, use_container_width=True, hide_index=True, height=480)

            csv = df_hist.to_csv(index=False).encode()
            st.download_button("⬇️ Export Filtered History (CSV)", data=csv,
                               file_name="agritrust_history.csv", mime="text/csv")
        else:
            st.info("No applications match the selected filter.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="at-footer">© 2026 <span>AgriTrust AI</span> – National Rural Fintech Innovation &nbsp;|&nbsp; '
    'Powered by XGBoost &amp; Streamlit &nbsp;|&nbsp; All assessments subject to credit officer review</div>',
    unsafe_allow_html=True
)
