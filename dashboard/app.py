from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.database import initialize, recent_predictions, save_prediction  # noqa: E402
from app.scoring import FEATURES, MODEL_VERSION, score_leads  # noqa: E402

# st.set_page_config MUST be the very first Streamlit call
st.set_page_config(
    page_title="AFLIX AI Lead Scoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# High-Contrast, Ultra-Readable Theme & Styling
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force clean text rendering */
    p, span, label, div {
        color-scheme: dark;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 6px solid #38BDF8;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94A3B8 !important;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }

    /* High-contrast KPI cards */
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8 !important;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
        color: #CBD5E1 !important;
    }

    /* Clear, high-contrast score output card */
    .score-container {
        background-color: #1E293B;
        border: 2px solid #334155;
        border-radius: 14px;
        padding: 22px;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .badge-high {
        background-color: #064E3B;
        color: #34D399 !important;
        border: 1px solid #059669;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .badge-medium {
        background-color: #78350F;
        color: #FCD34D !important;
        border: 1px solid #D97706;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .badge-low {
        background-color: #334155;
        color: #E2E8F0 !important;
        border: 1px solid #475569;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .recommendation-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 14px;
        color: #F1F5F9 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Header Banner
# -------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">⚡ AFLIX AI Lead Scoring Intelligence</div>
        <p class="hero-subtitle">
            Enterprise conversion probability ranking engine • Instant qualification • Live database persistence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

initialize()

# -------------------------------------------------------------
# Model Evaluation Expander
# -------------------------------------------------------------
evaluation_path = ROOT / "artifacts" / "evaluation.json"
if evaluation_path.exists():
    with st.expander("📊 Held-Out Model Evaluation & Benchmark Metrics", expanded=False):
        evaluation = pd.read_json(evaluation_path, typ="series")
        metric_frame = pd.json_normalize(evaluation.to_dict(), sep="_")
        st.dataframe(metric_frame, use_container_width=True, hide_index=True)
else:
    st.info("💡 Run `python train.py` first to generate benchmark reports.")

# -------------------------------------------------------------
# Main Navigation Tabs
# -------------------------------------------------------------
tab_single, tab_bulk, tab_overview = st.tabs([
    "🎯 Real-Time Lead Scorer", 
    "📁 Bulk CSV Processing", 
    "📊 Pipeline Analytics & KPIs"
])

# -------------------------------------------------------------
# TAB 1: Real-Time Lead Scorer
# -------------------------------------------------------------
with tab_single:
    st.markdown("#### ⚡ Prospect Qualification Parameters")
    
    # Quick presets
    st.caption("Quick Presets for Fast Testing:")
    p_c1, p_c2, p_c3 = st.columns(3)
    if p_c1.button("🔥 Hot Enterprise Prospect", use_container_width=True):
        st.session_state["f_cs"] = "1000+"
        st.session_state["f_ind"] = "Technology"
        st.session_state["f_src"] = "Referral"
        st.session_state["f_ctry"] = "US"
        st.session_state["f_days"] = 3
        st.session_state["f_wv"] = 25
        st.session_state["f_eo"] = 12
        st.session_state["f_mtg"] = True

    if p_c2.button("💼 Mid-Market Warm Lead", use_container_width=True):
        st.session_state["f_cs"] = "201-1000"
        st.session_state["f_ind"] = "Finance"
        st.session_state["f_src"] = "Webinar"
        st.session_state["f_ctry"] = "GB"
        st.session_state["f_days"] = 14
        st.session_state["f_wv"] = 12
        st.session_state["f_eo"] = 5
        st.session_state["f_mtg"] = True

    if p_c3.button("🧊 Cold Inbound Visitor", use_container_width=True):
        st.session_state["f_cs"] = "1-50"
        st.session_state["f_ind"] = "Retail"
        st.session_state["f_src"] = "Website"
        st.session_state["f_ctry"] = "IN"
        st.session_state["f_days"] = 75
        st.session_state["f_wv"] = 2
        st.session_state["f_eo"] = 1
        st.session_state["f_mtg"] = False

    st.markdown("")
    col_input, col_output = st.columns([1.1, 1], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown("##### 🏢 Company & Acquisition Profile")
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                company_size = st.selectbox(
                    "Company Size", 
                    ["1-50", "51-200", "201-1000", "1000+"],
                    index=["1-50", "51-200", "201-1000", "1000+"].index(st.session_state.get("f_cs", "1-50")),
                )
                source = st.selectbox(
                    "Lead Source", 
                    ["Website", "Outbound", "Referral", "Webinar", "Conference"],
                    index=["Website", "Outbound", "Referral", "Webinar", "Conference"].index(st.session_state.get("f_src", "Website")),
                )
            with sub_c2:
                industry = st.selectbox(
                    "Industry Sector", 
                    ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"],
                    index=["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"].index(st.session_state.get("f_ind", "Technology")),
                )
                country = st.selectbox(
                    "Target Region / Country", 
                    ["US", "GB", "CA", "AU", "IN"],
                    index=["US", "GB", "CA", "AU", "IN"].index(st.session_state.get("f_ctry", "US")),
                )

            st.markdown("##### 📈 Digital Engagement Signals")
            days_since_contact = st.slider("Days Since Last Contact", 0, 120, int(st.session_state.get("f_days", 10)))
            website_visits = st.slider("Website Visits Count", 0, 30, int(st.session_state.get("f_wv", 8)))
            emails_opened = st.slider("Marketing Emails Opened", 0, 15, int(st.session_state.get("f_eo", 3)))
            meeting_attended = st.toggle("Lead Attended Sales Demo / Meeting", value=bool(st.session_state.get("f_mtg", False)))

            score_btn = st.button("⚡ Score & Qualify Lead Now", type="primary", use_container_width=True)

with col_output:
        st.markdown("##### 🎯 AI Qualification Intelligence")
        
        if score_btn:
            single_record = {
                "company_size": company_size,
                "industry": industry,
                "source": source,
                "country": country,
                "days_since_contact": int(days_since_contact),
                "website_visits": float(website_visits),
                "emails_opened": int(emails_opened),
                "meeting_attended": bool(meeting_attended),
            }
            res = score_leads([single_record])[0]
            save_prediction(single_record, res["score"], res["priority"], res["model_version"])
            st.session_state["single_lead_result"] = res
            st.session_state["single_lead_input"] = single_record
            st.toast(f"✅ Lead evaluated ({res['score']}/100) and saved to database!", icon="🚀")
            st.rerun()

        if "single_lead_result" in st.session_state:
            res = st.session_state["single_lead_result"]
            inp = st.session_state.get("single_lead_input", {})
            score = float(res["score"])
            priority = res["priority"]

            # ---------------------------------------------------------
            # 1. Executive Tier Alert Banner
            # ---------------------------------------------------------
            if priority == "High":
                st.success(f"### 🟢 High Priority Lead • Score: {score:.1f} / 100")
                sla_text = "⚡ **Sales SLA:** Immediate outreach within **2 to 4 business hours**."
                rec_action = "High conversion propensity. Route directly to senior sales executive for discovery/demo closing."
            elif priority == "Medium":
                st.warning(f"### 🟡 Medium Priority Lead • Score: {score:.1f} / 100")
                sla_text = "⏱️ **Sales SLA:** Follow-up scheduled within **24 to 48 hours**."
                rec_action = "Moderate purchase intent. Enroll in targeted product nurture cadence and share relevant case studies."
            else:
                st.info(f"### ⚪ Low Priority Lead • Score: {score:.1f} / 100")
                sla_text = "💤 **Sales SLA:** Automated marketing nurturing only."
                rec_action = "Passive top-of-funnel visitor. Retain in periodic newsletter; do not allocate direct sales rep calling hours."

            # ---------------------------------------------------------
            # 2. Interactive Speedometer Gauge
            # ---------------------------------------------------------
            gauge_accent = "#10B981" if priority == "High" else "#F59E0B" if priority == "Medium" else "#64748B"
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": " / 100", "font": {"size": 32, "color": "#FFFFFF", "family": "Inter"}},
                    title={"text": "Conversion Propensity Index", "font": {"size": 14, "color": "#94A3B8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#64748B", "tickvals": [0, 40, 70, 100]},
                        "bar": {"color": "#38BDF8", "thickness": 0.28},
                        "bgcolor": "#0F172A",
                        "borderwidth": 2,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 40], "color": "#1E293B"},
                            {"range": [40, 70], "color": "#78350F"},
                            {"range": [70, 100], "color": "#064E3B"},
                        ],
                        "threshold": {
                            "line": {"color": gauge_accent, "width": 5},
                            "thickness": 0.85,
                            "value": score,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#FFFFFF"},
                height=220,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ---------------------------------------------------------
            # 3. Priority Tier Reference Guide
            # ---------------------------------------------------------
            st.caption("Active Priority Threshold Ranges:")
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.markdown("<small style='color:#94A3B8;'>⚪ Low Tier<br><b>< 40.0</b></small>", unsafe_allow_html=True)
            with col_t2:
                st.markdown("<small style='color:#FBBF24;'>🟡 Medium Tier<br><b>40.0 – 69.9</b></small>", unsafe_allow_html=True)
            with col_t3:
                st.markdown("<small style='color:#34D399;'>🟢 High Tier<br><b>≥ 70.0</b></small>", unsafe_allow_html=True)

            st.markdown("---")


# ---------------------------------------------------------
            # 4. Explainability & Key Drivers Analysis (High Contrast)
            # ---------------------------------------------------------
            st.markdown("#### 🔍 Primary Conversion Drivers")
            positives, cautions = [], []

            if inp.get("meeting_attended"):
                positives.append("Attended live sales meeting / demo call (+ High Intent)")
            else:
                cautions.append("No sales demo or meeting attended yet")

            if inp.get("website_visits", 0) >= 15:
                positives.append(f"High website engagement ({int(inp['website_visits'])} site visits)")
            elif inp.get("website_visits", 0) <= 2:
                cautions.append("Minimal website activity (≤ 2 visits)")

            if inp.get("emails_opened", 0) >= 6:
                positives.append(f"Strong email campaign responsiveness ({int(inp['emails_opened'])} opened)")
            elif inp.get("emails_opened", 0) == 0:
                cautions.append("Unopened marketing communications (0 emails opened)")

            if inp.get("days_since_contact", 0) <= 7:
                positives.append(f"Recent prospect touchpoint ({int(inp['days_since_contact'])} days ago)")
            elif inp.get("days_since_contact", 0) >= 45:
                cautions.append(f"Dormant contact history ({int(inp['days_since_contact'])} days since last contact)")

            if inp.get("company_size") in ["201-1000", "1000+"]:
                positives.append(f"Enterprise account profile ({inp['company_size']} employees)")

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("### 🟢 Positive Catalysts")
                if positives:
                    for p in positives:
                        st.markdown(f"✅ **{p}**")
                else:
                    st.markdown("⚪ *Baseline profile indicators only*")

            with d_col2:
                st.markdown("### 🔴 Friction Factors")
                if cautions:
                    for c in cautions:
                        st.markdown(f"⚠️ **{c}**")
                else:
                    st.markdown("⚪ *No significant engagement friction*")

            # ---------------------------------------------------------
            # 5. Prescribed Sales Playbook & Footer (Native Streamlit)
            # ---------------------------------------------------------
            st.markdown("---")
            st.info(f"📋 **Prescribed Sales Action:**  \n{rec_action}  \n\n{sla_text}")

            st.caption(f"⚙️ Model Version: `{res['model_version']}` • Persisted to SQLite (`leads.sqlite3`)")
# -------------------------------------------------------------
# TAB 2: Bulk CSV Upload & Batch Scoring
# -------------------------------------------------------------
with tab_bulk:
    st.markdown("#### 📁 Batch Lead Upload & Automated Scoring")
    st.caption("Upload up to 5,000 enterprise leads in CSV format for bulk qualification.")

    uploaded = st.file_uploader("Drop leads CSV file here", type=["csv"], help="Columns required: " + ", ".join(FEATURES))
    if uploaded:
        try:
            input_df = pd.read_csv(uploaded)
            missing = sorted(set(FEATURES) - set(input_df.columns))
            if missing:
                st.error("❌ CSV is missing required columns: " + ", ".join(missing))
            elif len(input_df) > 5000:
                st.error("⚠️ Maximum limit is 5,000 rows per file.")
            else:
                st.markdown(f"**Uploaded Dataset Preview ({len(input_df):,} rows):**")
                st.dataframe(input_df.head(4), use_container_width=True, hide_index=True)

                if st.button("⚡ Score Entire Batch Now", type="primary", use_container_width=True):
                    with st.spinner("Scoring leads via AI pipeline..."):
                        records = input_df[FEATURES].where(pd.notnull(input_df[FEATURES]), None).to_dict("records")
                        results = score_leads(records)
                        output = pd.concat([input_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                        for lead, result in zip(records, results):
                            save_prediction(lead, result["score"], result["priority"], result["model_version"])

                    st.success(f"🎉 Successfully scored and persisted {len(output):,} leads to SQLite!")

                    c_act1, c_act2 = st.columns(2)
                    with c_act1:
                        st.download_button(
                            "📥 Download Scored Results (CSV)",
                            output.to_csv(index=False),
                            "lead_scores_output.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                    with c_act2:
                        if st.button("🔄 Refresh Analytics Overview", use_container_width=True):
                            st.rerun()

                    st.dataframe(output, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Could not score file: {exc}")

# -------------------------------------------------------------
# TAB 3: Global Analytics & Real-Time KPIs
# -------------------------------------------------------------
with tab_overview:
    stored = recent_predictions(1000)
    st.markdown("#### 📊 Real-Time Database Analytics (Latest 1,000 Leads)")
    
    if stored:
        df = pd.DataFrame(stored)
        total_count = len(df)
        high_count = int((df["priority"] == "High").sum())
        med_count = int((df["priority"] == "Medium").sum())
        low_count = int((df["priority"] == "Low").sum())
        avg_score = float(df["score"].mean()) if total_count > 0 else 0.0

        # KPI Cards (High Contrast & Clear Typography)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Leads Scored</div>
                    <div class="kpi-value">{total_count:,}</div>
                    <div class="kpi-subtext">Verified in SQLite database</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Average Pipeline Score</div>
                    <div class="kpi-value" style="color: #38BDF8 !important;">{avg_score:.1f}</div>
                    <div class="kpi-subtext">Overall pipeline index / 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            high_pct = (high_count / total_count * 100) if total_count > 0 else 0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">High Priority Leads</div>
                    <div class="kpi-value" style="color: #34D399 !important;">{high_count:,}</div>
                    <div class="kpi-subtext">🔥 {high_pct:.1f}% immediate sales focus</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            med_pct = (med_count / total_count * 100) if total_count > 0 else 0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Nurture Pipeline</div>
                    <div class="kpi-value" style="color: #FBBF24 !important;">{med_count:,}</div>
                    <div class="kpi-subtext">🌱 {med_pct:.1f}% medium priority pool</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # High-Impact Visualizations
        # -------------------------------------------------------------
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_hist = px.histogram(
                df, 
                x="score", 
                nbins=25, 
                title="<b>Lead Score Distribution (0 - 100)</b>",
                color_discrete_sequence=["#38BDF8"],
                opacity=0.88
            )
            fig_hist.update_traces(
                marker_line_width=1.2,
                marker_line_color="#0284C7",
                hovertemplate="Score Range: %{x}<br>Count: %{y}<extra></extra>"
            )
            fig_hist.update_layout(
                paper_bgcolor="#1E293B",
                plot_bgcolor="#0F172A",
                font={"color": "#F8FAFC", "family": "Inter"},
                title_font={"size": 16, "color": "#FFFFFF"},
                margin=dict(l=20, r=20, t=55, b=25),
                xaxis=dict(gridcolor="#334155", title="Lead Score (0-100)", showgrid=True),
                yaxis=dict(gridcolor="#334155", title="Number of Leads", showgrid=True),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch2:
            priority_counts = df["priority"].value_counts()
            p_order = [p for p in ["High", "Medium", "Low"] if p in priority_counts]
            p_vals = [priority_counts[p] for p in p_order]
            color_map = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#64748B"}

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=p_order,
                        values=p_vals,
                        hole=0.55,
                        marker=dict(colors=[color_map[p] for p in p_order], line=dict(color="#1E293B", width=3)),
                        textinfo="label+percent",
                        textfont=dict(size=13, color="#FFFFFF", family="Inter"),
                        hovertemplate="Priority: %{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                    )
                ]
            )
            fig_donut.update_layout(
                title="<b>Priority Mix Breakdown</b>",
                title_font={"size": 16, "color": "#FFFFFF"},
                paper_bgcolor="#1E293B",
                plot_bgcolor="#1E293B",
                font={"color": "#F8FAFC", "family": "Inter"},
                margin=dict(l=20, r=20, t=55, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                annotations=[
                    dict(text="<b>LEADS</b>", x=0.5, y=0.5, font_size=15, font_color="#94A3B8", showarrow=False)
                ]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("##### 📜 Recent Scored Lead Audit Table")
        st.dataframe(
            df[["id", "score", "priority", "created_at", "model_version"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions recorded yet. Score a single lead or upload a CSV to generate analytics.")