# ─────────────────────────────────────────────
#  Repo_4 — Superstore Sales Analysis
#  Home.py  |  Multipage Launcher
#  Author : Mohamed · M3
# ─────────────────────────────────────────────
# =============================================================================
## path = streamlit run "E:\FINAL PROJECTS\P4_Superstore (2015-2018)\Home.py"
# ================================================================#
import streamlit as st
import pathlib

st.set_page_config(
    page_title  = "Superstore Sales · M3",
    page_icon   = "🛍️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

CLR = {
    "primary"   : "#1565c0",
    "success"   : "#2e7d32",
    "warning"   : "#e65100",
    "danger"    : "#c62828",
    "teal"      : "#00695c",
    "secondary" : "#455a64",
    "light"     : "#e3f2fd",
    "dark"      : "#1a237e",
    "purple"    : "#6a1b9a",
    "amber"     : "#f57f17",
    "pink"      : "#ad1457",
    "grey"      : "#546e7a",
    "white"     : "#ffffff",
    "black"     : "#212121",
    "teal2"     : "#00695c",
    "green2"    : "#1b5e20",
}

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🛍️ Superstore Sales")
    st.markdown("**Author:** Mohamed · M3")
    st.markdown("---")
    st.markdown("#### 📂 Navigation")
    st.markdown("""
- **🏠 Home** ← You are here
- **📊 EDA Dashboard** → Stage 1
- **🤖 ML Models** → Stage 2
""")
    st.markdown("---")
    st.markdown("#### 📁 Dataset Info")
    st.markdown("""
- **Source:** Tableau Superstore (Kaggle)
- **Rows:** 9,994 orders
- **Columns:** 20 features
- **Period:** 2015 – 2018
- **Regions:** 4 US regions
""")
    st.markdown("---")
    st.caption("© M3 · Data Analysis Portfolio")

# ── Header
st.markdown(f"""
<div style='background: linear-gradient(135deg, {CLR["green2"]} 0%, {CLR["teal2"]} 60%, {CLR["primary"]} 100%);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.4rem;'>
        🛍️ Superstore Sales Analysis
    </h1>
    <p style='color: #b2dfdb; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        End-to-End Business Intelligence & Machine Learning · Tableau Superstore Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards
col1, col2, col3, col4 = st.columns(4)
for col_w, label, val, color in zip(
    [col1, col2, col3, col4],
    ["Total Orders", "Profitable Orders", "Features", "Years Coverage"],
    ["9,994", "80.6%", "20", "2015–2018"],
    [CLR["primary"], CLR["success"], CLR["teal2"], CLR["amber"]]
):
    col_w.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {color}; text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,.06);'>
        <h2 style='color:{color}; margin:0;'>{val}</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>{label}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Stages
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["primary"]}; margin-top:0;'>📊 Stage 1 — EDA Dashboard</h3>
        <p style='color:{CLR["secondary"]};'>Deep business intelligence across 12 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Data Overview & Correlation</li>
            <li>Variables Analysis & Distributions</li>
            <li>IQR Cleaning & Outlier Detection</li>
            <li>Missing Values & Multicollinearity</li>
            <li>Insights & Recommendations</li>
            <li>📦 Business KPI Dashboard <b>(NEW)</b></li>
            <li>💰 Profit & Loss Analysis <b>(NEW)</b></li>
            <li>🗺️ Regional Performance <b>(NEW)</b></li>
            <li>📈 Time Series Trends <b>(NEW)</b></li>
            <li>🧪 Statistical Tests</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["success"]}; margin-top:0;'>🤖 Stage 2 — ML Models</h3>
        <p style='color:{CLR["secondary"]};'>Full machine learning pipeline across 5 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Regression Models (6) → predict <b>Profit</b></li>
            <li>Classification Models (6) → predict <b>is_profitable</b></li>
            <li>Model Comparison & Report</li>
            <li>Predict New Order</li>
            <li>Final Insights & Report (PDF + Word)</li>
        </ul>
        <br>
        <p style='color:{CLR["secondary"]}; font-size:0.85rem;'>
            ✅ 12 models · Parallel training · 80/20 imbalance handling · AUC evaluation
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Business Context
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;
            border-left:5px solid {CLR["teal2"]}; margin-bottom:1rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🎯 Why This Project Matters</h4>
    <p style='color:{CLR["black"]}; margin:0; line-height:1.8;'>
        19.4% of all Superstore orders are <b>unprofitable</b> — losing up to $6,600 per order.
        This project identifies <b>which products, regions, and discount levels destroy profit margins</b>,
        enabling data-driven pricing and inventory decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── How to Use
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🚀 How to Use</h4>
    <ol style='color:{CLR["black"]}; line-height:2;'>
        <li>Go to <b>📊 EDA Dashboard</b> → upload <code>superstore_clean.csv</code></li>
        <li>Explore all 12 EDA tabs — insights auto-saved to session</li>
        <li>Go to <b>🤖 ML Models</b> → data flows automatically from Stage 1</li>
        <li>Train models, compare results, export the final report</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align:center; color:{CLR["grey"]}; font-size:0.8rem; padding:1rem;'>
    Built with ❤️ by Mohamed · M3 · Data Analysis Portfolio · Tableau Superstore Dataset
</div>
""", unsafe_allow_html=True)
