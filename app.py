import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS (Premium Dark Theme)
# -------------------------------

st.markdown("""
<style>

/* ---------- Global ---------- */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ---------- Titles ---------- */
h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    color: #38bdf8;
    font-weight: 700;
}

/* ---------- Cards ---------- */
.card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #22c55e);
    color: #020617;
    border-radius: 14px;
    padding: 12px 26px;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 12px 30px rgba(56,189,248,0.5);
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 50px;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
<div class="card">
<h1>🧬 Data Mining Dashboard</h1>
<p style="font-size:1.1rem; max-width:900px;">
A professional analytical platform for <b>PCA visualization</b> and
<b>Association Rule Mining</b> on medical datasets.
Designed with a modern dark UI for clarity, presentation, and academic excellence.
</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------------
# Main Feature Cards (2 columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
    <h3>🧠 Kidney PCA</h3>
    <p>Perform complete PCA analysis on the Kidney dataset:</p>
    <ul>
        <li>Advanced data cleaning</li>
        <li>Manual PCA implementation</li>
        <li>2D & 3D visualizations</li>
        <li>Statistical boxplots</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>🔗 Association Rules</h3>
    <p>Discover hidden patterns using Apriori algorithm:</p>
    <ul>
        <li>Frequent itemsets mining</li>
        <li>Support, confidence & lift</li>
        <li>Rule generation</li>
        <li>Actionable insights</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------------
# How to Use Section
# -------------------------------
st.markdown("""
<div class="card">
<h2>📌 How to Use</h2>
<ol style="font-size:1.05rem;">
    <li>Select a page from the <b>sidebar</b></li>
    <li>Upload the required CSV dataset</li>
    <li>Use buttons to explore analysis results</li>
    <li>Interpret patterns and insights easily</li>
</ol>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
 Data Mining Project • PCA & Association Rules • Streamlit Dashboard
</div>
""", unsafe_allow_html=True)
