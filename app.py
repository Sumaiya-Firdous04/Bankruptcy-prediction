import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Bankruptcy AI Dashboard",
    page_icon="🏦",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

[data-testid="stMetricValue"] {
color:white !important;
font-size:38px !important;
font-weight:bold;
}

[data-testid="stMetricLabel"] {
color:#e0e0e0 !important;
}

.title{
text-align:center;
font-size:55px;
font-weight:800;
margin-top:10px;
color:white;
}

.subtitle{
text-align:center;
font-size:20px;
color:#d0d0d0;
margin-bottom:30px;
}

label {
color:white !important;
font-size:16px !important;
font-weight:500;
}

.stSelectbox div[data-baseweb="select"]{
background-color:#1f2c38;
color:white;
border-radius:10px;
border:1px solid rgba(255,255,255,0.2);
}

.stButton > button{
background: linear-gradient(90deg,#ff512f,#dd2476);
color:white;
font-weight:600;
font-size:18px;
height:45px;
width:260px;
border-radius:12px;
border:none;
}

.stButton > button:hover{
background: linear-gradient(90deg,#00c6ff,#4facfe);
transform:scale(1.05);
transition:0.3s;
}

.result-safe{
background:#00c853;
padding:18px;
border-radius:12px;
text-align:center;
font-size:22px;
font-weight:600;
}

.result-risk{
background:#ff1744;
padding:18px;
border-radius:12px;
text-align:center;
font-size:22px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("bankruptcy_model.pkl")

# -------------------------
# HEADER
# -------------------------
st.markdown("<div class='title'>🏦 Bankruptcy Prediction System</div>", unsafe_allow_html=True)

# -------------------------
# RISK VALUES
# -------------------------
risk = {
"Low (0)":0,
"Medium (0.5)":0.5,
"High (1)":1
}

# -------------------------
# INPUT SECTION
# -------------------------
st.markdown("### 📊 Company Risk Indicators")

col1,col2 = st.columns(2)

with col1:
    industrial_risk = st.selectbox("Industrial Risk", risk.keys())
    management_risk = st.selectbox("Management Risk", risk.keys())
    financial_flexibility = st.selectbox("Financial Flexibility", risk.keys())

with col2:
    credibility = st.selectbox("Credibility", risk.keys())
    competitiveness = st.selectbox("Competitiveness", risk.keys())
    operating_risk = st.selectbox("Operating Risk", risk.keys())

st.markdown("")

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Bankruptcy Risk"):

    features = np.array([[
        risk[industrial_risk],
        risk[management_risk],
        risk[financial_flexibility],
        risk[credibility],
        risk[competitiveness],
        risk[operating_risk]
    ]])

    prediction = model.predict(features)

    prob = model.predict_proba(features)[0]

    safe = prob[0] * 100
    bankrupt = prob[1] * 100

    st.markdown("## 📊 Prediction Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Safe Probability", f"{safe:.2f}%")
        st.progress(int(safe))

    with col4:
        st.metric("Bankruptcy Probability", f"{bankrupt:.2f}%")
        st.progress(int(bankrupt))

    # RESULT MESSAGE
    if prediction[0] == 1:
        st.markdown(
            f"<div class='result-risk'>⚠ High Bankruptcy Risk ({bankrupt:.2f}%)</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-safe'>✅ Company is Financially Safe ({safe:.2f}%)</div>",
            unsafe_allow_html=True
        )

    # -------------------------
    # DOUGHNUT CHART
    # -------------------------
    st.markdown("### 📈 Bankruptcy Probability Visualization")

    values = [safe, bankrupt]
    colors = ["#7fb3d5", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(2.5,2.5))

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        autopct="%1.0f%%",
        pctdistance=0.75,
        wedgeprops=dict(width=0.45)
    )

    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(10)
        text.set_weight("bold")

    ax.axis("equal")

    colA, colB, colC = st.columns([2,1,2])
    with colB:
        st.pyplot(fig, transparent=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("Machine Learning Model: Random Forest | Streamlit  Dashboard")
