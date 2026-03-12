# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import os

# st.set_page_config(page_title="Bankruptcy Prediction", layout="centered")
# st.title("🏦 Bankruptcy Prediction App")

# # Load model
# model = joblib.load("models/bankruptcy_model.pkl")

# # Load encoders
# feature_columns = ['industrial_risk', 'management_risk', 'financial_flexibility',
#                    'credibility', 'competitiveness', 'operating_risk']
# encoders = {}
# for col in feature_columns + ['class']:
#     encoders[col] = joblib.load(f"models/{col}_encoder.pkl")

# st.header("Enter company financial information:")

# # Create input form
# input_data = {}
# for col in feature_columns:
#     classes = encoders[col].classes_
#     input_data[col] = st.selectbox(col.replace("_", " ").title(), classes)

# input_df = pd.DataFrame(input_data, index=[0])

# # Encode inputs
# for col in feature_columns:
#     input_df[col] = encoders[col].transform(input_df[col])

# # Prediction
# if st.button("Predict Bankruptcy"):
#     prediction = model.predict(input_df)
#     result = encoders['class'].inverse_transform(prediction)[0]
#     if result.lower() in ["bankrupt", "1"]:
#         st.error("⚠️ The company is predicted to **BANKRUPT**")
#     else:
#         st.success("✅ The company is predicted to **NOT BANKRUPT**")

import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Bankruptcy Prediction Dashboard")

st.markdown("""
Welcome! Enter your company's financial information below to predict bankruptcy.
""")

# -----------------------------
# Load model and encoders
# -----------------------------
model_path = "models/bankruptcy_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found! Please run the training script first.")
    st.stop()

model = joblib.load(model_path)

feature_columns = [
    'industrial_risk', 'management_risk', 'financial_flexibility',
    'credibility', 'competitiveness', 'operating_risk'
]

encoders = {}
for col in feature_columns + ['class']:
    encoder_path = f"models/{col}_encoder.pkl"
    if not os.path.exists(encoder_path):
        st.error(f"Encoder for {col} not found! Please run the training script first.")
        st.stop()
    encoders[col] = joblib.load(encoder_path)

# -----------------------------
# Input Form with Columns
# -----------------------------
st.header("📊 Company Financial Details")
possible_values = [0, 0.5, 1, 2]

input_data = {}
cols = st.columns(3)  # 3 columns for layout

for idx, col in enumerate(feature_columns):
    input_data[col] = cols[idx % 3].selectbox(
        col.replace("_", " ").title(),
        possible_values
    )

input_df = pd.DataFrame(input_data, index=[0])

# -----------------------------
# Encode inputs safely
# -----------------------------
can_encode = True
invalid_feature = None
for col in feature_columns:
    try:
        input_df[col] = encoders[col].transform(input_df[col])
    except ValueError:
        can_encode = False
        invalid_feature = col
        break

# -----------------------------
# Prediction Button & Result
# -----------------------------
if st.button("Predict Bankruptcy"):
    if not can_encode:
        st.warning(f"⚠️ Prediction unavailable: Invalid input for **{invalid_feature.replace('_',' ').title()}**")
    else:
        prediction = model.predict(input_df)[0]
        result = encoders['class'].inverse_transform([prediction])[0]

        if result.lower() == "bankruptcy":
            st.markdown(
                "<div style='background-color:#ffcccc; padding:20px; border-radius:10px; text-align:center;'>"
                "<h2>⚠️ The company is predicted to <strong>BANKRUPT</strong></h2></div>", 
                unsafe_allow_html=True
            )
        elif result.lower() == "non-bankruptcy":
            st.markdown(
                "<div style='background-color:#ccffcc; padding:20px; border-radius:10px; text-align:center;'>"
                "<h2>✅ The company is predicted to <strong>NOT BANKRUPT</strong></h2></div>", 
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Prediction unavailable or invalid. Please check the input.")

# -----------------------------
# Footer / Notes
# -----------------------------
st.markdown("---")
st.markdown("💡 **Tip:** Use the dropdowns to select financial risk levels. Values 0, 0.5, 1, and 2 represent increasing risk levels.")
