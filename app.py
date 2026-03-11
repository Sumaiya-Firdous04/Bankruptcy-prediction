# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Bankruptcy Prediction", layout="centered")
st.title("🏦 Bankruptcy Prediction App")

# Load model
model = joblib.load("models/bankruptcy_model.pkl")

# Load encoders
feature_columns = ['industrial_risk', 'management_risk', 'financial_flexibility',
                   'credibility', 'competitiveness', 'operating_risk']
encoders = {}
for col in feature_columns + ['class']:
    encoders[col] = joblib.load(f"models/{col}_encoder.pkl")

st.header("Enter company financial information:")

# Create input form
input_data = {}
for col in feature_columns:
    classes = encoders[col].classes_
    input_data[col] = st.selectbox(col.replace("_", " ").title(), classes)

input_df = pd.DataFrame(input_data, index=[0])

# Encode inputs
for col in feature_columns:
    input_df[col] = encoders[col].transform(input_df[col])

# Prediction
if st.button("Predict Bankruptcy"):
    prediction = model.predict(input_df)
    result = encoders['class'].inverse_transform(prediction)[0]
    if result.lower() in ["bankrupt", "1"]:
        st.error("⚠️ The company is predicted to **BANKRUPT**")
    else:
        st.success("✅ The company is predicted to **NOT BANKRUPT**")
