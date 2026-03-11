# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bankruptcy Prediction", layout="centered")
st.title("🏦 Bankruptcy Prediction App")

# Load model
model = joblib.load("bankruptcy_model.pkl")

# Load encoders
feature_columns = ['industrial_risk', 'management_risk', 'financial_flexibility',
                   'credibility', 'competitiveness', 'operating_risk']

encoders = {}
for col in feature_columns + ['class']:
    encoders[col] = joblib.load(f"{col}_encoder.pkl")

# User input
st.header("Enter company financial information:")

input_data = {}
for col in feature_columns:
    # Get classes from encoder
    classes = encoders[col].classes_
    input_data[col] = st.selectbox(col.replace("_", " ").title(), classes)

# Convert input to DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Encode input
for col in feature_columns:
    input_df[col] = encoders[col].transform(input_df[col])

# Predict button
if st.button("Predict Bankruptcy"):
    prediction = model.predict(input_df)
    result = encoders['class'].inverse_transform(prediction)[0]  # Decode class
    if result == "Bankrupt" or result == 1:
        st.error(f"⚠️ The company is predicted to **BANKRUPT**")
    else:
        st.success(f"✅ The company is predicted to **NOT BANKRUPT**")
