import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("bankruptcy_model.pkl")

st.title("Bankruptcy Prediction App")

# User input
st.write("Enter company financial information:")

# Example: numeric or categorical features
industrial_risk = st.selectbox("Industrial Risk", ["low","medium","high"])
management_risk = st.selectbox("Management Risk", ["low","medium","high"])
financial_flexibility = st.selectbox("Financial Flexibility", ["low","medium","high"])
credibility = st.selectbox("Credibility", ["low","medium","high"])
competitiveness = st.selectbox("Competitiveness", ["low","medium","high"])
operating_risk = st.selectbox("Operating Risk", ["low","medium","high"])

# Convert inputs to DataFrame
input_df = pd.DataFrame({
    "industrial_risk":[industrial_risk],
    "management_risk":[management_risk],
    "financial_flexibility":[financial_flexibility],
    "credibility":[credibility],
    "competitiveness":[competitiveness],
    "operating_risk":[operating_risk]
})

# Encode categorical features (must match training)
from sklearn.preprocessing import LabelEncoder
for col in input_df.columns:
    le = LabelEncoder()
    # Fit on training column unique values (example, in production you save these encoders)
    le.fit(df[col])  # df is your training dataframe
    input_df[col] = le.transform(input_df[col])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Bankrupt" if prediction[0]==1 else "Not Bankrupt"
    st.success(f"The company is predicted to be: {result}")
