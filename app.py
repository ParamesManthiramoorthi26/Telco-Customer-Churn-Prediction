import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------------------
# Load saved artifacts
# ---------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# ---------------------------
# Title
# ---------------------------
st.title("📊 Telco Customer Churn Prediction App")
st.write("Predict whether a customer will leave the telecom company.")

# ---------------------------
# Input UI
# ---------------------------
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 2000)

contract_two_year = st.sidebar.selectbox("Two Year Contract", ["No", "Yes"])
internet_fiber = st.sidebar.selectbox("Fiber Optic Internet", ["No", "Yes"])

# ---------------------------
# Convert inputs to model format
# ---------------------------
input_data = pd.DataFrame([[
    tenure,
    monthly_charges,
    total_charges,
    1 if contract_two_year == "Yes" else 0,
    1 if internet_fiber == "Yes" else 0
]], columns=features)

# ---------------------------
# Scale numeric features
# ---------------------------
input_data_scaled = scaler.transform(input_data)

# ---------------------------
# Prediction
# ---------------------------
# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # ---------------------------
    # SIMPLE REASON ENGINE
    # ---------------------------
    reason = []

    if tenure < 12:
        reason.append("Low tenure (new customer) increases churn risk")

    if monthly_charges > 80:
        reason.append("High monthly charges increase churn risk")

    if total_charges < 2000:
        reason.append("Low total spending indicates weak loyalty")

    if contract_two_year == "No":
        reason.append("No long-term contract increases churn risk")

    if internet_fiber == "Yes":
        reason.append("Fiber optic users show higher churn tendency")

    # ---------------------------
    # OUTPUT
    # ---------------------------
    if prediction == 1:
        st.error(f"⚠️ Customer will CHURN (Risk: {probability:.2f})")

        st.markdown("###  Reasons for churn:")
        for r in reason:
            st.write("•", r)

    else:
        st.success(f"✅ Customer will NOT churn (Risk: {probability:.2f})")

        st.markdown("###  Positive factors:")
        st.write("• Long tenure helps retention")
        st.write("• Lower charges reduce churn risk")
        st.write("• Stable contract improves loyalty")