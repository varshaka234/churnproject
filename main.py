import streamlit as st
import pandas as pd
import pickle
from os import path

PIPELINE_PATH = "MModel/pipeline.pkl"

# Load pipeline
if path.exists(PIPELINE_PATH) and path.getsize(PIPELINE_PATH) > 0:
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
else:
    st.error("❌ Pipeline file missing or empty.")
    st.stop()

st.title("📊 Customer Churn Predictor")

# Sidebar inputs
def user_input():
    tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.sidebar.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    return pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment
    }])

df_input = user_input()
st.subheader("Customer Input")
st.write(df_input)

# Prediction
if st.button("Predict Churn"):
    prediction = pipeline.predict(df_input)
    probability = pipeline.predict_proba(df_input)[0][1]

    st.success(f"⚠️ Customer Status: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.info(f"📊 Probability of Churn: {probability * 100:.2f}%")
