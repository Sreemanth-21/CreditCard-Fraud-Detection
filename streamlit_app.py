import streamlit as st
import numpy as np
import pickle

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter scaled transaction data below to predict if it's **fraudulent**.")

st.header("📝 Input Transaction Details")

col1, col2 = st.columns(2)
scaled_time = col1.number_input("Scaled Time", value=0.0, format="%.5f")
scaled_amount = col2.number_input("Scaled Amount", value=0.0, format="%.5f")

v_inputs = []
for i in range(1, 29, 2):
    col1, col2 = st.columns(2)
    v1 = col1.number_input(f"V{i}", value=0.0, format="%.5f")
    v_inputs.append(v1)
    if i + 1 <= 28:
        v2 = col2.number_input(f"V{i+1}", value=0.0, format="%.5f")
        v_inputs.append(v2)

st.markdown("---")
if st.button("🔎 Predict Transaction"):
    input_data = np.array([scaled_time, scaled_amount] + v_inputs).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2%})")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + XGBoost | Dataset: Kaggle Credit Card Fraud")
