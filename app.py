import streamlit as st
import pandas as pd
import joblib
# Load Model
model = joblib.load("xgb_fraud_model.pkl")

st.title("Credit Card Fraud Detection App")
st.write("Predict whether a transcation is Fraud or not Fraud Using XGBoost Model.")

## Input Features
V1 = st.number_input("Enter V1 Value ", format="%.5f")
V2 = st.number_input("Enter V2 Value ", format="%.5f")
V3 = st.number_input("Enter V3 Value ", format="%.5f")
Amount = st.number_input("Enter the Amount(in dollars) : ",max_value=0.0)

#input dataframe
input_data = pd.DataFrame([[V1,V2,V3,Amount]],columns= [V1,V2,V3,'Amount'])

#--Predict--

if st.button("Prediction Fraud"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction ==1:
        st.error(f"Fraud Transaction Detected ! (Probabilty : {prob:.2f})")
    else:
        st.success(f"Safe Transaction (probabilty : {prob:.2f})")