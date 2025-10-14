import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("fraud_model.pkl")

# App Title and Description

st.title("Credit Card Fraud Detection App")
st.write("""
App predicts whether a given credit card transaction is **Fraud** or **Not Fraud**.
""")

# User Input Section

st.subheader("🔹 Enter Transaction Details")
v1 = st.number_input("Enter V1 Value", format="%.5f")
v2 = st.number_input("Enter V2 Value", format="%.5f")
v3 = st.number_input("Enter V3 Value", format="%.5f")
amount = st.number_input("Enter the Amount (in dollars)", format="%.2f")

# Prediction Section

if st.button("Predict Fraud"):
    # Define the same columns used during training
    columns = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
        'V28', 'Amount'
    ]
    input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    # Fill user inputs in corresponding columns
    input_data.loc[0, 'V1'] = v1
    input_data.loc[0, 'V2'] = v2
    input_data.loc[0, 'V3'] = v3
    input_data.loc[0, 'Amount'] = amount

    
    # Make Prediction
   
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Result

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"**Fraud**! (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"**Not Fraud** (Probability: {probability*100:.2f}%)")

#Accuracy Display
st.sidebar.header("Model Info")
st.sidebar.write("Model: RandomForestClassifier")
st.sidebar.write("Accuracy on test data: **~99.9%** approx")
st.sidebar.info("Note: Results depend on the model trained and data used.")
