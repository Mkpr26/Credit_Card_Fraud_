# Credit Card Fraud Detection using XGBoost

This project predicts whether a given credit card transaction is **Fraud or not Fraud** using the **Machine Learning Algorithm**.  
It also includes a **Streamlit web application** where users can input transaction details and instantly check if the transaction is likely to be fraudulent.

Use the App link to check whether the Transaction is **Fraud or not Fraud**:

---

## 🚀 Project Overview

Credit card fraud is one of the major issues in the banking and financial sectors.  
This project uses a **imbalanced dataset** and applies **Machine Learning techniques** to classify transactions as Fraud or Not Fraud.

---

## 🧠 Key Features
- Trained on **imbalanced data**.
- Uses **XGBoost Classifier** for high accuracy.
- Visualizes **performance metrics**.
- Saved model with **Joblib** for quick loading.

---

## 📂 Project Structure

Credit_Card_Fraud_Detection/
│
├── app.py # Streamlit app for deployment
├── xgb_fraud_model.pkl # Trained XGBoost model (saved using joblib)
├── creditcard.csv # Dataset (Kaggle)
├── requirements.txt # Project dependencies
├── README.md # Project documentation

---

## 🧾 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:**  
  The dataset contains transactions made by European cardholders in September 2013.  
  It has **284,807 transactions**, among which **492 are frauds (0.172%)**.

| Feature | Description |
|----------|--------------|
| `V1` - `V28` | PCA-transformed features (confidential data) |
| `Amount` | Transaction amount |
| `Class` | Target variable (0 = Not Fraud, 1 = Fraud) |
