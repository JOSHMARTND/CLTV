# app.py - Streamlit Interface for CLV Prediction

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("clv_model.pkl")

st.title("Customer Lifetime Value (CLV) Prediction")
st.write("Upload your dataset to analyze customer RFM values and predict CLV.")

# Upload File
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display data preview
    st.write("Dataset Preview:")
    st.write(df.head())

    # Compute RFM values
    df['TotalRevenue'] = df['Quantity'] * df['Price']
    df['Recency'] = (pd.to_datetime('today') - pd.to_datetime(df['InvoiceDate'])).dt.days
    rfm_df = df.groupby('Customer ID').agg({
        'Recency': 'min',
        'Invoice': 'nunique',
        'TotalRevenue': 'sum'
    }).reset_index()

    # ✅ Rename columns to match model's feature names
    rfm_df = rfm_df.rename(columns={
        'Invoice': 'Frequency',
        'TotalRevenue': 'Monetary'
    })

    # Prepare data for prediction
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]

    # Predict CLV
    rfm_df['Predicted CLV'] = model.predict(X)

    # Show results
    st.write("Predicted CLV Results:")
    st.write(rfm_df[['Customer ID', 'Predicted CLV']])
