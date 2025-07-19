import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('car_price_model.pkl')

st.title("Car Resale Price Predictor")

# User Inputs
name = st.text_input("Car Name (e.g., Maruti Swift)").title()
year = st.number_input("Purchase Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Preprocess manually
car_age = 2025 - year
name = " ".join(name.split()[:2])  # Keep first two words of car name

# Create input DataFrame
input_df = pd.DataFrame([{
    'name': name,
    'km_driven': km_driven,
    'fuel': fuel,
    'transmission': transmission,
    'owner': owner,
    'car_age': car_age
}])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Resale Price: ₹ {prediction:,.2f}")