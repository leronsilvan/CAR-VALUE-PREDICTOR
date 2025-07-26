import streamlit as st
import joblib
import pandas as pd
import gdown
import os

st.title("Car Resale Price Predictor")

# --- Load the model only once ---
@st.cache_resource
def load_model():
    url = "https://drive.google.com/file/d/17nOabOcxGztDR7VuXai7EsN08xkU1pDk/view?usp=drive_link"  # Replace with your actual file ID
    

    output = "car_price_model.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    return joblib.load(output)

model = load_model()

# --- User Inputs ---
name = st.text_input("Car Name (e.g., Maruti Swift)").title()
year = st.number_input("Purchase Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", [
    "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"
])

# --- Preprocess inputs ---
car_age = 2025 - year
name = " ".join(name.split()[:2])  # Only first 2 words of car name

input_df = pd.DataFrame([{
    'name': name,
    'km_driven': km_driven,
    'fuel': fuel,
    'transmission': transmission,
    'owner': owner,
    'car_age': car_age
}])

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Resale Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
