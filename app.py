import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("cardiac_model.pkl")

st.title("❤️ Cardiac Disease Prediction App")
st.write("Enter patient details below to predict the risk of cardiac disease.")

# Input fields
age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=10, max_value=250, step=1)
ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, step=1)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, step=1)
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])  # encoding in dataset
gluc = st.selectbox("Glucose", [1, 2, 3])
smoke = st.selectbox("Smoking", [0, 1])
alco = st.selectbox("Alcohol intake", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

# Encode categorical fields
gender_encoded = 1 if gender == "Male" else 2  # (check your dataset encoding: usually 1=male, 2=female)

# Feature vector (must match training order)
features = np.array([[age, gender_encoded, height, weight, ap_hi, ap_lo,
                      cholesterol, gluc, smoke, alco, active]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ High risk of cardiac disease!")
    else:
        st.success("✅ Low risk of cardiac disease.")
