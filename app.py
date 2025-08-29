import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('cardiac_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Error: 'cardiac_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Set page title and header
st.title("Cardiovascular Disease Prediction App")
st.header("Enter Patient Details")

# Define input fields based on the notebook's features
# The notebook uses the following features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
# Note: The 'id' column is dropped during processing in the notebook.
st.subheader("Physical and Lifestyle Information")
age = st.slider("Age (in years)", min_value=1, max_value=100, value=55)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=75.0)
gender = st.selectbox("Gender", options=[("Male", 2), ("Female", 1)], format_func=lambda x: x[0])
gender = gender[1] # Select the integer value (1 or 2)

st.subheader("Blood Pressure")
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=200, value=80)

st.subheader("Other Health Indicators")
cholesterol = st.selectbox("Cholesterol Level", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])
cholesterol = cholesterol[1]
gluc = st.selectbox("Glucose Level", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])
gluc = gluc[1]

st.subheader("Habits")
smoke = st.selectbox("Smoker?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
smoke = smoke[1]
alco = st.selectbox("Alcohol Consumer?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
alco = alco[1]
active = st.selectbox("Physically Active?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
active = active[1]

# Create a dictionary from the user inputs
user_data = {
    'age': [age * 365.25], # Convert age to days as done in the notebook
    'gender': [gender],
    'height': [height],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol': [cholesterol],
    'gluc': [gluc],
    'smoke': [smoke],
    'alco': [alco],
    'active': [active]
}

# Create a DataFrame from the user data
input_df = pd.DataFrame(user_data)

# Create a button to make a prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("The model predicts a high risk of cardiovascular disease.")
        else:
            st.success("The model predicts a low risk of cardiovascular disease.")
        
        st.write("---")
        st.subheader("User Input Summary")
        st.table(input_df)
        
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
      
