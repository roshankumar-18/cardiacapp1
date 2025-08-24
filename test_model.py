import pickle
import numpy as np

# Load the saved pipeline (StandardScaler + Model)
model = pickle.load(open("cardiac_model.pkl", "rb"))

# Example input (replace with actual feature values in correct order)
# Format: [age, gender, cholesterol, ap_hi, ap_lo, gluc, smoke, alco, active]
sample_input = np.array([[55, 1, 2, 140, 90, 1, 0, 0, 1]])  

# Make prediction
prediction = model.predict(sample_input)[0]

# Interpret result
result = "Disease" if prediction == 1 else "No Disease"

print("✅ Model Loaded Successfully!")
print("Input:", sample_input)
print("Prediction:", result)

