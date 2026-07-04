# ==========================================
# Smart Crop Recommendation System
# Prediction Script
# ==========================================
import joblib
import numpy as np
import pandas as pd

# Load Saved Model
model = joblib.load("models/random_forest_model.pkl")

# Load Label Encoder
label_encoder = joblib.load("models/label_encoder.pkl")

print("===================================")
print(" Smart Crop Recommendation System ")
print("===================================\n")

# Take User Input
nitrogen = float(input("Enter Nitrogen: "))
phosphorus = float(input("Enter Phosphorus: "))
potassium = float(input("Enter Potassium: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH Value: "))
rainfall = float(input("Enter Rainfall: "))

input_data = pd.DataFrame({
    "Nitrogen": [nitrogen],
    "Phosphorus": [phosphorus],
    "Potassium": [potassium],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "pH_Value": [ph],
    "Rainfall": [rainfall]
})

# Predict Crop
prediction = model.predict(input_data)

# Convert Number Back to Crop Name
crop = label_encoder.inverse_transform(prediction)

print("\n===================================")
print("Recommended Crop:", crop[0])
print("===================================")