
import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('Delhi House Price Prediction App')

# Input fields
st.subheader("Enter the property details:")
area = st.number_input('Area (in sq. ft.)', min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=2, step=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=1, step=1)
floors = st.number_input('Number of Floors', min_value=1, max_value=50, value=1, step=1)
age_of_property = st.number_input('Age of the Property (in years)', min_value=0, max_value=100, value=5, step=1)

# Additional features if available (customize based on your data)
location = st.selectbox('Location', ['South Delhi', 'North Delhi', 'East Delhi', 'West Delhi', 'Central Delhi'])
# Encoding location manually (this depends on your model's preprocessing)
location_encoded = {
    'South Delhi': 0,
    'North Delhi': 1,
    'East Delhi': 2,
    'West Delhi': 3,
    'Central Delhi': 4
}[location]

# Prepare the feature vector
features = np.array([[area, bedrooms, bathrooms, floors, age_of_property, location_encoded]], dtype=np.float64)

# Scale the features
features_scaled = scaler.transform(features)

# Predict the house price
predicted_price = model.predict(features_scaled)st.write(f'Predicted House Price: ₹{predicted_price[0]:,.2f}')
