
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
bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2, step=1)
bathroom = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=1, step=1)
furnishing = st.selectbox('Furnishing', ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
locality = st.selectbox('Locality', ['South Delhi', 'North Delhi', 'East Delhi', 'West Delhi', 'Central Delhi'])
parking = st.number_input('Number of Parking Spaces', min_value=0, max_value=5, value=1, step=1)
status = st.selectbox('Status', ['Ready to Move', 'Under Construction'])
transaction = st.selectbox('Transaction Type', ['New Property', 'Resale'])
property_type = st.selectbox('Property Type', ['Apartment', 'Independent House', 'Villa', 'Builder Floor'])
per_sqft = st.number_input('Per Square Foot Rate (₹)', min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

# Encoding categorical features
furnishing_encoded = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully-Furnished': 2}[furnishing]
locality_encoded = {
    'South Delhi': 0,
    'North Delhi': 1,
    'East Delhi': 2,
    'West Delhi': 3,
    'Central Delhi': 4
}[locality]
status_encoded = {'Ready to Move': 0, 'Under Construction': 1}[status]
transaction_encoded = {'New Property': 0, 'Resale': 1}[transaction]
property_type_encoded = {
    'Apartment': 0,
    'Independent House': 1,
    'Villa': 2,
    'Builder Floor': 3
}[property_type]

# Prepare the feature vector
features = np.array([[area, bhk, bathroom, furnishing_encoded, locality_encoded, parking, status_encoded, transaction_encoded, property_type_encoded, per_sqft]], dtype=np.float64)

# Scale the features
features_scaled = scaler.transform(features)

# Predict the house price
predicted_price = model.predict(features_scaled)

# Display the result
st.write(f'Predicted House Price: ₹{predicted_price[0]:,.2f}')
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
bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2, step=1)
bathroom = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=1, step=1)
furnishing = st.selectbox('Furnishing', ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
locality = st.selectbox('Locality', ['South Delhi', 'North Delhi', 'East Delhi', 'West Delhi', 'Central Delhi'])
parking = st.number_input('Number of Parking Spaces', min_value=0, max_value=5, value=1, step=1)
status = st.selectbox('Status', ['Ready to Move', 'Under Construction'])
transaction = st.selectbox('Transaction Type', ['New Property', 'Resale'])
property_type = st.selectbox('Property Type', ['Apartment', 'Independent House', 'Villa', 'Builder Floor'])
per_sqft = st.number_input('Per Square Foot Rate (₹)', min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

# Encoding categorical features
furnishing_encoded = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully-Furnished': 2}[furnishing]
locality_encoded = {
    'South Delhi': 0,
    'North Delhi': 1,
    'East Delhi': 2,
    'West Delhi': 3,
    'Central Delhi': 4
}[locality]
status_encoded = {'Ready to Move': 0, 'Under Construction': 1}[status]
transaction_encoded = {'New Property': 0, 'Resale': 1}[transaction]
property_type_encoded = {
    'Apartment': 0,
    'Independent House': 1,
    'Villa': 2,
    'Builder Floor': 3
}[property_type]

# Prepare the feature vector
features = np.array([[area, bhk, bathroom, furnishing_encoded, locality_encoded, parking, status_encoded, transaction_encoded, property_type_encoded, per_sqft]], dtype=np.float64)

# Scale the features
features_scaled = scaler.transform(features)

# Predict the house price
predicted_price = model.predict(features_scaled)

# Display the result
st.write(f'Predicted House Price: ₹{predicted_price[0]:,.2f}')
