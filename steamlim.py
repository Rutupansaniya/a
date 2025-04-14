# Import required libraries
import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(
    page_title="Timelytics: Supply Chain Optimization",
    layout="wide",
    page_icon="📦"
)

# Display the title and description
st.title("Timelytics: Optimize Your Supply Chain with Advanced Forecasting")
st.caption(
    "Timelytics leverages XGBoost, Random Forest, and SVM models to provide "
    "accurate predictions for Order to Delivery (OTD) times."
)
st.caption(
    "Identify bottlenecks, reduce delays, and optimize your supply chain with actionable insights."
)

# Define the path to the model file
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'voting_model.pkl')

# Load the trained model
@st.cache_resource
def load_model():
    """
    Load the trained ensemble model from the pickle file.
    """
    try:
        with open(MODEL_FILE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'voting_model.pkl' exists in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model_data = load_model()
if model_data:
    voting_model = model_data.get('model')

# Define the prediction function
def predict_wait_time(
    purchase_dow, purchase_month, year, product_size_cm3,
    product_weight_g, geolocation_state_customer,
    geolocation_state_seller, distance
):
    """
    Predict the wait time using the loaded model.
    """
    try:
        prediction = voting_model.predict(
            np.array(
                [[
                    purchase_dow, purchase_month, year,
                    product_size_cm3, product_weight_g,
                    geolocation_state_customer, geolocation_state_seller,
                    distance
                ]]
            )
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Sidebar for user inputs
with st.sidebar:
    st.image("./assets/supply_chain_optimisation.jpg", caption="Optimize Your Supply Chain", use_column_width=True)
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Customer's State Code", value=10)
    geolocation_state_seller = st.number_input("Seller's State Code", value=20)
    distance = st.number_input("Distance (in km)", value=475.35)
    submit = st.button("Predict Wait Time")

# Main container for results
with st.container():
    st.header("Output: Predicted Wait Time (in Days)")
    if submit and model_data:
        with st.spinner("Predicting..."):
            prediction = predict_wait_time(
                purchase_dow, purchase_month, year, product_size_cm3,
                product_weight_g, geolocation_state_customer,
                geolocation_state_seller, distance
            )
            if prediction is not None:
                st.success(f"The predicted wait time is: {prediction} days")

# Display a sample dataset
sample_data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm³": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Customer's State Code": [25, 25, 25],
    "Seller's State Code": [20, 7, 20],
    "Distance (in km)": [247.94, 250.35, 4.915],
}
st.header("Sample Dataset")
st.write(pd.DataFrame(sample_data))
