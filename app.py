import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Load trained model
# ==========================================

model = joblib.load("tip_model.pkl")


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Tip Prediction",
    page_icon="💰",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("💰 Restaurant Tip Prediction")

st.write(
    "Enter the restaurant bill details "
    "to predict the expected tip."
)


# ==========================================
# User inputs
# ==========================================

total_bill = st.number_input(
    "Total Bill ($)",
    min_value=0.0,
    value=20.0,
    step=0.50
)


sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)


smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)


day = st.selectbox(
    "Day",
    ["Thur", "Fri", "Sat", "Sun"]
)


time = st.selectbox(
    "Time",
    ["Lunch", "Dinner"]
)


size = st.number_input(
    "Number of People",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)


# ==========================================
# Prediction button
# ==========================================

if st.button("Predict Tip"):

    input_data = pd.DataFrame({
        "total_bill": [total_bill],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time],
        "size": [size]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Tip: ${prediction:.2f}"
    )

    st.write(
        f"Total Bill: ${total_bill:.2f}"
    )

    tip_percentage = (prediction / total_bill) * 100

    st.write(
        f"Predicted Tip Percentage: {tip_percentage:.1f}%"
    )