# frontend/app.py
import streamlit as st
import requests
import os

st.title("Iris Prediction App")

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(f"{backend_url}/predict", json=data)

    if response.status_code == 200:
        st.success(f"Prediction: {response.json()['prediction']}")
    else:
        st.error(f"Error contacting API: {response.text}")

