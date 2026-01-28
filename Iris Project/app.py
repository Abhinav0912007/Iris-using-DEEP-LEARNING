import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("iris_model.h5")
scaler = joblib.load("scaler.pkl")

# Class labels
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Page config
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Enter flower measurements to predict the Iris species.")

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Input array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Species"):
    prediction = model.predict(input_scaled)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🌼 Predicted Species: **{predicted_class}**")

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")
