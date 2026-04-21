# app.py

import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------------
# SIMPLE CLEAN CSS
# -----------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #1e3c72, #2a5298);
}

h1 {
    text-align: center;
    color: white;
}

div.stButton > button {
    width: 100%;
    background-color: #00c6ff;
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

div.stButton > button:hover {
    background-color: #0072ff;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🚀 AI Prediction App")
st.write("Enter values below and click Predict")

# -----------------------------------
# INPUT FIELDS
# Change these according to your model
# -----------------------------------
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# -----------------------------------
# BUTTON
# -----------------------------------
if st.button("🔮 Predict"):

    if model is not None:
        try:
            input_data = np.array([[feature1, feature2, feature3, feature4]])
            prediction = model.predict(input_data)

            st.success(f"Prediction Result: {prediction[0]}")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    else:
        st.warning("Model not loaded")

# -----------------------------------
# FOOTER
# -----------------------------------
st.write("---")
st.write("Made with ❤️ using Streamlit")
