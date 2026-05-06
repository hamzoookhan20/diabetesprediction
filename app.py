import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_assets():
    model = joblib.load('best_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Error loading model files. Please ensure 'best_diabetes_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

# --- HEADER ---
st.title("🩺 Early Diabetes Detection System")
st.markdown("""
This application uses a **Gradient Boosted (XGBoost)** model to predict the likelihood of diabetes 
based on clinical diagnostic measurements.
""")

# --- INPUT SECTION ---
st.header("Patient Diagnostic Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skinthickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Weight in kg/(height in m)^2)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=21, max_value=120, value=30)

# --- PREDICTION LOGIC ---
if st.button("Analyze Risk Level"):
    # 1. Organize input data into a DataFrame (must match training order)
    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skinthickness, 
        insulin, bmi, dpf, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # 2. Scale the data using the saved scaler
    scaled_data = scaler.transform(input_data)

    # 3. Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # --- DISPLAY RESULTS ---
    st.divider()
    if prediction == 1:
        st.error(f"### Result: Positive Indication (High Risk)")
        st.write(f"The model predicts a **{probability*100:.1f}%** probability of diabetes.")
        st.warning("Recommendation: Follow up with a clinical glucose tolerance test.")
    else:
        st.success(f"### Result: Negative Indication (Low Risk)")
        st.write(f"The model predicts a **{probability*100:.1f}%** probability of diabetes.")
        st.balloons()

# --- FOOTER ---
st.sidebar.info("Model: XGBoost Classifier\nAccuracy: High (SOTA Baseline)")
st.sidebar.markdown("---")
st.sidebar.write("Developed by: Muhammad Hamza and Umar Faruk Ibrahim")
