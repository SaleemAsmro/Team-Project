import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown(
    """
    <style>
    div[role="combobox"] > div > div > select:hover {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

st.title("Heart Disease Prediction")

# User inputs for all features
sex = st.selectbox("Sex", options=['Male', 'Female'])
age = st.slider("Age", 20, 100, 50)
chest_pain = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.selectbox("Resting ECG", options=['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", options=['No', 'Yes'])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", options=['Up', 'Flat'])

# Mapping inputs to numerical as per preprocessing
sex_num = 1 if sex == 'Male' else 0
cp_map = {'ATA':1, 'NAP':2, 'ASY':3, 'TA':4}
chest_pain_num = cp_map[chest_pain]
resting_ecg_map = {'Normal':1, 'ST':2, 'LVH':3}
resting_ecg_num = resting_ecg_map[resting_ecg]
exercise_angina_num = 1 if exercise_angina == 'Yes' else 0
st_slope_map = {'Up':1, 'Flat':0}
st_slope_num = st_slope_map[st_slope]

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_num],
    'ChestPainType': [chest_pain_num],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg_num],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_num],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_num]
})

# Scale numerical columns
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Predict on click
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Prediction: The model predicts presence of heart disease.\nProbability: {proba:.2f}")
    else:
        st.success(f"Prediction: The model predicts NO heart disease.\nProbability: {1 - proba:.2f}")
