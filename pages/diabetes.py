import streamlit as st
import pickle
import numpy as np
from utils.preprocessing import preprocess_input_data
from models.model_utils import display_model_performance
import os

# Set the path to the images directory
IMAGES_DIR = r"D:\Software Development\Beyond-Symptoms\static\images"

def diabetes_page():
    st.title('Diabetes Prediction')
    
    # Display the diabetes image
    image_path = os.path.join(IMAGES_DIR, "diabetes.png")
    if os.path.exists(image_path):
        st.image(image_path, width=200)
    else:
        st.warning("Diabetes image not found.")
    
    st.write("""
    This module predicts the likelihood of diabetes based on several health metrics. 
    Please input your health data below:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=1000, value=79)
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input('Age', min_value=0, max_value=120, value=33)
    
    if st.button('Predict'):
        # Load the model and scaler
        model = pickle.load(open('saved_models/diabetes_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/diabetes_scaler.pkl', 'rb'))
        
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
        
        # Calculate the new ratios
        glucose_bmi_ratio = glucose / bmi if bmi != 0 else 0
        insulin_glucose_ratio = insulin / (glucose + 1)

        # Create the input array with the new features
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, glucose_bmi_ratio, insulin_glucose_ratio]).reshape(1, -1)
        
        # Preprocess the input
        preprocessed_data = preprocess_input_data(input_data, scaler)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)
        
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.warning(f'High risk of diabetes. Probability: {prediction_proba[0][1]:.2f}')
        else:
            st.success(f'Low risk of diabetes. Probability: {prediction_proba[0][0]:.2f}')
        
        st.write("""
        Please note: This prediction is based on machine learning and should not be considered as a definitive diagnosis. 
        Always consult with a healthcare professional for proper medical advice and diagnosis.
        """)
        
        # Display model performance
        X_test = np.load('saved_models/diabetes_X_test.npy')
        y_test = np.load('saved_models/diabetes_y_test.npy')
        display_model_performance(model, X_test, y_test, 'Diabetes')