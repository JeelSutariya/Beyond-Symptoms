import streamlit as st
import pickle
import numpy as np
import pandas as pd
from utils.preprocessing import preprocess_input_data
from models.model_utils import display_model_performance
import os

# Set the path to the images directory
IMAGES_DIR = r"D:\Software Development\Beyond-Symptoms\static\images"

def heart_disease_page():
    st.title('Heart Disease Prediction')
    
    # Display the heart disease image
    image_path = os.path.join(IMAGES_DIR, "heart.png")
    if os.path.exists(image_path):
        st.image(image_path, width=200)
    else:
        st.warning("Heart disease image not found.")
    
    st.write("""
    This module predicts the likelihood of heart disease based on various health metrics. 
    Please input your health data below:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=0.0)
    
    with col3:
        slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    if st.button('Predict'):
        # Load the model and scaler
        model = pickle.load(open('saved_models/heart_disease_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/heart_disease_scaler.pkl', 'rb'))
        
        # Convert categorical variables
        sex = 1 if sex == 'Male' else 0
        cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
        fbs = 1 if fbs == 'Yes' else 0
        restecg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
        exang = 1 if exang == 'Yes' else 0
        slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
        thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
        
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        
        # Create a DataFrame with column names
        column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'CA', 'Thal']
        input_df = pd.DataFrame(input_data, columns=column_names)
        
        # Apply one-hot encoding
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all columns from training are present
        for col in scaler.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[scaler.feature_names_in_]
        
        # Preprocess the input
        preprocessed_data = preprocess_input_data(input_df.values, scaler)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)
        
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.warning(f'High risk of heart disease. Probability: {prediction_proba[0][1]:.2f}')
        else:
            st.success(f'Low risk of heart disease. Probability: {prediction_proba[0][0]:.2f}')
        
        st.write("""
        Please note: This prediction is based on machine learning and should not be considered as a definitive diagnosis. 
        Always consult with a healthcare professional for proper medical advice and diagnosis.
        """)
        
        # Display model performance
        X_test = np.load('saved_models/heart_X_test.npy')
        y_test = np.load('saved_models/heart_y_test.npy')
        display_model_performance(model, X_test, y_test, 'Heart Disease')