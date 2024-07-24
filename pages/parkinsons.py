import streamlit as st
import pickle
import numpy as np
from utils.preprocessing import preprocess_input_data
from models.model_utils import display_model_performance
import os

# Set the path to the images directory
IMAGES_DIR = r"D:\Software Development\Beyond-Symptoms\static\images"

def parkinsons_page():
    st.title("Parkinson's Disease Prediction")
    
    # Display the Parkinson's disease image
    image_path = os.path.join(IMAGES_DIR, "parkinsons.png")
    if os.path.exists(image_path):
        st.image(image_path, width=200)
    else:
        st.warning("Parkinson's disease image not found.")
    
    st.write("""
    This module predicts the likelihood of Parkinson's disease based on various voice measures. 
    Please input the voice data below:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz) - Average vocal fundamental frequency', value=119.992)
        fhi = st.number_input('MDVP:Fhi(Hz) - Maximum vocal fundamental frequency', value=157.302)
        flo = st.number_input('MDVP:Flo(Hz) - Minimum vocal fundamental frequency', value=74.997)
        jitter_percent = st.number_input('MDVP:Jitter(%) - Variation in fundamental frequency', value=0.00784)
        jitter_abs = st.number_input('MDVP:Jitter(Abs) - Absolute jitter in microseconds', value=0.00007)
        rap = st.number_input('MDVP:RAP - Relative amplitude perturbation', value=0.00370)
    
    with col2:
        ppq = st.number_input('MDVP:PPQ - Five-point period perturbation quotient', value=0.00554)
        ddp = st.number_input('Jitter:DDP - Average absolute difference of differences between cycles', value=0.01109)
        shimmer = st.number_input('MDVP:Shimmer - Variation in amplitude', value=0.04374)
        shimmer_db = st.number_input('MDVP:Shimmer(dB) - Shimmer in decibels', value=0.426)
        apq3 = st.number_input('Shimmer:APQ3 - Three-point amplitude perturbation quotient', value=0.02182)
        apq5 = st.number_input('Shimmer:APQ5 - Five-point amplitude perturbation quotient', value=0.03130)
    
    with col3:
        apq = st.number_input('MDVP:APQ - Amplitude perturbation quotient', value=0.02971)
        dda = st.number_input('Shimmer:DDA - Average absolute differences between consecutive differences', value=0.06545)
        nhr = st.number_input('NHR - Noise-to-harmonics ratio', value=0.02211)
        hnr = st.number_input('HNR - Harmonics-to-noise ratio', value=21.033)
        rpde = st.number_input('RPDE - Recurrence period density entropy measure', value=0.414783)
        dfa = st.number_input('DFA - Signal fractal scaling exponent', value=0.815285)
    
    with col4:
        spread1 = st.number_input('spread1 - Nonlinear measure of fundamental frequency variation', value=-4.813031)
        spread2 = st.number_input('spread2 - Nonlinear measure of fundamental frequency variation', value=0.266482)
        d2 = st.number_input('D2 - Correlation dimension', value=2.301442)
        ppe = st.number_input('PPE - Pitch period entropy', value=0.284654)
    
    if st.button('Predict'):
        # Load the model and scaler
        model = pickle.load(open('saved_models/parkinsons_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/parkinsons_scaler.pkl', 'rb'))
        
        input_data = np.array([fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, 
                               apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]).reshape(1, -1)
        
        # Preprocess the input
        preprocessed_data = preprocess_input_data(input_data, scaler)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)
        
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.warning(f"High likelihood of Parkinson's disease. Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.success(f"Low likelihood of Parkinson's disease. Probability: {prediction_proba[0][0]:.2f}")
        
        st.write("""
        Please note: This prediction is based on machine learning and should not be considered as a definitive diagnosis. 
        Always consult with a healthcare professional for proper medical advice and diagnosis.
        """)
        
        # Display model performance
        X_test = np.load('saved_models/parkinsons_X_test.npy')
        y_test = np.load('saved_models/parkinsons_y_test.npy')
        display_model_performance(model, X_test, y_test, "Parkinson's Disease")