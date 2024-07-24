import numpy as np

def preprocess_input_data(input_data, scaler):
    """
    Preprocess the input data using the provided scaler.
    
    Args:
    input_data (numpy.ndarray): Input data to be preprocessed.
    scaler (sklearn.preprocessing.StandardScaler): Fitted StandardScaler object.
    
    Returns:
    numpy.ndarray: Preprocessed input data.
    """
    return scaler.transform(input_data)