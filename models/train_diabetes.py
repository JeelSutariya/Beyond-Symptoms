import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

def train_diabetes_model():
    # Load and preprocess data
    DATA_DIR = r"D:\Software Development\Beyond-Symptoms\data"
    data = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
    
    # Feature engineering
    data['glucose_bmi_ratio'] = np.where(data['BMI'] != 0, data['Glucose'] / data['BMI'], 0)
    data['insulin_glucose_ratio'] = data['Insulin'] / (data['Glucose'] + 1)  # Adding 1 to avoid division by zero
    
    # Replace infinity with a large number
    data = data.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Remove any remaining NaN values
    data = data.dropna()

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Define models
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)
    
    # Define parameter grids
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Perform GridSearchCV for each model
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, n_jobs=-1, verbose=1)
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, n_jobs=-1, verbose=1)
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, n_jobs=-1, verbose=1)
    
    # Fit models
    rf_grid.fit(X_train_resampled, y_train_resampled)
    gb_grid.fit(X_train_resampled, y_train_resampled)
    xgb_grid.fit(X_train_resampled, y_train_resampled)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_grid.best_estimator_),
            ('gb', gb_grid.best_estimator_),
            ('xgb', xgb_grid.best_estimator_)
        ],
        voting='soft'
    )
    
    # Fit ensemble
    ensemble.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = ensemble.predict(X_test_scaled)
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    with open('saved_models/diabetes_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    with open('saved_models/diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved in saved_models directory")
    
    # Save test data for later use
    np.save('saved_models/diabetes_X_test.npy', X_test_scaled)
    np.save('saved_models/diabetes_y_test.npy', y_test)
    print("Test data saved for later use")

if __name__ == "__main__":
    train_diabetes_model()