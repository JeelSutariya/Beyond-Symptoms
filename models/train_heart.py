import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_heart_model():
    # Set the path to the data directory
    DATA_DIR = r"D:\Software Development\Beyond-Symptoms\data"

    # Load the dataset
    data = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))

    # Separate features and target
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    # Convert categorical variables to numerical
    X = pd.get_dummies(X, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create and train the model using GridSearchCV
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate the model
    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    with open('saved_models/heart_disease_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save the scaler
    with open('saved_models/heart_disease_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved in saved_models directory")

    # Save test data for later use
    np.save('saved_models/heart_X_test.npy', X_test_scaled)
    np.save('saved_models/heart_y_test.npy', y_test)

    print("Test data saved for later use")

if __name__ == "__main__":
    train_heart_model()