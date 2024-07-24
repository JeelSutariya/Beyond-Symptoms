import unittest
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_diabetes_model(self):
        model = pickle.load(open('saved_models/diabetes_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/diabetes_scaler.pkl', 'rb'))
        
        X_scaled = scaler.transform(self.X_test)
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.7, "Diabetes model accuracy should be greater than 0.7")

    def test_heart_disease_model(self):
        model = pickle.load(open('saved_models/heart_disease_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/heart_disease_scaler.pkl', 'rb'))
        
        X_scaled = scaler.transform(self.X_test)
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.7, "Heart disease model accuracy should be greater than 0.7")

    def test_parkinsons_model(self):
        model = pickle.load(open('saved_models/parkinsons_model.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/parkinsons_scaler.pkl', 'rb'))
        
        X_scaled = scaler.transform(self.X_test)
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.7, "Parkinson's disease model accuracy should be greater than 0.7")

if __name__ == '__main__':
    unittest.main()