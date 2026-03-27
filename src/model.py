import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

class PhishingModel:
    def __init__(self):
        # 1. Use balanced class weights to prevent overwhelming the model 
        # with massive Safe class disparity (which causes Overgeneralization)
        base_estimator = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        # 2. Wrap the estimator in a Calibrator to ensure predict_proba() 
        # outputs legitimate confidence probabilities (e.g., 0.85 = 85%), not arbitrary RF leaf scores.
        self.model = CalibratedClassifierCV(estimator=base_estimator, method='isotonic', cv=3)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def save(self, path='models/model.pkl'):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load(self, path='models/model.pkl'):
        self.model = joblib.load(path)
        
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
