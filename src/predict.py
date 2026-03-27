from src.model import PhishingModel
from src.feature_extraction import extract_features_batch

class PredictionPipeline:
    def __init__(self, model_path='models/model.pkl'):
        self.model = PhishingModel()
        try:
            self.model.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Could not load model from {model_path}. Please train it first.") from e
        
    def predict_url(self, url: str) -> str:
        # 1. Extract features (returns a DataFrame of 1 row)
        features_df = extract_features_batch([url])
        # 2. Predict using the loaded model
        prediction = self.model.predict(features_df)[0]
        # 3. Format output based on the binary label (assuming 1 = phishing)
        return "phishing" if prediction == 1 else "safe"
