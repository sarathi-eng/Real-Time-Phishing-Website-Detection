class HybridDecisionEngine:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        
    def evaluate(self, feature_df, heuristic_signal: float = None) -> dict:
        """
        Soft-Voting Classifier logic: Final score is a weighted average 
        of the purely transparent ML prediction and an external Heuristic Signal (Deep Scan).
        """
        ml_raw_score = float(self.ml_model.predict_proba(feature_df)[0][1])
        
        if heuristic_signal is None:
            # Pure ML mode (Lexical Fast-Path) -> Handled 100% by the Random Forest math
            final_confidence = ml_raw_score
            heuristic_val = 0.0
        else:
            # Deep Scan Soft Voting Weighting
            # Guarantees the ML model is ALWAYS the primary brain mathematically.
            ml_weight = 0.75
            heuristic_weight = 0.25
            
            final_confidence = (ml_raw_score * ml_weight) + (heuristic_signal * heuristic_weight)
            # Ensure boundaries
            final_confidence = min(max(final_confidence, 0.0), 1.0)
            heuristic_val = heuristic_signal
            
        return {
            "ml_raw_score": ml_raw_score,
            "heuristic_signal": heuristic_val,
            "final_confidence": final_confidence
        }
