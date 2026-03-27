from src.typo_detector import TyposquattingDetector
from src.safelist import SafelistManager
from src.feature_extraction import extract_features_batch

class FastLexicalPipeline:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        # Pre-load instances into memory for 0ms instantiation queries
        self.typo_detector = TyposquattingDetector()
        self.safelist_mgr = SafelistManager()

    def evaluate(self, url: str) -> dict:
        """
        Synchronous, CPU-bound Fast-Path evaluation.
        Evaluates purely lexical & contextual features in under 15ms.
        """
        # 1. Safelist Base Domain check (O(1))
        is_safelisted = self.safelist_mgr.is_safelisted(url)
        
        # 2. Typosquatting & Edit Distances (O(N) against limited Top 100 brands)
        typo_features = self.typo_detector.analyze_domain(url)
        
        # 3. Predict Proba with Scikit-Learn
        lexical_features_df = extract_features_batch([url])
        lexical_prob = self.ml_model.predict_proba(lexical_features_df)[0][1]
        
        confidence = float(lexical_prob)
        decision_reason = []
        is_hard_override = False

        # --- SAFELIST OVERRIDE ---
        if is_safelisted:
            confidence = 0.01
            decision_reason.append("URL explicitly verified against Top 10K Trusted Safelist.")
            is_hard_override = True

        # --- RED FLAG OVERRIDES ---
        if not is_hard_override:
            if typo_features.get("red_flag_override") and not typo_features.get("is_official_domain"):
                confidence = max(confidence, 0.99)
                reasons = " | ".join(typo_features["typo_reasons"])
                decision_reason.append(f"Typosquatting Override: {reasons}")
                is_hard_override = True

            if typo_features.get("is_punycode"):
                confidence = max(confidence, 0.98)
                decision_reason.append("Punycode (IDN) abuse detected (Spoofing characters).")
                is_hard_override = True
                
        if not is_hard_override:
            decision_reason.append("Lexical ML Model probability calculation.")

        if confidence > 0.75:
            verdict = "PHISHING (Red)"
        elif confidence >= 0.40:
            verdict = "SUSPICIOUS (Yellow)"
        else:
            verdict = "SAFE (Green)"

        return {
            "verdict": verdict,
            "confidence_score": round(confidence, 4),
            "applied_safelist_bypass": bool(is_safelisted),
            "decision_reason": " | ".join(decision_reason),
            "raw_features": {
                "base_ml_prob": round(float(lexical_prob), 4),
                "typosquatting": typo_features
            }
        }
