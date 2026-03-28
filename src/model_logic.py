class HybridDecisionEngine:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        
    def evaluate(self, feature_df, heuristic_signal: float = None) -> dict:
        """
        Gating Network & Non-Linear Scaling Logic:
        Mathematically resolves the Heuristic-ML Imbalance and False Negative Boundary Risks.
        """
        ml_raw_score = float(self.ml_model.predict_proba(feature_df)[0][1])
        
        if heuristic_signal is None:
            # Pure Lexical Mode -> Fast Path
            final_confidence = ml_raw_score
            heuristic_val = 0.0
        else:
            heuristic_val = heuristic_signal
            
            # Non-Linear Amplification (Resolving Boundary False Negatives)
            # If the Threat Intel or Semantic Scanner returns an aggressive danger signal (> 0.8),
            # we quadratically converge the ML probability toward 1.0, rather than a linear blend 
            # that might leave a dangerous URL rotting at 49% (False Negative).
            if heuristic_signal >= 0.85:
                # Gating Function: Strongly amplifies the remaining safe margin
                final_confidence = ml_raw_score + ((1.0 - ml_raw_score) * heuristic_signal)
            elif heuristic_signal <= 0.2:
                # Gating Function: Strongly amplifies safety validations
                final_confidence = ml_raw_score * (heuristic_signal + 0.1)
            else:
                # Standard Soft-Voting Equilibrium
                ml_weight = 0.70
                heuristic_weight = 0.30
                final_confidence = (ml_raw_score * ml_weight) + (heuristic_signal * heuristic_weight)

            # Cap statistical bounds
            final_confidence = min(max(final_confidence, 0.0), 1.0)
            
        # -------------------------------------------------------------
        # 3-TIER DECISION PRIORITY ENGINE: Safelist vs Attack
        # -------------------------------------------------------------
        hard_override_reason = None
        
        if 'levenshtein_distance_feature' in feature_df.columns and 'is_trusted_brand_score' in feature_df.columns:
            typo_signal = float(feature_df['levenshtein_distance_feature'].iloc[0])
            puny_signal = float(feature_df.get('is_punycode_abuse', [0.0])[0])
            is_safelisted_base = float(feature_df.get('is_safelisted_base', [0.0])[0])
            suspicious_kw = float(feature_df.get('suspicious_keyword_context', [0.0])[0])
            
            # An authoritative domain fundamentally cannot be a typosquat of itself.
            # We nullify the mathematical fuzzy-matching False Positive collision.
            if is_safelisted_base == 1.0:
                typo_signal = 0.0
                
            # PRIORITY 1: HARD ATTACK OVERRIDE -> PHISHING
            if (typo_signal >= 0.8 or puny_signal >= 0.8) and is_safelisted_base == 0.0:
                final_confidence = max(final_confidence, 0.99)
                hard_override_reason = "Typosquatting detected: high similarity to trusted brand (Hard Override)."
                
            # PRIORITY 2: SAFELIST PROTECTION -> SAFE
            elif is_safelisted_base == 1.0 and suspicious_kw < 0.8:
                # If heuristic_val >= 0.85, the deep Deep-Scan found actual malicious 
                # password fields (i.e. a Compromised Site). Otherwise, force SAFE.
                if heuristic_val < 0.85:
                    final_confidence = min(final_confidence, 0.01)
                    hard_override_reason = "Trusted domain (safelisted) — no attack indicators found."
                    
            # PRIORITY 3: ML PREDICTION (Fallback defaults automatically if gates 1 & 2 fail)
            
        return {
            "ml_raw_score": ml_raw_score,
            "heuristic_signal": heuristic_val,
            "final_confidence": final_confidence,
            "hard_override_reason": hard_override_reason
        }
