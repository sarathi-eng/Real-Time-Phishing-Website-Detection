import pandas as pd

from src.model_logic import HybridDecisionEngine


class DummyModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, _):
        return [[1 - self.score, self.score]]


def test_evaluate_hard_override_for_typosquatting():
    engine = HybridDecisionEngine(DummyModel(0.35))
    feature_df = pd.DataFrame([
        {
            "levenshtein_distance_feature": 1.0,
            "is_punycode_abuse": 0.0,
            "is_safelisted_base": 0.0,
            "is_trusted_brand_score": 0.0,
            "suspicious_keyword_context": 1.0,
        }
    ])

    out = engine.evaluate(feature_df, heuristic_signal=0.5)

    assert out["final_confidence"] >= 0.99
    assert "Typosquatting detected" in out["hard_override_reason"]


def test_evaluate_safelist_override_to_safe_when_no_attack_signal():
    engine = HybridDecisionEngine(DummyModel(0.72))
    feature_df = pd.DataFrame([
        {
            "levenshtein_distance_feature": 1.0,
            "is_punycode_abuse": 0.0,
            "is_safelisted_base": 1.0,
            "is_trusted_brand_score": 1.0,
            "suspicious_keyword_context": 0.0,
        }
    ])

    out = engine.evaluate(feature_df, heuristic_signal=0.4)

    assert out["final_confidence"] <= 0.01
    assert "Trusted domain" in out["hard_override_reason"]
