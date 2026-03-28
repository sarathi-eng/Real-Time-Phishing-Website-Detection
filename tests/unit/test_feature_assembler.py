from src.feature_assembler import FeatureAssembler


def test_assemble_adds_expected_features_for_safelisted_domain():
    assembler = FeatureAssembler()
    result = assembler.assemble("https://google.com/login")

    assert result["is_trusted_brand_score"] == 1.0
    assert "levenshtein_distance_feature" in result
    assert "is_punycode_abuse" in result
    assert "suspicious_content_density" in result


def test_assemble_normalizes_url_without_scheme():
    assembler = FeatureAssembler()
    result = assembler.assemble("example.com")

    assert result["url_length"] >= len("http://example.com")
    assert result["has_at_symbol"] == 0
