import pandas as pd
from urllib.parse import urlparse
from src.context_features import extract_contextual_features
from src.typo_detector import TyposquattingDetector
from src.safelist import SafelistManager

class FeatureAssembler:
    def __init__(self):
        self.typo_detector = TyposquattingDetector()
        self.safelist_mgr = SafelistManager()

    def assemble(self, url: str) -> dict:
        """
        Feature Fusion Layer: Converts hard heuristics (Safelisting, Typosquatting) 
        into numerical ML features to eliminate Hard-Rule Bias.
        """
        if not url.startswith('http'):
            url = 'http://' + url
            
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        
        # 1. Base Lexical
        features = {
            'url_length': len(url),
            'has_at_symbol': 1 if '@' in url else 0,
            'num_subdomains': len(hostname.split('.')) - 1 if hostname else 0,
            'is_https': 1 if parsed.scheme == 'https' else 0,
        }
        
        # 2. Contextual Features
        context_features = extract_contextual_features(url)
        features.update(context_features)
        
        # 3. Safelisting -> converted to a High-Weight numerical Feature
        is_safelisted = self.safelist_mgr.is_safelisted(url)
        features['is_trusted_brand_score'] = 1.0 if is_safelisted else 0.0
        
        # 4. Typosquatting -> converted to numerical features instead of exit gates
        typo_features = self.typo_detector.analyze_domain(url)
        
        # Normalizes levenshtein distance threat. (1.0 = highly likely typosquatting)
        features['levenshtein_distance_feature'] = 1.0 if typo_features.get('red_flag_override') else 0.0
        features['is_punycode_abuse'] = 1.0 if typo_features.get('is_punycode') else 0.0
        
        # 5. Suspicious Content Density
        features['suspicious_content_density'] = context_features.get('suspicious_keyword_context', 0.0)
        
        return features

    def assemble_batch(self, urls: list) -> pd.DataFrame:
        """Process a list of URLs into a fused DataFrame."""
        return pd.DataFrame([self.assemble(url) for url in urls])

feature_assembler = FeatureAssembler()
