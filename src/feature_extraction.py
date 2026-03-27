import pandas as pd
from urllib.parse import urlparse
from src.context_features import extract_contextual_features

def extract_features(url: str) -> dict:
    """Extract basic lexical + advanced contextual features from a URL."""
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    
    # 1. Base Lexical (Maintains baseline compatibility)
    features = {
        'url_length': len(url),
        'has_at_symbol': 1 if '@' in url else 0,
        'num_subdomains': len(hostname.split('.')) - 1 if hostname else 0,
        'is_https': 1 if parsed.scheme == 'https' else 0,
    }
    
    # 2. Advanced Contextual Features (Preventing Overgeneralization)
    context_features = extract_contextual_features(url)
    features.update(context_features)
    
    return features

def extract_features_batch(urls: list) -> pd.DataFrame:
    """Process a list of URLs into a DataFrame of features."""
    return pd.DataFrame([extract_features(url) for url in urls])
