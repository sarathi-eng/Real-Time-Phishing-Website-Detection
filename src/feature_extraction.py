import pandas as pd
from src.feature_assembler import feature_assembler

def extract_features(url: str) -> dict:
    """Extract fused lexical/contextual/typosquatting features from a URL."""
    return feature_assembler.assemble(url)

def extract_features_batch(urls: list) -> pd.DataFrame:
    """Process a list of URLs into a DataFrame of features."""
    return feature_assembler.assemble_batch(urls)
