from urllib.parse import urlparse
import tldextract
from src.safelist import SafelistManager

# Initialize the O(1) Safelist Manager
safelist_mgr = SafelistManager()

SENSITIVE_KEYWORDS = {"login", "secure", "verify", "account", "update", "bank", "password", "auth"}
VOWELS = set("aeiou")

def _vowel_consonant_ratio(s: str) -> float:
    """Helper to detect DGA (Domain Generation Algorithms) by analyzing phonetic structure."""
    if not s:
        return 0.0
    vowels = sum(1 for c in s if c in VOWELS)
    consonants = sum(1 for c in s if c.isalpha() and c not in VOWELS)
    if consonants == 0:
        return float(vowels)
    return vowels / consonants

def extract_contextual_features(url: str) -> dict:
    """
    Extracts ratio-based and context-aware lexical features to prevent ML overgeneralization.
    """
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    base_domain = f"{ext.domain}.{ext.suffix}".lower()
    path = parsed.path or ""
    
    # Safelist Context Generation
    is_safelisted = safelist_mgr.is_safelisted(url)
    
    # 1. Path to Domain Ratio 
    # (Attackers compromise legitimate sites and hide phish deep in long sub-paths)
    domain_len = len(base_domain)
    path_len = len(path)
    path_to_domain_ratio = (path_len / domain_len) if domain_len > 0 else 0.0
    
    # 2. Suspicious Keyword Context
    # Bias Fix: Only penalize "login" or "secure" if the domain is NOT safely explicitly known.
    suspicious_keyword_context = 0
    if not is_safelisted:
        if any(kw in url.lower() for kw in SENSITIVE_KEYWORDS):
            suspicious_keyword_context = 1
            
    # 3. Vowel-Consonant Ratio (DGA Detection)
    # Replaces raw "URL Length" logic with phonetic string structure checks.
    vc_ratio = _vowel_consonant_ratio(ext.domain.lower())
    
    return {
        "is_safelisted_base": 1 if is_safelisted else 0,
        "path_to_domain_ratio": round(path_to_domain_ratio, 4),
        "suspicious_keyword_context": suspicious_keyword_context,
        "vc_ratio": round(vc_ratio, 4)
    }
