from urllib.parse import urlparse
import tldextract
import math
from collections import Counter
from src.safelist import SafelistManager

# Initialize the O(1) Safelist Manager
safelist_mgr = SafelistManager()

SENSITIVE_KEYWORDS = {
    "login": 0.9,
    "secure": 0.8,
    "verify": 0.9,
    "account": 0.6,
    "update": 0.7,
    "bank": 0.8,
    "password": 1.0,
    "auth": 0.8,
    "signin": 0.8,
    "confirm": 0.7,
    "billing": 0.9,
    "wallet": 0.7,
}
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "gq", "ml", "cf", "click", "work", "support"}
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


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return float(entropy)


def _character_distribution_anomaly_score(s: str) -> float:
    if not s:
        return 0.0
    digits = sum(1 for c in s if c.isdigit())
    hyphens = s.count("-")
    alphabetic = sum(1 for c in s if c.isalpha())
    ratio_digits = digits / len(s)
    ratio_hyphen = hyphens / max(1, len(s))
    low_alpha_penalty = 0.25 if alphabetic < max(3, len(s) // 4) else 0.0
    score = (ratio_digits * 0.55) + (ratio_hyphen * 0.45) + low_alpha_penalty
    return float(min(1.0, score))


def _weighted_keyword_score(url: str, is_safelisted: bool) -> float:
    if is_safelisted:
        return 0.0
    lower_url = url.lower()
    score = sum(weight for keyword, weight in SENSITIVE_KEYWORDS.items() if keyword in lower_url)
    return float(min(1.0, score / 2.2))

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
    
    # 2. Weighted Suspicious Keyword Context
    suspicious_keyword_score = _weighted_keyword_score(url, is_safelisted=is_safelisted)
    suspicious_keyword_context = 1 if suspicious_keyword_score >= 0.4 else 0
             
    # 3. Vowel-Consonant Ratio (DGA Detection)
    # Replaces raw "URL Length" logic with phonetic string structure checks.
    vc_ratio = _vowel_consonant_ratio(ext.domain.lower())

    # 4. Domain entropy + character anomalies
    domain_entropy = _shannon_entropy(ext.domain.lower())
    char_distribution_anomaly = _character_distribution_anomaly_score(ext.domain.lower())

    # 5. Suspicious TLD
    tld = (ext.suffix or "").split(".")[-1].lower() if ext.suffix else ""
    suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
     
    return {
        "is_safelisted_base": 1 if is_safelisted else 0,
        "path_to_domain_ratio": round(path_to_domain_ratio, 4),
        "suspicious_keyword_context": suspicious_keyword_context,
        "suspicious_keyword_weighted_score": round(suspicious_keyword_score, 4),
        "vc_ratio": round(vc_ratio, 4),
        "domain_entropy": round(domain_entropy, 4),
        "char_distribution_anomaly": round(char_distribution_anomaly, 4),
        "suspicious_tld": suspicious_tld,
    }
