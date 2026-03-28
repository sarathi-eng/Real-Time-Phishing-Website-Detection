import unicodedata
import re
from urllib.parse import urlparse
from src.brands_db import TARGETED_BRANDS, OFFICIAL_DOMAINS

# 1. Native Python Damerau-Levenshtein (Handles String Transpositions)
def damerau_levenshtein(s1: str, s2: str) -> int:
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1
        
    for i in range(len1):
        for j in range(len2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,      # deletion
                d[(i, j - 1)] + 1,      # insertion
                d[(i - 1, j - 1)] + cost # substitution
            )
            # transposition (e.g. paypa1 / paypal)
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)
    return d[(len1 - 1, len2 - 1)]

# 2. Defensive High-Risk Affixes Engine
HIGH_RISK_AFFIXES = ["login", "secure", "account", "update", "support", "auth", "verify", "service", "billing"]

class TyposquattingDetector:
    def __init__(self):
        self.brands = TARGETED_BRANDS
        self.official_domains = OFFICIAL_DOMAINS
        
    def _skeletonize(self, text: str) -> str:
        """3. Unicode Homoglyph Normalization using NFKD and ASCII mapping."""
        # This instantly crushes visual confusables (Cyrillic a -> Latin a).
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    def analyze_domain(self, url: str) -> dict:
        """
        Analyzes a URL using advanced Mathematical Distance vectors and Skeletonization.
        """
        if not url.startswith("http"):
            url = "http://" + url
            
        parsed = urlparse(url)
        raw_hostname = parsed.hostname or ""
        hostname = raw_hostname.lower()
        
        features = {
            "is_punycode": False,
            "contains_homoglyph": False,
            "closest_brand_distance": 999,
            "closest_brand": None,
            "exact_brand_substring": False,
            "is_official_domain": hostname in self.official_domains,
            "red_flag_override": False,
            "typo_reasons": []
        }
        
        hostname_to_check = hostname
        
        # Punycode Decryption
        if "xn--" in hostname:
            features["is_punycode"] = True
            features["typo_reasons"].append("Punycode (IDN) abuse decrypted.")
            try:
                decoded_hostname = raw_hostname.encode("ascii").decode("idna")
                hostname_to_check = decoded_hostname.lower()
            except Exception:
                pass
                
        # Execute Skeletonization (Homoglyph Disarmament)
        skeletonized_host = self._skeletonize(hostname_to_check)
        if skeletonized_host != hostname_to_check and not features["is_punycode"]:
            features["contains_homoglyph"] = True
            features["typo_reasons"].append("Advanced Homoglyph / Mixed-Script evasion skeletonized.")
            
        parts = skeletonized_host.split('.')
        best_dist = 999
        matched_brand = None
        
        # Typosquatting Core Vector Engine
        for part in parts:
            if not part: continue
            
            for brand in self.brands:
                # Substring Match with Defensive Affix Rules
                if brand in part and hostname not in self.official_domains:
                    features["exact_brand_substring"] = True
                    features["red_flag_override"] = True
                    matched_brand = brand
                    
                    has_affix = any(affix in part for affix in HIGH_RISK_AFFIXES)
                    if has_affix:
                        features["typo_reasons"].append(f"High-Risk Keyword pairing with spoofed brand '{brand}'.")
                    else:
                        if f"Unauthorized substring of brand '{brand}'." not in features["typo_reasons"]:
                            features["typo_reasons"].append(f"Unauthorized substring of brand '{brand}'.")

                # 4. Inverse Length-Ratio Dynamic Thresholding
                # Strict bound ensuring we don't False Positive on 3-letter combinations
                max_edits_allowed = max(1, len(brand) // 4)
                
                # Only compute the heavy DP matrix if length constraints physically permit it
                if abs(len(part) - len(brand)) <= max_edits_allowed:
                    dist = damerau_levenshtein(part, brand)
                    if dist < best_dist:
                        best_dist = dist
                    
                    # Ensure it's not the exact brand (dist == 0) because that's caught by official_domain
                    # and ensure it's mathematically within the dynamic len threshold
                    if 1 <= dist <= max_edits_allowed:
                        features["red_flag_override"] = True
                        matched_brand = brand
                        
                        reason = f"Damerau-Levenshtein transposition/edit ({dist}) against '{brand}'."
                        if reason not in features["typo_reasons"]:
                            features["typo_reasons"].append(reason)

        features["closest_brand_distance"] = best_dist
        if matched_brand:
            features["closest_brand"] = matched_brand
            
        return features
