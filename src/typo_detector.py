import unicodedata
from urllib.parse import urlparse

# Use python-Levenshtein for rapid C-level edit distance computation
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback to pure python if library isn't available (though it should be installed)
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

from src.brands_db import TARGETED_BRANDS, OFFICIAL_DOMAINS

class TyposquattingDetector:
    def __init__(self):
        self.brands = TARGETED_BRANDS
        self.official_domains = OFFICIAL_DOMAINS
        
    def analyze_domain(self, url: str) -> dict:
        """
        Analyzes a URL for Punycode, Homoglyph, and fuzzy Typosquatting attacks.
        Returns a dictionary of structured features.
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
        
        # 1. Punycode Detection & Decoding
        if "xn--" in hostname:
            features["is_punycode"] = True
            features["typo_reasons"].append("Punycode (IDN) abuse detected (xn-- prefix).")
            try:
                # Decode to actual unicode representation to see the spoofed characters
                decoded_hostname = raw_hostname.encode("ascii").decode("idna")
                hostname_to_check = decoded_hostname.lower()
            except Exception:
                hostname_to_check = hostname
        else:
            hostname_to_check = hostname
            
        # 2. Homoglyph Checks (Mixed Scripts / Confusables)
        # If the domain contains non-ASCII characters that aren't punycode, or decoded punycode
        if not all(ord(c) < 128 for c in hostname_to_check):
            # Check script blocks. A standard domain shouldn't mix deeply distinct scripts
            scripts = set()
            for char in hostname_to_check:
                if char.isalpha():
                    try:
                        scripts.add(unicodedata.name(char).split()[0]) # E.g., 'LATIN', 'CYRILLIC', 'GREEK'
                    except ValueError:
                        pass
            if len(scripts) > 1 and "LATIN" in scripts:
                features["contains_homoglyph"] = True
                features["typo_reasons"].append("Homoglyph attack detected (Mixed unicode scripts).")
                
        # 3. Fuzzy Matching (Edit Distance) & Substrings
        # Extract just the main domain entity (strip TLDs and wide subdomains for the fuzzy check)
        parts = hostname_to_check.split('.')
        # Check all parts (subdomains + sld) against the brand list
        
        for part in parts:
            if not part: continue
            
            # Substring check
            for brand in self.brands:
                if brand in part and hostname not in self.official_domains:
                    features["exact_brand_substring"] = True
                    features["typo_reasons"].append(f"Suspicious substring: URL contains target brand '{brand}' but is not official.")
                    features["red_flag_override"] = True
                    features["closest_brand"] = brand
                    
                # Edit Distance (Typosquatting like netfflix vs netflix)
                # Only compare if lengths are somewhat similar to save compute
                if abs(len(part) - len(brand)) <= 2:
                    dist = levenshtein_distance(part, brand)
                    if dist < features["closest_brand_distance"]:
                        features["closest_brand_distance"] = dist
                    
                    # If distance is 1 or 2, and it's NOT the exact brand (which would make distance 0)
                    if 1 <= dist <= 2:
                        features["red_flag_override"] = True
                        features["closest_brand"] = brand
                        if f"Typosquatting detected: Edit distance {dist} to brand '{brand}'" not in features["typo_reasons"]:
                            features["typo_reasons"].append(f"Typosquatting detected: Edit distance {dist} to brand '{brand}'.")

        return features
