import tldextract

# Simulated highly optimized O(1) safelist matching the Top 10,000 domains
# (In production, this would load from a remote CSV or Redis cache)
TOP_10K_TRUSTED = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "netflix.com",
    "github.com", "aws.amazon.com", "wikipedia.org", "youtube.com", "facebook.com",
    "yahoo.com", "whatsapp.com", "instagram.com", "linkedin.com", "twitter.com",
    "chase.com", "paypal.com", "adobe.com", "cloudflare.com"
}

class SafelistManager:
    def __init__(self):
        self.trusted_domains = TOP_10K_TRUSTED
        # Note: tldextract is extremely fast and accurately parses complex TLDs (.co.uk)
        self.extract = tldextract.TLDExtract(suffix_list_urls=None)
        
    def is_safelisted(self, url: str) -> bool:
        """
        Parses the base domain (e.g. drive.google.com -> google.com) 
        and performs an O(1) set lookup to determine if it is trusted.
        """
        ext = self.extract(url)
        # Reconstruct the base root domain (e.g., "google.com")
        base_domain = f"{ext.domain}.{ext.suffix}".lower()
        
        return base_domain in self.trusted_domains
