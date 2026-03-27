from bs4 import BeautifulSoup
from urllib.parse import urlparse

SENSITIVE_KEYWORDS = {"login", "update account", "bank", "paypal", "microsoft", "verify", "secure", "signin", "password", "support"}

def extract_semantic_features(html: str, url: str) -> dict:
    """
    Parses HTML to extract intent-revealing semantic features.
    """
    if not html:
        return {
            'brand_spoofing_flag': 0,
            'suspicious_form_action': 0,
            'hidden_iframes_scripts': 0,
            'external_link_ratio': 0.0
        }
        
    soup = BeautifulSoup(html, 'html.parser')
    parsed_url = urlparse(url)
    domain_text = parsed_url.hostname.lower() if parsed_url.hostname else ""
    
    # Combine all textual data
    page_text = soup.get_text(separator=' ', strip=True).lower()
    title_text = soup.title.string.lower() if soup.title and soup.title.string else ""
    combined_text = page_text + " " + title_text
    
    # 1. Brand Spoofing Check (Looking for keywords if domain doesn't contain them)
    brand_spoofing_flag = 0
    found_keywords = [kw for kw in SENSITIVE_KEYWORDS if kw in combined_text]
    if found_keywords:
        # Check if the domain itself is actually related to the brand
        brand_in_domain = any(kw in domain_text for kw in found_keywords)
        if not brand_in_domain:
            brand_spoofing_flag = 1
            
    # 2. Form Analysis
    suspicious_form_action = 0
    for form in soup.find_all('form'):
        has_password = form.find('input', type='password') is not None
        action = form.get('action', '').lower()
        
        # If posting to an external HTTP domain that doesn't match the current hostname
        if has_password and action.startswith('http') and domain_text not in action:
            suspicious_form_action = 1
            break
            
    # 3. Hidden Elements
    hidden_count = 0
    hidden_count += len(soup.find_all(style=lambda value: value and 'display: none' in value.lower()))
    hidden_count += len(soup.find_all(style=lambda value: value and 'visibility:hidden' in value.lower()))
    
    # Some phishers use hidden iframes
    hidden_count += len(soup.find_all('iframe', style=lambda value: value and ('display:none' in value.lower() or 'width:0' in value.lower())))
    
    # 4. External Links Ratio
    internal_links = 0
    external_links = 0
    for link in soup.find_all('a', href=True):
        href = link['href'].lower()
        if href.startswith('http'):
            if domain_text in href:
                internal_links += 1
            else:
                external_links += 1
        elif href.startswith('/') or not href.startswith('#'):
            internal_links += 1
            
    total_links = internal_links + external_links
    external_link_ratio = (external_links / total_links) if total_links > 0 else 0.0
    
    return {
        'brand_spoofing_flag': brand_spoofing_flag,
        'suspicious_form_action': suspicious_form_action,
        'hidden_iframes_scripts': hidden_count,
        'external_link_ratio': round(external_link_ratio, 4)
    }
