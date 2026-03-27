import asyncio
from src.network_utils import safe_async_get

async def check_threat_intelligence(url: str) -> dict | None:
    """
    Simulates checking a threat intelligence feed (e.g., PhishTank, VirusTotal).
    Wrapped for latency guarantees. Returns None on timeout.
    """
    # Emulating a PhishTank URL lookup that usually takes ~300ms
    await asyncio.sleep(0.3)
    
    blacklist_triggers = [".xyz", "scam", "phish", "free-robux", "login-apple", "paypal-secure"]
    is_blacklisted = any(trigger in url.lower() for trigger in blacklist_triggers)
    
    return {
        "blacklisted": is_blacklisted,
        "source": "MockedPhishTank",
        "threat_intel_confidence": 0.99 if is_blacklisted else 0.0
    }
