from cachetools import TTLCache
import time

# ---------------------------------------------------------
# SMART CACHE STRATEGY: Resolving the "Cache Bias Effect"
# ---------------------------------------------------------

# 1. Deep Scan Cache (TTL = 4 Hours / 14400 seconds)
# Stores extremely accurate, fully saturated Semantic & Threat Intel results.
deep_scan_cache = TTLCache(maxsize=15000, ttl=14400)

# 2. Degraded / Fast-Path Cache (TTL = 60 Seconds)
# Stores purely Lexical or Timeout-degraded results. Forces the system
# to attempt a live Deep Scan roughly every minute if under attack, ensuring
# temporary "safe" or "timeout" modes never falsely persist.
degraded_cache = TTLCache(maxsize=10000, ttl=60)

def get_cached_prediction(url: str):
    """
    Retrieves the prediction, strictly preferring the high-fidelity deep scan cache.
    """
    if url in deep_scan_cache:
        result = deep_scan_cache[url].copy()
        result["cache_tier"] = "deep_long_term"
        return result
        
    if url in degraded_cache:
        result = degraded_cache[url].copy()
        result["cache_tier"] = "degraded_volatile"
        return result
        
    return None

def set_cached_prediction(url: str, result: dict):
    """
    Dynamically routes caching to prevent Stale Predictions and Cache Bias.
    """
    depth = result.get("analysis_depth", "")
    
    if result.get("degraded_mode") or "fast_path" in depth:
        # Route volatile, unverified, or timed-out results to the 60-second cache
        degraded_cache[url] = result
    else:
        # Route fully validated deep-scan network results to the 4-hour cache
        deep_scan_cache[url] = result
        
        # Actively purge the volatile cache to bust any stale degraded references
        degraded_cache.pop(url, None)
