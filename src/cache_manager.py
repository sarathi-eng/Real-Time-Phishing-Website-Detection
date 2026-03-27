from cachetools import TTLCache
import copy

# High-Speed Caching Layer: Store up to 10,000 URLs
# Expires records strictly after 1 hour (3600 seconds) to prevent stale threat intel
prediction_cache = TTLCache(maxsize=10000, ttl=3600)

def get_cached_prediction(url: str) -> dict | None:
    """O(1) memory lookup to retrieve a cached prediction payload."""
    record = prediction_cache.get(url)
    if record:
        # Return a deepcopy to prevent mutable state bleeding across API requests
        return copy.deepcopy(record)
    return None

def set_cached_prediction(url: str, result: dict):
    """Stores the fully processed prediction result in the TTL cache."""
    # Strip processing time and cache hit flags from the stored payload to keep it clean
    store_payload = copy.deepcopy(result)
    store_payload["cache_hit"] = True 
    prediction_cache[url] = store_payload
