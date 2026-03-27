from src.network_utils import safe_async_get

async def fetch_html_async(url: str, timeout: float = 1.5) -> str | None:
    """Async scraper wrapped behind SRE resilient network client to enforce max 1.5s latency."""
    return await safe_async_get(url, timeout=timeout)
