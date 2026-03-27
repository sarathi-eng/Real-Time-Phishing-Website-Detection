import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def safe_async_get(url: str, timeout: float = 1.5) -> Optional[str]:
    """
    Highly resilient SRE wrapper for outbound HTTP requests.
    Enforces strict timeouts and catches all connection exceptions.
    Returns the decoded text payload if successful, or None if the network fails.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    if not url.startswith('http'):
        url = 'http://' + url
        
    try:
        async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException:
        logger.warning(f"Aggressive Timeout (>{timeout}s) triggered for {url}")
        return None
    except (httpx.ConnectError, httpx.RequestError):
        logger.warning(f"Connection Error while fetching {url}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected network failure for {url}: {e}")
        return None
