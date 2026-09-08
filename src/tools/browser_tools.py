"""
Browser Tools for Expera AI.

Browser automation:
- Web scraping
- Form filling
- Navigation
- Screenshot

Usage:
    browser = BrowserTool()
    html = await browser.get("https://example.com")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BrowserResult:
    """Result of browser operation."""

    success: bool
    content: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


class BrowserTool:
    """
    Browser automation tool.

    Features:
    - HTTP requests
    - HTML parsing
    - Basic automation
    """

    def __init__(self):
        """Initialize browser tool."""
        import aiohttp
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self):
        """Get HTTP session."""
        import aiohttp
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get(
        self,
        url: str,
        headers: Optional[dict] = None,
    ) -> BrowserResult:
        """Get page content."""
        try:
            session = await self._get_session()

            default_headers = {
                "User-Agent": "ExperaAI/1.0",
            }
            if headers:
                default_headers.update(headers)

            async with session.get(url, headers=default_headers) as response:
                content = await response.text()

                return BrowserResult(
                    success=True,
                    content=content,
                    url=str(response.url),
                )

        except Exception as e:
            logger.error(f"Get failed: {e}")
            return BrowserResult(
                success=False,
                error=str(e),
            )

    async def post(
        self,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> BrowserResult:
        """Post to URL."""
        try:
            session = await self._get_session()

            async with session.post(url, data=data, json=json) as response:
                content = await response.text()

                return BrowserResult(
                    success=True,
                    content=content,
                    url=str(response.url),
                )

        except Exception as e:
            logger.error(f"Post failed: {e}")
            return BrowserResult(
                success=False,
                error=str(e),
            )

    def parse_html(
        self,
        html: str,
        selector: str,
    ) -> list[str]:
        """Parse HTML with selector."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            elements = soup.select(selector)

            return [str(el) for el in elements]

        except ImportError:
            logger.error("BeautifulSoup not available")
            return []
        except Exception as e:
            logger.error(f"Parse failed: {e}")
            return []

    def extract_links(self, html: str) -> list[str]:
        """Extract all links from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            links = []

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href:
                    links.append(href)

            return links

        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Link extraction failed: {e}")
            return []

    def extract_images(self, html: str) -> list[str]:
        """Extract all image URLs from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            images = []

            for img in soup.find_all("img", src=True):
                src = img["src"]
                if src:
                    images.append(src)

            return images

        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return []

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


# Export
__all__ = [
    "BrowserTool",
    "BrowserResult",
]