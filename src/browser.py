"""
Browser controller module - wraps Playwright for web page interaction.
"""

import asyncio
from typing import Optional
from playwright.async_api import async_playwright, Page, BrowserContext


class BrowserController:
    """Controls a browser instance via Playwright."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self):
        """Launch the browser and create a new context + page."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()

    async def navigate(self, url: str) -> str:
        """Navigate to a URL and return the page title."""
        await self._page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)  # Let the page settle
        return await self._page.title()

    async def click(self, selector: str) -> bool:
        """Click an element identified by CSS selector."""
        try:
            await self._page.click(selector, timeout=5000)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to click '{selector}': {e}")

    async def fill(self, selector: str, text: str) -> bool:
        """Fill a form field."""
        try:
            await self._page.fill(selector, text, timeout=5000)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to fill '{selector}': {e}")

    async def type_text(self, selector: str, text: str):
        """Type text into an element (useful for search inputs)."""
        await self._page.locator(selector).press_sequentially(text, delay=50)
        await asyncio.sleep(0.3)

    async def press_key(self, key: str):
        """Press a keyboard key (Enter, Escape, etc.)."""
        await self._page.keyboard.press(key)
        await asyncio.sleep(0.5)

    async def get_text(self, selector: str = "body") -> str:
        """Get visible text content from the page or a specific element."""
        try:
            element = self._page.locator(selector)
            text = await element.inner_text(timeout=3000)
            return text.strip()
        except Exception:
            return ""

    async def get_page_text(self) -> str:
        """Get the main visible text content of the current page."""
        text = await self.get_text("body")
        # Truncate to avoid token overflow
        return text[:8000] if text else ""

    async def get_url(self) -> str:
        """Get the current page URL."""
        return self._page.url

    async def screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        return await self._page.screenshot(type="png")

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
        """Wait for an element to appear."""
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def get_clickable_elements(self) -> list[dict]:
        """Get a list of clickable elements (a, button, input, etc.) with basic info."""
        elements = []
        try:
            # Get links
            links = await self._page.eval_on_selector_all(
                "a, button, input[type='submit'], input[type='button'], [role='button']",
                """
                (els) => els.slice(0, 50).map((el, i) => ({
                    index: i,
                    tag: el.tagName.toLowerCase(),
                    text: (el.textContent || '').trim().slice(0, 60),
                    href: el.href || '',
                    selector: getSelector(el)
                }));
                function getSelector(el) {
                    if (el.id) return '#' + el.id;
                    if (el.className && typeof el.className === 'string') {
                        const cls = el.className.trim().split(/\\s+/)[0];
                        return el.tagName.toLowerCase() + '.' + cls;
                    }
                    return el.tagName.toLowerCase();
                }
                """
            )
            elements = links
        except Exception:
            pass
        return elements

    async def close(self):
        """Close the browser."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()