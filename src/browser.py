"""
Browser controller module - wraps Playwright for web page interaction.
"""

import asyncio
import os
from typing import Optional
from playwright.async_api import async_playwright, Page, BrowserContext


class BrowserController:
    """Controls a browser instance via Playwright."""

    def __init__(self, headless: bool = False, cookie_file: Optional[str] = None):
        self.headless = headless
        self.cookie_file = cookie_file
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

        # Restore cookies if available
        if self.cookie_file and os.path.exists(self.cookie_file):
            import json
            try:
                with open(self.cookie_file, "r") as f:
                    cookies = json.load(f)
                if cookies:
                    await self._context.add_cookies(cookies)
            except Exception:
                pass

        self._page = await self._context.new_page()

    async def get_title(self) -> str:
        """Get the current page title."""
        if self._page:
            return await self._page.title()
        return ""

    async def navigate(self, url: str, wait_until: str = "networkidle") -> str:
        """Navigate to a URL and return the page title.
        
        Args:
            url: The URL to navigate to.
            wait_until: 'load' | 'domcontentloaded' | 'networkidle' | 'commit'
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        await self._page.goto(url, wait_until=wait_until, timeout=30000)
        await self._page.wait_for_load_state("networkidle", timeout=15000)
        return await self._page.title()

    async def click(self, selector: str) -> bool:
        """Click an element identified by CSS selector."""
        try:
            await self._page.click(selector, timeout=8000)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to click '{selector}': {e}")

    async def click_xpath(self, xpath: str) -> bool:
        """Click an element identified by XPath."""
        try:
            locator = self._page.locator(f"xpath={xpath}")
            await locator.click(timeout=8000)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to click XPath '{xpath}': {e}")

    async def click_by_text(self, text: str) -> bool:
        """Click an element containing specific text."""
        try:
            locator = self._page.locator(f"text={text}")
            await locator.first.click(timeout=8000)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to click text '{text}': {e}")

    async def fill(self, selector: str, text: str, delay_ms: int = 60) -> bool:
        """Fill a form field with human-like typing delay."""
        try:
            await self._page.locator(selector).fill("")  # clear first
            await asyncio.sleep(0.2)
            await self._page.locator(selector).press_sequentially(text, delay=delay_ms)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to fill '{selector}': {e}")

    async def press_key(self, key: str):
        """Press a keyboard key (Enter, Escape, etc.)."""
        await self._page.keyboard.press(key)
        await asyncio.sleep(0.5)

    async def scroll(self, direction: str = "down", amount: int = 500):
        """Scroll the page."""
        if direction == "down":
            await self._page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            await self._page.evaluate(f"window.scrollBy(0, -{amount})")
        elif direction == "bottom":
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif direction == "top":
            await self._page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.3)

    async def select_option(self, selector: str, value: str):
        """Select an option from a dropdown/select element."""
        await self._page.select_option(selector, value)

    async def check(self, selector: str):
        """Check a checkbox or select a radio button."""
        await self._page.check(selector)

    async def uncheck(self, selector: str):
        """Uncheck a checkbox."""
        await self._page.uncheck(selector)

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
        return text[:8000] if text else ""

    async def get_url(self) -> str:
        """Get the current page URL."""
        if not self._page:
            return ""
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
        try:
            links = await self._page.eval_on_selector_all(
                "a, button, input[type='submit'], input[type='button'], [role='button']",
                """
                (els) => {
                    function getSelector(el) {
                        if (el.id) return '#' + el.id;
                        if (el.className && typeof el.className === 'string') {
                            const cls = el.className.trim().split(/\\s+/)[0];
                            return el.tagName.toLowerCase() + '.' + cls;
                        }
                        return el.tagName.toLowerCase();
                    }
                    return els.slice(0, 50).map((el, i) => ({
                        index: i,
                        tag: el.tagName.toLowerCase(),
                        text: (el.textContent || '').trim().slice(0, 60),
                        href: el.href || '',
                        selector: getSelector(el)
                    }));
                }
                """
            )
            return links
        except Exception:
            return []

    async def describe_page(self) -> str:
        """Get a textual description of the current page (better than screenshot for LLM)."""
        parts = []
        parts.append(f"URL: {await self.get_url()}")
        parts.append(f"Title: {await self.get_title()}")
        text = await self.get_page_text()
        parts.append(f"Content:\n{text[:3000]}")
        return "\n".join(parts)

    async def save_cookies(self, filepath: str):
        """Save current cookies to a JSON file."""
        if not self._context:
            return
        cookies = await self._context.cookies()
        import json
        with open(filepath, "w") as f:
            json.dump(cookies, f, indent=2)

    async def close(self):
        """Close the browser."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()