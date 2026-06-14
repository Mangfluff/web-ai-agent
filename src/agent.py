"""
AI Agent core - drives the browser using LLM decision-making (ReAct loop).
"""

import asyncio
import json
import re
import os
import logging
from datetime import datetime
from src.browser import BrowserController
from src.llm import LLMClient

logger = logging.getLogger("web-agent")

# System prompt that defines the agent's behavior
SYSTEM_PROMPT = """You are an AI web agent that controls a browser to complete tasks.
You receive the current page state and respond with ONE action at a time.

Available actions (respond with raw JSON, no markdown):

{"action": "navigate", "url": "<full_url>"}
  - Navigate to a URL. Include the protocol (https://).

{"action": "click", "selector": "<css_selector>"}
  - Click an element by CSS selector. Use tag, #id, .class, or [attr="value"].

{"action": "click_xpath", "xpath": "<xpath_expression>"}
  - Click an element by XPath. Use when CSS selector fails.

{"action": "click_text", "text": "<visible_text>"}
  - Click an element by its visible text content. Good for buttons and links.

{"action": "fill", "selector": "<css_selector>", "text": "<text>"}
  - Type text into an input/textarea field.

{"action": "check", "selector": "<css_selector>"}
  - Check a checkbox or select a radio button.

{"action": "uncheck", "selector": "<css_selector>"}
  - Uncheck a checkbox.

{"action": "select", "selector": "<css_selector>", "value": "<option_value>"}
  - Select an option from a dropdown/select element.

{"action": "press", "key": "<key_name>"}
  - Press a keyboard key: "Enter", "Escape", "ArrowDown", "Tab".

{"action": "extract"}
  - Extract the current page text content. Use this to read what's on the page.

{"action": "scroll", "direction": "down", "amount": 500}
  - Scroll the page. direction: "down", "up", "bottom", "top". amount: pixels.

{"action": "wait", "ms": <milliseconds>}
  - Wait for a period of time (1000 = 1 second).

{"action": "describe"}
  - Get a full summary of the current page (URL, title, content).

{"action": "done", "result": "<final_answer>"}
  - Task is complete. Provide the final result to the user.

Rules:
1. Always respond with EXACTLY ONE JSON action, no extra text or markdown.
2. First: navigate to a relevant starting page (Google, Wikipedia, etc.).
3. Then: use "extract" or "describe" to read the page content before making decisions.
4. For clicking, prefer CSS selectors first, then click_text, then click_xpath as fallback.
5. If an action fails, try a different approach (click_text, navigate back, etc.).
6. For search tasks: navigate to the search engine, fill the search box, press Enter, examine results.
7. When you find what you need, use "done" with a clear result summary.
8. Always verify the page is loaded before interacting. Use "wait" if needed.
9. If stuck on a page, try navigating to a fresh starting point.
"""


def _fmt(step, total, msg):
    return f"[{step}/{total}] {msg}"


class WebAgent:
    """AI agent that controls a browser to perform web tasks."""

    def __init__(self, llm_client: LLMClient, browser: BrowserController,
                 max_steps: int = 30, debug: bool = False):
        self.llm = llm_client
        self.browser = browser
        self.max_steps = max_steps
        self.debug = debug
        self.history: list[dict] = []
        self._started = False
        self._log_file = None

    def _log(self, msg: str, level: str = "info"):
        """Log a message to both logger and optional log file."""
        print(msg)
        if self._log_file:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_file.write(f"[{timestamp}] {msg}\n")
            self._log_file.flush()

    def _build_messages(self, task: str, page_info: dict) -> list[dict]:
        """Build the message list for the LLM chat completion."""
        url = page_info.get("url", "")
        title = page_info.get("title", "")
        text = page_info.get("text", "")

        if not url or url == "about:blank":
            state_msg = "No page loaded yet. Start by navigating to a relevant website."
        else:
            state_msg = (
                f"Current URL: {url}\n"
                f"Page title: {title}\n"
                f"Page text (first 3000 chars):\n{text[:3000]}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}\n\n{state_msg}"},
        ]

        # Add conversation history
        for entry in self.history:
            messages.append({"role": "assistant", "content": entry["action"]})
            messages.append({"role": "user", "content": entry["observation"]})

        return messages

    async def run(self, task: str) -> str:
        """Run the agent on a given task."""
        self.history = []
        self._started = True

        # Setup log file
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w]', '_', task[:30])
        log_path = os.path.join(
            log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.log"
        )
        self._log_file = open(log_path, "w", encoding="utf-8")

        self._log(f"Task: {task}")
        self._log(f"Log file: {log_path}")

        try:
            for step in range(1, self.max_steps + 1):
                self._log(f"\n{'='*60}")
                self._log(f"Step {step}/{self.max_steps}")

                # Get current page state (async)
                page_info = await self._get_page_state()

                # Ask LLM for the next action
                messages = self._build_messages(task, page_info)
                try:
                    response = await self.llm.chat(messages)
                except Exception as e:
                    error_msg = f"LLM API error: {e}"
                    self._log(f"  ERROR: {error_msg}")
                    return f"Task failed at step {step}: {error_msg}"

                # Parse the JSON action
                action = self._parse_action(response)
                if not action:
                    self._log(f"  Failed to parse action from: {response[:200]}")
                    self.history.append({
                        "action": response,
                        "observation": "ERROR: Invalid action format. Respond with raw JSON only.",
                    })
                    continue

                self._log(f"  Action: {json.dumps(action, ensure_ascii=False)}")

                if self.debug:
                    self._log(f"  Raw LLM response: {response[:300]}")

                # Execute the action
                observation = await self._execute_action(action)

                self._log(f"  Result: {observation[:300]}")

                # Check if done
                if action.get("action") == "done":
                    self._log(f"\n{'='*60}")
                    self._log("Task completed!")
                    return action.get("result", "Task finished.")

                # Store in history
                self.history.append({
                    "action": response,
                    "observation": observation[:500],
                })

            return f"Reached maximum steps ({self.max_steps}). The task may not be complete."

        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None

    async def _get_page_state(self) -> dict:
        """Get current page state (async)."""
        url = await self.browser.get_url()
        info = {"url": url, "title": "", "text": ""}
        if url and url != "about:blank":
            info["title"] = await self.browser.get_title()
            info["text"] = await self.browser.get_page_text()
        return info

    def _parse_action(self, text: str) -> dict | None:
        """Parse a JSON action from the LLM response with robust fallbacks."""
        text = text.strip()
        # Remove code block markers
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?\s*```$', '', text)
        text = text.strip()

        # Strategy 1: Direct JSON parse
        try:
            action = json.loads(text)
            if "action" in action:
                return action
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find first {..} block with balanced braces
        brace_depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if start == -1:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if "action" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    start = -1

        # Strategy 3: Regex fallback for simple cases
        match = re.search(r'"action"\s*:\s*"(\w+)"', text)
        if match:
            action = match.group(1)
            result = {"action": action}
            # Try to extract url
            url_match = re.search(r'"url"\s*:\s*"([^"]+)"', text)
            if url_match:
                result["url"] = url_match.group(1)
            # Try to extract selector
            sel_match = re.search(r'"selector"\s*:\s*"([^"]+)"', text)
            if sel_match:
                result["selector"] = sel_match.group(1)
            # Try to extract text
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', text)
            if text_match:
                result["text"] = text_match.group(1)
            # Try to extract result
            res_match = re.search(r'"result"\s*:\s*"([^"]+)"', text)
            if res_match:
                result["result"] = res_match.group(1)
            return result

        return None

    async def _execute_action(self, action: dict) -> str:
        """Execute a parsed action and return the observation."""
        action_type = action.get("action", "")

        try:
            if action_type == "navigate":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                title = await self.browser.navigate(url)
                return f"Navigated to {url}. Page title: {title}"

            elif action_type == "click":
                selector = action.get("selector", "")
                await self.browser.click(selector)
                url = await self.browser.get_url()
                return f"Clicked '{selector}'. Current URL: {url}"

            elif action_type == "click_xpath":
                xpath = action.get("xpath", "")
                await self.browser.click_xpath(xpath)
                url = await self.browser.get_url()
                return f"Clicked XPath '{xpath}'. Current URL: {url}"

            elif action_type == "click_text":
                text = action.get("text", "")
                await self.browser.click_by_text(text)
                url = await self.browser.get_url()
                return f"Clicked text '{text}'. Current URL: {url}"

            elif action_type == "fill":
                selector = action.get("selector", "")
                text = action.get("text", "")
                await self.browser.fill(selector, text)
                return f"Filled '{selector}' with text."

            elif action_type == "check":
                selector = action.get("selector", "")
                await self.browser.check(selector)
                return f"Checked '{selector}'."

            elif action_type == "uncheck":
                selector = action.get("selector", "")
                await self.browser.uncheck(selector)
                return f"Unchecked '{selector}'."

            elif action_type == "press":
                key = action.get("key", "Enter")
                await self.browser.press_key(key)
                return f"Pressed key: {key}"

            elif action_type == "extract":
                page_info = await self._get_page_state()
                return (
                    f"URL: {page_info['url']}\n"
                    f"Title: {page_info['title']}\n"
                    f"Content:\n{page_info['text'][:2000]}"
                )

            elif action_type == "describe":
                desc = await self.browser.describe_page()
                return desc[:3000]

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                amount = action.get("amount", 500)
                await self.browser.scroll(direction, amount)
                return f"Scrolled {direction}."

            elif action_type == "select":
                selector = action.get("selector", "")
                value = action.get("value", "")
                await self.browser.select_option(selector, value)
                return f"Selected '{value}' in '{selector}'."

            elif action_type == "wait":
                ms = action.get("ms", 1000)
                await asyncio.sleep(ms / 1000)
                return f"Waited {ms}ms."

            elif action_type == "done":
                return action.get("result", "Done")

            else:
                return f"Unknown action: {action_type}"

        except Exception as e:
            return f"Error executing {action_type}: {e}"