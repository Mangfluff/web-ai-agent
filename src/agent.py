"""
AI Agent core - drives the browser using LLM decision-making (ReAct loop).
"""

import json
import re
from src.browser import BrowserController
from src.llm import LLMClient

# System prompt that defines the agent's behavior
SYSTEM_PROMPT = """You are an AI web agent that controls a browser to complete tasks.
You receive the current page state and respond with ONE action at a time.

Available actions (respond with raw JSON, no markdown):

{"action": "navigate", "url": "<full_url>"}
  - Navigate to a URL.

{"action": "click", "selector": "<css_selector>"}
  - Click an element on the page. Use simple selectors like tag, #id, .class, or [attr="value"].

{"action": "fill", "selector": "<css_selector>", "text": "<text>"}
  - Type text into an input/textarea field.

{"action": "press", "key": "<key_name>"}
  - Press a keyboard key: "Enter", "Escape", "ArrowDown", "Tab".

{"action": "extract"}
  - Extract the current page text content and report what you find.

{"action": "screenshot"}
  - Take a screenshot (for visual inspection). Use this when you need to see the page layout.

{"action": "wait", "ms": <milliseconds>}
  - Wait for a period of time.

{"action": "done", "result": "<final_answer>"}
  - Task is complete. Provide the final result to the user.

Rules:
1. Always respond with EXACTLY ONE JSON action, no extra text.
2. Before clicking or filling, make sure you're on the right page.
3. If you're unsure what's on the page, use "extract" to read the text.
4. Use "navigate" to go to a starting page like Google or a specific site.
5. For search tasks, navigate to Google, fill the search box, press Enter, then examine results.
6. Keep track of what you've done. If something fails, try an alternative approach.
7. When the task is done, use the "done" action with a clear summary.
8. Use CSS selectors that actually exist on the page. Tag names (a, button, input) or common selectors work best.
"""


class WebAgent:
    """AI agent that controls a browser to perform web tasks."""

    def __init__(self, llm_client: LLMClient, browser: BrowserController, max_steps: int = 30):
        self.llm = llm_client
        self.browser = browser
        self.max_steps = max_steps
        self.history: list[dict] = []

    def _build_messages(self, task: str, page_info: dict) -> list[dict]:
        """Build the message list for the LLM chat completion."""
        state = (
            f"Current URL: {page_info.get('url', 'N/A')}\n"
            f"Page title: {page_info.get('title', 'N/A')}\n"
            f"Page text (first 3000 chars):\n{page_info.get('text', '')[:3000]}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}\n\nInitial state - no page loaded yet. Start by navigating to a relevant website."},
        ]

        # Add history
        for entry in self.history:
            messages.append({"role": "assistant", "content": entry["action"]})
            messages.append({"role": "user", "content": entry["observation"]})

        # Add current state
        messages.append({"role": "assistant", "content": "I'll analyze the current state and decide the next action."})
        messages.append({"role": "user", "content": f"Current page state:\n{state}\n\nWhat is the next action?"})

        return messages

    async def run(self, task: str) -> str:
        """Run the agent on a given task."""
        self.history = []

        for step in range(1, self.max_steps + 1):
            print(f"\n{'='*60}")
            print(f"Step {step}/{self.max_steps}")

            # Get current page state
            page_info = {
                "url": await self.browser.get_url(),
                "title": "",
                "text": "",
            }
            if page_info["url"] and page_info["url"] != "about:blank":
                page_info["title"] = await self.browser._page.title()
                page_info["text"] = await self.browser.get_page_text()

            # Ask LLM for the next action
            messages = self._build_messages(task, page_info)
            try:
                response = self.llm.chat(messages)
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                print(f"  {error_msg}")
                return f"Task failed at step {step}: {error_msg}"

            # Parse the JSON action
            action = self._parse_action(response)
            if not action:
                print(f"  Failed to parse action from: {response[:200]}")
                self.history.append({
                    "action": response,
                    "observation": "ERROR: Invalid action format. Respond with raw JSON only.",
                })
                continue

            print(f"  Action: {json.dumps(action, ensure_ascii=False)}")

            # Execute the action
            observation = await self._execute_action(action)

            print(f"  Observation: {observation[:200]}")

            # Check if done
            if action.get("action") == "done":
                print(f"\n{'='*60}")
                print("Task completed!")
                return action.get("result", "Task finished.")

            # Store in history
            self.history.append({
                "action": response,
                "observation": observation[:500],
            })

        return f"Reached maximum steps ({self.max_steps}). The task may not be complete."

    def _parse_action(self, text: str) -> dict | None:
        """Parse a JSON action from the LLM response."""
        # Try direct JSON parsing
        text = text.strip()
        # Remove code block markers if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            action = json.loads(text)
            if "action" in action:
                return action
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
        if match:
            try:
                action = json.loads(match.group())
                if "action" in action:
                    return action
            except json.JSONDecodeError:
                pass

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

            elif action_type == "fill":
                selector = action.get("selector", "")
                text = action.get("text", "")
                await self.browser.fill(selector, text)
                return f"Filled '{selector}' with text."

            elif action_type == "press":
                key = action.get("key", "Enter")
                await self.browser.press_key(key)
                return f"Pressed key: {key}"

            elif action_type == "extract":
                text = await self.browser.get_page_text()
                url = await self.browser.get_url()
                title = await self.browser._page.title()
                return f"URL: {url}\nTitle: {title}\nContent:\n{text[:2000]}"

            elif action_type == "wait":
                ms = action.get("ms", 1000)
                import asyncio
                await asyncio.sleep(ms / 1000)
                return f"Waited {ms}ms."

            elif action_type == "screenshot":
                return "Screenshot captured (visual analysis not available in text mode). Use 'extract' to read page content."

            elif action_type == "done":
                return action.get("result", "Done")

            else:
                return f"Unknown action: {action_type}"

        except Exception as e:
            return f"Error executing {action_type}: {e}"