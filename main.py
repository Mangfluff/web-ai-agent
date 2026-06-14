"""
Web AI Agent - CLI entry point.

A browser automation agent powered by LLM.
"""

import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv

from src.browser import BrowserController
from src.llm import LLMClient
from src.agent import WebAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web AI Agent - browser automation powered by LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a task
  python main.py "Search for Python tutorials on Google"

  # Use a specific model
  python main.py "Check the weather in Tokyo" --model claude-sonnet-4-20250514

  # Headless mode (no visible browser)
  python main.py "Visit GitHub" --headless
        """,
    )
    parser.add_argument("task", nargs="?", default=None,
                        help="The task for the AI agent to perform")
    parser.add_argument("--api-key", help="LLM API key (default: from .env or LLM_API_KEY env var)")
    parser.add_argument("--base-url", help="LLM API base URL (default: from .env or OpenAI)")
    parser.add_argument("--model", help="LLM model name (default: from .env or gpt-4o)")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode (no visible window)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Maximum number of agent steps (default: 30)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode - enter tasks interactively")
    return parser.parse_args()


async def run_task(task: str, args):
    """Run the agent on a single task."""
    # Read headless setting
    headless = args.headless
    if not headless:
        env_val = os.getenv("BROWSER_HEADLESS", "false").lower()
        headless = env_val in ("true", "1", "yes")

    # Initialize components
    llm = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    browser = BrowserController(headless=headless)
    agent = WebAgent(llm, browser, max_steps=args.max_steps)

    try:
        print(f"Starting browser...")
        await browser.start()
        print(f"Running task: {task}")
        result = await agent.run(task)
        print(f"\nFinal result:\n{result}")
        return result
    finally:
        await browser.close()
        llm.close()


async def interactive_mode(args):
    """Run in interactive mode."""
    print("=" * 60)
    print("Web AI Agent - Interactive Mode")
    print("Enter a task, or 'quit' to exit.")
    print("=" * 60)

    while True:
        task = input("\nTask: ").strip()
        if not task or task.lower() in ("quit", "exit", "q"):
            break
        await run_task(task, args)


def main():
    load_dotenv()
    args = parse_args()

    if args.interactive:
        asyncio.run(interactive_mode(args))
    elif args.task:
        asyncio.run(run_task(args.task, args))
    else:
        print("Please provide a task or use --interactive/-i mode.")
        print("Examples:")
        print("  python main.py \"Search Google for Python news\"")
        print("  python main.py -i")
        sys.exit(1)


if __name__ == "__main__":
    main()