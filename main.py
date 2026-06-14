"""
Web AI Agent - CLI entry point.

A browser automation agent powered by LLM.
"""

import asyncio
import argparse
import os
import sys
import logging
from dotenv import load_dotenv

from src.browser import BrowserController
from src.llm import LLMClient
from src.agent import WebAgent


# Colors for terminal output
class Colors:
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════╗
║         Web AI Agent v2.0            ║
║   Browser automation powered by LLM   ║
╚══════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def validate_config(args) -> list[str]:
    """Validate configuration and return list of warnings/issues."""
    issues = []

    # Check API key
    api_key = args.api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        issues.append("LLM_API_KEY not set. Use --api-key or create a .env file.")

    # Check base URL
    base_url = args.base_url or os.getenv("LLM_BASE_URL", "")
    if not base_url:
        issues.append(
            "No LLM_BASE_URL set. Defaulting to https://api.openai.com/v1"
        )

    return issues


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web AI Agent - browser automation powered by LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Search for Python tutorials on Google"
  python main.py "Check the weather in Tokyo" --model claude-sonnet-4-20250514
  python main.py "Visit GitHub" --headless
  python main.py -i                        # Interactive mode
  python main.py "Find top news" --debug   # Debug mode with detailed logs
        """,
    )
    parser.add_argument("task", nargs="?", default=None,
                        help="The task for the AI agent to perform")
    parser.add_argument("--api-key",
                        help="LLM API key (default: from LLM_API_KEY env var or .env)")
    parser.add_argument("--base-url",
                        help="LLM API base URL (default: from LLM_BASE_URL or https://api.openai.com/v1)")
    parser.add_argument("--model",
                        help="LLM model name (default: from LLM_MODEL or gpt-4o)")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode (no visible window)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Maximum number of agent steps (default: 30)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode - enter tasks interactively")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Debug mode - show detailed decision process")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save results to a file")
    return parser.parse_args()


async def run_task(task: str, args) -> str:
    """Run the agent on a single task."""
    # Read headless setting
    headless = args.headless
    if not headless:
        env_val = os.getenv("BROWSER_HEADLESS", "false").lower()
        headless = env_val in ("true", "1", "yes")

    print(f"{Colors.CYAN}Starting browser...{Colors.RESET}")

    # Initialize components
    llm = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    browser = BrowserController(headless=headless)
    agent = WebAgent(
        llm, browser,
        max_steps=args.max_steps,
        debug=args.debug,
    )

    try:
        print(f"{Colors.BOLD}Running:{Colors.RESET} {task}")
        await browser.start()
        result = await agent.run(task)
        print(f"\n{Colors.GREEN}{Colors.BOLD}Result:{Colors.RESET}\n{result}")

        # Save result to file if requested
        if args.save:
            os.makedirs("results", exist_ok=True)
            from datetime import datetime
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:40])
            result_path = f"results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.txt"
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n{'='*60}\n{result}\n")
            print(f"{Colors.CYAN}Result saved to: {result_path}{Colors.RESET}")

        return result
    finally:
        await browser.close()
        await llm.close()


async def interactive_mode(args):
    """Run in interactive mode with history."""
    print("=" * 60)
    print(f"{Colors.BOLD}Web AI Agent - Interactive Mode{Colors.RESET}")
    print("Enter a task, or 'quit'/'exit' to stop.")
    print("Prefix with '!' to run a shell command (e.g. !help)")
    print("=" * 60)

    history: list[str] = []

    while True:
        try:
            raw = input(f"\n{Colors.CYAN}Task{Colors.RESET}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.startswith("!"):
            cmd = raw[1:]
            if cmd == "history":
                for i, h in enumerate(history, 1):
                    print(f"  {i}. {h}")
            elif cmd == "help":
                print("Commands:  quit/exit  history  !<cmd>")
            else:
                print(f"Unknown command: {cmd}")
            continue

        history.append(raw)
        await run_task(raw, args)


def main():
    load_dotenv()
    args = parse_args()

    # Console frame
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    print_banner()

    # Validate configuration
    issues = validate_config(args)
    for issue in issues:
        if "not set" in issue:
            print(f"  {Colors.RED}ERROR: {issue}{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}WARNING: {issue}{Colors.RESET}")

    # Check if API key is missing
    api_key = args.api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        print(f"\n{Colors.RED}No API key configured.{Colors.RESET}")
        print("Create a .env file with LLM_API_KEY=your-key or use --api-key.")
        sys.exit(1)

    if args.interactive:
        asyncio.run(interactive_mode(args))
    elif args.task:
        asyncio.run(run_task(args.task, args))
    else:
        print(f"\n{Colors.YELLOW}No task provided.{Colors.RESET}")
        print("Examples:")
        print('  python main.py "Search Google for Python news"')
        print("  python main.py -i")
        sys.exit(1)


if __name__ == "__main__":
    main()