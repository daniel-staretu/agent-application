"""
DataSage — Interactive CLI
===========================
Run:
    python main.py

Commands inside the session
    /new        — start a new conversation session (clears short-term memory)
    /memory     — show memory statistics
    /tools      — list available tools
    /quit       — exit
"""

import os
import sys
import textwrap

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _check_api_key() -> str:
    """Validate that an Anthropic API key is available."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print(
            "\n[Error] ANTHROPIC_API_KEY is not set.\n"
            "  Option 1: Create a .env file with:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Option 2: Set it in your shell:     export ANTHROPIC_API_KEY=sk-ant-...\n"
        )
        sys.exit(1)
    return key


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║              DataSage — Data Science Agent  🔬                   ║
║                                                                  ║
║  Commands:  /new  /memory  /tools  /quit                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

TOOLS_HELP = """
Available tools (called automatically by DataSage):

  analyze_dataset      — Full EDA on a CSV / Excel file
  execute_python_code  — Run Python and return output
  list_data_files      — Discover datasets in a directory
  search_memory        — Query long-term memory from past sessions
  save_to_memory       — Persist facts / insights for future sessions

DataSage decides which tools to use based on your request.
You can hint by saying things like:
  "Analyse the file sales.csv"
  "Run this code: ..."
  "What do you remember about my dataset?"
"""


def _print_wrapped(text: str, width: int = 100) -> None:
    """Print text with basic word-wrap for long lines."""
    for line in text.split("\n"):
        if len(line) > width:
            for chunk in textwrap.wrap(line, width):
                print(chunk)
        else:
            print(line)


def main() -> None:
    _check_api_key()

    print(BANNER)

    # Late import so we get a clean error message if dependencies are missing
    try:
        from agent import DataSageAgent
    except ImportError as exc:
        print(f"[Error] Could not import agent: {exc}")
        print("Have you run:  pip install -r requirements.txt ?")
        sys.exit(1)

    agent = DataSageAgent(verbose=True)
    print(f"Session id: {agent.session_id}")

    stats = agent.memory_stats()
    if stats["total_memories"] > 0:
        print(f"Long-term memory: {stats['total_memories']} stored memories across {stats['sessions']} past session(s).")
    else:
        print("Long-term memory: empty (first run).")

    print("\nHello! I'm DataSage, your data science mentor. How can I help today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────────────
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/new":
            agent.new_session()
            print("Session cleared. Starting fresh!\n")
            continue

        if user_input.lower() == "/memory":
            stats = agent.memory_stats()
            print("\n── Memory Statistics ─────────────────────────────")
            print(f"  Total memories : {stats['total_memories']}")
            print(f"  Sessions saved : {stats['sessions']}")
            if stats["by_category"]:
                print("  By category    :")
                for cat, cnt in sorted(stats["by_category"].items()):
                    print(f"    {cat:<15} {cnt}")
            print()
            continue

        if user_input.lower() == "/tools":
            print(TOOLS_HELP)
            continue

        # ── Agent interaction ──────────────────────────────────────────────
        print("\nDataSage: ", end="", flush=True)
        try:
            response = agent.chat(user_input)
            print()
            _print_wrapped(response)
            print()
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    main()
