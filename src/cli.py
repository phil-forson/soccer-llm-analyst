"""
Command-line interface for Soccer Match LLM Analyst.

Provides a terminal-based interface for querying matches and asking questions.
Uses web search and YouTube for match information (no API keys required).
"""

import sys

from .web_search_agent import search_and_summarize
from .youtube_search_agent import search_and_display_highlights


# =============================================================================
# CLI Display Helpers
# =============================================================================

def print_banner():
    """Print the welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           ⚽ Soccer Match LLM Analyst (CLI) ⚽                ║
║                                                              ║
║  Find match info and highlights using AI-powered search      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_divider():
    """Print a visual divider."""
    print("\n" + "─" * 60 + "\n")


def show_youtube_highlights(query: str):
    """
    Automatically search and display YouTube highlights of the match.
    
    Args:
        query: The match query string.
    """
    print("\n⏳ Finding match highlights...\n")
    try:
        result = search_and_display_highlights(query, use_llm=False)
        print(result)
    except Exception as e:
        print(f"❌ Error searching for highlights: {e}")


# =============================================================================
# Main CLI Logic
# =============================================================================

def handle_match_search(query: str):
    """
    Handle a match search query.
    
    1. Searches the web for match information
    2. Shows YouTube highlights
    
    Args:
        query: The user's match query.
    """
    # Search web for match info
    print("\n🌐 Searching for match information...\n")
    
    try:
        result = search_and_summarize(query, use_llm=True)
        print("🔍 Match Information:\n")
        print(result)
    except Exception as e:
        print(f"❌ Error searching the web: {e}")
    
    # Show YouTube highlights
    show_youtube_highlights(query)


def main():
    """Main entry point for the CLI application."""
    print_banner()
    
    print("Enter a match description to find highlights and info.")
    print("Examples:")
    print("  • 'Chelsea vs Barcelona'  (finds most recent match)")
    print("  • 'Arsenal vs Chelsea 2024-12-01'  (specific date)")
    print("  • 'Real Madrid vs Man City Champions League'\n")
    print("Tip: First team = home team, second team = away team\n")
    
    while True:
        print_divider()
        
        try:
            query = input("🎮 Describe the match (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('q', 'quit', 'exit'):
            print("\n👋 Goodbye!")
            break
        
        print(f"\n🔍 Searching for: {query}")
        
        try:
            handle_match_search(query)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Returning to main menu...")
            continue
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different query.\n")
            continue
    
    sys.exit(0)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
