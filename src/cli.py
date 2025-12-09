"""
Command-line interface for Soccer Match LLM Analyst.

Provides a terminal-based interface for querying sports information.
Uses a smart query parser to understand intent and route appropriately.
"""

import sys
from datetime import datetime

from .query_parser_agent import parse_query, should_fetch_highlights, QueryIntent
from .web_search_agent import search_with_rag
from .youtube_search_agent import search_and_display_highlights_with_metadata


# =============================================================================
# CLI Display Helpers
# =============================================================================

def print_banner():
    """Print the welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ⚽ Soccer LLM Analyst (Smart Search) ⚽             ║
║                                                              ║
║  Ask anything about football - results, lineups, news, etc. ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_divider():
    """Print a visual divider."""
    print("\n" + "─" * 60 + "\n")


def format_match_summary(match_metadata: dict) -> str:
    """Format match metadata into a nice summary with key moments."""
    if not match_metadata:
        return ""
    
    home = match_metadata.get("home_team", "Unknown")
    away = match_metadata.get("away_team", "Unknown")
    score = match_metadata.get("score", "?-?")
    match_date = match_metadata.get("match_date", "")
    key_moments = match_metadata.get("key_moments", [])
    man_of_match = match_metadata.get("man_of_the_match")
    match_summary = match_metadata.get("match_summary")
    
    lines = []
    
    lines.append("")
    lines.append("╔" + "═" * 58 + "╗")
    lines.append(f"║  📊 MATCH SUMMARY{' ' * 40}║")
    lines.append("╠" + "═" * 58 + "╣")
    
    score_line = f"║  {home} {score} {away}"
    score_line += " " * max(0, 59 - len(score_line)) + "║"
    lines.append(score_line)
    
    if match_date:
        date_line = f"║  📅 {match_date}"
        date_line += " " * max(0, 59 - len(date_line)) + "║"
        lines.append(date_line)
    
    lines.append("╠" + "═" * 58 + "╣")
    
    if key_moments:
        lines.append(f"║  ⚡ KEY MOMENTS{' ' * 42}║")
        lines.append("║" + " " * 58 + "║")
        
        event_emojis = {
            "GOAL": "⚽", "goal": "⚽",
            "RED_CARD": "🟥", "red_card": "🟥",
            "YELLOW_CARD": "🟨", "yellow_card": "🟨",
            "PENALTY": "🎯", "penalty": "🎯",
            "OWN_GOAL": "😬", "own_goal": "😬",
            "VAR": "📺", "var": "📺",
            "SAVE": "🧤", "save": "🧤",
            "SUBSTITUTION": "🔄", "substitution": "🔄",
        }
        
        for moment in key_moments[:6]:
            minute = moment.get("minute", "?")
            event = moment.get("event", "EVENT")
            desc = moment.get("description", "")
            
            emoji = event_emojis.get(event, "📌")
            moment_text = f"{emoji} {minute}' - {desc}"
            if len(moment_text) > 54:
                moment_text = moment_text[:51] + "..."
            
            moment_line = f"║  {moment_text}"
            moment_line += " " * max(0, 59 - len(moment_line)) + "║"
            lines.append(moment_line)
        
        lines.append("║" + " " * 58 + "║")
    
    if man_of_match:
        lines.append("╠" + "═" * 58 + "╣")
        motm_line = f"║  🌟 Man of the Match: {man_of_match}"
        motm_line += " " * max(0, 59 - len(motm_line)) + "║"
        lines.append(motm_line)
    
    if match_summary:
        lines.append("╠" + "═" * 58 + "╣")
        lines.append(f"║  📝 SUMMARY{' ' * 46}║")
        
        words = match_summary.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= 52:
                current_line += (" " if current_line else "") + word
            else:
                sum_line = f"║  {current_line}"
                sum_line += " " * max(0, 59 - len(sum_line)) + "║"
                lines.append(sum_line)
                current_line = word
        if current_line:
            sum_line = f"║  {current_line}"
            sum_line += " " * max(0, 59 - len(sum_line)) + "║"
            lines.append(sum_line)
    
    lines.append("╚" + "═" * 58 + "╝")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main Query Handler
# =============================================================================

def handle_query(query: str):
    """
    Handle a user query with smart intent detection.
    
    The flow is:
    1. Parse the query to understand intent
    2. Search the web for relevant information
    3. Summarize based on intent
    4. Only show highlights if appropriate
    """
    print("\n🧠 Understanding your query...\n")
    parsed = parse_query(query)
    
    intent = parsed.get("intent", "general")
    search_query = parsed.get("search_query", query)
    show_highlights = should_fetch_highlights(parsed)
    
    intent_emoji = {
        QueryIntent.MATCH_RESULT: "🏆",
        QueryIntent.MATCH_HIGHLIGHTS: "🎬",
        QueryIntent.LINEUP: "📋",
        QueryIntent.PLAYER_INFO: "👤",
        QueryIntent.TRANSFER_NEWS: "💰",
        QueryIntent.TEAM_NEWS: "📰",
        QueryIntent.STANDINGS: "📊",
        QueryIntent.FIXTURES: "📅",
        QueryIntent.STATS: "📈",
        QueryIntent.GENERAL: "🔍",
    }
    
    emoji = intent_emoji.get(intent, "🔍")
    print(f"{emoji} I understand you want: {intent.replace('_', ' ').title()}")
    
    if parsed.get("teams"):
        print(f"   Teams: {', '.join(parsed['teams'])}")
    if parsed.get("players"):
        print(f"   Players: {', '.join(parsed['players'])}")
    if parsed.get("competition"):
        print(f"   Competition: {parsed['competition']}")
    
    print("\n🌐 Searching for information...\n")
    
    enhanced_search_query = search_query
    if intent in ["match_result", "match_highlights"] and not parsed.get("date_context"):
        current_month = datetime.now().strftime("%B %Y")
        enhanced_search_query = f"{search_query} {current_month}"
    
    match_metadata = {}
    web_summary = ""
    try:
        result, match_metadata = search_with_rag(
            query=enhanced_search_query,
            intent=intent,
            original_query=query,
            parsed_query=parsed
        )
        web_summary = result
        print("─" * 50)
        print(result)
        print("─" * 50)
        
        if intent in ["match_result", "match_highlights"] and match_metadata.get("score"):
            summary_display = format_match_summary(match_metadata)
            if summary_display:
                print(summary_display)
                
    except Exception as e:
        print(f"❌ Error searching the web: {e}")
        return
    
    if show_highlights:
        print("\n🎬 Finding match highlights (with RAG validation)...\n")
        try:
            home_team = match_metadata.get("home_team") or (parsed.get("teams", [None])[0])
            away_team = match_metadata.get("away_team") or (parsed.get("teams", [None, None])[1] if len(parsed.get("teams", [])) > 1 else None)
            match_date = match_metadata.get("match_date")
            
            highlights = search_and_display_highlights_with_metadata(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                web_summary=web_summary,
                match_metadata=match_metadata
            )
            
            if highlights:
                print("\n🎬 HIGHLIGHTS:\n")
                for i, video in enumerate(highlights[:5], 1):
                    title = video.get("title", "Unknown")
                    url = video.get("url", "")
                    print(f"  {i}. {title}")
                    if url:
                        print(f"     🔗 {url}")
                    print()
            else:
                print("No highlights found for this match.")
                
        except Exception as e:
            print(f"❌ Error searching for highlights: {e}")
    else:
        if intent == QueryIntent.LINEUP:
            print("\n💡 Tip: Lineup info doesn't need highlights. Ask about the match result to see highlights!")
        elif intent == QueryIntent.TRANSFER_NEWS:
            print("\n💡 Tip: For match highlights, ask about a specific game result!")


def main():
    """Main entry point for the CLI application."""
    print_banner()
    
    print("Ask me anything about football/soccer! Examples:")
    print("  • 'What was the score of Manchester City vs Liverpool?'")
    print("  • 'Show me the Arsenal vs Chelsea lineup'")
    print("  • 'Latest transfer news for Real Madrid'")
    print("  • 'Who is top of the Premier League?'")
    print("  • 'Tell me about Haaland's recent performance'\n")
    
    while True:
        print_divider()
        
        try:
            query = input("⚽ Your question (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('q', 'quit', 'exit'):
            print("\n👋 Goodbye!")
            break
        
        try:
            handle_query(query)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Returning to main menu...")
            continue
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different query.\n")
            continue
    
    sys.exit(0)


if __name__ == "__main__":
    main()
