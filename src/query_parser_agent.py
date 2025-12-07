"""
Query Parser Agent for Soccer LLM Analyst.

This agent parses natural language sports queries and determines:
1. What the user is asking about (match result, lineup, news, transfer, etc.)
2. What teams/players are involved
3. What date/competition context
4. Whether highlights should be shown

This is the "brain" that routes queries to the right search strategy.
"""

import json
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL


# =============================================================================
# Intent Types
# =============================================================================

class QueryIntent:
    """Enumeration of query intent types."""
    MATCH_RESULT = "match_result"       # User wants score/result of a match
    MATCH_HIGHLIGHTS = "match_highlights"  # User wants to watch highlights
    LINEUP = "lineup"                   # User wants starting XI/lineup
    PLAYER_INFO = "player_info"         # User wants info about a player
    TRANSFER_NEWS = "transfer_news"     # User wants transfer news
    TEAM_NEWS = "team_news"             # User wants general team news
    STANDINGS = "standings"             # User wants league table/standings
    FIXTURES = "fixtures"               # User wants upcoming matches
    STATS = "stats"                     # User wants statistics
    GENERAL = "general"                 # General sports query


# Intent categories that should include highlights
HIGHLIGHT_INTENTS = {
    QueryIntent.MATCH_RESULT,
    QueryIntent.MATCH_HIGHLIGHTS,
}


# =============================================================================
# LLM Client
# =============================================================================

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Get or initialize the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# =============================================================================
# Query Parser
# =============================================================================

def validate_query_relevance(query: str) -> dict:
    """
    Validate if the query is about football/soccer before processing.
    
    This prevents wasting API credits on irrelevant queries.
    
    Args:
        query: Natural language query from user.
        
    Returns:
        dict with:
            - is_relevant: bool (True if query is about football/soccer)
            - reason: str (explanation if not relevant)
            - suggestion: str (suggested query if not relevant)
    """
    try:
        client = _get_openai_client()
        
        validation_prompt = """You are a query validator for a football/soccer information system.

Your job is to determine if a user's query is about FOOTBALL/SOCCER.

VALID queries include:
- Match results, scores, highlights
- Team information, lineups, news
- Player information, stats, transfers
- League standings, fixtures
- Football/soccer related questions

INVALID queries (NOT about football/soccer):
- Questions about other sports (basketball, tennis, etc.) unless clearly football context
- Questions about pets, animals, colors, weather, cooking, etc.
- General knowledge questions unrelated to football
- Questions about other topics entirely

Return ONLY valid JSON:
{
    "is_relevant": true/false,
    "reason": "brief explanation (only if is_relevant is false)",
    "suggestion": "suggested football query like 'Barcelona vs Atletico Madrid' (only if is_relevant is false)"
}

Be strict - if the query is clearly NOT about football/soccer, set is_relevant to false."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": validation_prompt},
                {"role": "user", "content": f"Validate this query: \"{query}\"\n\nReturn ONLY the JSON object."}
            ],
            temperature=0,
            max_tokens=200,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up the response
        if answer.startswith("```"):
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
        answer = answer.strip()
        
        result = json.loads(answer)
        
        # Set defaults
        result.setdefault("is_relevant", True)
        result.setdefault("reason", "")
        result.setdefault("suggestion", "Try: Barcelona vs Atletico Madrid")
        
        return result
        
    except Exception as e:
        print(f"[QueryParser] Validation error: {e}")
        # Default to relevant if validation fails (to avoid blocking valid queries)
        return {
            "is_relevant": True,
            "reason": "",
            "suggestion": ""
        }


def parse_query(query: str) -> dict:
    """
    Parse a natural language sports query to extract intent and entities.
    
    This is the main function that determines:
    - What the user wants (intent)
    - What teams/players/competitions are involved
    - Whether to show highlights
    
    Args:
        query: Natural language query from user.
        
    Returns:
        dict with:
            - intent: str (one of QueryIntent values)
            - search_query: str (optimized search string for web)
            - teams: list of team names
            - players: list of player names
            - competition: str or None
            - date_context: str or None
            - should_show_highlights: bool
            - summary_focus: str (what to emphasize in summary)
            - is_relevant: bool (True if query is about football/soccer)
            - validation_error: dict (if not relevant, contains reason and suggestion)
    """
    print(f"\n[QueryParser] === Parsing Query ===")
    print(f"[QueryParser] Input: \"{query}\"")
    
    # Step 1: Validate query relevance FIRST (before wasting credits)
    print(f"[QueryParser] Step 1: Validating query relevance...")
    validation = validate_query_relevance(query)
    
    if not validation.get("is_relevant", True):
        print(f"[QueryParser] ❌ Query is NOT about football/soccer")
        print(f"[QueryParser] Reason: {validation.get('reason', 'Not relevant')}")
        print(f"[QueryParser] Suggestion: {validation.get('suggestion', '')}")
        
        return {
            "intent": QueryIntent.GENERAL,
            "search_query": query,
            "teams": [],
            "players": [],
            "competition": None,
            "date_context": None,
            "should_show_highlights": False,
            "summary_focus": "key information",
            "is_relevant": False,
            "validation_error": {
                "reason": validation.get("reason", "This query is not about football/soccer."),
                "suggestion": validation.get("suggestion", "Try asking about a match, like 'Barcelona vs Atletico Madrid'")
            }
        }
    
    print(f"[QueryParser] ✓ Query is relevant to football/soccer")
    
    try:
        client = _get_openai_client()
        
        # Get current date for context
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        current_day = today.strftime("%A, %B %d, %Y")
        
        system_prompt = """You are a sports query parser. Analyze the user's query about football/soccer and extract structured information.

INTENTS (pick the most specific one):
- "match_result": User wants score/result of a match that HAS ALREADY BEEN PLAYED (e.g., "what was the score", "who won", "how did the game go", "leeds vs liverpool", "arsenal chelsea result"). If user mentions two teams without words like "upcoming", "next", "when", "schedule" - assume it's a past match.
- "match_highlights": User explicitly wants to WATCH highlights (e.g., "show me highlights", "I want to see the goals")
- "lineup": User wants starting XI or lineup (e.g., "who started", "what was the lineup", "formation")
- "player_info": User wants info about a specific player (e.g., "how is Messi doing", "Haaland stats")
- "transfer_news": User wants transfer news/rumors (e.g., "transfer news", "is X joining Y")
- "team_news": User wants general team news (e.g., "what's happening with Arsenal", "latest on Chelsea")
- "standings": User wants league table/standings (e.g., "Premier League table", "who is top")
- "fixtures": User wants UPCOMING/FUTURE matches (e.g., "when do they play next", "upcoming matches", "schedule", "next game", "when is the match")
- "stats": User wants statistics (e.g., "top scorers", "possession stats")
- "general": Anything else sports related


TEAM NAME NORMALIZATION:
- "man city", "city" → "Manchester City"
- "man utd", "united" (in PL context) → "Manchester United"
- "spurs" → "Tottenham"
- "arsenal", "gunners" → "Arsenal"
- "liverpool", "reds" (in PL context) → "Liverpool"
- "chelsea", "blues" (in PL context) → "Chelsea"
- "barca" → "Barcelona"
- "real", "madrid" → "Real Madrid"
- "bayern" → "Bayern Munich"

Return ONLY valid JSON:
{
    "intent": "one of the intent types above",
    "search_query": "optimized search string for finding this info on the web",
    "teams": ["Team1", "Team2"],
    "players": ["Player1"],
    "competition": "Premier League or null",
    "date_context": "specific date or 'most recent' or 'yesterday' or null",
    "is_most_recent": true/false,
    "should_show_highlights": true/false,
    "summary_focus": "what to emphasize in the summary (e.g., 'score and key moments' or 'lineup and formation' or 'transfer details')"
}

CRITICAL RULES FOR TEAM ORDER:
- If user says "Team A vs Team B" or "Team A v Team B", preserve the order:
  * First team mentioned = home_team
  * Second team mentioned = away_team
- For "Team A vs Team B" queries, the teams array should be ["Team A", "Team B"] in that exact order
- Do NOT swap the order - respect the user's query order

RULES FOR should_show_highlights:
- TRUE only if intent is "match_result" or "match_highlights"
- FALSE for lineup, transfer, standings, fixtures, stats, player_info
- FALSE if user asks "don't show highlights" or similar

IMPORTANT FOR RECENCY:
- If user asks "what was the score" without specifying a date, set is_most_recent to true
- If user asks about a match without a specific date, assume they want the MOST RECENT match
- Only set is_most_recent to false if user specifies a particular date or year

"""

        user_prompt = f"""Today's date: {current_day}

Parse this query: "{query}"

Return ONLY the JSON object."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=400,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown code blocks if present)
        if answer.startswith("```"):
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
        answer = answer.strip()
        
        parsed = json.loads(answer)
        
        # Validate and set defaults
        parsed.setdefault("intent", QueryIntent.GENERAL)
        parsed.setdefault("search_query", query)
        parsed.setdefault("teams", [])
        parsed.setdefault("players", [])
        parsed.setdefault("competition", None)
        parsed.setdefault("date_context", None)
        parsed.setdefault("should_show_highlights", False)
        parsed.setdefault("summary_focus", "key information")
        parsed.setdefault("is_relevant", True)
        parsed.setdefault("validation_error", None)
        
        # Log parsed result
        print(f"[QueryParser] Intent: {parsed['intent']}")
        print(f"[QueryParser] Teams: {parsed['teams']}")
        print(f"[QueryParser] Search query: \"{parsed['search_query']}\"")
        print(f"[QueryParser] Show highlights: {parsed['should_show_highlights']}")
        print(f"[QueryParser] Summary focus: {parsed['summary_focus']}")
        
        return parsed
        
    except Exception as e:
        print(f"[QueryParser] Error: {e}")
        # Return a safe fallback
        return {
            "intent": QueryIntent.GENERAL,
            "search_query": query,
            "teams": [],
            "players": [],
            "competition": None,
            "date_context": None,
            "should_show_highlights": False,
            "summary_focus": "key information",
            "is_relevant": True,
            "validation_error": None
        }


def should_fetch_highlights(parsed_query: dict) -> bool:
    """
    Determine if highlights should be fetched for this query.
    
    Args:
        parsed_query: The parsed query dict from parse_query().
        
    Returns:
        True if highlights should be shown.
    """
    return parsed_query.get("should_show_highlights", False)


def get_search_query(parsed_query: dict) -> str:
    """
    Get the optimized search query string.
    
    Args:
        parsed_query: The parsed query dict from parse_query().
        
    Returns:
        Optimized search string for web search.
    """
    return parsed_query.get("search_query", "")


def get_intent(parsed_query: dict) -> str:
    """
    Get the intent from a parsed query.
    
    Args:
        parsed_query: The parsed query dict from parse_query().
        
    Returns:
        Intent string.
    """
    return parsed_query.get("intent", QueryIntent.GENERAL)

