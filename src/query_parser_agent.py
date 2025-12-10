"""
Query parser agent - Smart natural language understanding for football queries.

Handles flexible queries like:
- "Latest Premier League game result"
- "Liverpool Brentford game last year"
- "Who won the Champions League final?"
- "Arsenal vs Chelsea highlights"
- "What happened in the Manchester derby yesterday?"
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .utils import get_openai_client, safe_lower
from .config import DEFAULT_LLM_MODEL


# =============================================================================
# Intent Enum
# =============================================================================

class QueryIntent(str, Enum):
    MATCH_RESULT = "match_result"
    MATCH_HIGHLIGHTS = "match_highlights"
    LINEUP = "lineup"
    PLAYER_INFO = "player_info"
    TRANSFER_NEWS = "transfer_news"
    TEAM_NEWS = "team_news"
    STANDINGS = "standings"
    FIXTURES = "fixtures"
    STATS = "stats"
    COMPETITION_LATEST = "competition_latest"  # New: "latest PL game"
    GENERAL = "general"


# Intents that are about specific matches
MATCH_INTENTS = {
    QueryIntent.MATCH_RESULT.value,
    QueryIntent.MATCH_HIGHLIGHTS.value,
    QueryIntent.LINEUP.value,
    QueryIntent.STATS.value,
}

# Intents that can work with 0, 1, or 2 teams
FLEXIBLE_TEAM_INTENTS = {
    QueryIntent.COMPETITION_LATEST.value,
    QueryIntent.STANDINGS.value,
    QueryIntent.FIXTURES.value,
    QueryIntent.TRANSFER_NEWS.value,
    QueryIntent.TEAM_NEWS.value,
    QueryIntent.PLAYER_INFO.value,
    QueryIntent.GENERAL.value,
}


# =============================================================================
# Time Reference Parsing
# =============================================================================

def _parse_relative_time(text: str) -> Optional[str]:
    """
    Parse relative time references into approximate date ranges.
    Returns a string describing the date context for search.
    """
    text_lower = text.lower()
    today = datetime.now()
    
    # Yesterday
    if "yesterday" in text_lower:
        date = today - timedelta(days=1)
        return date.strftime("%Y-%m-%d")
    
    # Last week
    if "last week" in text_lower:
        start = today - timedelta(days=7)
        return f"between {start.strftime('%Y-%m-%d')} and {today.strftime('%Y-%m-%d')}"
    
    # This week / this weekend
    if "this week" in text_lower or "this weekend" in text_lower:
        start = today - timedelta(days=today.weekday())
        return f"week of {start.strftime('%B %d, %Y')}"
    
    # Last year
    if "last year" in text_lower:
        year = today.year - 1
        return f"in {year}"
    
    # Specific year pattern (e.g., "in 2023", "2022 season")
    year_match = re.search(r'\b(20[12]\d)\b', text_lower)
    if year_match:
        return f"in {year_match.group(1)}"
    
    # Last month
    if "last month" in text_lower:
        last_month = today.replace(day=1) - timedelta(days=1)
        return f"in {last_month.strftime('%B %Y')}"
    
    # Specific month patterns
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    for month in months:
        if month in text_lower:
            # Try to find year context
            year_match = re.search(r'\b(20[12]\d)\b', text_lower)
            year = year_match.group(1) if year_match else str(today.year)
            return f"in {month.title()} {year}"
    
    return None


def _enhance_search_query(query: str, parsed: Dict[str, Any]) -> str:
    """
    Build an optimal search query from parsed components.
    """
    parts = []
    
    teams = parsed.get("teams", [])
    competition = parsed.get("competition")
    date_context = parsed.get("date_context")
    intent = parsed.get("intent", "general")
    
    # Add teams
    if len(teams) == 2:
        parts.append(f"{teams[0]} vs {teams[1]}")
    elif len(teams) == 1:
        parts.append(teams[0])
    
    # Add competition for context
    if competition:
        parts.append(competition)
    
    # Add intent-specific keywords
    if intent == QueryIntent.MATCH_RESULT.value:
        parts.append("result score")
    elif intent == QueryIntent.MATCH_HIGHLIGHTS.value:
        parts.append("highlights goals")
    elif intent == QueryIntent.LINEUP.value:
        parts.append("starting lineup XI")
    elif intent == QueryIntent.STATS.value:
        parts.append("match statistics stats")
    elif intent == QueryIntent.COMPETITION_LATEST.value:
        parts.append("latest recent game result")
    elif intent == QueryIntent.STANDINGS.value:
        parts.append("table standings")
    elif intent == QueryIntent.FIXTURES.value:
        parts.append("fixtures schedule")
    
    # Add date context
    if date_context:
        parts.append(date_context)
    
    # Build final query
    search_query = " ".join(parts) if parts else query
    
    return search_query


# =============================================================================
# Heuristic Helpers
# =============================================================================

def _extract_teams_heuristic(query: str) -> List[str]:
    """
    Extract team names from query using heuristics.
    Handles: "Leeds vs Liverpool", "Liverpool Brentford", "the Manchester derby"
    """
    q = query.lower()
    
    # Pattern 1: "X vs Y" or "X v Y"
    q_norm = re.sub(r"\bv\.?\b", " vs ", q)
    if " vs " in q_norm:
        parts = q_norm.split(" vs ")
        if len(parts) == 2:
            left = re.sub(r"[^a-z0-9\s]+$", "", parts[0].strip()).strip()
            right = re.sub(r"[^a-z0-9\s]+$", "", parts[1].strip()).strip()
            # Clean up common prefixes
            left = re.sub(r"^(what was|show me|get|find|the)\s+", "", left)
            right = re.sub(r"\s+(game|match|result|score|highlights?).*$", "", right)
            teams = []
            if left:
                teams.append(left.title())
            if right:
                teams.append(right.title())
            return teams
    
    # Pattern 2: Known derby names
    derbies = {
        "manchester derby": ["Manchester United", "Manchester City"],
        "north london derby": ["Arsenal", "Tottenham"],
        "merseyside derby": ["Liverpool", "Everton"],
        "el clasico": ["Real Madrid", "Barcelona"],
        "milan derby": ["AC Milan", "Inter Milan"],
        "old firm": ["Celtic", "Rangers"],
    }
    for derby_name, teams in derbies.items():
        if derby_name in q:
            return teams
    
    # Pattern 3: Two consecutive team-like words (e.g., "Liverpool Brentford")
    # This is a simple heuristic - the LLM will do better
    
    return []


def _extract_competition_heuristic(query: str) -> Optional[str]:
    """Extract competition name from query."""
    q = query.lower()
    
    competitions = {
        "premier league": "Premier League",
        "la liga": "La Liga",
        "bundesliga": "Bundesliga",
        "serie a": "Serie A",
        "ligue 1": "Ligue 1",
        "champions league": "UEFA Champions League",
        "ucl": "UEFA Champions League",
        "europa league": "UEFA Europa League",
        "fa cup": "FA Cup",
        "carabao cup": "EFL Cup",
        "league cup": "EFL Cup",
        "world cup": "FIFA World Cup",
        "euros": "UEFA European Championship",
        "european championship": "UEFA European Championship",
    }
    
    for pattern, name in competitions.items():
        if pattern in q:
            return name
    
    return None


def _baseline_intent_from_text(query: str) -> str:
    """Determine intent from query text using heuristics."""
    q = safe_lower(query)
    
    # Latest/recent game patterns
    if any(w in q for w in ["latest", "recent", "last game", "last match"]):
        if any(c in q for c in ["premier league", "la liga", "champions league", 
                                 "bundesliga", "serie a", "ligue 1"]):
            return QueryIntent.COMPETITION_LATEST.value
    
    if any(w in q for w in ["highlight", "highlights", "watch goals", "goals video"]):
        return QueryIntent.MATCH_HIGHLIGHTS.value
    
    if any(w in q for w in ["score", "result", "who won", "full time", "final score"]):
        return QueryIntent.MATCH_RESULT.value
    
    if any(w in q for w in ["lineup", "starting xi", "starting 11", "formation", "who played", "who started"]):
        return QueryIntent.LINEUP.value
    
    if any(w in q for w in ["table", "standings", "league position", "league table"]):
        return QueryIntent.STANDINGS.value
    
    if any(w in q for w in ["fixture", "fixtures", "upcoming", "next game", "schedule"]):
        return QueryIntent.FIXTURES.value
    
    if any(w in q for w in ["stats", "statistics", "xg", "shots", "possession"]):
        return QueryIntent.STATS.value
    
    if any(w in q for w in ["transfer", "signing", "signed", "bought", "sold"]):
        return QueryIntent.TRANSFER_NEWS.value
    
    if any(w in q for w in ["injury", "injured", "news", "update"]):
        return QueryIntent.TEAM_NEWS.value
    
    if any(w in q for w in ["player", "who is", "about"]):
        return QueryIntent.PLAYER_INFO.value
    
    return QueryIntent.GENERAL.value


# =============================================================================
# LLM Parsing
# =============================================================================

def _build_llm_system_prompt() -> str:
    return """You are a smart football (soccer) query parser. Your job is to understand natural language queries about football.

Given a user question, respond ONLY with valid JSON containing these fields:

{
  "intent": string,  // One of: match_result, match_highlights, lineup, player_info, transfer_news, team_news, standings, fixtures, stats, competition_latest, general
  "teams": array,    // Team names found. Can be 0, 1, or 2 teams. Order: [home, away] if both present
  "competition": string | null,  // e.g., "Premier League", "Champions League"
  "date_context": string | null,  // Natural date description, e.g., "December 8, 2024", "last week", "2023 season"
  "is_most_recent": boolean,  // True if asking about most recent/latest match
  "specific_match_identifier": string | null,  // Any unique match identifier like "final", "semi-final", "round 16"
  "summary_focus": string,  // What to focus on in the answer
  "search_query": string,  // Optimized search query for web search
  "is_relevant": boolean,  // False if not about football/soccer
  "validation_error": object | null  // {reason, suggestion} if query is invalid
}

RULES:
1. "intent" meanings:
   - match_result: Score/result of a specific match
   - match_highlights: Video highlights of a match
   - competition_latest: "Latest Premier League game", "Recent Champions League match"
   - lineup: Starting XI / formation
   - stats: Match statistics (possession, shots, etc.)
   - standings: League table
   - fixtures: Upcoming matches
   - transfer_news: Transfer rumors/signings
   - team_news: Injuries, updates
   - player_info: About a specific player
   - general: Other football questions

2. Team extraction:
   - "Liverpool vs Brentford" → ["Liverpool", "Brentford"]
   - "Liverpool Brentford game" → ["Liverpool", "Brentford"]  
   - "Manchester derby" → ["Manchester United", "Manchester City"]
   - "latest Premier League game" → [] (no specific teams)
   - "Arsenal game" → ["Arsenal"] (one team)

3. Date understanding:
   - "yesterday" → compute actual date
   - "last year" → "2024" (or appropriate year)
   - "last week" → "past 7 days"
   - "Liverpool Brentford game last year" → teams + "2024 season"

4. search_query should be optimized for Google/web search, include:
   - Team names with "vs" between them
   - Competition name if known
   - Date context
   - Keywords like "result", "score", "highlights" based on intent

EXAMPLES:
Query: "What was the latest Premier League game result?"
→ intent: "competition_latest", teams: [], competition: "Premier League", is_most_recent: true

Query: "Liverpool Brentford game last year"  
→ intent: "match_result", teams: ["Liverpool", "Brentford"], date_context: "2024 season"

Query: "Arsenal highlights"
→ intent: "match_highlights", teams: ["Arsenal"], is_most_recent: true

Query: "Who won the Champions League final?"
→ intent: "match_result", competition: "UEFA Champions League", specific_match_identifier: "final"

Output ONLY valid JSON. No explanations."""


def _call_llm_for_parse(user_query: str) -> Dict[str, Any]:
    """Call the LLM and parse its JSON output."""
    client = get_openai_client()
    
    # Add current date context
    today = datetime.now()
    date_info = f"Today is {today.strftime('%A, %B %d, %Y')}."
    
    system_prompt = _build_llm_system_prompt()
    user_prompt = f"{date_info}\n\nUser query: {user_query!r}\n\nReturn ONLY JSON."
    
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    content = resp.choices[0].message.content
    
    # Clean up response (remove markdown code blocks if present)
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object")
        return parsed
    except Exception as e:
        print(f"[QueryParser] LLM JSON parse error: {e}")
        print("[QueryParser] Raw content:", content[:500])
        return {}


# =============================================================================
# Validation & Post-processing
# =============================================================================

def _validate_and_enhance(parsed: Dict[str, Any], raw_query: str) -> Dict[str, Any]:
    """
    Validate parsed query and enhance with computed fields.
    
    Now more flexible - doesn't require exactly 2 teams for all match queries.
    """
    intent = (parsed.get("intent") or QueryIntent.GENERAL.value).lower()
    teams = [t for t in (parsed.get("teams") or []) if t]
    competition = parsed.get("competition")
    date_context = parsed.get("date_context")
    is_most_recent = parsed.get("is_most_recent", False)
    
    # Set home/away teams if we have exactly 2
    home_team = teams[0] if len(teams) >= 1 else None
    away_team = teams[1] if len(teams) >= 2 else None
    
    # For match intents, we need EITHER:
    # - Two teams specified
    # - One team + most_recent flag
    # - Competition + most_recent flag (competition_latest intent)
    is_valid = True
    validation_error = None
    
    if intent in MATCH_INTENTS:
        if len(teams) == 0:
            # No teams - check if we have competition context for "latest" queries
            if competition and is_most_recent:
                # Convert to competition_latest intent
                intent = QueryIntent.COMPETITION_LATEST.value
            elif not competition:
                is_valid = False
                validation_error = {
                    "reason": "Please specify at least one team or a competition for match queries.",
                    "suggestion": "Try: 'Arsenal vs Chelsea result' or 'latest Premier League game'"
                }
        elif len(teams) == 1:
            # One team - that's fine, we'll search for their most recent match
            if is_most_recent:
                pass  # Good - "Arsenal's latest game"
            else:
                # Assume most recent if not specified
                is_most_recent = True
    
    # Build enhanced search query
    search_query = parsed.get("search_query") or raw_query
    if not search_query or search_query == raw_query:
        search_query = _enhance_search_query(raw_query, {
            "teams": teams,
            "competition": competition,
            "date_context": date_context,
            "intent": intent,
        })
    
    # Enhance date context from query if not found
    if not date_context:
        date_context = _parse_relative_time(raw_query)
    
    result = {
        "raw_query": raw_query,
        "intent": intent,
        "teams": teams,
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "date_context": date_context,
        "is_most_recent": is_most_recent,
        "specific_match_identifier": parsed.get("specific_match_identifier"),
        "summary_focus": parsed.get("summary_focus") or "key information",
        "search_query": search_query,
        "is_relevant": is_valid and parsed.get("is_relevant", True),
        "validation_error": validation_error or parsed.get("validation_error"),
        "emphasize_order": parsed.get("emphasize_order", False),  # Preserve this flag!
    }
    
    return result


# =============================================================================
# Public API
# =============================================================================

def parse_query(user_query: str, emphasize_order: bool = False) -> Dict[str, Any]:
    """
    Main entry point for query parsing.
    
    Handles flexible queries like:
    - "Latest Premier League game result"
    - "Liverpool Brentford game last year"  
    - "Who won the Champions League final?"
    - "Arsenal vs Chelsea highlights"
    
    Args:
        user_query: The user's natural language query
        emphasize_order: If True, the order teams appear matters.
                        First team = home, second team = away.
    
    Returns a dict with:
      - raw_query, intent, teams, competition, date_context
      - search_query, is_relevant, validation_error
      - home_team, away_team (if applicable)
      - is_most_recent, specific_match_identifier
      - emphasize_order: Whether team order should be preserved
    """
    raw = user_query.strip()
    
    print(f"[QueryParser] Emphasize team order: {emphasize_order}")
    
    # 1) Call LLM for intelligent parsing
    llm_data = _call_llm_for_parse(raw)
    
    # 2) Fallback to heuristics if LLM failed
    if not llm_data:
        print("[QueryParser] Falling back to heuristic parsing.")
        teams = _extract_teams_heuristic(raw)
        competition = _extract_competition_heuristic(raw)
        intent = _baseline_intent_from_text(raw)
        date_context = _parse_relative_time(raw)
        
        base = {
            "raw_query": raw,
            "intent": intent,
            "teams": teams,
            "competition": competition,
            "date_context": date_context,
            "is_most_recent": any(w in raw.lower() for w in ["latest", "recent", "last"]),
            "specific_match_identifier": None,
            "summary_focus": "key information",
            "search_query": raw,
            "is_relevant": True,
            "validation_error": None,
            "emphasize_order": emphasize_order,
        }
        result = _validate_and_enhance(base, raw)
        print("[QueryParser] Parsed (heuristic):", result)
        return result
    
    # 3) Validate and enhance LLM output
    llm_data["emphasize_order"] = emphasize_order
    result = _validate_and_enhance(llm_data, raw)
    print("[QueryParser] Parsed query:", result)
    return result


def should_fetch_highlights(parsed: Dict[str, Any]) -> bool:
    """
    Decide whether to auto-fetch highlights.
    
    Returns True if:
    - Intent is 'match_highlights'
    - Query text mentions highlights/goals
    """
    intent = (parsed.get("intent") or "").lower()
    if intent == QueryIntent.MATCH_HIGHLIGHTS.value:
        return True
    
    raw_q = safe_lower(parsed.get("raw_query"))
    search_q = safe_lower(parsed.get("search_query"))
    text = raw_q + " " + search_q
    
    highlight_keywords = ["highlight", "highlights", "watch goals", "extended highlights", "goals video"]
    return any(w in text for w in highlight_keywords)
