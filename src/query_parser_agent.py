"""
Query parser agent.

- Uses an LLM to interpret natural language football queries.
- Extracts intent, teams, competition, date context, etc.
- Enforces: for match-type intents we must have EXACTLY TWO teams.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL


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
    GENERAL = "general"


MATCH_INTENTS = {
    QueryIntent.MATCH_RESULT.value,
    QueryIntent.MATCH_HIGHLIGHTS.value,
}


# =============================================================================
# OpenAI client
# =============================================================================

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# =============================================================================
# Helpers
# =============================================================================

def _safe_lower(x: Any) -> str:
    return str(x).lower() if x is not None else ""


def _extract_teams_heuristic(query: str) -> List[str]:
    """
    Very simple fallback extraction if LLM JSON parsing fails.

    Looks for patterns like:
      "leeds vs liverpool"
      "barcelona v real madrid"
    """
    q = query.lower()
    # Replace ' v ' variations with ' vs ' for simpler splitting
    q_norm = re.sub(r"\bv\.?\b", " vs ", q)

    # Try to split around ' vs '
    parts = q_norm.split(" vs ")
    if len(parts) == 2:
        left = parts[0].strip()
        right = parts[1].strip()
        # Roughly strip non-letters at ends
        left = re.sub(r"[^a-z0-9\s]+$", "", left).strip()
        right = re.sub(r"[^a-z0-9\s]+$", "", right).strip()
        teams = []
        if left:
            teams.append(left.title())
        if right:
            teams.append(right.title())
        return teams

    return []


def _baseline_intent_from_text(query: str) -> str:
    """
    Heuristic baseline intent if LLM fails.
    """
    q = _safe_lower(query)

    if any(w in q for w in ["highlight", "highlights", "watch goals"]):
        return QueryIntent.MATCH_HIGHLIGHTS.value
    if any(w in q for w in ["score", "result", "who won", "full time"]):
        return QueryIntent.MATCH_RESULT.value
    if any(w in q for w in ["lineup", "starting xi", "starting 11", "formation"]):
        return QueryIntent.LINEUP.value
    if any(w in q for w in ["table", "standings", "league positions"]):
        return QueryIntent.STANDINGS.value
    if any(w in q for w in ["fixture", "fixtures", "upcoming games"]):
        return QueryIntent.FIXTURES.value
    if any(w in q for w in ["stats", "statistics", "xg", "shots on target"]):
        return QueryIntent.STATS.value

    return QueryIntent.GENERAL.value


def _build_llm_system_prompt() -> str:
    return (
        "You are a football (soccer) query parser.\n"
        "Given a user question, you MUST respond ONLY with valid JSON.\n\n"
        "The JSON object must have the following fields:\n"
        "  - intent: one of\n"
        "      'match_result', 'match_highlights', 'lineup', 'player_info',\n"
        "      'transfer_news', 'team_news', 'standings', 'fixtures', 'stats', 'general'\n"
        "  - teams: array of team names in the order they appear in the match expression.\n"
        "           e.g. ['Leeds United', 'Liverpool'] if the user says 'Leeds vs Liverpool'.\n"
        "  - competition: string or null (e.g. 'Premier League', 'La Liga').\n"
        "  - date_context: string or null describing any date info (e.g. '2023-10-28', 'yesterday', 'last season').\n"
        "  - is_most_recent: boolean indicating whether the user wants the most recent match.\n"
        "  - summary_focus: short string on what to focus on in the answer.\n"
        "  - search_query: improved search query string for web search.\n"
        "  - is_relevant: boolean indicating whether the query is about football/soccer.\n"
        "  - validation_error: null or an object with 'reason' and 'suggestion' if the query is invalid.\n\n"
        "Rules:\n"
        "  - If the query is not about football/soccer, set is_relevant=false and fill validation_error.\n"
        "  - Do NOT include any comments or extra text. Output JSON ONLY.\n"
    )


def _call_llm_for_parse(user_query: str) -> Dict[str, Any]:
    """
    Call the LLM and parse its JSON output.
    """
    client = _get_openai_client()

    system_prompt = _build_llm_system_prompt()
    user_prompt = f"User query: {user_query!r}\n\nReturn ONLY JSON."

    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object at top level")
        return parsed
    except Exception as e:
        print(f"[QueryParser] LLM JSON parse error: {e}")
        print("[QueryParser] Raw content:", content)
        # Fallback: empty dict – caller will use heuristics
        return {}


# =============================================================================
# Validation: enforce EXACTLY TWO teams for match intents
# =============================================================================

def _validate_team_count_for_match_intent(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce: for match-type intents we must have *exactly two* teams.
    If not, mark query as not relevant for the purposes of this pipeline.

    This prevents 'recent games for Leeds' (1 team) or 'Leeds, Liverpool, Arsenal'
    (3 teams) from flowing into the match_result / highlights pipeline.
    """
    intent = (parsed.get("intent") or QueryIntent.GENERAL.value).lower()
    teams: List[str] = [t for t in (parsed.get("teams") or []) if t]

    # For non-match intents, do nothing.
    if intent not in MATCH_INTENTS:
        parsed.setdefault("is_relevant", True)
        parsed.setdefault("validation_error", None)
        return parsed

    team_count = len(teams)

    if team_count == 2:
        # Good case
        parsed["teams"] = teams
        parsed["home_team"] = teams[0]
        parsed["away_team"] = teams[1]
        parsed.setdefault("is_relevant", True)
        parsed.setdefault("validation_error", None)
        return parsed

    # Anything else (0, 1, 3+): treat as a validation failure for match queries
    reason = (
        f"Expected exactly two teams for a match query, "
        f"but found {team_count} team(s): {teams or '[]'}."
    )
    suggestion = (
        "Ask about a single specific match using two teams, for example:\n"
        "  • 'Leeds vs Liverpool 4-3 highlights'\n"
        "  • 'Barcelona vs Real Madrid score 28 October 2023'"
    )

    return {
        "raw_query": parsed.get("raw_query"),
        "intent": intent,
        "search_query": parsed.get("search_query") or parsed.get("raw_query"),
        "teams": teams,
        "home_team": None,
        "away_team": None,
        "competition": parsed.get("competition"),
        "date_context": parsed.get("date_context"),
        "is_most_recent": parsed.get("is_most_recent", False),
        "summary_focus": parsed.get("summary_focus"),
        "is_relevant": False,
        "validation_error": {
            "reason": reason,
            "suggestion": suggestion,
            "team_count": team_count,
            "teams_found": teams,
        },
    }


# =============================================================================
# Public: parse_query
# =============================================================================

def parse_query(user_query: str) -> Dict[str, Any]:
    """
    Main entry point.

    Returns a dict with at least:
      - raw_query
      - intent
      - teams
      - competition
      - date_context
      - search_query
      - is_relevant
      - validation_error
      - home_team (optional)
      - away_team (optional)
      - is_most_recent (optional)
      - summary_focus (optional)
    """
    raw = user_query.strip()

    # 1) Call LLM
    llm_data = _call_llm_for_parse(raw)

    # 2) If LLM failed or gave garbage, fall back to heuristics
    if not llm_data:
        print("[QueryParser] Falling back to heuristic parsing.")
        teams = _extract_teams_heuristic(raw)
        intent = _baseline_intent_from_text(raw)

        base = {
            "raw_query": raw,
            "intent": intent,
            "teams": teams,
            "competition": None,
            "date_context": None,
            "is_most_recent": False,
            "summary_focus": "key information",
            "search_query": raw,
            "is_relevant": True,
            "validation_error": None,
        }
        validated = _validate_team_count_for_match_intent(base)
        print("[QueryParser] Parsed (heuristic):", validated)
        return validated

    # 3) Normalise fields from LLM
    intent = (llm_data.get("intent") or _baseline_intent_from_text(raw)).lower()
    teams = llm_data.get("teams") or []
    if not isinstance(teams, list):
        teams = []

    competition = llm_data.get("competition")
    date_context = llm_data.get("date_context")
    is_most_recent = bool(llm_data.get("is_most_recent", False))
    summary_focus = llm_data.get("summary_focus") or "key information"
    search_query = llm_data.get("search_query") or raw

    is_relevant = bool(llm_data.get("is_relevant", True))
    validation_error = llm_data.get("validation_error")

    base = {
        "raw_query": raw,
        "intent": intent,
        "teams": teams,
        "competition": competition,
        "date_context": date_context,
        "is_most_recent": is_most_recent,
        "summary_focus": summary_focus,
        "search_query": search_query,
        "is_relevant": is_relevant,
        "validation_error": validation_error,
    }

    # 4) If LLM already said "not relevant", respect that immediately
    if not is_relevant and validation_error:
        print("[QueryParser] LLM flagged query as not relevant:", validation_error)
        # Still enforce the same shape of object
        base.setdefault("home_team", None)
        base.setdefault("away_team", None)
        return base

    # 5) Enforce the EXACTLY-TWO-TEAMS rule for match intents
    validated = _validate_team_count_for_match_intent(base)
    print("[QueryParser] Parsed query:", validated)
    return validated


# =============================================================================
# Helper: should_fetch_highlights
# =============================================================================

def should_fetch_highlights(parsed: Dict[str, Any]) -> bool:
    """
    Decide whether to auto-fetch highlights.

    - True if intent is 'match_highlights'
    - True if user query (raw_query or search_query) strongly suggests highlights
    - Otherwise False
    """
    intent = (parsed.get("intent") or "").lower()
    if intent == QueryIntent.MATCH_HIGHLIGHTS.value:
        return True

    raw_q = _safe_lower(parsed.get("raw_query"))
    search_q = _safe_lower(parsed.get("search_query"))
    text = raw_q + " " + search_q

    if any(w in text for w in ["highlight", "highlights", "watch goals", "extended highlights"]):
        return True

    return False
