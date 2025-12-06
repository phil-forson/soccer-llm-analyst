"""
Web search agent for finding football match information.

Uses DuckDuckGo for web search (free, no API key required).
Implements RAG: chunks results, embeds, retrieves relevant context.
STRICT: Will NOT hallucinate - only reports what it actually finds.
"""

import re
import hashlib
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL

# Try to import RAG components (optional)
RAG_AVAILABLE = False
try:
    from .embeddings_store import _get_embedding_model, _get_collection
    RAG_AVAILABLE = True
except ImportError:
    print("[RAG] Note: embeddings_store not available, using simple search")


# =============================================================================
# Configuration
# =============================================================================

MAX_RESULTS = 5
MAX_SEARCH_RESULTS = 15  # Fetch more, then filter to best

# Trusted football sources (prioritized in relevance scoring)
TRUSTED_SOURCES = [
    # Tier 1 - Highly authoritative match data
    "espn.com",
    "bbc.com/sport",
    "bbc.co.uk/sport",
    "skysports.com",
    "goal.com",
    "transfermarkt",
    "flashscore.com",
    "sofascore.com",
    "whoscored.com",
    "fotmob.com",
    # Tier 2 - Major sports news
    "theguardian.com/football",
    "telegraph.co.uk/football",
    "90min.com",
    "football365.com",
    "givemesport.com",
    "sportskeeda.com",
    "marca.com",
    "as.com",
    # Tier 3 - Club/League official sources
    "premierleague.com",
    "laliga.com",
    "bundesliga.com",
    "seriea.com",
    "ligue1.com",
    "uefa.com",
    "fifa.com",
]

# Keywords that indicate match-relevant content
RELEVANCE_KEYWORDS = [
    "score", "result", "goals", "match report", "highlights",
    "vs", "v.", "versus", "final score", "full time", "ft",
    "win", "draw", "loss", "defeat", "victory",
    "lineup", "starting xi", "team news",
    "premier league", "champions league", "la liga", "serie a",
    "bundesliga", "europa league", "fa cup", "carabao cup",
]

# Keywords that indicate low-relevance content (filter out)
LOW_RELEVANCE_KEYWORDS = [
    "betting", "odds", "prediction", "preview", "upcoming",
    "tickets", "buy tickets", "merchandise", "shop",
    "fantasy", "fpl", "dream team",
]


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
# LLM-Based Natural Language Query Parser
# =============================================================================

def _parse_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse natural language queries into structured match info.
    
    Handles queries like:
        - "tell me about the most recent man city game"
        - "arsenal vs chelsea last week"
        - "what happened in the liverpool match yesterday"
        - "man united champions league result"
    
    Args:
        query: Natural language query from user.
        
    Returns:
        dict with parsed match information.
    """
    print(f"\n[QueryParser] === Parsing Natural Language Query ===")
    print(f"[QueryParser] Input: \"{query}\"")
    
    try:
        client = _get_openai_client()
        
        # Get current date for context
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        current_month = today.strftime("%B %Y")
        
        system_prompt = """You are a football/soccer query parser. Extract match information from natural language queries.

IMPORTANT: Normalize team nicknames to full official names:
- "man city", "city" → "Manchester City"
- "man utd", "united" → "Manchester United"  
- "spurs" → "Tottenham"
- "arsenal", "gunners" → "Arsenal"
- "liverpool", "reds" (in PL context) → "Liverpool"
- "chelsea", "blues" (in PL context) → "Chelsea"
- "barca" → "Barcelona"
- "real", "madrid" → "Real Madrid"
- "bayern" → "Bayern Munich"
- etc.

Extract and return ONLY valid JSON with these fields:
{
    "home_team": "full team name or null",
    "away_team": "full team name or null",
    "date": "YYYY-MM-DD or null",
    "date_context": "most recent/last week/yesterday/specific description or null",
    "competition": "Premier League/Champions League/etc or null",
    "is_most_recent": true/false
}

If user says "most recent", "latest", "last game" - set is_most_recent to true.
If user mentions a specific date, convert to YYYY-MM-DD format.
If only one team mentioned, put it in home_team, leave away_team null."""

        user_prompt = f"""Parse this football query. Today's date is {current_date} ({current_month}).

Query: "{query}"

Return ONLY the JSON object, no other text."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=200,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        import json
        
        # Clean up the response (remove markdown code blocks if present)
        if answer.startswith("```"):
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
        answer = answer.strip()
        
        parsed = json.loads(answer)
        
        print(f"[QueryParser] LLM parsed result:")
        print(f"[QueryParser]   Home team: {parsed.get('home_team', 'None')}")
        print(f"[QueryParser]   Away team: {parsed.get('away_team', 'None')}")
        print(f"[QueryParser]   Date: {parsed.get('date', 'None')}")
        print(f"[QueryParser]   Date context: {parsed.get('date_context', 'None')}")
        print(f"[QueryParser]   Competition: {parsed.get('competition', 'None')}")
        print(f"[QueryParser]   Most recent: {parsed.get('is_most_recent', False)}")
        
        return parsed
        
    except Exception as e:
        print(f"[QueryParser] LLM parsing failed: {e}")
        print(f"[QueryParser] Falling back to regex parsing")
        return {}


# =============================================================================
# Query Parsing (for logging)
# =============================================================================

def _parse_search_query(query: str) -> dict:
    """
    Parse a match query using LLM for natural language understanding.
    
    Handles queries like:
        - "tell me about the most recent man city game"
        - "arsenal vs chelsea result"
        - "what happened in the liverpool match"
    
    Args:
        query: User's search query (natural language).
        
    Returns:
        dict with home_team, away_team, date, year, competition, is_most_recent.
    """
    # First, try LLM parsing for natural language understanding
    llm_result = _parse_query_with_llm(query)
    
    result = {
        "home_team": None,
        "away_team": None,
        "date": None,
        "year": None,
        "competition": None,
        "is_most_recent": False,
    }
    
    # Use LLM results if available
    if llm_result:
        result["home_team"] = llm_result.get("home_team")
        result["away_team"] = llm_result.get("away_team")
        result["date"] = llm_result.get("date")
        result["competition"] = llm_result.get("competition")
        result["is_most_recent"] = llm_result.get("is_most_recent", False)
        
        # Extract year from date if present
        if result["date"]:
            result["year"] = result["date"][:4]
    
    # If LLM didn't find teams, fall back to regex parsing
    if not result["home_team"]:
        print(f"[WebSearch] LLM didn't extract teams, using regex fallback...")
        result = _parse_search_query_regex(query, result)
    
    return result


def _parse_search_query_regex(query: str, result: dict) -> dict:
    """
    Regex-based fallback parser for search queries.
    
    Args:
        query: Original query string.
        result: Existing result dict to update.
        
    Returns:
        Updated result dict.
    """
    working_query = query
    
    # Extract competition names first
    competition_patterns = [
        r"(champions\s*league)",
        r"(europa\s*league)",
        r"(premier\s*league)",
        r"(la\s*liga)",
        r"(serie\s*a)",
        r"(bundesliga)",
        r"(ligue\s*1)",
        r"(fa\s*cup)",
        r"(carabao\s*cup)",
        r"(efl\s*cup)",
        r"(world\s*cup)",
        r"(euro\s*\d{4})",
        r"(copa\s*america)",
    ]
    
    if not result.get("competition"):
        for pattern in competition_patterns:
            comp_match = re.search(pattern, working_query, re.IGNORECASE)
            if comp_match:
                result["competition"] = comp_match.group(1)
                working_query = working_query.replace(comp_match.group(1), "")
                break
    
    # Extract date (YYYY-MM-DD)
    if not result.get("date"):
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", working_query)
        if date_match:
            result["date"] = date_match.group(1)
            result["year"] = date_match.group(1)[:4]
            working_query = working_query.replace(date_match.group(1), "")
    
    # Extract year
    if not result.get("year"):
        year_match = re.search(r"\b(20\d{2})\b", working_query)
        if year_match:
            result["year"] = year_match.group(1)
            working_query = working_query.replace(year_match.group(1), "")
    
    # Parse teams
    if not result.get("home_team"):
        separators = r"\s+(?:vs\.?|v\.?|versus|-|against|@)\s+"
        parts = re.split(separators, working_query.strip(), flags=re.IGNORECASE)
        
        if len(parts) >= 2:
            result["home_team"] = parts[0].strip()
            result["away_team"] = parts[1].strip()
        elif len(parts) == 1 and parts[0].strip():
            result["home_team"] = parts[0].strip()
    
    return result


# =============================================================================
# Relevance Scoring
# =============================================================================

def _calculate_relevance_score(result: dict, parsed: dict) -> float:
    """
    Calculate relevance score for a search result.
    
    Higher score = more relevant.
    
    Args:
        result: Search result dict.
        parsed: Parsed query info.
        
    Returns:
        Relevance score (0.0 to 100.0).
    """
    score = 0.0
    
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    url = result.get("url", "").lower()
    combined = f"{title} {snippet} {url}"
    
    home = (parsed.get("home_team") or "").lower()
    away = (parsed.get("away_team") or "").lower()
    date = parsed.get("date", "")
    year = parsed.get("year", "")
    competition = (parsed.get("competition") or "").lower()
    
    # ========== TEAM MATCHING (up to 40 points) ==========
    if home:
        # Check for team name in title (most important)
        if home in title:
            score += 15
        elif home in snippet:
            score += 8
    
    if away:
        if away in title:
            score += 15
        elif away in snippet:
            score += 8
    
    # Bonus for both teams in title (strong match signal)
    if home and away and home in title and away in title:
        score += 10
    
    # ========== DATE/YEAR MATCHING (up to 15 points) ==========
    if date and date in combined:
        score += 15
    elif year and year in combined:
        score += 8
    
    # ========== COMPETITION MATCHING (up to 10 points) ==========
    if competition and competition in combined:
        score += 10
    
    # ========== SOURCE TRUST SCORE (up to 20 points) ==========
    for i, source in enumerate(TRUSTED_SOURCES):
        if source in url:
            # Higher score for sources at the top of the list
            trust_score = 20 - min(i, 15)  # 20 to 5 points
            score += trust_score
            break
    
    # ========== RELEVANCE KEYWORD BONUS (up to 15 points) ==========
    keyword_matches = 0
    for keyword in RELEVANCE_KEYWORDS:
        if keyword in combined:
            keyword_matches += 1
    score += min(keyword_matches * 2, 15)
    
    # ========== LOW RELEVANCE PENALTY (subtract up to 30 points) ==========
    for keyword in LOW_RELEVANCE_KEYWORDS:
        if keyword in combined:
            score -= 10
    
    # Clamp score to valid range
    return max(0.0, min(100.0, score))


def _filter_and_rank_results(results: list[dict], parsed: dict) -> list[dict]:
    """
    Filter and rank search results by relevance.
    
    Args:
        results: Raw search results.
        parsed: Parsed query info.
        
    Returns:
        Filtered and ranked results.
    """
    if not results:
        return []
    
    # Score all results
    scored = []
    for result in results:
        score = _calculate_relevance_score(result, parsed)
        result["_relevance_score"] = score
        scored.append((score, result))
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Filter out very low scoring results (likely irrelevant)
    MIN_SCORE_THRESHOLD = 10.0
    filtered = [r for (score, r) in scored if score >= MIN_SCORE_THRESHOLD]
    
    # Log scoring for debugging
    print(f"[WebSearch] Relevance scoring:")
    for score, result in scored[:8]:
        url_short = result.get("url", "")[:50]
        status = "✓" if score >= MIN_SCORE_THRESHOLD else "✗"
        print(f"  {status} Score {score:.1f}: {url_short}...")
    
    return filtered


# =============================================================================
# Web Search with DDG Library
# =============================================================================

def _search_ddg(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Search using DuckDuckGo ddgs library.
    
    Args:
        query: Search query.
        max_results: Maximum results.
        
    Returns:
        List of search results.
    """
    try:
        from ddgs import DDGS
        
        print(f"[WebSearch] DDG query: \"{query}\"")
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                result = {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source": "DuckDuckGo",
                }
                results.append(result)
                print(f"[WebSearch]   Found: {result['title'][:60]}...")
        
        print(f"[WebSearch] Total results: {len(results)}")
        return results
        
    except ImportError:
        print("[WebSearch] ERROR: ddgs library not installed")
        return []
    except Exception as e:
        print(f"[WebSearch] ERROR: {e}")
        return []


def _build_search_queries(parsed: dict, query: str) -> list[str]:
    """
    Build a list of highly targeted search queries.
    
    Handles both specific match queries and "most recent" style queries.
    
    Args:
        parsed: Parsed query info.
        query: Original query string.
        
    Returns:
        List of search queries (most specific first).
    """
    search_queries = []
    
    home = parsed.get("home_team", "")
    away = parsed.get("away_team", "")
    date = parsed.get("date", "")
    year = parsed.get("year", "")
    competition = parsed.get("competition", "")
    is_most_recent = parsed.get("is_most_recent", False)
    
    current_year = datetime.now().year
    current_month = datetime.now().strftime("%B")
    
    # ========== MOST RECENT MATCH QUERIES ==========
    if is_most_recent and home:
        # User wants the most recent match for a team
        search_queries.append(f"{home} latest match result score {current_month} {current_year}")
        search_queries.append(f"{home} most recent game result {current_year}")
        search_queries.append(f"site:espn.com {home} latest result {current_year}")
        search_queries.append(f"site:bbc.com/sport {home} latest match")
        search_queries.append(f"{home} last match score {current_month}")
        search_queries.append(f"{home} recent results {current_year}")
        
        if competition:
            search_queries.append(f"{home} {competition} latest result {current_year}")
        
        if away:
            search_queries.append(f"{home} vs {away} latest match {current_year}")
    
    # ========== SPECIFIC MATCH QUERIES ==========
    elif home and away:
        match_str = f"{home} vs {away}"
        
        # TIER 1: Most specific (date + teams)
        if date:
            search_queries.append(f"site:espn.com {match_str} {date}")
            search_queries.append(f"site:bbc.com {match_str} {date}")
            search_queries.append(f"{match_str} {date} score result final")
            search_queries.append(f"{match_str} {date} match report")
        
        # TIER 2: Year + teams
        if year:
            search_queries.append(f"site:espn.com {match_str} {year}")
            search_queries.append(f"{match_str} {year} score result")
            search_queries.append(f"{match_str} {year} match report highlights")
            
            if competition:
                search_queries.append(f"{match_str} {competition} {year}")
        
        # TIER 3: General match searches
        search_queries.append(f"{match_str} final score goals")
        search_queries.append(f"{match_str} match result score {current_year}")
        search_queries.append(f"{match_str} full time result")
        
        # Site-specific
        search_queries.append(f"site:skysports.com {match_str}")
        search_queries.append(f"site:goal.com {match_str}")
        search_queries.append(f"site:flashscore.com {home} {away}")
        search_queries.append(f"site:sofascore.com {home} {away}")
    
    # ========== SINGLE TEAM QUERIES ==========
    elif home:
        if date:
            search_queries.append(f"{home} match {date} score result")
        if year:
            search_queries.append(f"{home} matches {year} results")
        
        # Recent matches for single team
        search_queries.append(f"{home} latest match result score {current_year}")
        search_queries.append(f"{home} recent results {current_month} {current_year}")
        search_queries.append(f"site:espn.com {home} results {current_year}")
        
    else:
        # Fallback: use original query with enhancements
        search_queries.append(f"{query} score result")
        search_queries.append(f"{query} match report")
    
    return search_queries


def web_search(query: str) -> list[dict]:
    """
    Perform a web search for match information.
    
    Uses multiple targeted queries and relevance scoring to find
    the most relevant sources.
    
    Args:
        query: Search query.
        
    Returns:
        List of highly relevant search results.
    """
    # Parse the query
    parsed = _parse_search_query(query)
    
    print(f"\n[WebSearch] === Starting Web Search ===")
    print(f"[WebSearch] Original query: \"{query}\"")
    print(f"[WebSearch] Parsed home team: {parsed.get('home_team', 'Unknown')}")
    print(f"[WebSearch] Parsed away team: {parsed.get('away_team', 'Unknown')}")
    print(f"[WebSearch] Parsed date: {parsed.get('date', 'Not specified')}")
    print(f"[WebSearch] Parsed competition: {parsed.get('competition', 'Not specified')}")
    print(f"[WebSearch] Most recent: {parsed.get('is_most_recent', False)}")
    
    # Build targeted search queries
    search_queries = _build_search_queries(parsed, query)
    
    print(f"[WebSearch] Generated {len(search_queries)} targeted queries")
    
    # Execute searches and collect results
    all_results = []
    seen_urls = set()
    
    # Run multiple queries to gather a pool of results
    queries_to_run = search_queries[:6]  # Limit to avoid rate limiting
    
    for sq in queries_to_run:
        results = _search_ddg(sq, max_results=5)
        
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
        
        # If we have a good pool, we can stop
        if len(all_results) >= MAX_SEARCH_RESULTS:
            break
    
    print(f"[WebSearch] Collected {len(all_results)} raw results")
    
    # ========== RELEVANCE FILTERING ==========
    # Score and rank results by relevance
    ranked_results = _filter_and_rank_results(all_results, parsed)
    
    print(f"[WebSearch] After relevance filtering: {len(ranked_results)} results")
    print(f"[WebSearch] === Search Complete ===\n")
    
    # Return top results by relevance
    return ranked_results[:MAX_RESULTS]


# =============================================================================
# LLM Summarization (STRICT - NO HALLUCINATION)
# =============================================================================

def _summarize_with_llm(query: str, results: list[dict], parsed: dict) -> str:
    """
    Use LLM to summarize search results.
    
    STRICT RULES:
    - Only use information from the search results
    - If info not found, say so clearly
    - Never make up scores or events
    
    Args:
        query: Original query.
        results: Search results (already ranked by relevance).
        parsed: Parsed query info.
        
    Returns:
        Summary string.
    """
    client = _get_openai_client()
    
    # Build context from search results (include relevance for LLM context)
    context_lines = []
    for i, result in enumerate(results, 1):
        relevance = result.get("_relevance_score", 0)
        trust_indicator = "⭐" if relevance >= 40 else "📄"
        
        context_lines.append(f"[Source {i}] {trust_indicator} (relevance: {relevance:.0f}/100)")
        context_lines.append(f"Title: {result.get('title', 'Unknown')}")
        context_lines.append(f"Content: {result.get('snippet', '')}")
        context_lines.append(f"URL: {result.get('url', '')}")
        context_lines.append("")
    
    context = "\n".join(context_lines)
    
    home = parsed.get("home_team", "Unknown")
    away = parsed.get("away_team", "Unknown")
    date = parsed.get("date", "not specified")
    competition = parsed.get("competition", "")
    
    system_prompt = """You are a football match information assistant.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY report information that is EXPLICITLY stated in the search results below
2. If you cannot find the specific match score or result in the search results, say "I could not find information about this specific match"
3. NEVER make up scores, dates, or match events
4. NEVER guess or infer scores
5. If the search results are about a DIFFERENT match (wrong teams or wrong date), say so
6. Only mention scores/results if they are CLEARLY stated in the search results
7. PRIORITIZE sources with higher relevance scores (marked with ⭐)

If the search results don't contain the specific match requested, respond with:
"I could not find verified information about [Team A] vs [Team B] on [date]. The search results may contain information about different matches between these teams."
"""

    comp_context = f" ({competition})" if competition else ""
    user_message = f"""The user is looking for: {home} vs {away}{comp_context}
Date requested: {date}

Here are the search results, ranked by relevance (higher = more relevant):

{context}

Based ONLY on the search results above (do not make anything up):
1. Is there information about the specific match requested ({home} vs {away})?
2. If yes, what is the score and key information?
3. If no, clearly state that you could not find this specific match.

Remember: If you don't see the exact score in the search results, say you couldn't find it. Do NOT guess."""

    print(f"[WebSearch] Sending to LLM for summarization...")
    
    response = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Very low temperature for factual responses
        max_tokens=400,
    )
    
    answer = response.choices[0].message.content
    
    print(f"[WebSearch] LLM response received")
    
    # Add sources with relevance indicators
    answer += "\n\n📚 Sources (by relevance):"
    for result in results[:4]:
        url = result.get("url", "")
        relevance = result.get("_relevance_score", 0)
        if url:
            # Determine source tier
            tier = "🥇" if relevance >= 50 else "🥈" if relevance >= 30 else "📄"
            answer += f"\n  {tier} {url}"
    
    return answer


# =============================================================================
# Main Entry Point
# =============================================================================

def search_and_summarize(query: str, use_llm: bool = True) -> str:
    """
    Search the web for match information and summarize.
    
    STRICT: Will not hallucinate. If match not found, says so.
    
    Uses intelligent relevance scoring to return highly relevant sources.
    
    Args:
        query: Match query (e.g., "Arsenal vs Chelsea 2024-03-15").
        use_llm: Whether to use LLM for summarization.
        
    Returns:
        Summary string or "not found" message.
    """
    print(f"\n{'='*60}")
    print(f"[WebSearch] STARTING SEARCH")
    print(f"{'='*60}")
    
    # Parse the query
    parsed = _parse_search_query(query)
    
    # Perform web search (with relevance scoring)
    results = web_search(query)
    
    if not results:
        print(f"[WebSearch] NO RELEVANT RESULTS FOUND")
        return _not_found_message(query, parsed)
    
    # Log what we found with relevance scores
    print(f"\n[WebSearch] Top relevant results:")
    for i, r in enumerate(results, 1):
        score = r.get("_relevance_score", 0)
        print(f"  {i}. [Score: {score:.0f}] {r.get('title', 'No title')[:60]}...")
    
    if not use_llm:
        return _format_search_results(results)
    
    # Use LLM to summarize (with strict anti-hallucination)
    try:
        summary = _summarize_with_llm(query, results, parsed)
        print(f"[WebSearch] SEARCH COMPLETE")
        print(f"{'='*60}\n")
        return summary
    except Exception as e:
        print(f"[WebSearch] LLM ERROR: {e}")
        return _format_search_results(results)


def _format_search_results(results: list[dict]) -> str:
    """Format raw search results for display with relevance indicators."""
    if not results:
        return "No search results found."
    
    lines = ["🔍 Search Results (ranked by relevance):\n"]
    
    for i, result in enumerate(results, 1):
        relevance = result.get("_relevance_score", 0)
        tier = "🥇" if relevance >= 50 else "🥈" if relevance >= 30 else "📄"
        
        lines.append(f"{tier} {i}. {result.get('title', 'Untitled')}")
        lines.append(f"   Relevance: {relevance:.0f}/100")
        lines.append(f"   {result.get('snippet', '')[:200]}...")
        if result.get("url"):
            lines.append(f"   🔗 {result['url']}")
        lines.append("")
    
    return "\n".join(lines)


def _not_found_message(query: str, parsed: dict) -> str:
    """Return a message when no information is found."""
    home = parsed.get("home_team", "the team")
    away = parsed.get("away_team", "")
    
    match_str = f"{home} vs {away}" if away else home
    
    return (
        f"❌ Could not find match information for \"{match_str}\".\n\n"
        "This could mean:\n"
        "• The match hasn't been played yet\n"
        "• The match details aren't indexed by search engines yet\n"
        "• Try specifying the exact date (YYYY-MM-DD)\n"
        "• Try including the competition name (e.g., 'Champions League')"
    )


# =============================================================================
# Intent-Aware Search and Summarize
# =============================================================================

def search_and_summarize_with_intent(
    search_query: str,
    intent: str,
    summary_focus: str,
    original_query: str
) -> tuple[str, dict]:
    """
    Search the web and summarize based on user intent.
    
    Also extracts match metadata (teams, date) for YouTube search.
    
    Args:
        search_query: Optimized search query from query parser.
        intent: The detected intent (lineup, match_result, etc.)
        summary_focus: What to emphasize in the summary.
        original_query: The original user query for context.
        
    Returns:
        Tuple of (summary_string, match_metadata_dict)
        match_metadata contains: home_team, away_team, match_date, score
    """
    print(f"\n{'='*60}")
    print(f"[WebSearch] INTENT-AWARE SEARCH")
    print(f"{'='*60}")
    print(f"[WebSearch] Query: \"{search_query}\"")
    print(f"[WebSearch] Intent: {intent}")
    print(f"[WebSearch] Focus: {summary_focus}")
    
    # Default match metadata
    match_metadata = {
        "home_team": None,
        "away_team": None,
        "match_date": None,
        "score": None,
    }
    
    # Perform web search
    try:
        from ddgs import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=8):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        
        print(f"[WebSearch] Found {len(results)} results")
        
    except ImportError:
        return "❌ Error: ddgs library not installed. Run: pip install ddgs", match_metadata
    except Exception as e:
        return f"❌ Search error: {e}", match_metadata
    
    if not results:
        return f"❌ Could not find information for: {original_query}", match_metadata
    
    # Build context for LLM
    context_lines = []
    for i, result in enumerate(results, 1):
        context_lines.append(f"[Source {i}]")
        context_lines.append(f"Title: {result.get('title', 'Unknown')}")
        context_lines.append(f"Content: {result.get('snippet', '')}")
        context_lines.append(f"URL: {result.get('url', '')}")
        context_lines.append("")
    
    context = "\n".join(context_lines)
    
    # Build intent-specific prompt
    intent_instructions = _get_intent_instructions(intent, summary_focus)
    
    try:
        client = _get_openai_client()
        
        # First, get the summary
        system_prompt = f"""You are a football/soccer information assistant.

{intent_instructions}

IMPORTANT RULES:
1. ONLY use information from the search results provided
2. If the specific information requested is not found, say so clearly
3. NEVER make up scores, dates, names, or statistics
4. Be concise but complete
5. Use football/soccer terminology appropriately"""

        user_message = f"""User's question: "{original_query}"

Search results:
{context}

Based on the search results above, provide a summary focused on: {summary_focus}
If the requested information is not in the results, say so clearly."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Add sources
        answer += "\n\n📚 Sources:"
        for result in results[:3]:
            url = result.get("url", "")
            if url:
                answer += f"\n  • {url}"
        
        # Now extract match metadata for YouTube search
        if intent in ["match_result", "match_highlights"]:
            match_metadata = _extract_match_metadata(context, original_query, client)
        
        return answer, match_metadata
        
    except Exception as e:
        print(f"[WebSearch] LLM Error: {e}")
        return _format_raw_results(results), match_metadata


def _extract_match_metadata(context: str, query: str, client) -> dict:
    """
    Extract match metadata (teams, date, score, key moments) from search results.
    
    This info is used to validate YouTube highlights and display match summary.
    
    Args:
        context: Search results context.
        query: Original user query.
        client: OpenAI client.
        
    Returns:
        dict with home_team, away_team, match_date, score, key_moments
    """
    try:
        extraction_prompt = f"""From the search results below, extract the match information AND key moments.

Search results:
{context}

User query: "{query}"

Return ONLY valid JSON with these fields:
{{
    "home_team": "team that played at home (full name)",
    "away_team": "team that played away (full name)",
    "match_date": "YYYY-MM-DD format or null if not found",
    "score": "X-X format (home-away) or null if not found",
    "key_moments": [
        {{"minute": "45", "event": "GOAL", "description": "Player Name scored for Team", "team": "home/away"}},
        {{"minute": "67", "event": "RED_CARD", "description": "Player Name sent off", "team": "home/away"}},
        ...
    ],
    "man_of_the_match": "Player Name or null",
    "match_summary": "Brief 1-2 sentence summary of the match"
}}

Key moments should include:
- Goals (with scorer name and minute if available)
- Red cards
- Penalties (scored or missed)
- Own goals
- Key saves or near misses
- VAR decisions

Be precise. Today's date is {datetime.now().strftime("%Y-%m-%d")}."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "Extract match information and key moments as JSON. Be precise and comprehensive."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=500,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up JSON
        if answer.startswith("```"):
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
        answer = answer.strip()
        
        import json
        metadata = json.loads(answer)
        
        print(f"[WebSearch] Extracted match metadata:")
        print(f"[WebSearch]   Home: {metadata.get('home_team')}")
        print(f"[WebSearch]   Away: {metadata.get('away_team')}")
        print(f"[WebSearch]   Date: {metadata.get('match_date')}")
        print(f"[WebSearch]   Score: {metadata.get('score')}")
        
        # Log key moments
        key_moments = metadata.get('key_moments', [])
        if key_moments:
            print(f"[WebSearch]   Key moments: {len(key_moments)} events found")
            for moment in key_moments[:3]:
                print(f"[WebSearch]     - {moment.get('minute', '?')}' {moment.get('event', 'EVENT')}: {moment.get('description', '')[:40]}")
        
        return metadata
        
    except Exception as e:
        print(f"[WebSearch] Error extracting metadata: {e}")
        return {
            "home_team": None,
            "away_team": None,
            "match_date": None,
            "score": None,
            "key_moments": [],
            "man_of_the_match": None,
            "match_summary": None,
        }


def _get_intent_instructions(intent: str, summary_focus: str) -> str:
    """Get intent-specific instructions for the LLM."""
    
    instructions = {
        "match_result": """Focus on providing:
- The final score
- Key goalscorers and times
- Notable moments (red cards, penalties, etc.)
- Brief context about what the result means""",
        
        "match_highlights": """Focus on:
- The score and result
- Key moments that would be in highlights
- Goalscorers and notable plays""",
        
        "lineup": """Focus on providing:
- Starting XI for both teams if available
- Formation used
- Notable player inclusions/exclusions
- Any tactical notes
Do NOT focus on the match result unless asked.""",
        
        "player_info": """Focus on:
- Recent performances
- Statistics (goals, assists, etc.)
- Current form
- Any news about the player""",
        
        "transfer_news": """Focus on:
- Latest transfer rumors and confirmed deals
- Fee details if available
- Agent/club statements
- Timeline information""",
        
        "team_news": """Focus on:
- Latest news about the team
- Injury updates
- Manager comments
- Recent results or upcoming fixtures""",
        
        "standings": """Focus on:
- Current league positions
- Points totals
- Recent form
- Key battles for positions""",
        
        "fixtures": """Focus on:
- Upcoming match dates and times
- Opponents
- Competition (league, cup, etc.)
- Venue information""",
        
        "stats": """Focus on:
- Specific statistics requested
- Comparisons if relevant
- Context for the numbers""",
        
        "general": """Provide a comprehensive summary of the relevant information found."""
    }
    
    return instructions.get(intent, instructions["general"])


def _format_raw_results(results: list[dict]) -> str:
    """Format raw search results as fallback."""
    lines = ["🔍 Search Results:\n"]
    
    for i, result in enumerate(results[:5], 1):
        lines.append(f"{i}. {result.get('title', 'Untitled')}")
        lines.append(f"   {result.get('snippet', '')[:200]}...")
        if result.get("url"):
            lines.append(f"   🔗 {result['url']}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# RAG: Chunk, Index, and Retrieve
# =============================================================================

def _chunk_search_results(results: list[dict], query_id: str) -> list[dict]:
    """
    Chunk search results for RAG indexing.
    
    Each search result becomes one chunk with metadata.
    
    Args:
        results: List of search results.
        query_id: Unique ID for this query session.
        
    Returns:
        List of chunks ready for embedding.
    """
    chunks = []
    
    for i, result in enumerate(results):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")
        
        # Combine title and snippet as chunk text
        chunk_text = f"Title: {title}\nContent: {snippet}\nSource: {url}"
        
        chunks.append({
            "id": f"{query_id}_{i}",
            "text": chunk_text,
            "metadata": {
                "query_id": query_id,
                "source_url": url,
                "title": title,
                "chunk_index": i,
            }
        })
    
    return chunks


def _index_chunks_for_rag(chunks: list[dict]) -> None:
    """
    Index chunks into ChromaDB for RAG retrieval.
    
    Args:
        chunks: List of chunks with id, text, and metadata.
    """
    if not chunks or not RAG_AVAILABLE:
        return
    
    try:
        collection = _get_collection()
        model = _get_embedding_model()
        
        # Prepare for ChromaDB
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Generate embeddings
        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        print(f"[RAG] Indexed {len(chunks)} chunks")
        
    except Exception as e:
        print(f"[RAG] Indexing error: {e}")


def _retrieve_relevant_context(query: str, query_id: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks using semantic search OR simple fallback.
    
    Args:
        query: The user's question.
        query_id: Query session ID to filter results.
        chunks: Original chunks (used as fallback if RAG not available).
        k: Number of chunks to retrieve.
        
    Returns:
        List of relevant chunks with text and metadata.
    """
    if RAG_AVAILABLE:
        try:
            collection = _get_collection()
            model = _get_embedding_model()
            
            # Embed the query
            query_embedding = model.encode(query, show_progress_bar=False).tolist()
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"query_id": query_id},
                include=["documents", "metadatas"],
            )
            
            # Format results
            retrieved = []
            if results["documents"] and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    retrieved.append({
                        "text": doc,
                        "source": meta.get("source_url", ""),
                        "title": meta.get("title", ""),
                    })
            
            print(f"[RAG] Retrieved {len(retrieved)} relevant chunks via embeddings")
            return retrieved
            
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}, using fallback")
    
    # Fallback: just use all chunks
    print(f"[RAG] Using simple context (no embeddings)")
    return [{"text": c["text"], "source": c["metadata"].get("source_url", ""), "title": c["metadata"].get("title", "")} for c in chunks[:k]]


def search_with_rag(query: str, intent: str, original_query: str) -> tuple[str, dict]:
    """
    Full RAG pipeline: Search -> Chunk -> Index -> Retrieve -> Generate.
    
    Args:
        query: Optimized search query.
        intent: Query intent.
        original_query: Original user question.
        
    Returns:
        Tuple of (answer, match_metadata).
    """
    print(f"\n{'='*60}")
    print(f"[RAG] Starting RAG Pipeline")
    print(f"{'='*60}")
    print(f"[RAG] Query: \"{query}\"")
    
    match_metadata = {
        "home_team": None,
        "away_team": None,
        "match_date": None,
        "score": None,
    }
    
    # Step 1: Web Search
    try:
        from ddgs import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        
        print(f"[RAG] Step 1: Found {len(results)} search results")
        
    except Exception as e:
        return f"❌ Search error: {e}", match_metadata
    
    if not results:
        return f"❌ No results found for: {original_query}", match_metadata
    
    # Step 2: Chunk and Index
    query_id = hashlib.md5(f"{query}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    chunks = _chunk_search_results(results, query_id)
    _index_chunks_for_rag(chunks)
    print(f"[RAG] Step 2: Chunked {len(chunks)} results")
    
    # Step 3: Retrieve relevant context (semantic search or fallback)
    relevant_chunks = _retrieve_relevant_context(original_query, query_id, chunks, k=5)
    print(f"[RAG] Step 3: Using {len(relevant_chunks)} context chunks")
    
    # Step 4: Generate answer from retrieved context
    context = "\n\n".join([f"[Source: {c.get('title', 'Unknown')}]\n{c['text']}" for c in relevant_chunks])
    
    try:
        client = _get_openai_client()
        
        intent_instructions = _get_intent_instructions(intent, "key information")
        
        system_prompt = f"""You are a football information assistant using RAG (Retrieval Augmented Generation).

{intent_instructions}

IMPORTANT:
1. ONLY use information from the retrieved context below
2. If information is not in the context, say "I couldn't find this information"
3. NEVER make up facts, scores, or dates
4. Cite your sources when possible"""

        user_message = f"""Question: {original_query}

Retrieved Context:
{context}

Based on the retrieved context above, answer the question."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"[RAG] Step 4: Generated answer")
        
        # Add sources
        sources = set(c.get("source", "") for c in relevant_chunks if c.get("source"))
        if sources:
            answer += "\n\n📚 Sources:"
            for src in list(sources)[:3]:
                answer += f"\n  • {src}"
        
        # Extract match metadata if this is a match result query
        if intent in ["match_result", "match_highlights"]:
            match_metadata = _extract_match_metadata(context, original_query, client)
        
        return answer, match_metadata
        
    except Exception as e:
        print(f"[RAG] Generation error: {e}")
        return _format_raw_results(results), match_metadata
