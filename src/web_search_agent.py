"""
Web search agent for finding football match information.

Uses DuckDuckGo for web search (free, no API key required).
STRICT: Will NOT hallucinate - only reports what it actually finds.
"""

import re
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL


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
# Query Parsing (for logging)
# =============================================================================

def _parse_search_query(query: str) -> dict:
    """
    Parse a match query to extract teams and date for logging.
    
    Args:
        query: User's search query.
        
    Returns:
        dict with home_team, away_team, date, year info.
    """
    result = {
        "home_team": None,
        "away_team": None,
        "date": None,
        "year": None,
        "competition": None,
    }
    
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
    for pattern in competition_patterns:
        comp_match = re.search(pattern, working_query, re.IGNORECASE)
        if comp_match:
            result["competition"] = comp_match.group(1)
            working_query = working_query.replace(comp_match.group(1), "")
            break
    
    # Extract date (YYYY-MM-DD)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", working_query)
    if date_match:
        result["date"] = date_match.group(1)
        result["year"] = date_match.group(1)[:4]
        working_query = working_query.replace(date_match.group(1), "")
    
    # Extract year
    if not result["year"]:
        year_match = re.search(r"\b(20\d{2})\b", working_query)
        if year_match:
            result["year"] = year_match.group(1)
            working_query = working_query.replace(year_match.group(1), "")
    
    # Parse teams
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
    
    Strategy: Use multiple specific queries to maximize relevant hits.
    
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
    
    current_year = datetime.now().year
    
    if home and away:
        match_str = f"{home} vs {away}"
        
        # ==========  TIER 1: Most specific (date + teams) ==========
        if date:
            # Site-specific searches for authoritative sources
            search_queries.append(f"site:espn.com {match_str} {date}")
            search_queries.append(f"site:bbc.com {match_str} {date}")
            search_queries.append(f"{match_str} {date} score result final")
            search_queries.append(f"{match_str} {date} match report")
        
        # ========== TIER 2: Year + teams ==========
        if year:
            search_queries.append(f"site:espn.com {match_str} {year}")
            search_queries.append(f"{match_str} {year} score result")
            search_queries.append(f"{match_str} {year} match report highlights")
            
            # Add competition context if known
            if competition:
                search_queries.append(f"{match_str} {competition} {year}")
        
        # ========== TIER 3: General match searches ==========
        # Highly targeted keywords for match results
        search_queries.append(f"{match_str} final score goals")
        search_queries.append(f"{match_str} match result score {current_year}")
        search_queries.append(f"{match_str} full time result")
        
        # Site-specific for general match
        search_queries.append(f"site:skysports.com {match_str}")
        search_queries.append(f"site:goal.com {match_str}")
        
        # Flashscore/Sofascore for live scores (very accurate)
        search_queries.append(f"site:flashscore.com {home} {away}")
        search_queries.append(f"site:sofascore.com {home} {away}")
        
    elif home:
        # Single team search
        if date:
            search_queries.append(f"{home} match {date} score result")
        if year:
            search_queries.append(f"{home} matches {year} results")
        search_queries.append(f"{home} latest match result score")
        
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
