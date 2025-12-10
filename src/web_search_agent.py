"""
Web search agent for finding football match information.

- Uses DuckDuckGo (ddgs) for web search.
- Fetches full HTML from trusted sources.
- Uses semantic similarity to rank results.
- LLM extracts answer and metadata in one call.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import requests
from requests.exceptions import RequestException
import datefinder

from .utils import get_openai_client, domain_from_url
from .config import DEFAULT_LLM_MODEL

# Optional BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[WebSearch] Note: bs4 not installed, falling back to simple HTML stripping")

# Optional sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    print("[WebSearch] Note: sentence-transformers not installed, using keyword ranking only")


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)

MAX_RESULTS = 10
MAX_ARTICLE_FETCH = 4
ARTICLE_TEXT_LIMIT = 12000
# Embedding model for semantic similarity
# all-MiniLM-L6-v2 is a good balance of quality and size (~90MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRUSTED_SOURCES = [
    "espn.com", "bbc.com", "bbc.co.uk", "skysports.com", "goal.com",
    "flashscore.com", "sofascore.com", "whoscored.com", "fotmob.com",
    "premierleague.com", "theguardian.com",
]

# Lazy-loaded embedding model (only loaded when needed)
_embedding_model: Optional["SentenceTransformer"] = None


def _get_embedding_model() -> "SentenceTransformer":
    """Get or initialize the embedding model for similarity search."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[Similarity] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# =============================================================================
# Search Functions
# =============================================================================

def _search_with_source(
    query: str, source: Optional[str] = None, max_results: int = 10
) -> List[Dict[str, str]]:
    """Search ddgs with optional site restriction."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("[WebSearch] ERROR: ddgs library not installed")
        return []

    results: List[Dict[str, str]] = []
    search_query = f"site:{source} {query}" if source else query

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append({
                        "title": r.get("title", "") or "",
                        "snippet": r.get("body", "") or "",
                        "url": r.get("href", "") or "",
                })
    except Exception as e:
        print(f"[WebSearch] Search error for {source or 'general'}: {e}")

    return results


# =============================================================================
# HTML Fetching and Parsing
# =============================================================================

def _fetch_url(url: str, timeout: int = 8) -> Optional[str]:
    """Fetch raw HTML from a URL."""
    if not url.startswith("http"):
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept-Language": "en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.text
    except RequestException:
        return None


def _extract_main_text_from_html(html: str) -> str:
    """Extract main article text from HTML."""
    if not html:
        return ""

    text = ""

    if BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            main = soup.find("main")
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                body = soup.body or soup
                text = body.get_text(separator="\n", strip=True)
    else:
        tmp = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        tmp = re.sub(r"<style.*?</style>", " ", tmp, flags=re.DOTALL | re.IGNORECASE)
        tmp = re.sub(r"<[^>]+>", " ", tmp)
        text = re.sub(r"\s+", " ", tmp).strip()

    return text.strip()[:ARTICLE_TEXT_LIMIT]


def _enrich_results_with_page_content(results: List[dict]) -> None:
    """Fetch full HTML for top results from trusted sources."""
    fetched = 0
    for r in results:
        if fetched >= MAX_ARTICLE_FETCH:
            break

        url = r.get("url") or ""
        if not url:
            continue

        dom = domain_from_url(url)
        if not any(dom.endswith(ts) for ts in TRUSTED_SOURCES):
            continue

        html = _fetch_url(url)
        if not html:
            continue

        main_text = _extract_main_text_from_html(html)
        if main_text:
            r["page_content"] = main_text
        fetched += 1


# =============================================================================
# Date/Year Extraction & Filtering
# =============================================================================

def _extract_dates_from_text(text: str) -> List[datetime]:
    """
    Extract all dates from text using datefinder.
    
    Returns list of datetime objects found.
    """
    if not text:
        return []
    
    try:
        # Use datefinder to extract dates
        matches = list(datefinder.find_dates(text, strict=False))
        dates = [d for d in matches if d]
        return dates
    except Exception as e:
        print(f"[DateExtraction] Error extracting dates: {e}")
        return []


def _extract_target_date(date_context: Optional[str]) -> Optional[Dict[str, Optional[int]]]:
    """
    Extract target date information (year, month, day) from date_context using datefinder.
    
    Returns dict with 'year', 'month', 'day' keys, or None if no date found.
    
    Examples:
    - "2023 season" -> {'year': 2023, 'month': None, 'day': None} (regex fallback)
    - "January 2023" -> {'year': 2023, 'month': 1, 'day': None}
    - "2023" -> {'year': 2023, 'month': None, 'day': None} (regex fallback)
    - "last year" -> None (relative, can't extract)
    - "December 8, 2024" -> {'year': 2024, 'month': 12, 'day': 8}
    - "Jan 18, 2023" -> {'year': 2023, 'month': 1, 'day': 18}
    """
    if not date_context:
        return None

    try:
        # Use datefinder to extract dates
        matches = list(datefinder.find_dates(date_context, strict=False))
        
        if matches:
            # Use the most specific date found (prefer later dates if multiple)
            date_obj = max(matches, key=lambda d: (d.year, d.month if d.month else 0, d.day if d.day else 0))
            
            return {
                'year': date_obj.year,
                'month': date_obj.month if hasattr(date_obj, 'month') else None,
                'day': date_obj.day if hasattr(date_obj, 'day') else None,
            }
    except Exception as e:
        print(f"[DateExtraction] Error extracting target date: {e}")
    
    # Fallback to regex if datefinder found nothing or failed
    # This handles cases like "2023 season" or just "2023"
    try:
        year_pattern = r'\b(19\d{2}|20\d{2}|2100)\b'
        matches = re.findall(year_pattern, date_context)
        if matches:
            year = max(int(y) for y in matches)
            return {'year': year, 'month': None, 'day': None}
    except Exception as e:
        print(f"[DateExtraction] Regex fallback also failed: {e}")

    return None


# =============================================================================
# Semantic Similarity Ranking
# =============================================================================

# Team name aliases/nicknames mapping
TEAM_ALIASES = {
    "tottenham": ["spurs", "tottenham hotspur", "tottenham fc"],
    "tottenham hotspur": ["spurs", "tottenham", "tottenham fc"],
    "manchester united": ["man united", "man u", "united", "manchester utd"],
    "manchester city": ["man city", "city", "mancity"],
    "arsenal": ["gunners", "arsenal fc"],
    "liverpool": ["liverpool fc", "the reds"],
    "chelsea": ["chelsea fc", "the blues"],
    "manchester": ["man united", "man city"],  # Ambiguous, but try both
}

def _get_team_variations(team_name: str) -> List[str]:
    """Get all variations/aliases for a team name."""
    if not team_name:
        return []
    
    team_lower = team_name.lower()
    variations = [team_lower]
    
    # Add aliases if team is in mapping
    if team_lower in TEAM_ALIASES:
        variations.extend(TEAM_ALIASES[team_lower])
    
    # Also check if any alias maps to this team
    for alias_team, aliases in TEAM_ALIASES.items():
        if team_lower in aliases:
            variations.append(alias_team)
            variations.extend(aliases)
    
    return list(set(variations))  # Remove duplicates


def _check_team_order_in_text(text: str, home_team: Optional[str], away_team: Optional[str]) -> bool:
    """
    Check if text contains teams in the expected order (home vs away).
    
    Returns True if home_team appears before away_team in the text.
    Handles team name variations and aliases (e.g., "Tottenham" matches "Spurs", "Tottenham Hotspur").
    """
    if not home_team or not away_team:
        return False
    
    text_lower = text.lower()
    
    # Get all variations for both teams
    home_variations = _get_team_variations(home_team)
    away_variations = _get_team_variations(away_team)
    
    # Find positions for any variation of each team
    home_pos = -1
    away_pos = -1
    
    for variation in home_variations:
        pos = text_lower.find(variation)
        if pos != -1:
            if home_pos == -1 or pos < home_pos:
                home_pos = pos
    
    for variation in away_variations:
        pos = text_lower.find(variation)
        if pos != -1:
            if away_pos == -1 or pos < away_pos:
                away_pos = pos
    
    # Both teams must be present
    if home_pos == -1 or away_pos == -1:
        return False
    
    # Home team should appear before away team
    return home_pos < away_pos


def _compute_similarity_scores(
    query: str,
    results: List[dict],
    target_date: Optional[Dict[str, Optional[int]]] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    emphasize_order: bool = False,
) -> List[Tuple[float, dict]]:
    """
    Compute cosine similarity between query and each result.
    
    If target_date is provided, applies date-based filtering:
    - Full date match (year+month+day): +0.2 boost
    - Month+year match: +0.15 boost
    - Year match only: +0.1 boost
    - No match: -0.3 penalty
    
    If emphasize_order is True and home_team/away_team are provided:
    - Results matching team order (home before away): +0.2 boost
    - Results with teams in wrong order: -0.25 penalty
    
    Returns list of (similarity_score, result) tuples, sorted by score descending.
    """
    if not results:
        return []

    if not SIMILARITY_AVAILABLE:
        # Fallback: return results as-is with neutral scores
        return [(0.5, r) for r in results]
    
    try:
        model = _get_embedding_model()
        
        # Build text representation for each result
        result_texts = []
        for r in results:
            page_content = r.get("page_content", "")
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            
            # Use page content if available, otherwise snippet
            content = page_content[:2000] if page_content else snippet
            text = f"{title}. {content}"
            result_texts.append(text)
        
        # Encode query and results
        query_embedding = model.encode(query, show_progress_bar=False)
        result_embeddings = model.encode(result_texts, show_progress_bar=False)
        
        # Compute cosine similarities
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        result_norms = result_embeddings / np.linalg.norm(result_embeddings, axis=1, keepdims=True)
        
        # Dot product gives cosine similarity for normalized vectors
        similarities = np.dot(result_norms, query_norm)
        
        # Apply date filtering if target_date is specified
        if target_date is not None:
            target_year = target_date.get('year')
            target_month = target_date.get('month')
            target_day = target_date.get('day')
            
            for i, (sim, r) in enumerate(zip(similarities, results)):
                # Extract dates from result
                result_text = f"{r.get('title', '')} {r.get('snippet', '')} {r.get('page_content', '')[:500]}"
                result_dates = _extract_dates_from_text(result_text)
                
                if result_dates and target_year:
                    # Check for date matches
                    match_found = False
                    boost = 0.0
                    
                    for result_date in result_dates:
                        # Check full date match (year + month + day)
                        if (result_date.year == target_year and
                            target_month and result_date.month == target_month and
                            target_day and result_date.day == target_day):
                            boost = 0.2  # Strongest boost for exact date match
                            match_found = True
                            break
                        # Check month+year match
                        elif (result_date.year == target_year and
                              target_month and result_date.month == target_month):
                            boost = max(boost, 0.15)  # Good boost for month+year match
                            match_found = True
                        # Check year match only
                        elif result_date.year == target_year:
                            boost = max(boost, 0.1)  # Basic boost for year match
                            match_found = True
                    
                    if match_found:
                        # Apply boost
                        similarities[i] = min(1.0, sim + boost)
                    else:
                        # Penalty: results with different dates get -0.3 penalty
                        # But don't go below 0.1 to avoid completely removing them
                        similarities[i] = max(0.1, sim - 0.3)
        
        # Apply team order filtering if emphasize_order is True
        if emphasize_order and home_team and away_team:
            for i, (sim, r) in enumerate(zip(similarities, results)):
                result_text = f"{r.get('title', '')} {r.get('snippet', '')} {r.get('page_content', '')[:500]}"
                matches_order = _check_team_order_in_text(result_text, home_team, away_team)
                text_lower = result_text.lower()
                
                # Check if both teams are present (using variations/aliases)
                home_variations = _get_team_variations(home_team)
                away_variations = _get_team_variations(away_team)
                has_home = any(var in text_lower for var in home_variations)
                has_away = any(var in text_lower for var in away_variations)
                has_both_teams = has_home and has_away
                
                if matches_order:
                    # Boost: results matching team order get +0.2 boost (increased from 0.1)
                    old_score = similarities[i]
                    similarities[i] = min(1.0, sim + 0.2)
                    if i < 5:  # Log top 5 results
                        print(f"[TeamOrder] Result {i+1}: '{r.get('title', '')[:50]}' - ORDER MATCH: {old_score:.3f} → {similarities[i]:.3f} (+0.2)")
                elif has_both_teams:
                    # Penalty: teams present but in wrong order -0.25 penalty (increased from 0.15)
                    old_score = similarities[i]
                    similarities[i] = max(0.1, sim - 0.25)
                    if i < 5:  # Log top 5 results
                        print(f"[TeamOrder] Result {i+1}: '{r.get('title', '')[:50]}' - WRONG ORDER: {old_score:.3f} → {similarities[i]:.3f} (-0.25)")
        
        # Pair scores with results
        scored_results = [(float(sim), r) for sim, r in zip(similarities, results)]
        
        # Sort by similarity (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"[Similarity] Ranked {len(results)} results by semantic similarity")
        if target_date:
            date_str = f"{target_date.get('year', '?')}"
            if target_date.get('month'):
                date_str = f"{target_date['month']}/{date_str}"
            if target_date.get('day'):
                date_str = f"{target_date['day']}/{date_str}"
            print(f"[Similarity] Date filtering: prioritizing {date_str}")
        if emphasize_order and home_team and away_team:
            print(f"[Similarity] Team order filtering: prioritizing {home_team} vs {away_team}")
        for i, (score, r) in enumerate(scored_results[:3]):
            print(f"  {i+1}. Score: {score:.3f} | {r.get('title', 'No title')[:50]}")
        
        return scored_results
        
    except Exception as e:
        print(f"[Similarity] Error computing similarity: {e}")
        return [(0.5, r) for r in results]


def _rank_by_similarity(
    query: str,
    results: List[dict],
    target_date: Optional[Dict[str, Optional[int]]] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    emphasize_order: bool = False,
) -> List[dict]:
    """
    Rank results using pure semantic similarity.
    
    Uses sentence-transformers to compute cosine similarity between
    the query and each result's content.
    
    If target_date is provided, applies date-based filtering to prioritize
    results matching the date (year, month, day).
    """
    if not results:
        return []

    # Get semantic similarity scores (with date and team order filtering if provided)
    similarity_scored = _compute_similarity_scores(
        query, results, 
        target_date=target_date,
        home_team=home_team,
        away_team=away_team,
        emphasize_order=emphasize_order,
    )
    
    print(f"[Ranking] Ranked {len(results)} results by semantic similarity")
    if target_date:
        date_str = f"{target_date.get('year', '?')}"
        if target_date.get('month'):
            date_str = f"{target_date['month']}/{date_str}"
        if target_date.get('day'):
            date_str = f"{target_date['day']}/{date_str}"
        print(f"[Ranking] Date filtering active: prioritizing {date_str}")
    if emphasize_order and home_team and away_team:
        print(f"[Ranking] Team order filtering active: prioritizing {home_team} vs {away_team}")
    for i, (score, r) in enumerate(similarity_scored[:5]):
        print(f"  {i+1}. Score: {score:.3f} | {r.get('title', 'No title')[:60]}")
    
    return [r for _, r in similarity_scored]


# =============================================================================
# Context Building
# =============================================================================

def _build_context_from_results(results: List[dict], k: int = 5) -> List[dict]:
    """Build context from top-k ranked results."""
    context_chunks = []
    
    for r in results[:k]:
        page_text = r.get("page_content") or ""
        snippet = r.get("snippet", "") or ""
        body = page_text if page_text else snippet

        context_chunks.append({
            "text": f"Title: {r.get('title','')}\nContent: {body}\nSource: {r.get('url','')}",
            "source": r.get("url", ""),
                    "title": r.get("title", ""),
        })
    
    return context_chunks


# =============================================================================
# Main Search Pipeline
# =============================================================================

def search_with_rag(
    query: str,
    intent: str,
    original_query: str,
    parsed_query: Optional[dict] = None,
    summary_focus: Optional[str] = None,
    gender: str = "men",  # "men", "women", "any"
) -> tuple[str, dict]:
    """
    Web search pipeline with semantic similarity ranking.
    
    Flow:
    1. Search the web (DuckDuckGo)
    2. Fetch full articles from trusted sources
    3. Rank results by semantic similarity to query
    4. Use top results as context for LLM
    5. Extract match metadata

    Returns: (answer_text, match_metadata_dict)
    """
    print(f"\n{'='*60}")
    print("[WebSearch] Starting search pipeline")
    print(f"{'='*60}")

    # Get teams from parsed query
    teams = (parsed_query.get("teams") if parsed_query else []) or []
    emphasize_order = parsed_query.get("emphasize_order", False) if parsed_query else False
    
    # When emphasize_order is True, use teams in the exact order from the query
    # First team = home, second team = away (as they appear in user's query)
    home_team = teams[0] if len(teams) > 0 else None
    away_team = teams[1] if len(teams) > 1 else None

    print(f"[WebSearch] Teams: {home_team} vs {away_team}")
    print(f"[WebSearch] Emphasize order: {emphasize_order}")
    print(f"[WebSearch] Gender preference: {gender}")

    # Build search query - use parser's optimized query if available
    search_query = query
    if parsed_query:
        # Use the query parser's optimized search_query
        search_query = parsed_query.get("search_query") or query
        
        # Add date context if available and not already in query
        date_context = parsed_query.get("date_context")
        if date_context and date_context not in search_query.lower():
            search_query = f"{search_query} {date_context}"
        
        # Only add "result score" for match intents if not already present
        if intent in ("match_result", "match_highlights"):
            if "result" not in search_query.lower() and "score" not in search_query.lower():
                search_query = f"{search_query} result score"
    
    # Add gender bias to search query
    if gender == "men" and "women" not in search_query.lower():
        search_query = f"{search_query} men"
    elif gender == "women" and "women" not in search_query.lower():
        search_query = f"{search_query} women"
    # gender == "any" - don't add anything
    
    print(f'[WebSearch] Original query: "{query}"')
    print(f'[WebSearch] Search query: "{search_query}"')
    
    # Step 1: Web search
    all_results: List[dict] = []
    seen_urls: set[str] = set()
    search_sources = [None, "espn.com", "bbc.com/sport", "theguardian.com/football"]

    for source in search_sources:
        results = _search_with_source(search_query, source, max_results=6)
        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_results.append(r)
        if len(all_results) >= MAX_RESULTS:
            break

    print(f"[WebSearch] Found {len(all_results)} search results")

    if not all_results:
        metadata = {
            "home_team": home_team, "away_team": away_team,
            "match_date": None, "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [], "man_of_the_match": None, "match_summary": None,
        }
        # Only set no_match_found for match-related intents
        if intent in ("match_result", "match_highlights"):
            metadata["no_match_found"] = True
            return f"❌ No results found for: {original_query}", metadata
        else:
            # For non-match intents, just return a helpful message
            return f"❌ No results found for: {original_query}", metadata
    
    # Step 2: Fetch full articles from trusted sources
    _enrich_results_with_page_content(all_results)
    
    # Step 3: Extract target date (year, month, day) for filtering
    # Try date_context first, but fallback to raw_query if date_context is too generic
    target_date = None
    if parsed_query:
        date_context = parsed_query.get("date_context")
        raw_query = parsed_query.get("raw_query", original_query)
        
        # Extract from date_context
        target_date = _extract_target_date(date_context)
        
        # If date_context is too generic (e.g., "2023 season") and doesn't have month/day,
        # try extracting from raw_query to get more specific date info
        if target_date and not target_date.get('month') and not target_date.get('day'):
            # Check if raw_query has more specific date info
            raw_date = _extract_target_date(raw_query)
            if raw_date and (raw_date.get('month') or raw_date.get('day')):
                # Use the more specific date from raw_query
                target_date = raw_date
                print(f"[WebSearch] Using more specific date from raw query")
        
        # If no date found in date_context, try raw_query
        if not target_date:
            target_date = _extract_target_date(raw_query)
        
        if target_date:
            date_str = f"{target_date.get('year', '?')}"
            if target_date.get('month'):
                date_str = f"{target_date['month']}/{date_str}"
            if target_date.get('day'):
                date_str = f"{target_date['day']}/{date_str}"
            print(f"[WebSearch] Target date for filtering: {date_str}")
    
    # Step 4: Rank by semantic similarity (with date and team order filtering if applicable)
    ranked_results = _rank_by_similarity(
        query=original_query,
        results=all_results,
        target_date=target_date,
        home_team=home_team,
        away_team=away_team,
        emphasize_order=emphasize_order,
    )
    
    # Step 5: Build context from top results
    relevant_chunks = _build_context_from_results(ranked_results, k=5)
    print(f"[WebSearch] Using top {len(relevant_chunks)} results as context")
    
    if not relevant_chunks:
        metadata = {
            "home_team": home_team, "away_team": away_team,
            "match_date": None, "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [], "man_of_the_match": None, "match_summary": None,
        }
        # Only set no_match_found for match-related intents
        if intent in ("match_result", "match_highlights"):
            metadata["no_match_found"] = True
            return f"❌ Could not find reliable information for: {original_query}", metadata
        else:
            # For non-match intents, just return a helpful message
            return f"❌ Could not find reliable information for: {original_query}", metadata

    context = "\n\n".join(
        f"[Source: {c.get('title','Unknown')}]\n{c['text']}" for c in relevant_chunks
    )

    # Step 6: LLM extracts answer AND metadata in one call
    client = get_openai_client()
    
    # Build context with ranking information
    context_parts = []
    for idx, c in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Result #{idx} - Ranked {idx}]")
        context_parts.append(f"Title: {c.get('title','Unknown')}")
        context_parts.append(f"Content: {c.get('text','')}")
        context_parts.append(f"Source: {c.get('source','')}")
        context_parts.append("")  # Empty line between results
    
    context = "\n".join(context_parts)
    
    # Add team order hint if emphasize_order is set
    team_order_hint = ""
    if parsed_query and parsed_query.get("emphasize_order") and home_team and away_team:
        team_order_hint = f"\n\nIMPORTANT: The user's query specified team order as '{home_team} vs {away_team}'. Prioritize results that match this team order."
    
    system_prompt = """You are a football/soccer information assistant.

You must respond with valid JSON containing:
{
  "answer": "Your natural language answer to the user's question",
  "score": "X-Y" or null if not found,
  "home_team": "Team name" or null,
  "away_team": "Team name" or null,
  "match_date": "YYYY-MM-DD" or null,
  "competition": "Competition name" or null,
  "key_moments": [
    {"minute": "45'", "event": "Goal", "description": "Player scored", "team": "Team name"}
  ]
}

STRICT RULES:
1. Only use information from the provided context.
2. Results are ranked by relevance - Result #1 is most relevant, Result #2 is second most relevant, etc.
3. PRIORITIZE higher-ranked results (Result #1, #2, #3) over lower-ranked ones.
4. If multiple matches are found, prefer the one from the highest-ranked result.
5. Extract ALL available information from the context, even if it's not the exact match requested.
6. If the exact match isn't found, extract the closest match from the HIGHEST-RANKED results.
7. For teams: Extract team names. If team order was specified in the query, prioritize matches that respect that order.
8. For dates: Extract dates in YYYY-MM-DD format. If only year/month is available, use that.
9. For scores: Look for score patterns like "2-1", "3-0", "won 2-1", "beat 3-0", etc.
10. If information is truly not found after careful search, use null.
11. Do NOT guess or invent scores, dates, or player names.
12. The "answer" field should be a helpful, natural response to the user, explaining what was found.
13. Output ONLY valid JSON, no markdown or extra text."""

    user_message = f"""User question: {original_query}{team_order_hint}

Context (results ranked by relevance, #1 is most relevant):
{context}

Extract the answer and match metadata from the context above. Prioritize information from higher-ranked results (Result #1, #2, #3). Return JSON only."""

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        raw_response = resp.choices[0].message.content.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if raw_response.startswith("```"):
            raw_response = re.sub(r"^```(?:json)?\n?", "", raw_response)
            raw_response = re.sub(r"\n?```$", "", raw_response)
        
        # Parse JSON
        llm_data = json.loads(raw_response)
        
        answer_text = llm_data.get("answer", "")
        metadata = {
            "home_team": llm_data.get("home_team"),
            "away_team": llm_data.get("away_team"),
            "match_date": llm_data.get("match_date"),
            "score": llm_data.get("score"),
            "competition": llm_data.get("competition") or (parsed_query.get("competition") if parsed_query else None),
            "key_moments": llm_data.get("key_moments", []),
            "man_of_the_match": llm_data.get("man_of_the_match"),
            "match_summary": llm_data.get("match_summary"),
        }
        
        score = metadata.get('score')
        home = metadata.get('home_team')
        away = metadata.get('away_team')
        match_date = metadata.get('match_date')
        print(f"[WebSearch] LLM extracted - Score: {score}, Teams: {home} vs {away}, Date: {match_date}")
        
        # Warn if critical info is missing
        if not score and not home and not away:
            print(f"[WebSearch] WARNING: LLM extracted no match information. Answer: {answer_text[:100] if answer_text else 'None'}")
        
    except json.JSONDecodeError as e:
        print(f"[WebSearch] JSON parse error: {e}")
        print(f"[WebSearch] Raw response: {raw_response[:500]}")
        # Fallback: use raw response as answer
        answer_text = raw_response if raw_response else _format_raw_results(ranked_results)
        metadata = {
            "home_team": home_team, "away_team": away_team,
            "match_date": None, "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [], "man_of_the_match": None, "match_summary": None,
        }
    except Exception as e:
        print(f"[WebSearch] LLM error: {e}")
        answer_text = _format_raw_results(ranked_results)
        metadata = {
            "home_team": home_team, "away_team": away_team,
            "match_date": None, "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [], "man_of_the_match": None, "match_summary": None,
        }
    
    # Determine if match was found - only for match-related intents
    if intent in ("match_result", "match_highlights"):
        if not (metadata.get("score") or metadata.get("match_date") or metadata.get("key_moments")):
            metadata["no_match_found"] = True
        else:
            metadata["no_match_found"] = False
    else:
        # For non-match intents, don't set no_match_found
        # They should just return the answer with sources
        metadata["no_match_found"] = False

    # Append sources
    srcs = {c.get("source", "") for c in relevant_chunks if c.get("source")}
    if srcs:
        answer_text += "\n\n📚 Sources:"
        for s in list(srcs)[:3]:
            answer_text += f"\n  • {s}"

    return answer_text, metadata

    
def _format_raw_results(results: List[dict]) -> str:
    """Format raw results as fallback."""
    if not results:
        return "No search results found."
    lines = ["🔍 Search results:\n"]
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. {r.get('title','Untitled')}")
        lines.append(f"   {r.get('snippet','')[:200]}...")
        if r.get("url"):
            lines.append(f"   🔗 {r['url']}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# Legacy Compatibility
# =============================================================================

def search_and_summarize_with_intent(
    search_query: str,
    intent: str,
    summary_focus: str,
    original_query: str,
    parsed_query: Optional[dict] = None,
) -> tuple[str, dict]:
    """Backwards-compatible wrapper."""
    return search_with_rag(
        search_query, intent=intent, original_query=original_query,
        parsed_query=parsed_query, summary_focus=summary_focus,
    )


def search_and_summarize(query: str, use_llm: bool = True) -> str:
    """Simple wrapper for legacy API."""
    answer, _ = search_with_rag(
        query, intent="match_result", original_query=query,
        parsed_query=None, summary_focus=None,
    )
    return answer
