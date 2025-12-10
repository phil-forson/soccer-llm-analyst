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
# Semantic Similarity Ranking
# =============================================================================

def _compute_similarity_scores(
    query: str,
    results: List[dict],
) -> List[Tuple[float, dict]]:
    """
    Compute cosine similarity between query and each result.
    
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
        
        # Pair scores with results
        scored_results = [(float(sim), r) for sim, r in zip(similarities, results)]
        
        # Sort by similarity (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"[Similarity] Ranked {len(results)} results by semantic similarity")
        for i, (score, r) in enumerate(scored_results[:3]):
            print(f"  {i+1}. Score: {score:.3f} | {r.get('title', 'No title')[:50]}")
        
        return scored_results
        
    except Exception as e:
        print(f"[Similarity] Error computing similarity: {e}")
        return [(0.5, r) for r in results]


def _rank_by_similarity(
    query: str,
    results: List[dict],
) -> List[dict]:
    """
    Rank results using pure semantic similarity.
    
    Uses sentence-transformers to compute cosine similarity between
    the query and each result's content.
    """
    if not results:
        return []

    # Get semantic similarity scores
    similarity_scored = _compute_similarity_scores(query, results)
    
    print(f"[Ranking] Ranked {len(results)} results by semantic similarity")
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

    teams = (parsed_query.get("teams") if parsed_query else []) or []
    home_team = teams[0] if len(teams) > 0 else None
    away_team = teams[1] if len(teams) > 1 else None

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
            "no_match_found": True,
        }
        return f"❌ No results found for: {original_query}", metadata
    
    # Step 2: Fetch full articles from trusted sources
    _enrich_results_with_page_content(all_results)
    
    # Step 3: Rank by semantic similarity
    ranked_results = _rank_by_similarity(
        query=original_query,
        results=all_results,
    )
    
    # Step 4: Build context from top results
    relevant_chunks = _build_context_from_results(ranked_results, k=5)
    print(f"[WebSearch] Using top {len(relevant_chunks)} results as context")
    
    if not relevant_chunks:
        metadata = {
            "home_team": home_team, "away_team": away_team,
            "match_date": None, "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [], "man_of_the_match": None, "match_summary": None,
            "no_match_found": True,
        }
        return f"❌ Could not find reliable information for: {original_query}", metadata

    context = "\n\n".join(
        f"[Source: {c.get('title','Unknown')}]\n{c['text']}" for c in relevant_chunks
    )

    # Step 5: LLM extracts answer AND metadata in one call
    client = get_openai_client()
    
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
2. If information is not found, use null.
3. Do NOT guess or invent scores, dates, or player names.
4. The "answer" field should be a helpful, natural response to the user.
5. Output ONLY valid JSON, no markdown or extra text."""

    user_message = f"""User question: {original_query}

Context:
{context}

Extract the answer and match metadata from the context above. Return JSON only."""

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
        
        print(f"[WebSearch] LLM extracted - Score: {metadata.get('score')}, Teams: {metadata.get('home_team')} vs {metadata.get('away_team')}")
        
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
    
    # Determine if match was found
    if not (metadata.get("score") or metadata.get("match_date") or metadata.get("key_moments")):
        metadata["no_match_found"] = True
    else:
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
