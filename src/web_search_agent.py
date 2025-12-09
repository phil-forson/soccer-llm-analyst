"""
Web search agent for finding football match information.

- Uses DuckDuckGo (ddgs) for web search.
- Fetches full HTML from trusted sources.
- Uses semantic similarity to rank results.
- Extracts match metadata deterministically.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import requests
from requests.exceptions import RequestException

from .utils import get_openai_client, safe_lower, domain_from_url
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
# Score/Date Extraction
# =============================================================================

def _extract_score_from_context(
    context: str, home_team: Optional[str], away_team: Optional[str]
) -> Optional[str]:
    """Deterministically extract a score 'X-Y' from context."""
    if not context:
        return None

    text = context.lower().replace("–", "-")
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    score_hits: Dict[str, Dict[str, int]] = {}

    for match in re.finditer(r"(\d{1,2})\s*-\s*(\d{1,2})", text):
        left, right = int(match.group(1)), int(match.group(2))
        if left > 15 or right > 15:
            continue

        score = f"{left}-{right}"
        window_start = max(0, match.start() - 80)
        window_end = min(len(text), match.end() + 80)
        window = text[window_start:window_end]

        priority = 0
        if home and away and home in window and away in window:
            priority = 2
        elif (home and home in window) or (away and away in window):
            priority = 1

        hit = score_hits.get(score, {"count": 0, "priority": 0})
        hit["count"] += 1
        hit["priority"] = max(hit["priority"], priority)
        score_hits[score] = hit

    if not score_hits:
        return None

    best_score, best_meta = sorted(
        score_hits.items(),
        key=lambda item: (item[1]["priority"], item[1]["count"]),
        reverse=True,
    )[0]

    if (home or away) and best_meta["priority"] == 0:
        return None

    return best_score


def _extract_date_from_context(context: str, query: str) -> Optional[str]:
    """Extract a date from the context in YYYY-MM-DD format."""
    if not context:
        return None

    try:
        patterns = [
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
            r"((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        ]

        candidates: List[Tuple[datetime, str]] = []

        for pattern in patterns:
            for match in re.findall(pattern, context, re.IGNORECASE):
                raw = match[0] if isinstance(match, tuple) else match
                raw = raw.strip()
                try:
                    if re.match(r"\d{4}-\d{2}-\d{2}$", raw):
                        dt = datetime.strptime(raw, "%Y-%m-%d")
                    else:
                        dt = None
                        for fmt in ("%d %B %Y", "%B %d %Y", "%B %d, %Y"):
                            try:
                                dt = datetime.strptime(raw, fmt)
                                break
                            except ValueError:
                                continue
                        if dt is None:
                            continue
                    candidates.append((dt, raw))
                except Exception:
                    continue

        if candidates:
            today = datetime.now()
            past = [(dt, raw) for dt, raw in candidates if dt <= today]
            if past:
                past.sort(key=lambda x: x[0], reverse=True)
                return past[0][0].strftime("%Y-%m-%d")

        if "yesterday" in safe_lower(query) or "yesterday" in safe_lower(context):
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    except Exception:
        pass

    return None


# =============================================================================
# Team Order Correction
# =============================================================================

def _resolve_expected_teams(parsed_query: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """Extract expected home/away teams from parsed query."""
    if not parsed_query:
        return None, None

    teams = parsed_query.get("teams") or []
    home_team = teams[0] if len(teams) >= 1 else None
    away_team = teams[1] if len(teams) >= 2 else None
    return home_team, away_team


def _maybe_correct_team_order_with_score(
    context: str,
    score: Optional[str],
    home_team: Optional[str],
    away_team: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Correct team order if context shows reversed order."""
    if not (context and score and home_team and away_team):
        return home_team, away_team

    text = context.lower().replace("–", "-")
    h, a = home_team.lower(), away_team.lower()
    s = score.replace("–", "-")

    pattern_home_first = re.compile(
        rf"{re.escape(h)}[^\d]{{0,20}}{re.escape(s)}[^\w]{{0,20}}{re.escape(a)}",
        re.IGNORECASE,
    )
    pattern_away_first = re.compile(
        rf"{re.escape(a)}[^\d]{{0,20}}{re.escape(s)}[^\w]{{0,20}}{re.escape(h)}",
        re.IGNORECASE,
    )

    home_first = bool(pattern_home_first.search(text))
    away_first = bool(pattern_away_first.search(text))

    if away_first and not home_first:
        return away_team, home_team

    return home_team, away_team


# =============================================================================
# Goal Event Extraction
# =============================================================================

def _extract_goal_events_from_context(
    context: str,
    home_team: Optional[str],
    away_team: Optional[str],
    score: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract goal events as key moments from article text."""
    if not context:
        return []

    text = context.replace("–", "-")
    lower = text.lower()
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    expected_goals: Optional[int] = None
    if score and "-" in score:
        try:
            left, right = score.split("-", 1)
            expected_goals = int(left) + int(right)
        except ValueError:
            pass

    goal_keywords = (
        "goal", "scores", "scored", "nets", "netted",
        "strike", "header", "penalty", "spot-kick",
        "equaliser", "equalizer", "winner"
    )

    moments: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    pat1 = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\(\s*(\d{1,2})\s*(?:'|')?", re.MULTILINE)
    pat2 = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?-?\s*minute[^\.]{0,80}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.IGNORECASE | re.MULTILINE)
    pat3 = re.compile(r"in the (\d{1,2})(?:st|nd|rd|th)? minute[^\.]{0,80}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.IGNORECASE | re.MULTILINE)

    def _infer_team_for_name(idx: int) -> Optional[str]:
        window_start = max(0, idx - 140)
        window_end = min(len(lower), idx + 140)
        window = lower[window_start:window_end]

        if not any(kw in window for kw in goal_keywords):
            return None

        has_home = bool(home and home in window)
        has_away = bool(away and away in window)

        if has_home and not has_away:
            return home_team
        if has_away and not has_home:
            return away_team
        return None

    def _add_moment(minute: str, name: str, idx: int):
        try:
            m_int = int(minute)
            if m_int <= 0 or m_int > 130:
                return
        except ValueError:
            return

        team = _infer_team_for_name(idx)
        if not team:
            return

        minute_label = f"{minute}'"
        key = (minute_label, name, team)
        if key in moments:
            return

        description = f"GOAL for {team}: {name} scores in the {minute}th minute."
        moments[key] = {
            "minute": minute_label,
            "event": "Goal",
            "description": description,
            "team": team,
        }

    for m in pat1.finditer(text):
        _add_moment(m.group(2), m.group(1).strip(), m.start())
    for m in pat2.finditer(text):
        _add_moment(m.group(1), m.group(2).strip(), m.start())
    for m in pat3.finditer(text):
        _add_moment(m.group(1), m.group(2).strip(), m.start())

    result = list(moments.values())
    try:
        result.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x["minute"]) or 0))
    except Exception:
        pass

    if expected_goals is not None and len(result) > expected_goals:
        result = result[:expected_goals]

    return result


# =============================================================================
# Metadata Building
# =============================================================================

def _build_match_metadata_from_context(
    context: str, original_query: str, parsed_query: Optional[dict]
) -> dict:
    """Build match metadata deterministically from context."""
    home_team, away_team = _resolve_expected_teams(parsed_query)

    raw_score = _extract_score_from_context(context, home_team, away_team)
    match_date = _extract_date_from_context(context, original_query)

    home_team, away_team = _maybe_correct_team_order_with_score(
        context=context, score=raw_score, home_team=home_team, away_team=away_team
    )

    key_moments = _extract_goal_events_from_context(
        context, home_team, away_team=away_team, score=raw_score
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "score": raw_score,
        "competition": parsed_query.get("competition") if parsed_query else None,
        "key_moments": key_moments,
        "man_of_the_match": None,
        "match_summary": None,
    }


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
# LLM Summarization
# =============================================================================

def _get_intent_instructions(intent: str, summary_focus: Optional[str]) -> str:
    """Get intent-specific instructions for LLM summarization."""
    base = {
        "match_result": """Focus on:
- final score
- goalscorers and timings
- key moments (red cards, penalties)""",
        "match_highlights": """Focus on:
- score and result
- highlight moments (goals, big chances, red cards)""",
        "team_news": """Focus on:
- latest news
- injuries, manager comments
- recent form""",
        "transfer_news": """Focus on:
- confirmed transfers and rumours
- fees and contract details""",
    }.get(intent, "Provide a concise, factual summary.")
    
    return base + f"\n\nUser focus: {summary_focus or 'key information'}."


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

    # Bias query for match results
    biased_query = query
    if intent in ("match_result", "match_highlights"):
        year = datetime.now().year
        biased_query = f"{query} latest match result score {year} men"

    print(f'[WebSearch] Query: "{biased_query}"')
    
    # Step 1: Web search
    all_results: List[dict] = []
    seen_urls: set[str] = set()
    search_sources = [None, "espn.com", "bbc.com/sport", "theguardian.com/football"]

    for source in search_sources:
        results = _search_with_source(biased_query, source, max_results=6)
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

    # Step 5: LLM summary
    client = get_openai_client()
    intent_instructions = _get_intent_instructions(intent, summary_focus)

    system_prompt = f"""You are a football/soccer information assistant.

{intent_instructions}

STRICT RULES:
1. Only use information from the provided context.
2. If a score is not present, say it was not found.
3. Do NOT guess or invent scores, dates, or player names.
4. If unsure, say explicitly that information is not available."""

    user_message = f"""User question: {original_query}

Context:
{context}

Based ONLY on the context above, answer the user's question."""

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WebSearch] LLM error: {e}")
        answer_text = _format_raw_results(ranked_results)
    
    # Build metadata from context
    metadata = _build_match_metadata_from_context(context, original_query, parsed_query)
    
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
