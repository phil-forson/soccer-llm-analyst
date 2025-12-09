"""
YouTube search agent for finding match highlights.

Uses YouTube Data API with DDG fallback. Features:
- Competition-aware channel prioritization
- Semantic similarity ranking for video relevance
- Strict team name filtering
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .utils import normalize_team_name


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY is not set. YouTube search will fail.")

AGENT_VERSION = "youtube_search_agent_v8_similarity_2025-12-09"

# Preferred channels by competition
COMPETITION_CHANNEL_HINTS = {
    "premier league": [
        "UCqZQlzSHbVJrwrn5XvzrzcA",  # NBC Sports
        "UC6c1z7bA__85CIWZ_jpCK-Q",  # ESPN
        "UCET00YnetHT7tOpu12v8jxg",  # CBS Sports
    ],
    "champions league": [
        "UCET00YnetHT7tOpu12v8jxg",  # CBS Sports
        "UC6c1z7bA__85CIWZ_jpCK-Q",  # ESPN
        "UCqZQlzSHbVJrwrn5XvzrzcA",  # NBC Sports
    ],
}

GLOBAL_PREFERRED_CHANNEL_IDS = [
    "UC6c1z7bA__85CIWZ_jpCK-Q",  # ESPN
    "UCqZQlzSHbVJrwrn5XvzrzcA",  # NBC Sports
    "UCET00YnetHT7tOpu12v8jxg",  # CBS Sports
]

PREFERRED_BROADCASTERS = [
    "nbcsports", "nbc sports", "espn", "espnfc", "cbs sports", "golazo"
]

# Optional sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    logger.warning("sentence-transformers not installed, using keyword ranking only")

# Lazy-loaded embedding model
_embedding_model: Optional["SentenceTransformer"] = None
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embedding_model() -> "SentenceTransformer":
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# =============================================================================
# Helpers
# =============================================================================

def _get_youtube_client():
    """Get YouTube API client."""
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY environment variable is not set.")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def _safe_parse_date(value: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """Parse date from string or datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _build_context_text(match_context: Optional[Union[str, Dict, List]]) -> str:
    """Normalize match_context to plain text."""
    if match_context is None:
        return ""
    if isinstance(match_context, str):
        return match_context
    if isinstance(match_context, dict):
        return "\n".join(f"{k}: {v}" for k, v in match_context.items())
    if isinstance(match_context, list):
        return "\n".join(str(x) for x in match_context)
    return str(match_context)


def _extract_score_from_context(context_text: str) -> Optional[str]:
    """Extract score pattern from context."""
    if not context_text:
        return None
    matches = re.findall(r"\b\d+[-–:]\d+\b", context_text)
    return matches[0].lower() if matches else None


def _parse_publish_time(item: Dict[str, Any]) -> Optional[datetime]:
    """Parse video publish time."""
    ts = (
        item.get("publish_time")
        or item.get("publishTime")
        or item.get("raw", {}).get("snippet", {}).get("publishedAt")
    )
    return _safe_parse_date(ts) if ts else None


def _normalise_competition(comp: Optional[str]) -> Optional[str]:
    """Normalize competition name."""
    if not comp:
        return None
    c = comp.lower()
    if "premier" in c:
        return "premier league"
    if "champions" in c or "ucl" in c:
        return "champions league"
    return c


def _preferred_channel_order(comp: Optional[str]) -> List[str]:
    """Get preferred channel order for competition."""
    norm = _normalise_competition(comp)
    if norm and norm in COMPETITION_CHANNEL_HINTS:
        return COMPETITION_CHANNEL_HINTS[norm]
    return GLOBAL_PREFERRED_CHANNEL_IDS


# =============================================================================
# Search Query Building
# =============================================================================

def build_search_queries(
    home_team: str,
    away_team: str,
    match_date: Optional[Union[str, datetime]] = None,
    competition: Optional[str] = None,
) -> List[str]:
    """Build search queries for highlights."""
    dt = _safe_parse_date(match_date)
    comp_part = f" {competition}" if competition else ""

    queries = [
        f"{home_team} vs {away_team} highlights{comp_part}",
        f"{home_team} {away_team} extended highlights{comp_part}",
        f"{home_team} {away_team} goals{comp_part}",
    ]

    if dt:
        date_str = dt.strftime("%Y-%m-%d")
        queries.append(f"{home_team} vs {away_team} highlights {date_str}{comp_part}")
    
    return list(dict.fromkeys(queries))


# =============================================================================
# Semantic Similarity Ranking for Videos
# =============================================================================

def _compute_video_similarity_scores(
    query: str,
    match_context: str,
    candidates: List[Dict[str, Any]],
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Compute semantic similarity between query/context and video titles/descriptions.
    
    Returns list of (similarity_score, video) tuples, sorted by score descending.
    """
    if not candidates:
        return []
    
    if not SIMILARITY_AVAILABLE:
        return [(0.5, c) for c in candidates]
    
    try:
        model = _get_embedding_model()
        
        # Build the reference text (what we're looking for)
        reference_text = f"{query}. {match_context[:500]}" if match_context else query
        
        # Build text for each video
        video_texts = []
        for v in candidates:
            title = v.get("title", "")
            desc = v.get("description", "")[:300]
            channel = v.get("channel_title") or v.get("channelTitle", "")
            video_texts.append(f"{title}. {desc}. Channel: {channel}")
        
        # Encode
        ref_embedding = model.encode(reference_text, show_progress_bar=False)
        video_embeddings = model.encode(video_texts, show_progress_bar=False)
        
        # Compute cosine similarities
        ref_norm = ref_embedding / np.linalg.norm(ref_embedding)
        video_norms = video_embeddings / np.linalg.norm(video_embeddings, axis=1, keepdims=True)
        similarities = np.dot(video_norms, ref_norm)
        
        # Pair scores with videos
        scored = [(float(sim), v) for sim, v in zip(similarities, candidates)]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"[Similarity] Ranked {len(candidates)} videos by semantic similarity")
        for i, (score, v) in enumerate(scored[:3]):
            logger.info(f"  {i+1}. Score: {score:.3f} | {v.get('title', '')[:50]}")
        
        return scored
        
    except Exception as e:
        logger.error(f"[Similarity] Error: {e}")
        return [(0.5, c) for c in candidates]


# =============================================================================
# Filtering & Combined Ranking
# =============================================================================

def _candidate_has_teams(
    item: Dict[str, Any],
    home_team: Optional[str],
    away_team: Optional[str],
) -> bool:
    """Check if candidate mentions both teams."""
    title = (item.get("title") or "").lower()
    desc = (item.get("description") or "").lower()
    text = f"{title} {desc}"

    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    if h and a:
        return h in text and a in text
    if h:
        return h in text
    return True


def _rank_videos_by_similarity_and_heuristics(
    query: str,
    match_context: str,
    candidates: List[Dict[str, Any]],
    home_team: Optional[str],
    away_team: Optional[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rank videos using semantic similarity + heuristics.
    
    Combines:
    - Semantic similarity (how well video matches query/context)
    - Team name presence
    - Publication date proximity
    - Broadcaster preference
    """
    if not candidates:
        return []
    
    # Hard filter by team names first
    filtered = [c for c in candidates if _candidate_has_teams(c, home_team, away_team)]
    if not filtered:
        logger.warning("No candidates contained both team names.")
        return []
    
    # Get semantic similarity scores
    similarity_scored = _compute_video_similarity_scores(query, match_context, filtered)
    
    dt = _safe_parse_date(match_date)
    score_token = _extract_score_from_context(match_context)
    
    # Apply heuristic adjustments
    final_scored: List[Tuple[float, Dict[str, Any]]] = []
    
    for sim_score, item in similarity_scored:
        title = (item.get("title") or "").lower()
        score = sim_score  # Start with similarity (0-1)
        
        # Boost for highlight keywords
        if any(kw in title for kw in ["highlight", "highlights", "extended", "goals"]):
            score += 0.15
        
        # Boost for publication date proximity
        pub_dt = _parse_publish_time(item)
        if dt and pub_dt:
            days_diff = abs((pub_dt.date() - dt.date()).days)
            if days_diff <= 2:
                score += 0.2
            elif days_diff <= 7:
                score += 0.1
            else:
                score -= 0.1
        
        # Boost if video title contains score from context
        if score_token:
            t_compact = title.replace(" ", "").replace("–", "-")
            s_compact = score_token.replace(" ", "").replace("–", "-")
            if s_compact in t_compact:
                score += 0.15
        
        # Boost for preferred broadcasters
        chan = (item.get("channel_title") or item.get("channelTitle") or "").lower()
        desc = (item.get("description") or "").lower()
        url = (item.get("url") or "").lower()
        composite = f"{chan} {desc} {url}"
        if any(b in composite for b in PREFERRED_BROADCASTERS):
            score += 0.2
        
        final_scored.append((score, item))
    
    # Sort by final score
    final_scored.sort(key=lambda x: x[0], reverse=True)
    
    logger.info(f"[Ranking] Final video ranking (similarity + heuristics):")
    for i, (score, v) in enumerate(final_scored[:5]):
        logger.info(f"  {i+1}. Score: {score:.3f} | {v.get('title', '')[:60]}")
    
    return [item for _, item in final_scored[:max_results]]


# =============================================================================
# YouTube API Search
# =============================================================================

def _search_youtube(
    youtube,
    query: str,
    channel_id: Optional[str] = None,
    published_after: Optional[datetime] = None,
    published_before: Optional[datetime] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Low-level YouTube search."""
    params: Dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
    }

    if channel_id:
        params["channelId"] = channel_id
    if published_after:
        params["publishedAfter"] = published_after.isoformat("T") + "Z"
    if published_before:
        params["publishedBefore"] = published_before.isoformat("T") + "Z"

    try:
        response = youtube.search().list(**params).execute()
    except HttpError as e:
        logger.error("YouTube HttpError: %s", e)
        return []
    except Exception as e:
        logger.error("YouTube search failed: %s", e)
        return []

    results: List[Dict[str, Any]] = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        results.append({
                "video_id": video_id,
            "videoId": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "channelTitle": snippet.get("channelTitle", ""),
                "publish_time": snippet.get("publishedAt", ""),
                "publishTime": snippet.get("publishedAt", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnails": snippet.get("thumbnails", {}),
                "raw": item,
                "source_type": "channel" if channel_id else "global",
                "search_query": query,
        })

    return results


def _dedupe_by_video_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate videos."""
    seen = set()
    result = []
    for item in items:
        vid = item.get("video_id") or item.get("videoId")
        if vid and vid not in seen:
            seen.add(vid)
            result.append(item)
    return result


def _search_on_channel(
    youtube,
    channel_id: str,
    queries: List[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search specific channel with multiple queries."""
    dt = _safe_parse_date(match_date)
    published_after = dt - timedelta(days=2) if dt else None
    published_before = dt + timedelta(days=5) if dt else None

    all_items: List[Dict[str, Any]] = []
    for q in queries:
        all_items.extend(_search_youtube(
                youtube=youtube,
                query=q,
                channel_id=channel_id,
                published_after=published_after,
                published_before=published_before,
                max_results=max_results,
        ))
    
    return _dedupe_by_video_id(all_items)


def _search_globally(
    youtube,
    queries: List[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Global YouTube search with multiple queries."""
    dt = _safe_parse_date(match_date)
    published_after = dt - timedelta(days=2) if dt else None
    published_before = dt + timedelta(days=5) if dt else None

    all_items: List[Dict[str, Any]] = []
    for q in queries:
        all_items.extend(_search_youtube(
                youtube=youtube,
                query=q,
                channel_id=None,
                published_after=published_after,
                published_before=published_before,
                max_results=max_results,
        ))
    
    return _dedupe_by_video_id(all_items)


# =============================================================================
# DDG Fallback
# =============================================================================

def _parse_video_id_from_url(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    if "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    return None


def _ddg_search_highlights(
    home_team: str,
    away_team: str,
    competition: Optional[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Use DDG to find YouTube highlight links as fallback."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs not installed. Cannot use DDG fallback.")
        return []

    dt = _safe_parse_date(match_date)
    date_phrase = dt.strftime("%d %B %Y") if dt else ""
    comp_part = f" {competition}" if competition else ""
    
    base_q = f"{home_team} vs {away_team} highlights{comp_part}"
    queries = [base_q]
    if date_phrase:
        queries.append(f"{base_q} {date_phrase}")

    raw_results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for q in queries:
            for r in ddgs.text(q, max_results=max_results):
                url = r.get("href", "") or ""
                if "youtube.com" not in url:
                    continue
                vid = _parse_video_id_from_url(url)
                if not vid:
                    continue

                raw_results.append({
                        "video_id": vid,
                        "videoId": vid,
                    "title": r.get("title", "") or "",
                    "description": r.get("body", "") or "",
                    "channel_title": "",
                        "channelTitle": "",
                        "publish_time": "",
                        "publishTime": "",
                        "url": url,
                        "thumbnails": {},
                        "raw": r,
                        "source_type": "ddg_fallback",
                        "search_query": q,
                })
    
    return raw_results


# =============================================================================
# Main Entry Point
# =============================================================================

def search_and_display_highlights_with_metadata(
    home_team: str,
    away_team: str,
    match_date: Optional[Union[str, datetime]] = None,
    match_context: Optional[Union[str, Dict, List]] = None,
    max_results: int = 5,
    **extra: Any,
) -> List[Dict[str, Any]]:
    """
    Main function for searching highlight videos with semantic similarity ranking.
    
    Flow:
    1. Build search queries
    2. Try YouTube API (competition-aware channels)
    3. Fall back to DDG if needed
    4. Rank by semantic similarity + heuristics
    
    Returns list of video metadata dicts.
    """
    # Extract competition
    competition = extra.get("competition")
    match_metadata = extra.get("match_metadata")
    if not competition and isinstance(match_metadata, dict):
        competition = match_metadata.get("competition")

    logger.info(
        "Searching highlights | home=%s | away=%s | date=%s | competition=%s",
        home_team, away_team, match_date, competition
    )
    
    # Build search query string for similarity matching
    search_query = f"{home_team} vs {away_team} highlights"
    if competition:
        search_query += f" {competition}"
    
    # Build context text for similarity matching
    context_text = _build_context_text(match_context or extra.get("web_summary") or match_metadata)

    queries = build_search_queries(home_team, away_team, match_date, competition)

    candidates: List[Dict[str, Any]] = []
    youtube = None

    # Try YouTube API
    try:
        youtube = _get_youtube_client()
    except Exception as e:
        logger.error("YouTube client init failed: %s", e)

    if youtube:
        channel_order = _preferred_channel_order(competition)

        for chan in channel_order:
            channel_items = _search_on_channel(
                youtube=youtube,
                channel_id=chan,
                queries=queries,
                match_date=match_date,
                max_results=max_results * 2,
            )
            if channel_items:
                # Rank with similarity
                validated = _rank_videos_by_similarity_and_heuristics(
                    query=search_query,
                    match_context=context_text,
                candidates=channel_items,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                max_results=max_results,
            )
            if validated:
                candidates = validated
                break

        if not candidates:
            global_items = _search_globally(
                youtube=youtube,
                queries=queries,
                match_date=match_date,
                max_results=max_results * 2,
            )
            if global_items:
                candidates = _rank_videos_by_similarity_and_heuristics(
                    query=search_query,
                    match_context=context_text,
                    candidates=global_items,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    max_results=max_results,
                )

    # DDG fallback
    if not candidates:
        logger.info("Falling back to DDG search.")
        ddg_candidates = _ddg_search_highlights(
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            match_date=match_date,
            max_results=max_results * 2,
        )
        if ddg_candidates:
            candidates = _rank_videos_by_similarity_and_heuristics(
                query=search_query,
                match_context=context_text,
                candidates=ddg_candidates,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                max_results=max_results,
            )

    logger.info("Final highlight count: %d", len(candidates))
    return candidates
