"""
YouTube search agent for finding match highlights.

Uses YouTube Data API with DDG fallback.
Ranks videos by semantic similarity to the search query.
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY is not set. YouTube search will fail.")

# Preferred channels by competition (used for search order, not ranking)
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
# Smaller model for lower memory usage (~300MB vs ~500MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L3-v2"


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
# Similarity Ranking
# =============================================================================

def _rank_videos_by_similarity(
    query: str,
    match_context: str,
    candidates: List[Dict[str, Any]],
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rank videos using pure semantic similarity.
    
    Uses sentence-transformers to compute cosine similarity between
    the query/context and video titles/descriptions.
    """
    if not candidates:
        return []
    
    # Get semantic similarity scores
    similarity_scored = _compute_video_similarity_scores(query, match_context, candidates)
    
    logger.info(f"[Ranking] Ranked {len(candidates)} videos by semantic similarity")
    for i, (score, v) in enumerate(similarity_scored[:5]):
        logger.info(f"  {i+1}. Score: {score:.3f} | {v.get('title', '')[:60]}")
    
    return [item for _, item in similarity_scored[:max_results]]


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


# Preferred broadcasters by competition for DDG search
COMPETITION_BROADCASTERS = {
    "premier league": ["NBC Sports", "Sky Sports", "ESPN"],
    "champions league": ["CBS Sports Golazo", "CBS Sports", "TNT Sports"],
    "ucl": ["CBS Sports Golazo", "CBS Sports", "TNT Sports"],
    "la liga": ["ESPN", "beIN Sports"],
    "bundesliga": ["ESPN"],
    "serie a": ["CBS Sports", "Paramount+"],
    "ligue 1": ["beIN Sports"],
    "fa cup": ["ESPN", "BBC Sport"],
    "carabao cup": ["Sky Sports"],
    "europa league": ["CBS Sports Golazo", "CBS Sports", "TNT Sports"],
    "conference league": ["CBS Sports Golazo", "CBS Sports"],
}

# Broadcaster name patterns to detect from title/description
BROADCASTER_PATTERNS = {
    "golazo": "CBS Sports Golazo",
    "cbs sports golazo": "CBS Sports Golazo",
    "cbs golazo": "CBS Sports Golazo",
    "nbc": "NBC Sports",
    "nbcsports": "NBC Sports",
    "sky sports": "Sky Sports",
    "skysports": "Sky Sports",
    "espn": "ESPN",
    "espnfc": "ESPN",
    "cbs sports": "CBS Sports",
    "cbs": "CBS Sports",
    "paramount": "CBS Sports",
    "tnt sports": "TNT Sports",
    "bt sport": "TNT Sports",
    "bein": "beIN Sports",
    "bbc": "BBC Sport",
    "goal.com": "Goal",
}


def _infer_channel_from_text(title: str, description: str) -> str:
    """Try to infer channel name from title or description."""
    text = f"{title} {description}".lower()
    
    for pattern, channel_name in BROADCASTER_PATTERNS.items():
        if pattern in text:
            return channel_name
    
    return ""


def _get_preferred_broadcasters(competition: Optional[str]) -> List[str]:
    """Get preferred broadcasters for a competition."""
    if not competition:
        default = ["ESPN", "NBC Sports", "CBS Sports"]
        print(f"[YouTube] No competition specified, using default broadcasters: {default}")
        return default
    
    comp_lower = competition.lower()
    
    for comp_key, broadcasters in COMPETITION_BROADCASTERS.items():
        if comp_key in comp_lower:
            print(f"[YouTube] Competition '{competition}' matched '{comp_key}' → Broadcasters: {broadcasters}")
            return broadcasters
    
    default = ["ESPN", "NBC Sports", "CBS Sports"]
    print(f"[YouTube] Competition '{competition}' not found in config, using default: {default}")
    return default


def _ddg_search_highlights(
    home_team: str,
    away_team: str,
    competition: Optional[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Use DDG to find YouTube highlight links as fallback.
    
    Searches with competition-specific broadcaster names first,
    then falls back to generic search.
    """
    print(f"\n{'='*60}")
    print(f"[YouTube DDG Search]")
    print(f"{'='*60}")
    print(f"  Home: {home_team}")
    print(f"  Away: {away_team}")
    print(f"  Competition: {competition}")
    print(f"  Date: {match_date}")
    
    try:
        from ddgs import DDGS
    except ImportError:
        print("[YouTube] ERROR: ddgs not installed!")
        return []

    dt = _safe_parse_date(match_date)
    date_phrase = dt.strftime("%d %B %Y") if dt else ""
    comp_part = f" {competition}" if competition else ""
    
    # Get preferred broadcasters for this competition
    preferred_broadcasters = _get_preferred_broadcasters(competition)
    
    # Build queries: broadcaster-specific first, then generic
    queries = []
    
    # Priority 1: Broadcaster-specific queries
    for broadcaster in preferred_broadcasters[:2]:  # Top 2 broadcasters
        q = f"{home_team} vs {away_team} highlights {broadcaster}"
        if competition:
            q += f" {competition}"
        queries.append(q)
    
    # Priority 2: Generic queries
    base_q = f"{home_team} vs {away_team} highlights{comp_part}"
    queries.append(base_q)
    
    if date_phrase:
        queries.append(f"{base_q} {date_phrase}")

    print(f"\n[YouTube] Search queries (in priority order):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. \"{q}\"")

    raw_results: List[Dict[str, Any]] = []
    seen_video_ids: set = set()
    
    with DDGS() as ddgs:
        for q in queries:
            print(f"\n[YouTube] Searching: \"{q}\"")
            results_from_query = 0
            
            for r in ddgs.text(q, max_results=max_results):
                url = r.get("href", "") or ""
                if "youtube.com" not in url:
                    continue
                vid = _parse_video_id_from_url(url)
                if not vid or vid in seen_video_ids:
                    continue

                seen_video_ids.add(vid)
                title = r.get("title", "") or ""
                description = r.get("body", "") or ""
                
                # Try to infer channel from title/description
                inferred_channel = _infer_channel_from_text(title, description)
                
                raw_results.append({
                        "video_id": vid,
                        "videoId": vid,
                        "title": title,
                    "description": description,
                    "channel_title": inferred_channel,
                    "channelTitle": inferred_channel,
                        "publish_time": "",
                        "publishTime": "",
                        "url": url,
                        "thumbnails": {},
                        "raw": r,
                        "source_type": "ddg_fallback",
                        "search_query": q,
                })
                results_from_query += 1
                
                print(f"    Found: {title[:60]}...")
                if inferred_channel:
                    print(f"           Channel (inferred): {inferred_channel}")
            
            print(f"  → {results_from_query} videos from this query")
            
            # If we have enough results from broadcaster-specific queries, stop
            if len(raw_results) >= max_results:
                print(f"\n[YouTube] Got {len(raw_results)} results, stopping search")
            break

    print(f"\n[YouTube] Total: {len(raw_results)} YouTube videos found")
    print(f"{'='*60}\n")
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
    4. Rank by semantic similarity
    
    Returns list of video metadata dicts.
    """
    print(f"\n{'='*60}")
    print(f"[YouTube Highlights Search]")
    print(f"{'='*60}")
    
    # Extract competition
    competition = extra.get("competition")
    match_metadata = extra.get("match_metadata")
    if not competition and isinstance(match_metadata, dict):
        competition = match_metadata.get("competition")

    print(f"  Home Team: {home_team}")
    print(f"  Away Team: {away_team}")
    print(f"  Competition: {competition}")
    print(f"  Match Date: {match_date}")
    
    # Build search query string for similarity matching
    search_query = f"{home_team} vs {away_team} highlights"
    if competition:
        search_query += f" {competition}"
    
    print(f"  Similarity Query: \"{search_query}\"")
    
    # Build context text for similarity matching
    context_text = _build_context_text(match_context or extra.get("web_summary") or match_metadata)
    print(f"  Context Length: {len(context_text)} chars")

    queries = build_search_queries(home_team, away_team, match_date, competition)
    print(f"\n[YouTube] Built {len(queries)} search queries")

    candidates: List[Dict[str, Any]] = []
    youtube = None

    # Try YouTube API
    try:
        youtube = _get_youtube_client()
        print(f"[YouTube] API client initialized ✓")
    except Exception as e:
        print(f"[YouTube] API client FAILED: {e}")
        print(f"[YouTube] Will use DDG fallback")

    if youtube:
        channel_order = _preferred_channel_order(competition)
        print(f"[YouTube API] Searching preferred channels: {channel_order}")

        for chan in channel_order:
            print(f"[YouTube API] Searching channel: {chan}")
            channel_items = _search_on_channel(
                youtube=youtube,
                channel_id=chan,
                queries=queries,
                match_date=match_date,
                max_results=max_results * 2,
            )
            if channel_items:
                print(f"[YouTube API] Found {len(channel_items)} videos from channel {chan}")
                # Rank with similarity
                validated = _rank_videos_by_similarity(
                    query=search_query,
                    match_context=context_text,
                    candidates=channel_items,
                    max_results=max_results,
                )
                if validated:
                    candidates = validated
                    print(f"[YouTube API] Using {len(candidates)} videos from preferred channel")
                    break
            else:
                print(f"[YouTube API] No videos from channel {chan}")

        if not candidates:
            print(f"[YouTube API] No results from preferred channels, trying global search...")
            global_items = _search_globally(
                youtube=youtube,
                queries=queries,
                match_date=match_date,
                max_results=max_results * 2,
            )
            if global_items:
                print(f"[YouTube API] Found {len(global_items)} videos globally")
                candidates = _rank_videos_by_similarity(
                    query=search_query,
                    match_context=context_text,
                    candidates=global_items,
                    max_results=max_results,
                )

    # DDG fallback
    if not candidates:
        print(f"\n[YouTube] No API results, falling back to DDG search...")
        ddg_candidates = _ddg_search_highlights(
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            match_date=match_date,
            max_results=max_results * 2,
        )
        if ddg_candidates:
            print(f"[YouTube DDG] Ranking {len(ddg_candidates)} videos by similarity...")
            candidates = _rank_videos_by_similarity(
                query=search_query,
                match_context=context_text,
                candidates=ddg_candidates,
                max_results=max_results,
            )

    # Final summary
    print(f"\n{'='*60}")
    print(f"[YouTube] FINAL RESULTS: {len(candidates)} highlights")
    print(f"{'='*60}")
    for i, v in enumerate(candidates[:5], 1):
        title = v.get("title", "")[:55]
        channel = v.get("channel_title") or v.get("channelTitle") or "Unknown"
        source = v.get("source_type", "unknown")
        print(f"  {i}. [{source}] {title}...")
        print(f"     Channel: {channel}")
    print(f"{'='*60}\n")

    return candidates
