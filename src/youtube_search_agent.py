import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import re
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY is not set. YouTube search will fail.")

# Preferred channels (searched in this order) before falling back to global search.
PREFERRED_CHANNEL_IDS: List[str] = [
    "UC6c1z7bA__85CIWZ_jpCK-Q",  # ESPN
    "UCqZQlzSHbVJrwrn5XvzrzcA",  # NBC Sports
    "UCET00YnetHT7tOpu12v8jxg",  # CBS Sports
]

AGENT_VERSION = "youtube_search_agent_v5_strict_teams_2025-12-08"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_youtube_client():
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY environment variable is not set. "
            "Set it in your .env or shell before running the server."
        )
    logger.info("Initialising YouTube client | agent_version=%s", AGENT_VERSION)
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def _safe_parse_date(value: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """Accept None, datetime, or string and normalise to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    logger.warning("Could not parse match_date '%s'; ignoring date filter.", value)
    return None


def build_search_queries(
    home_team: str,
    away_team: str,
    match_date: Optional[Union[str, datetime]] = None,
) -> List[str]:
    """
    Build a set of queries for the highlights search.
    """
    dt = _safe_parse_date(match_date)
    base1 = f"{home_team} vs {away_team} highlights"
    base2 = f"{home_team} {away_team} extended highlights"
    base3 = f"{home_team} {away_team} goals"

    queries: List[str] = [base1, base2, base3]

    if dt:
        date_str_ymd = dt.strftime("%Y-%m-%d")
        date_str_dmy = dt.strftime("%d %B %Y")
        queries.append(f"{base1} {date_str_ymd}")
        queries.append(f"{base1} {date_str_dmy}")
        queries.append(f"{home_team} {away_team} highlights {date_str_ymd}")

    # Deduplicate while preserving order
    seen = set()
    uniq_queries: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq_queries.append(q)

    logger.info("Highlight search queries (agent=%s): %s", AGENT_VERSION, uniq_queries)
    return uniq_queries


def _search_youtube(
    youtube,
    query: str,
    channel_id: Optional[str] = None,
    published_after: Optional[datetime] = None,
    published_before: Optional[datetime] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
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

    logger.info(
        "YouTube search | agent=%s | query=%s | channel=%s | after=%s | before=%s",
        AGENT_VERSION,
        query,
        channel_id,
        params.get("publishedAfter"),
        params.get("publishedBefore"),
    )

    response = youtube.search().list(**params).execute()
    items = response.get("items", [])

    results: List[Dict[str, Any]] = []
    for item in items:
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        results.append(
            {
                "video_id": video_id,
                "videoId": video_id,  # alias
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
            }
        )

    logger.info("YouTube search returned %d items", len(results))
    return results


def _dedupe_by_video_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        vid = item.get("video_id") or item.get("videoId")
        if not vid:
            continue
        if vid in seen:
            continue
        seen.add(vid)
        result.append(item)
    return result


def _search_on_channel_with_queries(
    youtube,
    channel_id: str,
    queries: List[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    dt = _safe_parse_date(match_date)
    published_after = published_before = None
    if dt:
        published_after = dt - timedelta(days=2)
        published_before = dt + timedelta(days=5)

    all_items: List[Dict[str, Any]] = []
    for q in queries:
        all_items.extend(
            _search_youtube(
                youtube=youtube,
                query=q,
                channel_id=channel_id,
                published_after=published_after,
                published_before=published_before,
                max_results=max_results,
            )
        )

    deduped = _dedupe_by_video_id(all_items)
    logger.info(
        "Channel %s produced %d candidates (before dedupe %d)",
        channel_id,
        len(deduped),
        len(all_items),
    )
    return deduped


def _search_globally_with_queries(
    youtube,
    queries: List[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    dt = _safe_parse_date(match_date)
    published_after = published_before = None
    if dt:
        published_after = dt - timedelta(days=2)
        published_before = dt + timedelta(days=5)

    all_items: List[Dict[str, Any]] = []
    for q in queries:
        all_items.extend(
            _search_youtube(
                youtube=youtube,
                query=q,
                channel_id=None,
                published_after=published_after,
                published_before=published_before,
                max_results=max_results,
            )
        )

    deduped = _dedupe_by_video_id(all_items)
    logger.info(
        "Global search produced %d candidates (before dedupe %d)",
        len(deduped),
        len(all_items),
    )
    return deduped


def _build_context_text(
    match_context: Optional[Union[str, Dict[str, Any], List[Any]]]
) -> str:
    """
    Normalise match_context (str / dict / list) into plain text for ranking.
    No .split() on non-strings – we string-ify safely.
    """
    if match_context is None:
        return ""

    if isinstance(match_context, str):
        return match_context

    if isinstance(match_context, dict):
        parts = []
        for k, v in match_context.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts)

    if isinstance(match_context, list):
        # e.g. list of events, dicts, etc.
        return "\n".join(str(x) for x in match_context)

    # Fallback
    return str(match_context)


def _normalize_team_name(name: Optional[str]) -> str:
    """Very light normalisation – lower-case and strip."""
    if not name:
        return ""
    return name.lower().strip()


def _candidate_has_teams(
    item: Dict[str, Any],
    home_team: Optional[str],
    away_team: Optional[str],
) -> bool:
    """
    HARD FILTER:
    - if both teams known: require BOTH in title/description;
    - if only home_team known: require that one;
    - if none known: accept everything.
    """
    title = (item.get("title") or "").lower()
    desc = (item.get("description") or "").lower()
    text = f"{title} {desc}"

    h = _normalize_team_name(home_team)
    a = _normalize_team_name(away_team)

    if h and a:
        return h in text and a in text
    if h:
        return h in text
    # no team info – allow (should be rare in your pipeline)
    return True


def _title_matches_context(
    title: str,
    context_text: str,
    home_team: Optional[str],
    away_team: Optional[str],
) -> bool:
    """
    Highlight-like title that also respects team names.
    """
    t = title.lower()
    h = _normalize_team_name(home_team)
    a = _normalize_team_name(away_team)

    if h and h not in t:
        return False
    if a and a not in t:
        return False

    keywords = ("highlight", "highlights", "extended", "goals", "match", " vs ", " v ")
    if any(kw in t for kw in keywords):
        return True

    return False


def _extract_score_from_context(context_text: str) -> Optional[str]:
    """
    Try to pull something like '3-1', '2–0', '4:2' from the match context.
    """
    if not context_text:
        return None
    matches = re.findall(r"\b\d+[-–:]\d+\b", context_text)
    if not matches:
        return None
    # Take the first plausible score
    return matches[0].lower()


def _parse_publish_time(item: Dict[str, Any]) -> Optional[datetime]:
    """
    Try to get a datetime for when the video was published.
    """
    ts = (
        item.get("publish_time")
        or item.get("publishTime")
        or item.get("raw", {})
        .get("snippet", {})
        .get("publishedAt")
    )
    if not ts:
        return None
    return _safe_parse_date(ts)


def _validate_with_rag(
    candidates: List[Dict[str, Any]],
    match_context: Optional[Union[str, Dict[str, Any], List[Any]]],
    home_team: Optional[str],
    away_team: Optional[str],
    match_date: Optional[Union[str, datetime]] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ranking / validation for highlight candidates.

    Heuristics used:
    - HARD FILTER by team names: if we know the teams, only accept videos whose
      title or description contain both.
    - Prefer titles that contain highlight-like words.
    - Prefer videos whose publish time is close to match_date.
    - Slight boost if title seems to contain the score from match_context.
    - Bonus for preferred channels / official broadcasters.
    """
    if not candidates:
        return []

    # Hard filter by team names first
    filtered = [
        c for c in candidates
        if _candidate_has_teams(c, home_team, away_team)
    ]

    if not filtered:
        logger.warning(
            "No candidates contained both team names. Returning 0 highlights "
            "(better empty than Mbappé vs random team)."
        )
        return []

    candidates = filtered

    context_text = _build_context_text(match_context)
    logger.info(
        "Running RAG-style ranking on %d candidates | context_preview=%s",
        len(candidates),
        context_text[:200],
    )

    dt = _safe_parse_date(match_date)
    score_token = _extract_score_from_context(context_text)

    scored: List[tuple[float, Dict[str, Any]]] = []
    for item in candidates:
        title = item.get("title", "") or ""
        t = title.lower()
        score = 0.0

        # Strong boost if our simple heuristic says "this looks like true highlights"
        if _title_matches_context(title, context_text, home_team, away_team):
            score += 2.0

        # Score proximity of publish date to match date
        pub_dt = _parse_publish_time(item)
        if dt and pub_dt:
            days_diff = abs((pub_dt.date() - dt.date()).days)
            if days_diff <= 2:
                score += 1.5
            elif days_diff <= 7:
                score += 0.75
            else:
                score -= 0.25  # likely wrong season / old game

        # Try to match score pattern in title if we extracted one
        if score_token:
            t_compact = t.replace(" ", "").replace("–", "-").replace(":", "-")
            s_compact = score_token.replace(" ", "").replace("–", "-").replace(":", "-")
            if s_compact in t_compact:
                score += 1.2

        # small bonus for preferred-channel hits
        if item.get("source_type") == "channel":
            score += 0.5

        # If channel name itself looks like an official club or major broadcaster, small bump
        chan = (item.get("channel_title") or item.get("channelTitle") or "").lower()
        if any(
            kw in chan
            for kw in ("official", "fc", "cf", "tv", "nbc sports", "espn", "sky sports")
        ):
            score += 0.5

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [item for score, item in scored]

    if scored:
        logger.info(
            "Ranking complete. Top score=%.2f, bottom score=%.2f",
            scored[0][0],
            scored[-1][0],
        )

    return ranked[:max_results]


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def search_and_display_highlights_with_metadata(
    home_team: str,
    away_team: str,
    match_date: Optional[Union[str, datetime]] = None,
    match_context: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    max_results: int = 5,
    **_: Any,
) -> List[Dict[str, Any]]:
    """
    Main function used by the API layer.

    It will:
    1. Build search queries from home_team, away_team, match_date
    2. Try each preferred channel in PREFERRED_CHANNEL_IDS in order
    3. If all channels fail, fall back to a global YouTube search
    4. Apply RAG-style heuristic to rank and trim results
    5. Return a list of video metadata dicts
    """
    logger.info(
        "Searching highlights | agent=%s | home_team=%s | away_team=%s | match_date=%s | "
        "match_context_type=%s",
        AGENT_VERSION,
        home_team,
        away_team,
        match_date,
        type(match_context).__name__ if match_context is not None else "None",
    )

    youtube = _get_youtube_client()
    queries = build_search_queries(home_team, away_team, match_date)

    # 1) Try preferred channels in order
    channel_candidates: List[Dict[str, Any]] = []
    for channel_id in PREFERRED_CHANNEL_IDS:
        logger.info("Trying preferred channel: %s", channel_id)
        items = _search_on_channel_with_queries(
            youtube=youtube,
            channel_id=channel_id,
            queries=queries,
            match_date=match_date,
            max_results=max_results * 2,  # grab a few extra for ranking
        )
        if items:
            channel_candidates = items
            break

    if channel_candidates:
        logger.info(
            "Using candidates from preferred channels (%d items)",
            len(channel_candidates),
        )
        candidates = channel_candidates
    else:
        logger.info("No channel hits; falling back to global YouTube search.")
        candidates = _search_globally_with_queries(
            youtube=youtube,
            queries=queries,
            match_date=match_date,
            max_results=max_results * 2,
        )

    # 2) Apply RAG-style validation / ranking
    try:
        validated = _validate_with_rag(
            candidates=candidates,
            match_context=match_context,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            max_results=max_results,
        )
    except Exception as e:
        # This ensures we NEVER bubble up things like "'list' object has no attribute 'split'"
        logger.exception("RAG validation failed (agent=%s): %s", AGENT_VERSION, e)
        validated = candidates[:max_results]

    logger.info("Final validated highlight count: %d", len(validated))
    return validated