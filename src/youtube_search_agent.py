import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY is not set. YouTube search will be skipped and DDG fallback will be used.")

# Known channels
ESPN_CHANNEL_ID = "UC6c1z7bA__85CIWZ_jpCK-Q"   # ESPN
NBC_CHANNEL_ID = "UCqZQlzSHbVJrwrn5XvzrzcA"   # NBC Sports
CBS_CHANNEL_ID = "UCET00YnetHT7tOpu12v8jxg"   # CBS Sports

AGENT_VERSION = "youtube_search_agent_v7_competition_aware_quota_safe_2025-12-08"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_youtube_client():
    """
    Safe YouTube client initialiser.

    - If no API key, returns None (do not raise).
    - If build fails, returns None.
    """
    if not YOUTUBE_API_KEY:
        logger.warning("YOUTUBE_API_KEY missing; YouTube client disabled, will use DDG fallback.")
        return None
    try:
        logger.info("Initialising YouTube client | agent_version=%s", AGENT_VERSION)
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        logger.warning("Failed to initialise YouTube client (%s). Will use DDG fallback.", e)
        return None


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


def _competition_profile(competition: Optional[str]) -> Dict[str, Any]:
    """
    Decide which channels and fallback domains to prefer based on competition.

    Returns:
        {
          "primary_channels": [channel_ids...],
          "secondary_channels": [channel_ids...],
          "ddg_domains": [domains...],
        }
    """
    comp = (competition or "").lower()

    # Champions League
    if any(k in comp for k in ["champions league", "ucl", "uefa champions"]):
        logger.info("Competition profile: Champions League – prioritising CBS, then ESPN/NBC")
        return {
            "primary_channels": [CBS_CHANNEL_ID],
            "secondary_channels": [ESPN_CHANNEL_ID, NBC_CHANNEL_ID],
            "ddg_domains": ["cbssports.com", "uefa.com", "youtube.com"],
        }

    # Premier League
    if any(k in comp for k in ["premier league", "epl", "english premier"]):
        logger.info("Competition profile: Premier League – prioritising NBC, then ESPN")
        return {
            "primary_channels": [NBC_CHANNEL_ID],
            "secondary_channels": [ESPN_CHANNEL_ID, CBS_CHANNEL_ID],
            "ddg_domains": ["nbcsports.com", "premierleague.com", "youtube.com"],
        }

    # La Liga
    if "la liga" in comp:
        logger.info("Competition profile: La Liga – prioritising ESPN")
        return {
            "primary_channels": [ESPN_CHANNEL_ID],
            "secondary_channels": [CBS_CHANNEL_ID, NBC_CHANNEL_ID],
            "ddg_domains": ["espn.com", "youtube.com"],
        }

    # Default / unknown competition
    logger.info("Competition profile: Default – ESPN, NBC, CBS")
    return {
        "primary_channels": [ESPN_CHANNEL_ID, NBC_CHANNEL_ID],
        "secondary_channels": [CBS_CHANNEL_ID],
        "ddg_domains": ["youtube.com"],
    }


def build_search_queries(
    home_team: str,
    away_team: str,
    match_date: Optional[Union[str, datetime]] = None,
    competition: Optional[str] = None,
) -> List[str]:
    """
    Build a set of queries for the highlights search.
    We include competition name (if any) to nudge YouTube / DDG towards the right game.
    """
    dt = _safe_parse_date(match_date)
    comp = (competition or "").strip()

    base1 = f"{home_team} vs {away_team} highlights"
    base2 = f"{home_team} {away_team} extended highlights"
    base3 = f"{home_team} {away_team} goals"

    queries: List[str] = [base1, base2, base3]

    if comp:
        queries.append(f"{base1} {comp}")
        queries.append(f"{home_team} vs {away_team} {comp} full highlights")

    if dt:
        date_str_ymd = dt.strftime("%Y-%m-%d")
        date_str_dmy = dt.strftime("%d %B %Y")
        queries.append(f"{base1} {date_str_ymd}")
        queries.append(f"{base1} {date_str_dmy}")
        queries.append(f"{home_team} {away_team} highlights {date_str_ymd}")
        if comp:
            queries.append(f"{home_team} vs {away_team} {comp} {date_str_ymd}")

    # Deduplicate while preserving order
    seen = set()
    uniq_queries: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq_queries.append(q)

    logger.info(
        "Highlight search queries (agent=%s): %s",
        AGENT_VERSION,
        uniq_queries,
    )
    return uniq_queries


def _search_youtube(
    youtube,
    query: str,
    channel_id: Optional[str] = None,
    published_after: Optional[datetime] = None,
    published_before: Optional[datetime] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Low-level YouTube search.

    IMPORTANT:
    - Never raises on quota / HTTP errors.
    - Returns [] and logs, so caller can fall back to DDG.
    - Every returned item carries `"search_query": query`.
    """
    if youtube is None:
        # Should not be called in that case, but be defensive
        logger.info("YouTube client is None, skipping YouTube search for query='%s'", query)
        return []

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

    try:
        response = youtube.search().list(**params).execute()
    except HttpError as e:
        # This is where daily / per-100-seconds quota errors show up
        logger.warning(
            "YouTube API HttpError (likely quota) for query='%s'. "
            "Status=%s, Content=%s. Using DDG fallback instead.",
            query,
            getattr(e, "status_code", None),
            getattr(e, "content", "")[:200],
        )
        return []
    except Exception as e:
        logger.warning(
            "YouTube API unexpected error for query='%s'. Error=%s. "
            "Using DDG fallback instead.",
            query,
            e,
        )
        return []

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
                # which text we actually searched for this item
                "search_query": query,
            }
        )

    logger.info("YouTube search returned %d items for query='%s'", len(results), query)
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
    max_results: int = 5,
    max_queries: int = 4,
) -> List[Dict[str, Any]]:
    """
    Search a single channel with a *limited* number of queries to save credits.
    Quota errors inside will return [] (handled in _search_youtube).
    """
    if youtube is None:
        return []

    dt = _safe_parse_date(match_date)
    published_after = published_before = None
    if dt:
        published_after = dt - timedelta(days=2)
        published_before = dt + timedelta(days=5)

    all_items: List[Dict[str, Any]] = []
    for q in queries[:max_queries]:
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
    max_results: int = 5,
    max_queries: int = 4,
) -> List[Dict[str, Any]]:
    """
    Global YouTube search (no channel filter), with limited queries to save credits.
    Quota errors inside will return [] (handled in _search_youtube).
    """
    if youtube is None:
        return []

    dt = _safe_parse_date(match_date)
    published_after = published_before = None
    if dt:
        published_after = dt - timedelta(days=2)
        published_before = dt + timedelta(days=5)

    all_items: List[Dict[str, Any]] = []
    for q in queries[:max_queries]:
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


# ---------------------------------------------------------------------------
# DDG FALLBACK (competition-aware domains)
# ---------------------------------------------------------------------------

def _ddg_search_highlights(
    home_team: str,
    away_team: str,
    competition: Optional[str],
    domains: List[str],
    match_date: Optional[Union[str, datetime]],
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fallback when YouTube Data API does not give good candidates or is quota-limited.

    Uses DuckDuckGo (ddgs) to search competition-relevant domains
    and wraps results into the same candidate format used for RAG.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("[Highlights/DDG] ddgs not installed – skipping DDG fallback.")
        return []

    dt = _safe_parse_date(match_date)
    date_str = ""
    if dt:
        date_str = dt.strftime("%Y-%m-%d")

    comp = (competition or "").strip()
    base_query = f"{home_team} vs {away_team} highlights"
    if comp:
        base_query += f" {comp}"
    if date_str:
        base_query += f" {date_str}"

    logger.info(
        "[Highlights/DDG] Fallback search for '%s' across domains: %s",
        base_query,
        domains,
    )

    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    with DDGS() as ddgs:
        for domain in domains:
            search_query = f"site:{domain} {base_query}"
            logger.info("[Highlights/DDG] Query='%s'", search_query)
            try:
                for r in ddgs.text(search_query, max_results=max_results):
                    url = r.get("href") or ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = r.get("title", "") or ""
                    snippet = r.get("body", "") or ""
                    results.append(
                        {
                            "video_id": None,
                            "videoId": None,
                            "title": title,
                            "description": snippet,
                            "channel_title": domain,
                            "channelTitle": domain,
                            "publish_time": "",
                            "publishTime": "",
                            "url": url,
                            "thumbnails": {},
                            "raw": r,
                            "source_type": "ddg",
                            "search_query": search_query,
                        }
                    )
            except Exception as e:
                logger.warning("[Highlights/DDG] Error searching domain %s: %s", domain, e)

    logger.info("[Highlights/DDG] Collected %d DDG candidates", len(results))
    return results


# ---------------------------------------------------------------------------
# RANKING / VALIDATION
# ---------------------------------------------------------------------------

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
        return "\n".join(str(x) for x in match_context)

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
    - HARD FILTER by team names.
    - Require highlight-like titles (includes "highlights" in practice).
    - Prefer videos whose publish time is close to match_date.
    - Slight boost if title seems to contain the score from match_context.
    - Bonus for preferred channels / official broadcasters.
    """
    if not candidates:
        return []

    # Filter by team names
    filtered = [
        c for c in candidates
        if _candidate_has_teams(c, home_team, away_team)
    ]

    if not filtered:
        logger.warning(
            "No candidates contained both team names. Returning 0 highlights "
            "(better empty than random mismatch)."
        )
        return []

    # Require highlight-like titles
    filtered = [
        c for c in filtered
        if _title_matches_context(c.get("title", "") or "", "", home_team, away_team)
    ]

    if not filtered:
        logger.warning(
            "No candidates with highlight-like titles after team filter."
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

        # Already passed highlight-like check; small base bump
        score += 1.0

        # Publish date vs match date
        pub_dt = _parse_publish_time(item)
        if dt and pub_dt:
            days_diff = abs((pub_dt.date() - dt.date()).days)
            if days_diff <= 2:
                score += 1.5
            elif days_diff <= 7:
                score += 0.75
            else:
                score -= 0.25

        # Score pattern in title
        if score_token:
            t_compact = t.replace(" ", "").replace("–", "-").replace(":", "-")
            s_compact = score_token.replace(" ", "").replace("–", "-").replace(":", "-")
            if s_compact in t_compact:
                score += 1.2

        # Bonus for YouTube channel hits
        if item.get("source_type") == "channel":
            score += 0.5

        chan = (item.get("channel_title") or item.get("channelTitle") or "").lower()
        if any(
            kw in chan
            for kw in ("official", "fc", "cf", "tv", "nbc sports", "espn", "cbs", "sky sports")
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

    for i, (sc, item) in enumerate(scored[:5]):
        logger.info(
            "Top-%d candidate | score=%.2f | title=%s | query=%s | url=%s | source_type=%s",
            i + 1,
            sc,
            item.get("title"),
            item.get("search_query"),
            item.get("url"),
            item.get("source_type"),
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
    competition: Optional[str] = None,
    max_results: int = 5,
    **extra: Any,
) -> List[Dict[str, Any]]:
    """
    Main function used by the API layer.

    Behaviour:
    1. Derive competition-aware channel + domain profile.
    2. Build search queries.
    3. If YouTube client available:
       - Try primary channels (few queries).
       - If no validated, try secondary channels.
       - If still nothing, try global YouTube search.
    4. If YouTube unavailable OR all YouTube phases yield nothing,
       fall back to DDG with competition-aware domains.
    5. Apply RAG-style validation.
    6. Finally, if even DDG fails, return a small raw subset to avoid hard failure.
    """
    # If competition not explicitly passed, try to pull from match_metadata
    if competition is None:
        mm = extra.get("match_metadata")
        if isinstance(mm, dict):
            competition = mm.get("competition")

    logger.info(
        "Searching highlights | agent=%s | home_team=%s | away_team=%s | match_date=%s | "
        "competition=%s | match_context_type=%s",
        AGENT_VERSION,
        home_team,
        away_team,
        match_date,
        competition,
        type(match_context).__name__ if match_context is not None else "None",
    )

    youtube = _get_youtube_client()
    queries = build_search_queries(home_team, away_team, match_date, competition)
    profile = _competition_profile(competition)

    primary_channels = profile["primary_channels"]
    secondary_channels = profile["secondary_channels"]
    ddg_domains = profile["ddg_domains"]

    # If YouTube completely unavailable (no key / init failed), jump straight to DDG
    if youtube is None:
        logger.info("YouTube unavailable; using DDG-only fallback for highlights.")
        ddg_candidates = _ddg_search_highlights(
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            domains=ddg_domains,
            match_date=match_date,
            max_results=max_results * 2,
        )
        if ddg_candidates:
            validated = _validate_with_rag(
                candidates=ddg_candidates,
                match_context=match_context,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                max_results=max_results,
            )
            if validated:
                logger.info(
                    "Returning validated highlights from DDG-only fallback (%d videos)",
                    len(validated),
                )
                return validated
        return ddg_candidates[:max_results]

    # ------------------------------------------------------------------
    # 1) Primary competition-aware channels
    # ------------------------------------------------------------------
    candidates: List[Dict[str, Any]] = []
    for channel_id in primary_channels:
        logger.info("Trying primary channel: %s", channel_id)
        items = _search_on_channel_with_queries(
            youtube=youtube,
            channel_id=channel_id,
            queries=queries,
            match_date=match_date,
            max_results=max_results * 2,
            max_queries=4,
        )
        candidates.extend(items)

    validated = _validate_with_rag(
        candidates=candidates,
        match_context=match_context,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        max_results=max_results,
    )
    if validated:
        logger.info(
            "Returning validated highlights from primary channels (%d videos)",
            len(validated),
        )
        return validated

    # ------------------------------------------------------------------
    # 2) Secondary channels
    # ------------------------------------------------------------------
    secondary_candidates: List[Dict[str, Any]] = []
    for channel_id in secondary_channels:
        logger.info("Trying secondary channel: %s", channel_id)
        items = _search_on_channel_with_queries(
            youtube=youtube,
            channel_id=channel_id,
            queries=queries,
            match_date=match_date,
            max_results=max_results * 2,
            max_queries=3,
        )
        secondary_candidates.extend(items)

    validated = _validate_with_rag(
        candidates=secondary_candidates,
        match_context=match_context,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        max_results=max_results,
    )
    if validated:
        logger.info(
            "Returning validated highlights from secondary channels (%d videos)",
            len(validated),
        )
        return validated

    # ------------------------------------------------------------------
    # 3) Global YouTube search
    # ------------------------------------------------------------------
    logger.info("No good channel hits; falling back to global YouTube search.")
    global_candidates = _search_globally_with_queries(
        youtube=youtube,
        queries=queries,
        match_date=match_date,
        max_results=max_results * 2,
        max_queries=4,
    )

    validated = _validate_with_rag(
        candidates=global_candidates,
        match_context=match_context,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        max_results=max_results,
    )
    if validated:
        logger.info(
            "Returning validated highlights from global YouTube search (%d videos)",
            len(validated),
        )
        return validated

    # ------------------------------------------------------------------
    # 4) DDG fallback – competition-aware domains
    # ------------------------------------------------------------------
    logger.info("No validated YouTube highlights – trying DDG fallback.")
    ddg_candidates = _ddg_search_highlights(
        home_team=home_team,
        away_team=away_team,
        competition=competition,
        domains=ddg_domains,
        match_date=match_date,
        max_results=max_results * 2,
    )

    if ddg_candidates:
        validated = _validate_with_rag(
            candidates=ddg_candidates,
            match_context=match_context,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            max_results=max_results,
        )
        if validated:
            logger.info(
                "Returning validated highlights from DDG fallback (%d videos)",
                len(validated),
            )
            return validated

    # ------------------------------------------------------------------
    # 5) Absolute fallback – return raw few results to avoid total failure
    # ------------------------------------------------------------------
    logger.warning(
        "No validated highlights from YouTube or DDG. Returning up to %d raw candidates.",
        max_results,
    )
    all_raw = candidates + secondary_candidates + global_candidates + ddg_candidates
    return all_raw[:max_results]
