"""
Web search + RAG agent for finding football match information.

- Uses DuckDuckGo (ddgs) for web search.
- Optionally fetches full HTML from trusted sources and extracts main article text.
- Optional embeddings-based RAG via Chroma.
- Extracts match metadata deterministically (no JSON-from-LLM parsing).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse
import json


import logging

import requests
from requests.exceptions import RequestException

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL

# -----------------------------------------------------------------------------
# Optional RAG support
# -----------------------------------------------------------------------------
RAG_AVAILABLE = False
try:
    from .embeddings_store import _get_embedding_model, _get_collection

    RAG_AVAILABLE = True
except ImportError:
    print("[RAG] Note: embeddings_store not available, using simple context only")

# Optional BeautifulSoup for better HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[RAG] Note: bs4 not installed, falling back to simple HTML stripping")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Search configuration
# -----------------------------------------------------------------------------
MAX_RESULTS = 10
MAX_ARTICLE_FETCH = 4  # how many full pages we fetch per search
ARTICLE_TEXT_LIMIT = 12000  # max chars per article we keep

TRUSTED_SOURCES = [
    "espn.com",
    "bbc.com",
    "bbc.co.uk",
    "skysports.com",
    "goal.com",
    "flashscore.com",
    "sofascore.com",
    "whoscored.com",
    "fotmob.com",
    "premierleague.com",
    "theguardian.com",
]

# -----------------------------------------------------------------------------
# OpenAI client singleton
# -----------------------------------------------------------------------------
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    return str(value).lower() if value is not None else ""


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


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
                results.append(
                    {
                        "title": r.get("title", "") or "",
                        "snippet": r.get("body", "") or "",
                        "url": r.get("href", "") or "",
                    }
                )
    except Exception as e:
        print(f"[WebSearch] Search error for {source or 'general'}: {e}")

    return results


# -----------------------------------------------------------------------------
# Full-page fetching and HTML → text
# -----------------------------------------------------------------------------
def _fetch_url(url: str, timeout: int = 8) -> Optional[str]:
    """Fetch raw HTML from a URL."""
    if not url.startswith("http"):
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en;q=0.9",
    }

    try:
        logger.info("[RAG] Fetching article HTML from %s", url)
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(
                "[RAG] Non-200 status for %s: %s", url, resp.status_code
            )
            return None
        return resp.text
    except RequestException as e:
        logger.warning("[RAG] Error fetching %s: %s", url, e)
        return None


def _extract_main_text_from_html(html: str) -> str:
    """
    Extract main article text from HTML.

    Prefer BeautifulSoup if available; otherwise a simple regex-based fallback.
    """
    if not html:
        return ""

    text = ""

    if BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Heuristic: try <article>, then <main>, else full body
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
        # Very rough fallback: strip scripts/styles & tags
        tmp = re.sub(
            r"<script.*?</script>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        tmp = re.sub(
            r"<style.*?</style>", " ", tmp, flags=re.DOTALL | re.IGNORECASE
        )
        tmp = re.sub(r"<[^>]+>", " ", tmp)
        text = re.sub(r"\s+", " ", tmp).strip()

    text = text.strip()
    if len(text) > ARTICLE_TEXT_LIMIT:
        text = text[:ARTICLE_TEXT_LIMIT]

    logger.info(
        "[RAG] Extracted %d characters of main text from HTML", len(text)
    )
    return text


def _enrich_results_with_page_content(results: List[dict]) -> None:
    """
    For the top-ranked results, fetch full HTML and attach 'page_content'
    to each result dict when successful.
    """
    fetched = 0
    for r in results:
        if fetched >= MAX_ARTICLE_FETCH:
            break

        url = r.get("url") or ""
        if not url:
            continue

        dom = _domain_from_url(url)
        if not any(dom.endswith(ts) for ts in TRUSTED_SOURCES):
            # Only bother fetching from known football sources
            continue

        html = _fetch_url(url)
        if not html:
            continue

        main_text = _extract_main_text_from_html(html)
        if not main_text:
            continue

        r["page_content"] = main_text
        fetched += 1
        logger.info("[RAG] Attached page_content to result from %s", url)

    logger.info("[RAG] Enriched %d result(s) with full article content", fetched)

def _extract_score_from_answer_text(
    answer_text: str,
    context: str,
    home_team: Optional[str],
    away_team: Optional[str],
) -> Optional[str]:
    """
    Extract a score like '3-1' from the LLM's answer, but only accept it if
    the *same* score also appears in the raw context text.

    This way we still do not trust the LLM blindly – it must match something
    that existed in the scraped/snippet context.
    """
    if not answer_text:
        return None

    ans = answer_text.lower().replace("–", "-")
    ctx = context.lower().replace("–", "-")

    home = (home_team or "").lower()
    away = (away_team or "").lower()

    # Look for X-Y in the answer
    for m in re.finditer(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", ans):
        left = int(m.group(1))
        right = int(m.group(2))
        if left > 15 or right > 15:
            continue

        score = f"{left}-{right}"

        # 1) Require the same score to exist in the raw context
        if score not in ctx:
            continue

        # 2) Optionally require teams to appear near the score in the answer
        window_start = max(0, m.start() - 80)
        window_end = min(len(ans), m.end() + 80)
        window = ans[window_start:window_end]

        if home and home not in window:
            continue
        if away and away not in window:
            continue

        return score

    return None


# -----------------------------------------------------------------------------
# Deterministic metadata extraction (regex only)
# -----------------------------------------------------------------------------
def _extract_score_from_context(
    context: str, home_team: Optional[str], away_team: Optional[str]
) -> Optional[str]:
    """
    Deterministically extract a score 'X-Y' from context.
    Prefer scores that appear near the expected teams.
    """
    if not context:
        return None

    text = context.lower().replace("–", "-")
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    score_hits: Dict[str, Dict[str, int]] = {}

    for match in re.finditer(r"(\d{1,2})\s*-\s*(\d{1,2})", text):
        left = int(match.group(1))
        right = int(match.group(2))

        # Hard cap: football scores rarely exceed this on either side
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

    # If we expected teams but never saw them near the score, treat as unreliable
    if (home or away) and best_meta["priority"] == 0:
        return None

    return best_score


def _extract_date_from_context(context: str, query: str) -> Optional[str]:
    """
    Extract a date from the context if possible, in YYYY-MM-DD format.
    """
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
                if isinstance(match, tuple):
                    raw = match[0]
                else:
                    raw = match
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
                                dt = None
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

        # Relative "yesterday"
        if "yesterday" in _safe_lower(query) or "yesterday" in _safe_lower(context):
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    except Exception as e:
        print(f"[WebSearch] Date extraction error: {e}")

    return None


def _resolve_expected_teams(parsed_query: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Use parsed_query (from query_parser_agent) to figure out expected home/away.
    We assume teams[0] = home, teams[1] = away if present, but this can be
    corrected later based on context.
    """
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
    """
    Fix cases like:
      - query: "chelsea leeds"
      - parsed teams: home=Chelsea, away=Leeds
      - article: "Leeds 3-1 Chelsea"

    If context clearly contains "<Away> score <Home>" but not "<Home> score <Away>",
    swap home/away so metadata matches the article (Leeds 3-1 Chelsea).
    """
    if not (context and score and home_team and away_team):
        return home_team, away_team

    text = context.lower().replace("–", "-")
    h = home_team.lower()
    a = away_team.lower()
    s = score.replace("–", "-")

    # Allow some punctuation/words between team and score
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

    # If only the reversed ordering appears, swap
    if away_first and not home_first:
        print(
            "[WebSearch] Detected score pattern with reversed team order – swapping home/away"
        )
        return away_team, home_team

    return home_team, away_team


def _extract_goal_events_from_answer_text(
    answer_text: str,
    context: str,
    home_team: Optional[str],
    away_team: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Extract goal events from the *LLM summary text*.

    We rely on patterns like:
      • Jaka Bijol (6')
      • Dominic Calvert-Lewin (72') for Leeds

    To keep things grounded, we only keep a goal if the scorer's last name
    also appears somewhere in the raw context.
    """
    if not answer_text:
        return []

    txt = answer_text
    lower_ctx = context.lower().replace("–", "-")
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    # Allow optional bullet at the start and optional "for Team" at the end
    pat = re.compile(
        r"(?:^[\s•\-]*|\s)"                     # optional bullet / leading spaces
        r"([A-Z][A-Za-z\-]+(?:\s[A-Z][A-Za-z\-]+)*)"  # player name (1–3 words, allows hyphens)
        r"\s*\(\s*(\d{1,3})\s*['’]\s*\)"       # (minute')
        r"(?:\s*for\s+([A-Za-z ]+))?",          # optional "for Team"
        re.MULTILINE,
    )

    moments: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for m in pat.finditer(txt):
        name = m.group(1).strip()
        minute = m.group(2).strip()
        team_raw = (m.group(3) or "").strip()

        # Very light grounding: last name must appear in raw context
        last_name = name.split()[-1].lower().replace("-", " ")
        if last_name and last_name not in lower_ctx:
            continue

        # Try to map "for X" to home/away
        team: Optional[str] = None
        tr_low = team_raw.lower()
        if home and tr_low and home in tr_low:
            team = home_team
        elif away and tr_low and away in tr_low:
            team = away_team

        key = (minute, name)
        if key in moments:
            continue

        minute_label = f"{minute}'"
        desc_team = team or "Unknown team"
        description = f"GOAL for {desc_team}: {name} scores in the {minute}th minute."

        moments[key] = {
            "minute": minute_label,
            "event": "Goal",
            "description": description,
            "team": team,
        }

    # Sort by minute
    result = list(moments.values())
    try:
        result.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x["minute"]) or 0))
    except Exception:
        pass

    print(f"[RAG] Extracted {len(result)} goal events from LLM summary")
    return result

def _merge_goal_moments(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge two goal lists, deduping by (minute, description).
    Used if we want to combine regex-from-context and summary-based goals.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _add_all(lst: List[Dict[str, Any]]):
        for g in lst:
            key = (g.get("minute") or "", g.get("description") or "")
            if key not in merged:
                merged[key] = g

    _add_all(primary)
    _add_all(secondary)

    out = list(merged.values())
    try:
        out.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x.get("minute", "") or "0")))
    except Exception:
        pass
    return out



# -----------------------------------------------------------------------------
# Goal / key-moment extraction
# -----------------------------------------------------------------------------
def _extract_goal_events_from_context(
    context: str,
    home_team: Optional[str],
    away_team: Optional[str],
    score: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract goal events as key moments from article text.

    Much more conservative:

    - Require 'goal' / 'scores' / 'scored' / 'header' / 'penalty' etc.
      in the local window around the pattern.
    - Require that we can confidently assign the scorer to one of the
      two teams (otherwise drop the event).
    - Optionally cap the number of events based on the final score.
    """
    if not context:
        return []

    text = context.replace("–", "-")
    lower = text.lower()
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    # Rough upper bound on how many goals we expect
    expected_goals: Optional[int] = None
    if score and "-" in score:
        try:
            left, right = score.split("-", 1)
            expected_goals = int(left) + int(right)
        except ValueError:
            expected_goals = None

    goal_keywords = (
        "goal", "scores", "scored", "nets", "netted",
        "strike", "header", "penalty", "spot-kick",
        "equaliser", "equalizer", "winner"
    )

    moments: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    # Pattern 1: "Name (23')" or "Name (23rd minute)"
    pat1 = re.compile(
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\(\s*(\d{1,2})\s*(?:'|’)?",
        re.MULTILINE,
    )
    # Pattern 2: "23rd-minute strike from Name" or "23rd minute from Name"
    pat2 = re.compile(
        r"(\d{1,2})(?:st|nd|rd|th)?-?\s*minute[^\.]{0,80}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        re.IGNORECASE | re.MULTILINE,
    )
    # Pattern 3: "in the 23rd minute, Name scored"
    pat3 = re.compile(
        r"in the (\d{1,2})(?:st|nd|rd|th)? minute[^\.]{0,80}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        re.IGNORECASE | re.MULTILINE,
    )

    def _infer_team_for_name(idx: int) -> Optional[str]:
        """
        Look in a small window around idx to see if only one of
        home/away is mentioned -> assign that team.
        """
        window_start = max(0, idx - 140)
        window_end = min(len(lower), idx + 140)
        window = lower[window_start:window_end]

        # Must actually talk about a goal / scoring in this window
        if not any(kw in window for kw in goal_keywords):
            return None

        has_home = bool(home and home in window)
        has_away = bool(away and away in window)

        if has_home and not has_away:
            return home_team
        if has_away and not has_home:
            return away_team

        # If both or neither appear, do not guess.
        return None

    def _add_moment(minute: str, name: str, idx: int):
        # Ignore nonsense minutes
        try:
            m_int = int(minute)
            if m_int <= 0 or m_int > 130:
                return
        except ValueError:
            return

        team = _infer_team_for_name(idx)
        if not team:
            # We only keep events we can confidently attach to a team
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
        name = m.group(1).strip()
        minute = m.group(2)
        _add_moment(minute, name, m.start())

    for m in pat2.finditer(text):
        minute = m.group(1)
        name = m.group(2).strip()
        _add_moment(minute, name, m.start())

    for m in pat3.finditer(text):
        minute = m.group(1)
        name = m.group(2).strip()
        _add_moment(minute, name, m.start())

    result = list(moments.values())

    # Sort by minute numerically
    try:
        result.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x["minute"]) or 0))
    except Exception:
        pass

    # If we know the final score, cap the number of goals
    if expected_goals is not None and len(result) > expected_goals:
        # Prefer to keep the earliest N goals (typical match report order)
        result = result[:expected_goals]

    print(f"[WebSearch] Extracted {len(result)} goal events from context (strict)")
    return result

def _extract_goal_events_from_summary(
    summary_text: str,
    home_team: Optional[str],
    away_team: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Use the LLM summary to extract goal events.

    This is *stricter* than the main summariser:
    - Only extract goals explicitly mentioned in the summary_text.
    - Do NOT invent new players, teams or minutes.
    - If unsure, return an empty list.
    """
    summary_text = (summary_text or "").strip()
    if not summary_text:
        return []

    client = _get_openai_client()

    system_prompt = """
You extract *only* explicit goal events from a football match summary.

Rules:
- Use ONLY information that appears in the provided summary text.
- Do NOT invent new players, minutes, or team names.
- If a detail (minute, team) is missing for a goal, set it to null.
- If no goals are clearly described, return an empty list.

Output JSON ONLY, with this exact shape:
{
  "events": [
    {
      "player": "Full Player Name",
      "team": "Team Name or null",
      "minute": "6'"  or null,
      "description": "Short natural-language description of the goal"
    },
    ...
  ]
}
"""

    user_prompt = f"""Summary text:

{summary_text}

From this text, extract all goal events that are clearly described.
Remember: if a goal is not clearly stated, do NOT include it.
Return JSON only, conforming to the schema described above.
"""

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[WebSearch] LLM goal extraction error: {e}")
        return []

    events = data.get("events") or []
    key_moments: List[Dict[str, Any]] = []

    for ev in events:
        player = (ev.get("player") or "").strip()
        team = (ev.get("team") or None)
        minute = (ev.get("minute") or None)
        description = ev.get("description") or ""

        if not player:
            # If we do not even have a player name, skip.
            continue

        # Normalise minute string: ensure something like "6'" not "6"
        if isinstance(minute, str) and minute and not minute.endswith("'"):
            minute = minute + "'"

        if not description:
            if minute:
                description = f"GOAL for {team or 'Unknown team'}: {player} scores in the {minute} minute."
            else:
                description = f"GOAL for {team or 'Unknown team'}: {player} scores."

        key_moments.append(
            {
                "minute": minute,
                "event": "Goal",
                "description": description,
                "team": team,
            }
        )

    print(f"[WebSearch] Extracted {len(key_moments)} goal events from LLM summary")
    return key_moments


def _build_match_metadata_from_context(
    context: str, original_query: str, parsed_query: Optional[dict]
) -> dict:
    """
    Deterministic metadata:
    - home_team / away_team start from parsed_query
    - score and date extracted via regex
    - team order corrected if article shows reversed order
    - key_moments populated with goal scorers when patterns are explicit
    """
    home_team, away_team = _resolve_expected_teams(parsed_query)

    raw_score = _extract_score_from_context(context, home_team, away_team)
    match_date = _extract_date_from_context(context, original_query)

    # Correct orientation if context clearly indicates reversed order
    home_team, away_team = _maybe_correct_team_order_with_score(
        context=context,
        score=raw_score,
        home_team=home_team,
        away_team=away_team,
    )

    # goal → key_moments extraction
    key_moments = _extract_goal_events_from_context(context,home_team,away_team=away_team,score=raw_score)

    metadata = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "score": raw_score,
        "competition": parsed_query.get("competition") if parsed_query else None,
        "key_moments": key_moments,
        "man_of_the_match": None,
        "match_summary": None,
    }

    print("[WebSearch] Extracted metadata (deterministic):")
    print(f"  Home: {metadata['home_team']}")
    print(f"  Away: {metadata['away_team']}")
    print(f"  Date: {metadata['match_date']}")
    print(f"  Score: {metadata['score']}")
    print(f"  Key moments: {len(metadata['key_moments'])}")
    return metadata


# -----------------------------------------------------------------------------
# RAG: chunk, index, retrieve
# -----------------------------------------------------------------------------
def _chunk_search_results(results: List[dict], query_id: str) -> List[dict]:
    chunks: List[dict] = []
    for i, r in enumerate(results):
        # Prefer full page_content if available; otherwise fall back to snippet
        page_text = r.get("page_content") or ""
        snippet = r.get("snippet", "") or ""
        body = page_text if page_text else snippet

        text = (
            f"Title: {r.get('title','')}\n"
            f"Content: {body}\n"
            f"Source: {r.get('url','')}"
        )
        chunks.append(
            {
                "id": f"{query_id}_{i}",
                "text": text,
                "metadata": {
                    "query_id": query_id,
                    "source_url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "chunk_index": i,
                },
            }
        )
    return chunks


def _index_chunks_for_rag(chunks: List[dict]) -> None:
    if not (chunks and RAG_AVAILABLE):
        return
    try:
        collection = _get_collection()
        model = _get_embedding_model()

        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        embs = model.encode(docs, show_progress_bar=False).tolist()

        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        print(f"[RAG] Indexed {len(chunks)} chunks")
    except Exception as e:
        print(f"[RAG] Indexing error: {e}")


def _retrieve_relevant_context(
    query: str, query_id: str, chunks: List[dict], k: int = 5
) -> List[dict]:
    if RAG_AVAILABLE:
        try:
            collection = _get_collection()
            model = _get_embedding_model()
            q_emb = model.encode(query, show_progress_bar=False).tolist()
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                where={"query_id": query_id},
                include=["documents", "metadatas"],
            )

            retrieved: List[dict] = []
            if res["documents"] and res["documents"][0]:
                for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                    retrieved.append(
                        {
                            "text": doc,
                            "source": meta.get("source_url", ""),
                            "title": meta.get("title", ""),
                        }
                    )
            print(f"[RAG] Retrieved {len(retrieved)} chunks via embeddings")
            return retrieved
        except Exception as e:
            print(f"[RAG] Retrieval error: {e} – falling back to simple context")

    print("[RAG] Using simple context (no embeddings)")
    return [
        {
            "text": c["text"],
            "source": c["metadata"].get("source_url", ""),
            "title": c["metadata"].get("title", ""),
        }
        for c in chunks[:k]
    ]


# -----------------------------------------------------------------------------
# Result ranking to avoid “team overview” pages
# -----------------------------------------------------------------------------
def _rank_match_results(
    results: List[dict], home_team: Optional[str], away_team: Optional[str]
) -> List[dict]:
    """
    Heuristically rank web results so that actual match reports/score pages
    float to the top, and generic team overview/fixtures pages are pushed down.
    """
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    def _score_result(r: dict) -> int:
        title = (r.get("title") or "").lower()
        snippet = (r.get("snippet") or "").lower()
        url = (r.get("url") or "").lower()
        text = f"{title} {snippet}"

        score = 0

        # 1) Do we see home/away names?
        has_home = bool(home and home in text)
        has_away = bool(away and away in text)

        if has_home:
            score += 10
        if has_away:
            score += 10
        if has_home and has_away:
            score += 20  # both teams mentioned somewhere is good

        # 2) Patterns like "leeds vs chelsea" or 'leeds 3-1 chelsea'
        if home and away:
            vs_pattern = rf"{home}.*(vs|v|v\.)\s*{away}|{away}.*(vs|v|v\.)\s*{home}"
            score_pattern = (
                rf"{home}.*\d+\s*[-–]\s*\d+.*{away}"
                rf"|{away}.*\d+\s*[-–]\s*\d+.*{home}"
            )
            if re.search(vs_pattern, text):
                score += 25
            if re.search(score_pattern, text):
                score += 25

        # 3) Trusted domains bump
        dom = _domain_from_url(url)
        if any(dom.endswith(ts) for ts in TRUSTED_SOURCES):
            score += 10

        # 4) Penalise “generic” pages (overview, fixtures, tables, etc.)
        generic_words = [
            "overview",
            "team",
            "teams",
            "fixtures",
            "fixture",
            "schedule",
            "schedules",
            "table",
            "tables",
            "results",
        ]
        if any(w in url for w in generic_words) or any(
            w in title for w in generic_words
        ):
            if not (has_home and has_away):
                score -= 40  # generic team page with only one side
            else:
                score -= 10  # slightly penalise even if both sides are present

        return score

    ranked = sorted(results, key=_score_result, reverse=True)

    # Debug: see what we are doing
    print("[RAG] Step 1b: ranked results (top 5)")
    for r in ranked[:5]:
        print("   •", r.get("title"), "=>", r.get("url"))

    return ranked


# -----------------------------------------------------------------------------
# Intent-specific instructions for LLM summarisation
# -----------------------------------------------------------------------------
def _get_intent_instructions(intent: str, summary_focus: Optional[str]) -> str:
    base = {
        "match_result": """Focus on:
- final score
- goalscorers and timings if present
- key moments (red cards, penalties)
- what the result means (briefly)""",
        "match_highlights": """Focus on:
- score and overall result
- moments that would appear in highlights (goals, big chances, red cards)""",
        "team_news": """Focus on:
- latest news
- injuries, manager comments
- recent form""",
        "transfer_news": """Focus on:
- confirmed transfers and serious rumours
- fees / contract details if present""",
    }.get(
        intent,
        "Provide a concise, factual summary of the retrieved information.",
    )

    return base + f"\n\nUser requested focus: {summary_focus or 'key information'}."


# -----------------------------------------------------------------------------
# MAIN: RAG search pipeline
# -----------------------------------------------------------------------------
def search_with_rag(
    query: str,
    intent: str,
    original_query: str,
    parsed_query: Optional[dict] = None,
    summary_focus: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Full RAG pipeline:
    - web search (ddgs)
    - select good URLs and (for some) fetch full HTML
    - chunk + optional embeddings index
    - retrieve most relevant chunks
    - summarise with LLM
    - deterministically extract match metadata (score/date + corrected team order + goal key moments)

    Returns: (answer_text, match_metadata_dict)
    """
    print("\n" + "=" * 60)
    print("[RAG] Starting RAG pipeline")
    print("=" * 60)

    teams = (parsed_query.get("teams") if parsed_query else []) or []
    home_team = teams[0] if len(teams) > 0 else None
    away_team = teams[1] if len(teams) > 1 else None

    # Bias query to men's latest match for match_result / highlights
    biased_query = query
    if intent in ("match_result", "match_highlights"):
        year = datetime.now().year
        biased_query = f"{query} latest match result score {year} men"

    print(f'[RAG] Biased query: "{biased_query}"')
    if parsed_query:
        print("[RAG] Parsed query:")
        print(f"  Intent: {parsed_query.get('intent')}")
        print(f"  Teams:  {parsed_query.get('teams')}")
        print(f"  Competition: {parsed_query.get('competition')}")
        print(f"  Date context: {parsed_query.get('date_context')}")
        print(f"  Is most recent: {parsed_query.get('is_most_recent')}")

    # Step 1 – web search
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

    print(f"[RAG] Step 1: collected {len(all_results)} search results")

    if not all_results:
        metadata = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": None,
            "score": None,
            "competition": parsed_query.get("competition") if parsed_query else None,
            "key_moments": [],
            "man_of_the_match": None,
            "match_summary": None,
        }
        return f"❌ No results found for: {original_query}", metadata

    # Step 1b – rank results so true match pages come first
    all_results = _rank_match_results(all_results, home_team, away_team)

    # NEW: fetch and attach full article content for top trusted results
    _enrich_results_with_page_content(all_results)

    # Step 2 – chunk + index
    query_id = hashlib.md5(
        f"{biased_query}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    chunks = _chunk_search_results(all_results, query_id)
    _index_chunks_for_rag(chunks)
    print(f"[RAG] Step 2: chunked {len(chunks)} results")

    # Step 3 – retrieve relevant context
    relevant_chunks = _retrieve_relevant_context(
        original_query, query_id, chunks, k=5
    )
    print(f"[RAG] Step 3: using {len(relevant_chunks)} chunks as context")

    context = "\n\n".join(
        f"[Source: {c.get('title','Unknown')}]\n{c['text']}" for c in relevant_chunks
    )

    # Step 4 – LLM summary (STRICT: no fabrication)
    client = _get_openai_client()
    intent_instructions = _get_intent_instructions(intent, summary_focus)

    system_prompt = f"""You are a football/soccer information assistant using RAG.

{intent_instructions}

STRICT RULES:
1. Only use information that appears in the provided context.
2. If a score is not clearly present, say the score was not found.
3. Do NOT guess or invent scores, dates, player names or events.
4. If you are unsure about a detail, say explicitly that it is not available.
"""

    user_message = f"""User question: {original_query}

Context:
{context}

Based ONLY on the context above, answer the user's question.
If the requested information is missing, say so clearly."""

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
        print("[RAG] Step 4: generated summary")
    except Exception as e:
        print(f"[RAG] LLM error during summary: {e}")
        answer_text = _format_raw_results(all_results)

    # Deterministic metadata from the same context (regex only)
    metadata = _build_match_metadata_from_context(
        context, original_query, parsed_query
    )



    # ---- NEW: fallback from summary if score missing ----
    summary_score = _extract_score_from_answer_text(
        answer_text=answer_text,
        context=context,
        home_team=home_team,
        away_team=away_team,
    )

    if metadata.get("score") is None and summary_score:
        print("[RAG] Filling missing score from LLM summary (validated against context)")
        metadata["score"] = summary_score
    # (optional) if you *really* want to override even when different:
    # elif metadata.get("score") and summary_score and metadata["score"] != summary_score:
    #     print("[RAG] Score mismatch; keeping deterministic score:", metadata["score"])
        # Deterministic metadata from the same context (regex based)
    metadata = _build_match_metadata_from_context(
        context, original_query, parsed_query
    )

    # --- NEW: let the LLM summary refine score and key moments ---

    # 1) If we can see a score in the LLM answer, prefer that over the raw HTML score.
    try:
        summary_score = _extract_score_from_context(
            answer_text,
            metadata.get("home_team"),
            metadata.get("away_team"),
        )
        if summary_score:
            metadata["score"] = summary_score
    except Exception as e:
        print(f"[WebSearch] Score extraction from summary failed: {e}")

    # 2) Extract goal events from the summary text and override key_moments
    try:
        llm_key_moments = _extract_goal_events_from_summary(
            answer_text,
            metadata.get("home_team"),
            metadata.get("away_team"),
        )
        if llm_key_moments:
            metadata["key_moments"] = llm_key_moments
    except Exception as e:
        print(f"[WebSearch] LLM key-moment override failed: {e}")

    # -----------------------------------------------------------

    # Append a compact source list
    srcs = {c.get("source", "") for c in relevant_chunks if c.get("source")}
    if srcs:
        answer_text += "\n\n📚 Sources:"
        for s in list(srcs)[:3]:
            answer_text += f"\n  • {s}"

    return answer_text, metadata


    


# -----------------------------------------------------------------------------
# Simple wrappers for compatibility
# -----------------------------------------------------------------------------
def _format_raw_results(results: List[dict]) -> str:
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


def search_and_summarize_with_intent(
    search_query: str,
    intent: str,
    summary_focus: str,
    original_query: str,
    parsed_query: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Backwards-compatible wrapper around search_with_rag.
    """
    return search_with_rag(
        search_query,
        intent=intent,
        original_query=original_query,
        parsed_query=parsed_query,
        summary_focus=summary_focus,
    )


def search_and_summarize(query: str, use_llm: bool = True) -> str:
    """
    Simple wrapper if something still calls the old API.
    Assumes match_result intent.
    """
    answer, _ = search_with_rag(
        query,
        intent="match_result",
        original_query=query,
        parsed_query=None,
        summary_focus=None,
    )
    return answer
