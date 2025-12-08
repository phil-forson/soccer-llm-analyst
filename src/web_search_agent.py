"""
Web search + RAG agent for finding football match information.

- Uses DuckDuckGo (ddgs) for web search.
- Optional embeddings-based RAG via Chroma.
- Extracts match metadata deterministically (no JSON-from-LLM parsing).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

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

# -----------------------------------------------------------------------------
# Search configuration
# -----------------------------------------------------------------------------
MAX_RESULTS = 10

TRUSTED_SOURCES = [
    "espn.com",
    "bbc.com/sport",
    "bbc.co.uk/sport",
    "skysports.com",
    "goal.com",
    "flashscore.com",
    "sofascore.com",
    "whoscored.com",
    "fotmob.com",
    "premierleague.com",
    "theguardian.com/football",
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


def _search_with_source(query: str,
                        source: Optional[str] = None,
                        max_results: int = 10) -> List[Dict[str, str]]:
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


# -----------------------------------------------------------------------------
# Deterministic metadata extraction
# -----------------------------------------------------------------------------
def _extract_score_from_context(context: str,
                                home_team: Optional[str],
                                away_team: Optional[str]) -> Optional[str]:
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

    for match in re.finditer(r'(\d{1,2})\s*-\s*(\d{1,2})', text):
        score = f"{match.group(1)}-{match.group(2)}"
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
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
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


def _maybe_correct_team_order_with_score(context: str,
                                         score: Optional[str],
                                         home_team: Optional[str],
                                         away_team: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
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
        print("[WebSearch] Detected score pattern with reversed team order – swapping home/away")
        return away_team, home_team

    return home_team, away_team


# -----------------------------------------------------------------------------
# Goal / key-moment extraction
# -----------------------------------------------------------------------------
def _extract_goal_events_from_context(context: str,
                                      home_team: Optional[str],
                                      away_team: Optional[str]) -> List[Dict[str, Any]]:
    """
    Extract goal events as key moments from article text.

    Very conservative: only returns events when we see an explicit pattern,
    e.g. "Bamford (23')" or "in the 23rd minute, Patrick Bamford scored".

    Returns list of dicts:
      { "minute": "23'", "event": "Goal",
        "description": "...", "team": "Leeds" }
    """
    if not context:
        return []

    text = context.replace("–", "-")
    lower = text.lower()
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    moments: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Pattern 1: "Name (23')" or "Name (23rd minute)"
    pat1 = re.compile(
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\(\s*(\d{1,2})\s*(?:\'|’)?',
        re.MULTILINE,
    )
    # Pattern 2: "23rd-minute strike from Name" or "23rd minute from Name"
    pat2 = re.compile(
        r'(\d{1,2})(?:st|nd|rd|th)?-?\s*minute[^\.]{0,60}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
        re.IGNORECASE | re.MULTILINE,
    )
    # Pattern 3: "in the 23rd minute, Name scored"
    pat3 = re.compile(
        r'in the (\d{1,2})(?:st|nd|rd|th)? minute[^\.]{0,60}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
        re.IGNORECASE | re.MULTILINE,
    )

    def _infer_team_for_name(idx: int, scorer_name: str) -> Optional[str]:
        """
        Look in a small window around idx to see if 'chelsea' or 'leeds'
        is mentioned near the scorer.
        """
        window_start = max(0, idx - 120)
        window_end = min(len(lower), idx + 120)
        window = lower[window_start:window_end]

        if home and home in window and (not away or away not in window):
            return home_team
        if away and away in window and (not home or home not in window):
            return away_team
        # If both or neither appear, do not guess.
        return None

    # Helper to register a moment
    def _add_moment(minute: str, name: str, idx: int):
        key = (minute, name)
        if key in moments:
            return
        team = _infer_team_for_name(idx, name)
        minute_label = f"{minute}'"
        desc_team = team or "Unknown team"
        description = f"GOAL for {desc_team}: {name} scores in the {minute}th minute."
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

    # Sort by minute numerically
    result = list(moments.values())
    try:
        result.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x["minute"]) or 0))
    except Exception:
        pass

    print(f"[WebSearch] Extracted {len(result)} goal events from context")
    return result


def _build_match_metadata_from_context(context: str,
                                       original_query: str,
                                       parsed_query: Optional[dict]) -> dict:
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

    # NEW: goal → key_moments extraction
    key_moments = _extract_goal_events_from_context(context, home_team, away_team)

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
        text = f"Title: {r.get('title','')}\nContent: {r.get('snippet','')}\nSource: {r.get('url','')}"
        chunks.append({
            "id": f"{query_id}_{i}",
            "text": text,
            "metadata": {
                "query_id": query_id,
                "source_url": r.get("url", ""),
                "title": r.get("title", ""),
                "chunk_index": i,
            }
        })
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


def _retrieve_relevant_context(query: str,
                               query_id: str,
                               chunks: List[dict],
                               k: int = 5) -> List[dict]:
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
                    retrieved.append({
                        "text": doc,
                        "source": meta.get("source_url", ""),
                        "title": meta.get("title", ""),
                    })
            print(f"[RAG] Retrieved {len(retrieved)} chunks via embeddings")
            return retrieved
        except Exception as e:
            print(f"[RAG] Retrieval error: {e} – falling back to simple context")

    print("[RAG] Using simple context (no embeddings)")
    return [{
        "text": c["text"],
        "source": c["metadata"].get("source_url", ""),
        "title": c["metadata"].get("title", "")
    } for c in chunks[:k]]


# -----------------------------------------------------------------------------
# Intent-specific instructions for LLM summarisation
# -----------------------------------------------------------------------------
def _get_intent_instructions(intent: str, summary_focus: str) -> str:
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
    }.get(intent, "Provide a concise, factual summary of the retrieved information.")

    return base + f"\n\nUser requested focus: {summary_focus or 'key information'}."


# -----------------------------------------------------------------------------
# MAIN: RAG search pipeline
# -----------------------------------------------------------------------------
def search_with_rag(
    query: str,
    intent: str,
    original_query: str,
    parsed_query: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Full RAG pipeline:
    - web search (ddgs)
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

    print(f"[RAG] Biased query: \"{biased_query}\"")
    if parsed_query:
        print(f"[RAG] Parsed query:")
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

    # Step 2 – chunk + index
    query_id = hashlib.md5(f"{biased_query}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    chunks = _chunk_search_results(all_results, query_id)
    _index_chunks_for_rag(chunks)
    print(f"[RAG] Step 2: chunked {len(chunks)} results")

    # Step 3 – retrieve relevant context
    relevant_chunks = _retrieve_relevant_context(original_query, query_id, chunks, k=5)
    print(f"[RAG] Step 3: using {len(relevant_chunks)} chunks as context")

    context = "\n\n".join(
        f"[Source: {c.get('title','Unknown')}]\n{c['text']}" for c in relevant_chunks
    )

    # Step 4 – LLM summary (STRICT: no fabrication)
    client = _get_openai_client()
    intent_instructions = _get_intent_instructions(intent, "key information")

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

    # Deterministic metadata from the same context
    metadata = _build_match_metadata_from_context(context, original_query, parsed_query)

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
    Backwards-compatible wrapper that just calls search_with_rag.
    """
    return search_with_rag(search_query, intent, original_query, parsed_query)


def search_and_summarize(query: str, use_llm: bool = True) -> str:
    """
    Simple wrapper if something still calls the old API.
    Assumes match_result intent.
    """
    answer, _ = search_with_rag(query, intent="match_result", original_query=query, parsed_query=None)
    return answer