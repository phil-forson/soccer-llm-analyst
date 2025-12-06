"""
YouTube search agent for finding football match highlights.

Uses DuckDuckGo video search to find YouTube highlights (no API key required).
Intelligently handles seasons, home/away teams, and date-specific searches.
"""

import re
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL


# =============================================================================
# Configuration
# =============================================================================

# Maximum number of video results to fetch
MAX_RESULTS = 5

# Maximum duration for highlights (in seconds) - videos longer than this are likely simulations/full matches
MAX_HIGHLIGHT_DURATION_SECONDS = 60 * 60  # 1 hour = 3600 seconds
MAX_HIGHLIGHT_DURATION_MINUTES = 60  # 1 hour

# YouTube domain filter
YOUTUBE_DOMAINS = ["youtube.com", "youtu.be"]

# TOP PRIORITY channels (official clubs, Champions League, TNT Sports)
TOP_PRIORITY_CHANNELS = [
    # Official competitions
    "champions league",
    "uefa champions league",
    "uefa",
    "premier league",
    
    # TNT Sports (top priority broadcaster)
    "tnt sports",
    "bt sport",
    
    # Official club channels (Premier League)
    "arsenal",
    "chelsea fc", 
    "liverpool fc",
    "manchester united",
    "manchester city",
    "tottenham hotspur",
    "newcastle united",
    "aston villa",
    
    # Big European clubs
    "real madrid",
    "fc barcelona",
    "bayern munich",
    "psg",
    "juventus",
    "ac milan",
    "inter milan",
    "borussia dortmund",
]

# Trusted sports channels (prioritize these)
TRUSTED_CHANNELS = [
    # Major broadcasters (after top priority)
    "tnt sports",
    "nbc sports",
    "sky sports",
    "bt sport",
    "espn",
    "cbs sports",
    "bein sports",
    "dazn",
    "optus sport",
    "paramount+",
    # League official channels
    "premier league",
    "bundesliga",
    "la liga",
    "serie a",
    "ligue 1",
    "uefa",
    "champions league",
    "fa cup",
    "efl",
    "mls",
    # Official club channels (Premier League)
    "arsenal",
    "chelsea fc",
    "liverpool fc",
    "manchester united",
    "manchester city",
    "tottenham hotspur",
    "newcastle united",
    "aston villa",
    "west ham",
    "brighton",
    "wolves",
    "bournemouth",
    "fulham",
    "crystal palace",
    "brentford",
    "everton",
    "nottingham forest",
    "luton town",
    "burnley",
    "sheffield united",
    # Other big clubs
    "real madrid",
    "fc barcelona",
    "bayern munich",
    "psg",
    "juventus",
    "ac milan",
    "inter milan",
    "borussia dortmund",
    # Other trusted sources
    "match of the day",
    "football daily",
    "goal",
    "bt sport football",
]

# =============================================================================
# Simulation/Fake Content Detection
# =============================================================================

# Keywords that indicate video game simulations (NOT real highlights)
SIMULATION_KEYWORDS = [
    # Video games
    "fifa",
    "ea fc",
    "eafc",
    "fc 24",
    "fc 25",
    "fc24",
    "fc25",
    "efootball",
    "pes",
    "pro evolution soccer",
    "football manager",
    "fm24",
    "fm25",
    # Simulation indicators
    "simulation",
    "simulated",
    "gameplay",
    "game play",
    "video game",
    "videogame",
    "ps5",
    "ps4",
    "playstation",
    "xbox",
    "pc gameplay",
    "4k gameplay",
    "8k gameplay",
    "ultra graphics",
    "realistic graphics",
    "modded",
    # Prediction content (not actual match)
    "score prediction",
    "match prediction",
    "predicted lineup",
    "potential lineup",
    "what if",
    # Fan-made content
    "fan made",
    "fanmade",
    "recreation",
    "remake",
]

# Channels known for simulation/gaming content (blocklist)
SIMULATION_CHANNELS = [
    "fifazz",
    "ea sports fc",
    "konami",
    "pes mobile",
    "efootball mobile",
    "fc mobile",
    "droid gamer",
    "gaming",
    "gameplay",
    "vs legend",
    "vslegend",
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
# YouTube Search Functions
# =============================================================================

# =============================================================================
# Match Query Parsing
# =============================================================================

def _parse_match_info(query: str) -> dict:
    """
    Parse a match query to extract teams, date, and season info.
    
    Understands:
        - Team names (home team listed first is significant!)
        - Dates: "2024-03-15", "March 15 2024", "15/03/2024"
        - Seasons: "2024-25", "24/25", "2024/2025", "season 2024"
    
    Args:
        query: Match description string.
        
    Returns:
        dict with:
            - home_team: str (first team mentioned - plays at home)
            - away_team: str (second team - visiting)
            - date: str or None (YYYY-MM-DD if found)
            - season: str or None (e.g., "2024-25")
            - year: str or None (e.g., "2024")
    """
    print(f"\n[YouTubeParse] === Parsing Query ===")
    print(f"[YouTubeParse] Input: \"{query}\"")
    
    result = {
        "home_team": None,
        "away_team": None,
        "date": None,
        "season": None,
        "year": None,
        "month_year": None,
    }
    
    working_query = query
    
    # Extract season patterns: "2024-25", "24/25", "2024/2025", "season 2024"
    season_patterns = [
        r"(\d{4}[-/]\d{2,4})",  # 2024-25, 2024/2025
        r"(\d{2}/\d{2})",       # 24/25
        r"season\s*(\d{4})",    # season 2024
    ]
    
    for pattern in season_patterns:
        match = re.search(pattern, working_query, re.IGNORECASE)
        if match:
            result["season"] = match.group(1)
            working_query = re.sub(pattern, "", working_query, flags=re.IGNORECASE)
            break
    
    # Extract date: YYYY-MM-DD
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", working_query)
    if date_match:
        result["date"] = date_match.group(1)
        working_query = working_query.replace(date_match.group(1), "")
        # Extract year and month from date
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            result["year"] = str(dt.year)
            result["month_year"] = dt.strftime("%B %Y")  # e.g., "March 2024"
        except ValueError:
            pass
    
    # Extract standalone year if not already found
    if not result["year"] and not result["season"]:
        year_match = re.search(r"\b(20\d{2})\b", working_query)
        if year_match:
            result["year"] = year_match.group(1)
            working_query = working_query.replace(year_match.group(1), "")
    
    # Parse team names - IMPORTANT: First team = HOME, Second team = AWAY
    separators = r"\s+(?:vs\.?|v\.?|versus|-|against|@)\s+"
    parts = re.split(separators, working_query.strip(), flags=re.IGNORECASE)
    
    if len(parts) >= 2:
        result["home_team"] = parts[0].strip()
        result["away_team"] = parts[1].strip()
    elif len(parts) == 1 and parts[0].strip():
        result["home_team"] = parts[0].strip()
    
    # Default to current season if nothing specified
    if not result["season"] and not result["year"] and not result["date"]:
        current_year = datetime.now().year
        current_month = datetime.now().month
        # Football seasons span two years (Aug-May typically)
        if current_month >= 8:  # Aug onwards = start of new season
            result["season"] = f"{current_year}-{str(current_year + 1)[2:]}"
        else:  # Before Aug = end of previous season
            result["season"] = f"{current_year - 1}-{str(current_year)[2:]}"
        print(f"[YouTubeParse] No date specified, defaulting to current season: {result['season']}")
    
    # Log parsed results
    print(f"[YouTubeParse] Parsed results:")
    print(f"[YouTubeParse]   Home team: {result.get('home_team', 'None')}")
    print(f"[YouTubeParse]   Away team: {result.get('away_team', 'None')}")
    print(f"[YouTubeParse]   Date: {result.get('date', 'None')}")
    print(f"[YouTubeParse]   Season: {result.get('season', 'None')}")
    print(f"[YouTubeParse]   Year: {result.get('year', 'None')}")
    
    return result


def _build_highlight_queries(match_info: dict) -> list[str]:
    """
    Build smart search queries for highlights based on parsed match info.
    
    Creates queries that:
        - Target REAL match footage (not simulations)
        - Use correct home vs away order
        - PRIORITIZE RECENT MATCHES when no date specified
        - Include date/season context for accuracy
        - Target official broadcasters and clubs
    
    Args:
        match_info: Parsed match info dict.
        
    Returns:
        List of search query strings, best first.
    """
    print(f"\n[YouTubeQuery] === Building Search Queries ===")
    
    queries = []
    
    home = match_info.get("home_team", "")
    away = match_info.get("away_team", "")
    date = match_info.get("date", "")
    season = match_info.get("season", "")
    year = match_info.get("year", "")
    month_year = match_info.get("month_year", "")
    
    # Build the base match string (home vs away order matters!)
    if home and away:
        match_str = f"{home} vs {away}"
    elif home:
        match_str = home
    else:
        print(f"[YouTubeQuery] ERROR: No teams found in query")
        return []  # No teams found
    
    print(f"[YouTubeQuery] Base match string: \"{match_str}\"")
    
    # Get current date info for recency
    current_year = str(datetime.now().year)
    current_month = datetime.now().strftime("%B")  # e.g., "December"
    
    date_context = month_year or season or year or ""
    
    # Priority 1: Most recent match (current year/month if no date specified)
    if not date:
        queries.append(f"{match_str} highlights {current_year} {current_month}")
        queries.append(f"{match_str} highlights {current_year}")
    
    # Priority 2: Date-specific searches (if date provided)
    if date:
        queries.append(f"{match_str} highlights {date}")
        if month_year:
            queries.append(f"{match_str} extended highlights {month_year}")
    
    # Priority 3: Official broadcaster highlights with recency
    recency = date_context or current_year
    queries.append(f"{match_str} official highlights {recency} TNT Sports".strip())
    queries.append(f"{match_str} match highlights {recency} Champions League".strip())
    
    # Priority 4: Season-specific from official sources
    if season:
        queries.append(f"{match_str} highlights {season} Champions League")
        queries.append(f"{match_str} extended highlights {season}")
    
    # Priority 5: Club official channels
    if home:
        queries.append(f"{home} official extended highlights vs {away} {recency}".strip())
    
    # Fallback with real-footage indicators
    queries.append(f"{match_str} match highlights goals {current_year}")
    
    # Log all queries
    print(f"[YouTubeQuery] Generated {len(queries)} search queries:")
    for i, q in enumerate(queries, 1):
        print(f"[YouTubeQuery]   {i}. \"{q}\"")
    
    return queries


def _is_trusted_channel(publisher: str) -> bool:
    """Check if the publisher is a trusted sports channel."""
    publisher_lower = publisher.lower()
    return any(channel in publisher_lower for channel in TRUSTED_CHANNELS)


def _is_top_priority_channel(publisher: str) -> bool:
    """Check if the publisher is a TOP PRIORITY channel (official clubs, UEFA, TNT)."""
    publisher_lower = publisher.lower()
    return any(channel in publisher_lower for channel in TOP_PRIORITY_CHANNELS)


def _parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """
    Parse a duration string to seconds.
    
    Handles formats like:
        - "12:34" (mm:ss)
        - "1:23:45" (h:mm:ss)
        - "2:48:24" (h:mm:ss)
        - "45" (seconds only)
    
    Args:
        duration_str: Duration string from video metadata.
        
    Returns:
        Duration in seconds, or None if unparseable.
    """
    if not duration_str or duration_str == "Unknown":
        return None
    
    try:
        # Clean the string
        duration_str = duration_str.strip()
        
        # Split by colon
        parts = duration_str.split(":")
        
        if len(parts) == 3:
            # h:mm:ss format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # mm:ss format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 1:
            # Just seconds
            return int(parts[0])
        else:
            return None
            
    except (ValueError, TypeError):
        return None


def _is_duration_too_long(duration_str: str) -> bool:
    """
    Check if a video duration is too long to be highlights.
    
    Videos longer than 1 hour are likely:
        - Full match replays
        - Simulations/gameplay
        - Live streams
    
    Args:
        duration_str: Duration string (e.g., "12:34", "2:48:24").
        
    Returns:
        True if too long (>1 hour), False otherwise.
    """
    seconds = _parse_duration_to_seconds(duration_str)
    
    if seconds is None:
        return False  # Can't determine, allow it
    
    return seconds > MAX_HIGHLIGHT_DURATION_SECONDS


# =============================================================================
# Simulation Detection Agent
# =============================================================================

def _is_simulation(title: str, description: str = "", publisher: str = "") -> bool:
    """
    Detect if a video is a simulation/video game rather than real highlights.
    
    Checks title, description, and publisher for simulation indicators.
    
    Args:
        title: Video title.
        description: Video description.
        publisher: Channel/publisher name.
        
    Returns:
        True if likely a simulation, False if likely real highlights.
    """
    # Combine all text for checking
    combined = f"{title} {description} {publisher}".lower()
    
    # Check for simulation keywords
    for keyword in SIMULATION_KEYWORDS:
        if keyword in combined:
            return True
    
    # Check for simulation channels
    publisher_lower = publisher.lower()
    for channel in SIMULATION_CHANNELS:
        if channel in publisher_lower:
            return True
    
    return False


# =============================================================================
# YouTube Video Data Fetching (Description & Transcript)
# =============================================================================

def _extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
    
    Args:
        url: YouTube video URL.
        
    Returns:
        Video ID string or None if not found.
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'[?&]v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def _fetch_video_description(video_id: str) -> Optional[str]:
    """
    Fetch the full description of a YouTube video using yt-dlp.
    
    Args:
        video_id: YouTube video ID.
        
    Returns:
        Full video description or None if failed.
    """
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('description', '')
            
    except ImportError:
        print("[YouTubeValidator] yt-dlp not installed. Run: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"[YouTubeValidator] Error fetching description: {e}")
        return None


def _fetch_video_transcript(video_id: str) -> Optional[str]:
    """
    Fetch the transcript/captions of a YouTube video.
    
    Tries to get English captions first, then auto-generated.
    
    Args:
        video_id: YouTube video ID.
        
    Returns:
        Transcript text or None if unavailable.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        try:
            # New API: directly fetch transcript
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'en-GB']
            )
            
            # Combine all text segments
            full_text = ' '.join([entry['text'] for entry in transcript_data])
            
            # Limit to first 2000 chars for analysis
            return full_text[:2000] if len(full_text) > 2000 else full_text
            
        except Exception:
            # Try without language preference (auto-generated)
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = ' '.join([entry['text'] for entry in transcript_data])
                return full_text[:2000] if len(full_text) > 2000 else full_text
            except Exception:
                return None
            
    except ImportError:
        return None  # Silently fail if library not installed
    except Exception as e:
        print(f"[YouTubeValidator] Error fetching transcript: {e}")
        return None


def _fetch_video_metadata(url: str) -> dict:
    """
    Fetch full metadata for a YouTube video (description + transcript).
    
    Args:
        url: YouTube video URL.
        
    Returns:
        dict with:
            - description: str (full video description)
            - transcript: str (video transcript/captions)
            - video_id: str
    """
    video_id = _extract_video_id(url)
    
    if not video_id:
        return {"description": "", "transcript": "", "video_id": None}
    
    description = _fetch_video_description(video_id) or ""
    transcript = _fetch_video_transcript(video_id) or ""
    
    return {
        "description": description,
        "transcript": transcript,
        "video_id": video_id,
    }


# =============================================================================
# Deep Validation Agent (Description + Transcript Analysis)
# =============================================================================

def validate_video_is_real(video: dict, match_info: dict, deep_check: bool = True) -> dict:
    """
    Comprehensive validation to determine if a video shows REAL match highlights.
    
    This is the main validation function that:
    1. Quick check: Keyword detection on title/publisher
    2. Deep check: Fetches full description and transcript from YouTube
    3. LLM analysis: Uses AI to analyze all available data
    
    Args:
        video: Video result dict with title, url, publisher.
        match_info: Parsed match info for context.
        deep_check: If True, fetches description and transcript (slower but more accurate).
        
    Returns:
        dict with:
            - is_real: bool (True if real highlights)
            - confidence: float (0-1)
            - reason: str (explanation)
            - evidence: dict (what was checked)
    """
    title = video.get("title", "")
    url = video.get("url", "")
    publisher = video.get("publisher", "")
    search_description = video.get("description", "")  # From search results
    
    result = {
        "is_real": True,
        "confidence": 0.5,
        "reason": "Initial",
        "evidence": {
            "title_check": None,
            "description_check": None,
            "transcript_check": None,
        }
    }
    
    # Quick check: Title and publisher keywords
    if _is_simulation(title, search_description, publisher):
        return {
            "is_real": False,
            "confidence": 0.9,
            "reason": "Simulation keywords detected in title/publisher",
            "evidence": {"title_check": "FAILED - simulation keywords found"}
        }
    
    result["evidence"]["title_check"] = "PASSED - no simulation keywords"
    
    # If trusted channel, likely real
    if _is_trusted_channel(publisher):
        return {
            "is_real": True,
            "confidence": 0.95,
            "reason": "Trusted official broadcaster/club channel",
            "evidence": {"title_check": "PASSED", "publisher": f"Trusted: {publisher}"}
        }
    
    # Deep check: Fetch and analyze description + transcript
    if deep_check:
        print(f"[YouTubeValidator] Deep checking: {title[:50]}...")
        
        metadata = _fetch_video_metadata(url)
        full_description = metadata.get("description", "")
        transcript = metadata.get("transcript", "")
        
        # Check description for simulation indicators
        if full_description:
            if _is_simulation(title, full_description, publisher):
                return {
                    "is_real": False,
                    "confidence": 0.95,
                    "reason": "Simulation keywords found in video description",
                    "evidence": {
                        "title_check": "PASSED",
                        "description_check": "FAILED - simulation content detected"
                    }
                }
            result["evidence"]["description_check"] = "PASSED"
            
            # Look for real highlight indicators in description
            real_indicators = [
                "goal", "score", "minute", "half", "match report",
                "premier league", "champions league", "fa cup",
                "kick-off", "full-time", "half-time", "assist",
                "substitution", "yellow card", "red card", "penalty"
            ]
            desc_lower = full_description.lower()
            found_indicators = [ind for ind in real_indicators if ind in desc_lower]
            
            if found_indicators:
                result["confidence"] = min(0.9, result["confidence"] + 0.2)
                result["evidence"]["real_indicators"] = found_indicators[:5]
        
        # Check transcript for simulation indicators
        if transcript:
            transcript_lower = transcript.lower()
            
            # Simulation transcript indicators
            sim_transcript = [
                "press x", "press a", "controller", "button",
                "menu", "settings", "difficulty", "squad",
                "ultimate team", "career mode", "manager mode",
                "coins", "points", "pack", "sbc"
            ]
            
            # Real commentary indicators
            real_transcript = [
                "goal", "scores", "shoots", "pass", "tackle",
                "referee", "foul", "corner", "free kick",
                "manager", "substitution", "injury time",
                "crowd", "fans", "stadium"
            ]
            
            sim_found = any(term in transcript_lower for term in sim_transcript)
            real_found = sum(1 for term in real_transcript if term in transcript_lower)
            
            if sim_found:
                return {
                    "is_real": False,
                    "confidence": 0.98,
                    "reason": "Game/simulation language found in transcript",
                    "evidence": {
                        "title_check": "PASSED",
                        "description_check": result["evidence"].get("description_check", "NOT CHECKED"),
                        "transcript_check": "FAILED - game terminology detected"
                    }
                }
            
            if real_found >= 3:
                result["confidence"] = min(0.95, result["confidence"] + 0.3)
                result["evidence"]["transcript_check"] = f"PASSED - {real_found} real commentary terms found"
            else:
                result["evidence"]["transcript_check"] = "INCONCLUSIVE"
    
    # Final decision
    if result["confidence"] >= 0.7:
        result["is_real"] = True
        result["reason"] = "Passed validation checks"
    else:
        # Use LLM for uncertain cases
        result = _validate_with_llm_deep(video, match_info, metadata if deep_check else None)
    
    return result


def _validate_with_llm_deep(video: dict, match_info: dict, metadata: Optional[dict] = None) -> dict:
    """
    Use LLM to validate with all available data (description + transcript).
    
    Args:
        video: Video result dict.
        match_info: Parsed match info.
        metadata: Optional dict with full description and transcript.
        
    Returns:
        Validation result dict.
    """
    try:
        client = _get_openai_client()
        
        title = video.get("title", "")
        publisher = video.get("publisher", "")
        home = match_info.get("home_team", "")
        away = match_info.get("away_team", "")
        
        description = ""
        transcript = ""
        if metadata:
            description = metadata.get("description", "")[:1000]  # Limit for token usage
            transcript = metadata.get("transcript", "")[:1000]
        
        prompt = f"""Analyze this YouTube video and determine if it shows REAL football match highlights or a VIDEO GAME SIMULATION (FIFA, EA FC, eFootball, etc.).

VIDEO TITLE: {title}
CHANNEL: {publisher}

FULL DESCRIPTION:
{description if description else "(Not available)"}

TRANSCRIPT EXCERPT:
{transcript if transcript else "(Not available)"}

MATCH BEING SEARCHED: {home} vs {away}

SIMULATION INDICATORS:
- Mentions of FIFA, EA FC, EA Sports, eFootball, PES, gameplay
- Gaming terminology: controls, buttons, settings, ultimate team, career mode
- "4K", "ultra graphics", "realistic", "modded"
- Prediction or "what if" scenarios

REAL HIGHLIGHT INDICATORS:
- Official broadcaster language (NBC, Sky, TNT, ESPN)
- Match report details: scores, goalscorers, minute marks
- Real commentary: referee decisions, substitutions, injuries
- Stadium atmosphere descriptions

Based on ALL available evidence, is this REAL match highlights or a SIMULATION?
Respond in this format:
VERDICT: REAL or SIMULATION
CONFIDENCE: HIGH, MEDIUM, or LOW
REASON: One sentence explanation"""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at distinguishing real football match highlights from video game simulations. Analyze all evidence carefully."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100,
        )
        
        answer = response.choices[0].message.content.strip()
        
        is_real = "REAL" in answer.upper() and "SIMULATION" not in answer.upper().split("VERDICT")[1].split("\n")[0] if "VERDICT" in answer.upper() else "REAL" in answer.upper()
        
        confidence = 0.8
        if "HIGH" in answer.upper():
            confidence = 0.95
        elif "LOW" in answer.upper():
            confidence = 0.6
        
        return {
            "is_real": is_real,
            "confidence": confidence,
            "reason": answer.split("REASON:")[-1].strip() if "REASON:" in answer else "LLM deep analysis",
            "evidence": {"llm_analysis": "Complete"}
        }
        
    except Exception as e:
        print(f"[YouTubeValidator] LLM deep validation failed: {e}")
        return {
            "is_real": True,  # Default to showing if can't verify
            "confidence": 0.5,
            "reason": "Could not complete deep validation",
            "evidence": {"error": str(e)}
        }


def _validate_highlight_with_llm(video: dict, match_info: dict) -> dict:
    """
    Use LLM to validate if a video is real highlights or simulation.
    
    This is a quick check using only title/publisher info.
    For deeper validation, use validate_video_is_real() with deep_check=True.
    
    Args:
        video: Video result dict with title, description, publisher.
        match_info: Parsed match info for context.
        
    Returns:
        dict with:
            - is_real: bool (True if real highlights)
            - confidence: float (0-1)
            - reason: str (explanation)
    """
    try:
        client = _get_openai_client()
        
        title = video.get("title", "")
        description = video.get("description", "")
        publisher = video.get("publisher", "")
        
        home = match_info.get("home_team", "")
        away = match_info.get("away_team", "")
        
        prompt = f"""Analyze this YouTube video and determine if it shows REAL football match highlights or a VIDEO GAME SIMULATION.

Video Title: {title}
Channel: {publisher}
Description: {description}

Match being searched: {home} vs {away}

Signs of SIMULATION/FAKE:
- FIFA, EA FC, eFootball, PES in title
- "gameplay", "simulation", "4K", "ultra graphics"
- Gaming channels
- "prediction", "what if", "recreation"

Signs of REAL HIGHLIGHTS:
- Official broadcaster (NBC Sports, Sky Sports, TNT Sports, etc.)
- Official club channel
- "Official highlights", "extended highlights"
- Match report language

Respond with ONLY one word: REAL or SIMULATION"""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a football video classifier. Respond with only REAL or SIMULATION."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10,
        )
        
        answer = response.choices[0].message.content.strip().upper()
        
        is_real = "REAL" in answer
        
        return {
            "is_real": is_real,
            "confidence": 0.9 if is_real else 0.85,
            "reason": "LLM classification"
        }
        
    except Exception as e:
        print(f"[YouTubeSearch] LLM validation failed: {e}")
        # Fall back to keyword detection
        is_sim = _is_simulation(
            video.get("title", ""),
            video.get("description", ""),
            video.get("publisher", "")
        )
        return {
            "is_real": not is_sim,
            "confidence": 0.7,
            "reason": "Keyword detection fallback"
        }


def _search_youtube_highlights(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Search for YouTube highlight videos using DuckDuckGo video search.
    
    Filters out:
        - Videos longer than 1 hour (likely full matches or simulations)
        - Simulation/video game content (FIFA, eFootball, etc.)
        - Preview/prediction content
        - Gaming channels
    
    Prioritizes:
        - TOP PRIORITY: Official clubs, Champions League, TNT Sports
        - Trusted sports channels (NBC Sports, Sky Sports, etc.)
        - Real match footage
    
    Args:
        query: Search query (e.g., "Arsenal vs Chelsea highlights").
        max_results: Maximum number of results to return.
        
    Returns:
        List of video results with title, url, duration, and thumbnail.
    """
    try:
        from ddgs import DDGS
        
        results = []
        simulations_filtered = 0
        too_long_filtered = 0
        
        with DDGS() as ddgs:
            # Search for videos - fetch more to filter
            for video in ddgs.videos(query, max_results=max_results * 8):
                video_url = video.get("content", "")
                publisher = video.get("publisher", "")
                title = video.get("title", "Untitled")
                description = video.get("description", "")
                duration = video.get("duration", "Unknown")
                
                is_youtube = any(domain in video_url.lower() for domain in YOUTUBE_DOMAINS)
                
                if not is_youtube:
                    continue
                
                # FILTER: Videos longer than 1 hour (likely full match/simulation)
                if _is_duration_too_long(duration):
                    too_long_filtered += 1
                    continue
                
                # FILTER: Simulations
                if _is_simulation(title, description, publisher):
                    simulations_filtered += 1
                    continue
                
                is_top_priority = _is_top_priority_channel(publisher)
                is_trusted = is_top_priority or _is_trusted_channel(publisher)
                
                results.append({
                    "title": title,
                    "url": video_url,
                    "duration": duration,
                    "thumbnail": video.get("images", {}).get("large", ""),
                    "description": description,
                    "publisher": publisher,
                    "is_youtube": is_youtube,
                    "is_top_priority": is_top_priority,
                    "is_trusted": is_trusted,
                    "is_simulation": False,
                })
        
        if too_long_filtered > 0:
            print(f"[YouTubeSearch] Filtered out {too_long_filtered} videos >1 hour")
        if simulations_filtered > 0:
            print(f"[YouTubeSearch] Filtered out {simulations_filtered} simulation/game videos")
        
        # Sort: TOP PRIORITY first, then trusted, then others
        results.sort(key=lambda x: (
            not x.get("is_top_priority", False),
            not x.get("is_trusted", False),
        ))
        
        return results[:max_results]
        
    except ImportError:
        print("[YouTubeSearch] ddgs library not installed. Run: pip install ddgs")
        return []
    except Exception as e:
        print(f"[YouTubeSearch] Error searching videos: {e}")
        return []


def _video_matches_query(video: dict, match_info: dict) -> bool:
    """
    Check if a video actually matches the teams we're looking for.
    
    Prevents returning highlights from different matches.
    
    Args:
        video: Video result dict.
        match_info: Parsed match info.
        
    Returns:
        True if video matches the requested teams.
    """
    title_lower = video.get("title", "").lower()
    
    home = (match_info.get("home_team") or "").lower()
    away = (match_info.get("away_team") or "").lower()
    
    # Check if at least one team name appears in the title
    home_in_title = home and (home in title_lower or home.split()[0] in title_lower)
    away_in_title = away and (away in title_lower or away.split()[0] in title_lower)
    
    # If both teams specified, both should be in title (or at least one clear match)
    if home and away:
        if home_in_title and away_in_title:
            return True
        # Allow if one full name matches
        if home in title_lower or away in title_lower:
            return True
        return False
    
    # If only one team specified
    if home:
        return home_in_title
    
    return True


def search_match_highlights(match_query: str, deep_validate: bool = False) -> list[dict]:
    """
    Search for match highlights on YouTube from trusted sports channels.
    
    Smart features:
        - Understands home vs away team order (first team = home)
        - Defaults to current season if none specified
        - Validates videos actually match the requested teams
        - Prioritizes official club channels and broadcasters
    
    Args:
        match_query: Match description (e.g., "Arsenal vs Chelsea 2024-03-15").
        deep_validate: If True, fetches video description/transcript for validation.
        
    Returns:
        List of video results (verified as real highlights).
    """
    print(f"\n{'='*60}")
    print(f"[YouTubeSearch] STARTING HIGHLIGHT SEARCH")
    print(f"{'='*60}")
    print(f"[YouTubeSearch] Original query: \"{match_query}\"")
    
    # Parse the match query intelligently
    match_info = _parse_match_info(match_query)
    
    # Log what we understood
    home = match_info.get("home_team", "Unknown")
    away = match_info.get("away_team", "Unknown")
    date = match_info.get("date", "Not specified")
    season = match_info.get("season") or match_info.get("year") or "current"
    
    print(f"\n[YouTubeSearch] === Search Parameters ===")
    print(f"[YouTubeSearch] Home team: {home}")
    print(f"[YouTubeSearch] Away team: {away}")
    print(f"[YouTubeSearch] Date: {date}")
    print(f"[YouTubeSearch] Season: {season}")
    
    # Build smart search queries
    queries = _build_highlight_queries(match_info)
    
    if not queries:
        print(f"[YouTubeSearch] ERROR: Could not build queries")
        return []
    
    all_results = []
    seen_urls = set()
    wrong_match_filtered = 0
    
    print(f"\n[YouTubeSearch] === Executing Searches ===")
    
    for query in queries[:4]:  # Limit to first 4 queries to save time
        print(f"\n[YouTubeSearch] Searching: \"{query}\"")
        results = _search_youtube_highlights(query, max_results=5)
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # VALIDATE: Check if video matches the teams we're looking for
                if not _video_matches_query(result, match_info):
                    wrong_match_filtered += 1
                    print(f"[YouTubeSearch]   ✗ Wrong match: {title[:50]}...")
                    continue
                
                print(f"[YouTubeSearch]   ✓ Match found: {title[:50]}...")
                result["match_info"] = match_info
                all_results.append(result)
        
        # Stop if we have enough results
        if len(all_results) >= MAX_RESULTS:
            break
    
    # Log filtering summary
    print(f"\n[YouTubeSearch] === Search Summary ===")
    print(f"[YouTubeSearch] Total results found: {len(all_results)}")
    print(f"[YouTubeSearch] Wrong match filtered: {wrong_match_filtered}")
    
    if not all_results:
        print(f"[YouTubeSearch] ❌ NO VALID HIGHLIGHTS FOUND")
        print(f"[YouTubeSearch] Could not find highlights for {home} vs {away}")
        print(f"{'='*60}\n")
        return []
    
    # Score and sort results by relevance
    all_results = _score_and_sort_results(all_results, match_info)
    
    # Log top results
    print(f"\n[YouTubeSearch] Top {min(len(all_results), MAX_RESULTS)} results after scoring:")
    for i, r in enumerate(all_results[:MAX_RESULTS], 1):
        priority = "⭐" if r.get("is_top_priority") else ("✅" if r.get("is_trusted") else "")
        print(f"[YouTubeSearch]   {i}. {priority} {r.get('title', '')[:50]}...")
    
    # Deep validation (optional - disabled by default for speed)
    if deep_validate and all_results:
        print(f"\n[YouTubeSearch] === Deep Validation ===")
        validated_results = []
        
        for video in all_results[:MAX_RESULTS + 3]:
            # Skip deep validation for trusted channels (they're reliable)
            if video.get("is_trusted") or video.get("is_top_priority"):
                video["validation"] = {"is_real": True, "confidence": 0.95, "reason": "Trusted channel"}
                validated_results.append(video)
                continue
            
            # Deep validate non-trusted videos
            validation = validate_video_is_real(video, match_info, deep_check=True)
            
            if validation.get("is_real"):
                video["validation"] = validation
                validated_results.append(video)
                print(f"[YouTubeValidator] ✓ Verified: {video.get('title', '')[:50]}...")
            else:
                print(f"[YouTubeValidator] ✗ Rejected (simulation): {video.get('title', '')[:50]}...")
            
            # Stop once we have enough validated results
            if len(validated_results) >= MAX_RESULTS:
                break
        
        print(f"[YouTubeSearch] Validated results: {len(validated_results)}")
        print(f"{'='*60}\n")
        return validated_results[:MAX_RESULTS]
    
    print(f"[YouTubeSearch] SEARCH COMPLETE")
    print(f"{'='*60}\n")
    return all_results[:MAX_RESULTS]


def _score_and_sort_results(results: list[dict], match_info: dict) -> list[dict]:
    """
    Score and sort results by relevance to the specific match.
    
    Considers:
        - Simulation detection (HEAVY penalty)
        - Trusted channel status
        - Title matches team names
        - Title contains date/season info
        - Prefers "extended highlights"
    
    Args:
        results: List of video results.
        match_info: Parsed match info.
        
    Returns:
        Sorted list with most relevant first, simulations removed.
    """
    home = (match_info.get("home_team") or "").lower()
    away = (match_info.get("away_team") or "").lower()
    date = match_info.get("date") or ""
    season = match_info.get("season") or ""
    year = match_info.get("year") or ""
    
    # Current year for recency scoring
    current_year = datetime.now().year
    
    # First pass: filter out any remaining simulations
    filtered_results = []
    for r in results:
        title = r.get("title", "")
        description = r.get("description", "")
        publisher = r.get("publisher", "")
        
        # Double-check simulation detection
        if _is_simulation(title, description, publisher):
            print(f"[YouTubeSearch] Filtering simulation: {title[:50]}...")
            continue
        
        filtered_results.append(r)
    
    def score_result(r: dict) -> tuple:
        title_lower = r.get("title", "").lower()
        title_raw = r.get("title", "")
        duration = r.get("duration", "")
        score = 0
        
        # TOP PRIORITY CHANNEL BONUS (official clubs, Champions League, TNT Sports)
        if r.get("is_top_priority"):
            score += 200  # Massive bonus for top priority channels
        elif r.get("is_trusted"):
            score += 80
        
        # DURATION BONUS - prefer typical highlight length (5-20 minutes)
        duration_seconds = _parse_duration_to_seconds(duration)
        if duration_seconds:
            if 300 <= duration_seconds <= 1200:  # 5-20 minutes = ideal highlight length
                score += 50
            elif 120 <= duration_seconds < 300:  # 2-5 minutes = short highlights
                score += 30
            elif duration_seconds > 2400:  # >40 minutes = too long, penalize
                score -= 50
        
        # RECENCY BONUS - heavily prioritize recent videos
        if str(current_year) in title_raw:
            score += 150  # Big bonus for current year
        elif str(current_year - 1) in title_raw:
            score += 100  # Good bonus for last year
        
        # Check for recent season patterns (e.g., "2024-25", "24/25")
        current_season = f"{current_year}-{str(current_year + 1)[2:]}"
        prev_season = f"{current_year - 1}-{str(current_year)[2:]}"
        if current_season in title_raw or f"{str(current_year)[2:]}/{str(current_year + 1)[2:]}" in title_raw:
            score += 150
        elif prev_season in title_raw:
            score += 100
        
        # Penalize old matches (REWIND, throwback, classic, etc.)
        if any(word in title_lower for word in ["rewind", "throwback", "classic", "retro", "old", "history", "2018", "2019", "2020", "2021", "2022"]):
            score -= 200  # Heavy penalty for old content
        
        # Team name matches in title
        if home and home in title_lower:
            score += 30
        if away and away in title_lower:
            score += 30
        
        # Date/season matches from user query
        if date and date in title_raw:
            score += 100
        if season and season in title_lower:
            score += 80
        if year and year in title_raw:
            score += 60
        
        # Prefer extended/official highlights
        if "extended" in title_lower:
            score += 25
        if "official" in title_lower:
            score += 30
        
        # Bonus for match report language (indicates real coverage)
        if any(word in title_lower for word in ["goals", "score", "result"]):
            score += 15
        
        # Penalize non-highlight content
        if any(word in title_lower for word in ["preview", "reaction", "analysis", "prediction", "press conference"]):
            score -= 50
        
        # Penalize live streams (usually not highlights)
        if "live" in title_lower and "highlight" not in title_lower:
            score -= 100  # Heavy penalty
        
        # Extra penalty for any remaining simulation hints
        if any(word in title_lower for word in ["4k", "8k", "ultra", "graphics", "mod"]):
            score -= 100
        
        return (-score, not r.get("is_top_priority", False), not r.get("is_trusted", False))
    
    return sorted(filtered_results, key=score_result)


# =============================================================================
# Display and Formatting
# =============================================================================

def format_highlight_results(results: list[dict]) -> str:
    """
    Format highlight results for display.
    
    Shows only verified real match highlights (no simulations).
    Includes channel priority status for each video.
    
    Args:
        results: List of video result dicts.
        
    Returns:
        Formatted string for CLI display.
    """
    if not results:
        return "🎬 No highlight videos found."
    
    lines = ["🎬 Match Highlights:\n"]
    
    for i, video in enumerate(results, 1):
        title = video.get("title", "Untitled")
        url = video.get("url", "")
        duration = video.get("duration", "")
        publisher = video.get("publisher", "")
        is_top_priority = video.get("is_top_priority", False)
        is_trusted = video.get("is_trusted", False)
        
        # Build channel badge
        if is_top_priority:
            badge = " ⭐ Official"
        elif is_trusted:
            badge = " ✅ Trusted"
        else:
            badge = ""
        
        lines.append(f"  {i}. {title}")
        if duration:
            lines.append(f"     ⏱️  {duration}")
        if publisher:
            lines.append(f"     📺 {publisher}{badge}")
        if url:
            lines.append(f"     🔗 {url}")
        lines.append("")
    
    return "\n".join(lines)


def _pick_best_highlight(results: list[dict], match_query: str) -> Optional[dict]:
    """
    Use LLM to pick the best highlight video from results.
    
    Args:
        results: List of video results.
        match_query: Original match query.
        
    Returns:
        The best matching video dict, or None.
    """
    if not results:
        return None
    
    if len(results) == 1:
        return results[0]
    
    try:
        client = _get_openai_client()
        
        # Build options list
        options = []
        for i, video in enumerate(results, 1):
            options.append(f"{i}. Title: {video.get('title', 'Unknown')}")
            options.append(f"   Duration: {video.get('duration', 'Unknown')}")
            options.append(f"   Publisher: {video.get('publisher', 'Unknown')}")
            options.append("")
        
        options_text = "\n".join(options)
        
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a football expert. Pick the best highlight video for a match. Respond with ONLY the number (1, 2, 3, etc.) of the best option. Prefer official channels, longer durations, and titles that match the specific match."
                },
                {
                    "role": "user",
                    "content": f"Match: {match_query}\n\nVideos:\n{options_text}\n\nWhich is the best highlight video? Reply with just the number."
                }
            ],
            temperature=0,
            max_tokens=10,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract number from response
        import re
        match = re.search(r'\d+', answer)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(results):
                return results[idx]
        
        return results[0]
        
    except Exception as e:
        print(f"[YouTubeSearch] LLM selection failed: {e}")
        return results[0] if results else None


# =============================================================================
# Main Entry Point
# =============================================================================

def search_and_display_highlights(match_query: str, use_llm: bool = True) -> str:
    """
    Search for match highlights and return formatted results.
    
    This is the main entry point for the YouTube search agent.
    
    Args:
        match_query: The match description (e.g., "Arsenal vs Chelsea").
        use_llm: Whether to use LLM to pick the best result.
        
    Returns:
        Formatted string with highlight results.
    """
    results = search_match_highlights(match_query)
    
    if not results:
        return _not_found_message(match_query)
    
    # Format all results
    output = format_highlight_results(results)
    
    return output


def _not_found_message(query: str) -> str:
    """Return a friendly message when no highlights are found."""
    # Parse to give better feedback
    parsed = _parse_match_info(query)
    home = parsed.get("home_team", "Unknown")
    away = parsed.get("away_team", "")
    
    match_str = f"{home} vs {away}" if away else home
    
    return (
        f"❌ Could not find highlight videos for \"{match_str}\".\n\n"
        "Possible reasons:\n"
        "• The match hasn't been played yet\n"
        "• Highlights haven't been uploaded yet (usually 24-48 hours after match)\n"
        "• The teams/spelling might be different\n\n"
        "Suggestions:\n"
        "• Try including the date (e.g., '2024-12-01')\n"
        "• Try the competition name (e.g., 'Champions League')\n"
        "• Check team spelling (e.g., 'Barcelona' not 'Barca')"
    )

