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

# Try to import RAG components (optional)
RAG_EMBEDDINGS_AVAILABLE = False
try:
    from .embeddings_store import _get_embedding_model, _get_collection
    RAG_EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("[YouTubeRAG] Note: embeddings_store not available, using LLM-only validation")


# =============================================================================
# Configuration
# =============================================================================

# Maximum number of video results to return
MAX_RESULTS = 2  # Keep it focused - just 1-2 good highlights
MAX_SEARCH_RESULTS = 15  # Fetch some, then filter to best

# Minimum relevance score to include a result (0-100)
MIN_RELEVANCE_THRESHOLD = 25.0

# Maximum duration for highlights (in seconds) - videos longer than this are likely simulations/full matches
MAX_HIGHLIGHT_DURATION_SECONDS = 60 * 60  # 1 hour = 3600 seconds
MAX_HIGHLIGHT_DURATION_MINUTES = 60  # 1 hour

# YouTube domain filter
YOUTUBE_DOMAINS = ["youtube.com", "youtu.be"]

# Highlight-relevant keywords (boost score when found)
HIGHLIGHT_KEYWORDS = [
    "highlights", "extended highlights", "all goals", "goals",
    "match highlights", "full highlights", "official highlights",
    "hd highlights", "best moments", "key moments",
    "goal", "score", "win", "draw", "defeat", "victory",
]

# Competition keywords (boost relevance when match competition context)
COMPETITION_KEYWORDS = [
    "premier league", "champions league", "europa league",
    "la liga", "serie a", "bundesliga", "ligue 1",
    "fa cup", "carabao cup", "efl cup", "community shield",
    "world cup", "euro", "copa america", "nations league",
    "club world cup", "super cup", "conference league",
]

# Low-relevance keywords (REJECT these - not actual match highlights)
LOW_RELEVANCE_KEYWORDS = [
    "preview", "prediction", "press conference", "interview",
    "analysis", "reaction", "opinion", "debate", "discussion",
    "transfer", "news", "rumour", "rumor", "update",
    "compilation", "best of", "season review", "rewind",
    "throwback", "classic", "retro", "history", "memories",
    "watch along", "watchalong", "fan reaction", "fan cam",
    "training", "behind the scenes", "documentary",
    "pro soccer talk", "preview matchweek",  # NBC Sports talk shows, not highlights
    "takeaways", "takeaway", "the 2 robbies", "2 robbies",  # NBC Sports talk shows
    "thoughts on", "what we learned", "talking points",
]

# =============================================================================
# STRICT ALLOWED SOURCES - ONLY these are allowed
# =============================================================================

# NBC Sports - Always allowed (primary broadcaster)
NBC_SPORTS_CHANNELS = [
    "nbc sports",
    "nbc sports soccer", 
    "nbcsports",
    "nbc sports football",
]

# Official club channel name mappings
# Maps team names to their official YouTube channel identifiers
OFFICIAL_CLUB_CHANNELS = {
    # Premier League clubs
    "arsenal": ["arsenal", "arsenal fc", "arsenal official"],
    "chelsea": ["chelsea fc", "chelsea football club", "chelsea"],
    "liverpool": ["liverpool fc", "liverpool football club", "liverpool"],
    "manchester united": ["manchester united", "man united", "manutd"],
    "manchester city": ["manchester city", "man city", "mancity"],
    "tottenham": ["tottenham hotspur", "spurs official", "spurs", "tottenham"],
    "newcastle": ["newcastle united", "newcastle utd", "nufc"],
    "aston villa": ["aston villa", "aston villa fc", "avfc"],
    "west ham": ["west ham united", "west ham", "whufc"],
    "brighton": ["brighton & hove albion", "brighton", "bhafc"],
    "wolves": ["wolverhampton wanderers", "wolves", "wolves official"],
    "bournemouth": ["afc bournemouth", "bournemouth", "afcb"],
    "fulham": ["fulham fc", "fulham", "fulham football club"],
    "crystal palace": ["crystal palace fc", "crystal palace", "cpfc"],
    "brentford": ["brentford fc", "brentford"],
    "everton": ["everton fc", "everton", "everton football club"],
    "nottingham forest": ["nottingham forest", "nffc"],
    "ipswich": ["ipswich town", "ipswich"],
    "leicester": ["leicester city", "lcfc", "leicester"],
    "southampton": ["southampton fc", "southampton", "saints"],
    # Big European clubs
    "real madrid": ["real madrid", "real madrid cf"],
    "barcelona": ["fc barcelona", "barcelona", "barca"],
    "bayern": ["bayern munich", "fc bayern munich", "bayern"],
    "psg": ["psg", "paris saint-germain"],
    "juventus": ["juventus", "juventus fc"],
    "ac milan": ["ac milan", "milan"],
    "inter": ["inter milan", "inter", "fc internazionale"],
    "dortmund": ["borussia dortmund", "bvb", "dortmund"],
}

# For backwards compatibility - not used in strict mode
TOP_PRIORITY_CHANNELS = NBC_SPORTS_CHANNELS
TRUSTED_CHANNELS = NBC_SPORTS_CHANNELS
TIER1_PRIORITY_CHANNELS = NBC_SPORTS_CHANNELS

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
# LLM-Based Natural Language Query Parser
# =============================================================================

def _parse_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse natural language queries into structured match info.
    
    Handles queries like:
        - "tell me about the most recent man city game"
        - "arsenal vs chelsea last week"
        - "liverpool champions league highlights"
        - "what happened in the tottenham match yesterday"
    
    Args:
        query: Natural language query from user.
        
    Returns:
        dict with:
            - home_team: str (full official name)
            - away_team: str or None
            - date: str (YYYY-MM-DD) or None
            - date_context: str (e.g., "most recent", "last week")
            - competition: str or None
            - is_most_recent: bool
    """
    print(f"\n[QueryParser] === Parsing Natural Language Query ===")
    print(f"[QueryParser] Input: \"{query}\"")
    
    try:
        client = _get_openai_client()
        
        # Get current date for context
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        current_month = today.strftime("%B %Y")
        
        system_prompt = """You are a football/soccer query parser. Extract match information from natural language queries.

IMPORTANT: Normalize team nicknames to full official names:
- "man city", "city" → "Manchester City"
- "man utd", "united" → "Manchester United"  
- "spurs" → "Tottenham"
- "arsenal", "gunners" → "Arsenal"
- "liverpool", "reds" (in PL context) → "Liverpool"
- "chelsea", "blues" (in PL context) → "Chelsea"
- "barca" → "Barcelona"
- "real", "madrid" → "Real Madrid"
- "bayern" → "Bayern Munich"
- etc.

Extract and return ONLY valid JSON with these fields:
{
    "home_team": "full team name or null",
    "away_team": "full team name or null",
    "date": "YYYY-MM-DD or null",
    "date_context": "most recent/last week/yesterday/specific description or null",
    "competition": "Premier League/Champions League/etc or null",
    "is_most_recent": true/false
}

If user says "most recent", "latest", "last game" - set is_most_recent to true.
If user mentions a specific date, convert to YYYY-MM-DD format.
If only one team mentioned, put it in home_team, leave away_team null."""

        user_prompt = f"""Parse this football query. Today's date is {current_date} ({current_month}).

Query: "{query}"

Return ONLY the JSON object, no other text."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=200,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        import json
        
        # Clean up the response (remove markdown code blocks if present)
        if answer.startswith("```"):
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
        answer = answer.strip()
        
        parsed = json.loads(answer)
        
        print(f"[QueryParser] LLM parsed result:")
        print(f"[QueryParser]   Home team: {parsed.get('home_team', 'None')}")
        print(f"[QueryParser]   Away team: {parsed.get('away_team', 'None')}")
        print(f"[QueryParser]   Date: {parsed.get('date', 'None')}")
        print(f"[QueryParser]   Date context: {parsed.get('date_context', 'None')}")
        print(f"[QueryParser]   Competition: {parsed.get('competition', 'None')}")
        print(f"[QueryParser]   Most recent: {parsed.get('is_most_recent', False)}")
        
        return parsed
        
    except Exception as e:
        print(f"[QueryParser] LLM parsing failed: {e}")
        print(f"[QueryParser] Falling back to regex parsing")
        return {}


# =============================================================================
# YouTube Search Functions
# =============================================================================

# =============================================================================
# Match Query Parsing
# =============================================================================

def _parse_match_info(query: str) -> dict:
    """
    Parse a match query to extract teams, date, season, and competition info.
    
    Uses LLM for natural language understanding, with regex fallback.
    
    Handles queries like:
        - "tell me about the most recent man city game"
        - "arsenal vs chelsea 2024-03-15"
        - "liverpool champions league highlights"
        - "what happened in the spurs match yesterday"
    
    Args:
        query: Match description string (natural language).
        
    Returns:
        dict with:
            - home_team: str (full official name)
            - away_team: str or None
            - date: str or None (YYYY-MM-DD)
            - season: str or None
            - year: str or None
            - competition: str or None
            - is_most_recent: bool
    """
    print(f"\n[YouTubeParse] === Parsing Query ===")
    print(f"[YouTubeParse] Input: \"{query}\"")
    
    # First, try LLM parsing for natural language understanding
    llm_result = _parse_query_with_llm(query)
    
    result = {
        "home_team": None,
        "away_team": None,
        "date": None,
        "season": None,
        "year": None,
        "month_year": None,
        "competition": None,
        "is_most_recent": False,
    }
    
    # Use LLM results if available
    if llm_result:
        result["home_team"] = llm_result.get("home_team")
        result["away_team"] = llm_result.get("away_team")
        result["date"] = llm_result.get("date")
        result["competition"] = llm_result.get("competition")
        result["is_most_recent"] = llm_result.get("is_most_recent", False)
        
        # Extract year from date if present
        if result["date"]:
            try:
                dt = datetime.strptime(result["date"], "%Y-%m-%d")
                result["year"] = str(dt.year)
                result["month_year"] = dt.strftime("%B %Y")
            except ValueError:
                pass
    
    # If LLM didn't find teams, fall back to regex parsing
    if not result["home_team"]:
        print(f"[YouTubeParse] LLM didn't extract teams, using regex fallback...")
        result = _parse_match_info_regex(query, result)
    
    # Default to current season if nothing specified
    if not result["season"] and not result["year"] and not result["date"]:
        current_year = datetime.now().year
        current_month = datetime.now().month
        if current_month >= 8:
            result["season"] = f"{current_year}-{str(current_year + 1)[2:]}"
        else:
            result["season"] = f"{current_year - 1}-{str(current_year)[2:]}"
    
    # Log final parsed results
    print(f"\n[YouTubeParse] Final parsed results:")
    print(f"[YouTubeParse]   Home team: {result.get('home_team', 'None')}")
    print(f"[YouTubeParse]   Away team: {result.get('away_team', 'None')}")
    print(f"[YouTubeParse]   Date: {result.get('date', 'None')}")
    print(f"[YouTubeParse]   Season: {result.get('season', 'None')}")
    print(f"[YouTubeParse]   Year: {result.get('year', 'None')}")
    print(f"[YouTubeParse]   Competition: {result.get('competition', 'None')}")
    print(f"[YouTubeParse]   Most recent: {result.get('is_most_recent', False)}")
    
    return result


def _parse_match_info_regex(query: str, result: dict) -> dict:
    """
    Regex-based fallback parser for match queries.
    
    Args:
        query: Original query string.
        result: Existing result dict to update.
        
    Returns:
        Updated result dict.
    """
    working_query = query
    
    # Extract competition names
    competition_patterns = [
        (r"(champions\s*league)", "Champions League"),
        (r"(europa\s*league)", "Europa League"),
        (r"(conference\s*league)", "Conference League"),
        (r"(premier\s*league)", "Premier League"),
        (r"(la\s*liga)", "La Liga"),
        (r"(serie\s*a)", "Serie A"),
        (r"(bundesliga)", "Bundesliga"),
        (r"(ligue\s*1)", "Ligue 1"),
        (r"(fa\s*cup)", "FA Cup"),
        (r"(carabao\s*cup)", "Carabao Cup"),
        (r"(efl\s*cup)", "EFL Cup"),
        (r"(community\s*shield)", "Community Shield"),
        (r"(world\s*cup)", "World Cup"),
        (r"(euro\s*\d{4})", None),
        (r"(copa\s*america)", "Copa America"),
        (r"(nations\s*league)", "Nations League"),
    ]
    
    if not result.get("competition"):
        for pattern, normalized in competition_patterns:
            comp_match = re.search(pattern, working_query, re.IGNORECASE)
            if comp_match:
                result["competition"] = normalized or comp_match.group(1)
                working_query = working_query.replace(comp_match.group(1), "")
                break
    
    # Extract season patterns
    if not result.get("season"):
        season_patterns = [
            r"(\d{4}[-/]\d{2,4})",
            r"(\d{2}/\d{2})",
            r"season\s*(\d{4})",
        ]
        for pattern in season_patterns:
            match = re.search(pattern, working_query, re.IGNORECASE)
            if match:
                result["season"] = match.group(1)
                working_query = re.sub(pattern, "", working_query, flags=re.IGNORECASE)
                break
    
    # Extract date: YYYY-MM-DD
    if not result.get("date"):
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", working_query)
        if date_match:
            result["date"] = date_match.group(1)
            working_query = working_query.replace(date_match.group(1), "")
            try:
                dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                result["year"] = str(dt.year)
                result["month_year"] = dt.strftime("%B %Y")
            except ValueError:
                pass
    
    # Extract standalone year
    if not result.get("year") and not result.get("season"):
        year_match = re.search(r"\b(20\d{2})\b", working_query)
        if year_match:
            result["year"] = year_match.group(1)
            working_query = working_query.replace(year_match.group(1), "")
    
    # Parse team names
    if not result.get("home_team"):
        separators = r"\s+(?:vs\.?|v\.?|versus|-|against|@)\s+"
        parts = re.split(separators, working_query.strip(), flags=re.IGNORECASE)
        
        if len(parts) >= 2:
            result["home_team"] = parts[0].strip()
            result["away_team"] = parts[1].strip()
        elif len(parts) == 1 and parts[0].strip():
            result["home_team"] = parts[0].strip()
    
    return result


def _build_highlight_queries(match_info: dict) -> list[str]:
    """
    Build FOCUSED search queries for highlights.
    
    Strategy:
        - Prioritize NBC Sports (they always include dates in titles)
        - Also check official club channels
        - Use specific date if provided, otherwise "most recent"
    
    Args:
        match_info: Parsed match info dict.
        
    Returns:
        List of search query strings (focused, not too many).
    """
    print(f"\n[YouTubeQuery] === Building Search Queries ===")
    
    queries = []
    
    home = match_info.get("home_team", "")
    away = match_info.get("away_team", "")
    date = match_info.get("date", "")  # YYYY-MM-DD format
    month_year = match_info.get("month_year", "")
    competition = match_info.get("competition", "")
    is_most_recent = match_info.get("is_most_recent", False)
    
    if not home:
        print(f"[YouTubeQuery] ERROR: No teams found in query")
        return []
    
    # Build match string
    match_str = f"{home} vs {away}" if away else home
    match_str_alt = f"{away} vs {home}" if away else home  # Also search reverse order
    
    print(f"[YouTubeQuery] Match: \"{match_str}\"")
    print(f"[YouTubeQuery] Date: {date or 'most recent'}")
    
    # Get current date info
    now = datetime.now()
    current_year = now.year
    current_month = now.strftime("%B")
    current_month_num = now.month
    
    # Format date for search (NBC Sports uses MM/DD/YYYY in titles)
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            nbc_date_format = dt.strftime("%m/%d/%Y")  # e.g., "11/30/2025"
            month_name = dt.strftime("%B")  # e.g., "November"
            year = dt.year
        except ValueError:
            nbc_date_format = ""
            month_name = current_month
            year = current_year
    else:
        nbc_date_format = ""
        month_name = current_month
        year = current_year
    
    # ========== PRIMARY: NBC Sports with specific date ==========
    # NBC Sports always has dates in titles like "11/30/2025"
    if nbc_date_format:
        queries.append(f"{home} {away} NBC Sports {nbc_date_format}")
        queries.append(f"{match_str} NBC Sports highlights {nbc_date_format}")
    
    # NBC Sports with month/year
    queries.append(f"{match_str} NBC Sports highlights {month_name} {year}")
    queries.append(f"{match_str_alt} NBC Sports highlights {month_name} {year}")
    
    # ========== SECONDARY: Official club channels ==========
    if home:
        queries.append(f"{home} official highlights vs {away} {month_name} {year}")
    if away:
        queries.append(f"{away} official highlights vs {home} {month_name} {year}")
    
    # ========== FALLBACK: General search with year ==========
    if is_most_recent:
        queries.append(f"{match_str} highlights {current_month} {current_year}")
    else:
        queries.append(f"{match_str} highlights {year}")
    
    # Remove duplicates
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)
    
    print(f"[YouTubeQuery] Search queries:")
    for i, q in enumerate(unique_queries, 1):
        print(f"[YouTubeQuery]   {i}. \"{q}\"")
    
    return unique_queries


def _is_nbc_sports(publisher: str) -> bool:
    """Check if the publisher is NBC Sports."""
    publisher_lower = publisher.lower()
    return any(nbc in publisher_lower for nbc in NBC_SPORTS_CHANNELS)


def _is_official_club_channel(publisher: str, team_name: str) -> bool:
    """
    Check if the publisher is the official channel for a specific team.
    
    Args:
        publisher: The YouTube channel name.
        team_name: The team name to check for.
        
    Returns:
        True if publisher is the official channel for this team.
    """
    if not team_name:
        return False
    
    publisher_lower = publisher.lower()
    team_lower = team_name.lower()
    
    # Check if team has known official channels
    for team_key, channel_names in OFFICIAL_CLUB_CHANNELS.items():
        if team_key in team_lower or team_lower in team_key:
            # Found the team - check if publisher matches any official channel
            return any(ch in publisher_lower for ch in channel_names)
    
    # Fallback: check if team name is in publisher
    return team_lower in publisher_lower


def _is_allowed_source(publisher: str, home_team: str, away_team: str) -> bool:
    """
    STRICT CHECK: Is this video from an allowed source?
    
    Allowed sources are ONLY:
    1. NBC Sports
    2. Official channel of the HOME team
    3. Official channel of the AWAY team
    
    Args:
        publisher: YouTube channel name.
        home_team: Home team name.
        away_team: Away team name.
        
    Returns:
        True ONLY if from NBC Sports or official club channel.
    """
    # Check NBC Sports first (always allowed)
    if _is_nbc_sports(publisher):
        return True
    
    # Check if it's the official channel of either team
    if home_team and _is_official_club_channel(publisher, home_team):
        return True
    
    if away_team and _is_official_club_channel(publisher, away_team):
        return True
    
    return False


def _is_tier1_channel(publisher: str) -> bool:
    """Check if NBC Sports (for backwards compatibility)."""
    return _is_nbc_sports(publisher)


def _is_trusted_channel(publisher: str) -> bool:
    """Check if NBC Sports (for backwards compatibility)."""
    return _is_nbc_sports(publisher)


def _is_top_priority_channel(publisher: str) -> bool:
    """Check if NBC Sports (for backwards compatibility)."""
    return _is_nbc_sports(publisher)


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
# Relevance Scoring System
# =============================================================================

def _calculate_video_relevance(video: dict, match_info: dict) -> float:
    """
    Calculate a comprehensive relevance score for a video result.
    
    Score ranges from 0 to 100. Higher = more relevant.
    
    Scoring factors:
        - Team name matching (up to 30 points)
        - Date/season matching (up to 20 points)
        - Competition matching (up to 15 points)
        - Channel trust (up to 25 points)
        - Highlight keywords (up to 15 points)
        - Duration appropriateness (up to 10 points)
        - Recency bonus (up to 15 points)
        - Low-relevance penalties (subtract up to 40 points)
    
    Args:
        video: Video result dict.
        match_info: Parsed match info.
        
    Returns:
        Relevance score (0.0 to 100.0).
    """
    score = 0.0
    
    title = video.get("title", "").lower()
    title_raw = video.get("title", "")
    description = video.get("description", "").lower()
    publisher = video.get("publisher", "").lower()
    duration = video.get("duration", "")
    combined = f"{title} {description} {publisher}"
    
    home = (match_info.get("home_team") or "").lower()
    away = (match_info.get("away_team") or "").lower()
    date = match_info.get("date", "")
    season = match_info.get("season", "")
    year = match_info.get("year", "")
    competition = (match_info.get("competition") or "").lower()
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # ========== TEAM MATCHING (up to 30 points) ==========
    home_in_title = home and home in title
    away_in_title = away and away in title
    
    if home:
        if home_in_title:
            score += 12
        elif home in description:
            score += 6
    
    if away:
        if away_in_title:
            score += 12
        elif away in description:
            score += 6
    
    # Bonus for BOTH teams in title (strong signal)
    if home and away and home_in_title and away_in_title:
        score += 6
    
    # CRITICAL: HEAVY PENALTY for wrong team in title
    # If user searched for "Arsenal vs Chelsea" but video is "Arsenal vs Tottenham",
    # the video has the wrong team and should be penalized
    if home and away:
        # Check if there's a different team in the title
        # Look for "vs" or "v." pattern and extract teams
        vs_pattern = r'([a-zA-Z\s]+)\s+(?:vs?\.?|versus)\s+([a-zA-Z\s]+)'
        vs_match = re.search(vs_pattern, title_raw, re.IGNORECASE)
        if vs_match:
            title_team1 = vs_match.group(1).strip().lower()
            title_team2 = vs_match.group(2).strip().lower()
            
            # Check if either title team is NOT one of our target teams
            def team_matches(title_team, target_team):
                if not target_team:
                    return True
                target_lower = target_team.lower()
                # Check if team names match (partial match for short names like "City")
                return target_lower in title_team or title_team in target_lower
            
            # Both title teams should match our target teams (in either order)
            team1_matches_home = team_matches(title_team1, home)
            team1_matches_away = team_matches(title_team1, away)
            team2_matches_home = team_matches(title_team2, home)
            team2_matches_away = team_matches(title_team2, away)
            
            valid_match = (team1_matches_home and team2_matches_away) or \
                         (team1_matches_away and team2_matches_home)
            
            if not valid_match:
                # This is the WRONG MATCH - massive penalty
                score -= 200
    
    # ========== DATE/SEASON/YEAR MATCHING (up to 20 points) ==========
    if date:
        if date in title_raw or date in description:
            score += 20  # Exact date match is very strong
    
    if season:
        if season in title_raw or season in combined:
            score += 15
    
    if year:
        if year in title_raw:
            score += 10
    
    # ========== COMPETITION MATCHING (up to 15 points) ==========
    if competition:
        if competition in combined:
            score += 15
        # Check for competition abbreviations
        comp_abbrevs = {
            "champions league": ["ucl", "cl"],
            "europa league": ["uel", "el"],
            "premier league": ["epl", "pl"],
        }
        for full, abbrevs in comp_abbrevs.items():
            if competition == full:
                if any(abbr in combined for abbr in abbrevs):
                    score += 10
                break
    
    # Also check for any competition keywords in video
    for comp_kw in COMPETITION_KEYWORDS:
        if comp_kw in combined:
            score += 3
            break  # Only count once
    
    # ========== CHANNEL TRUST (up to 50 points) ==========
    # NBC Sports gets HIGHEST priority
    if _is_nbc_sports(publisher):
        score += 50  # Massive bonus for NBC Sports
    # Official club channels also get high bonus
    elif video.get("is_official_club"):
        score += 45
    elif video.get("is_tier1"):
        score += 40
    else:
        # No bonus for non-allowed sources (they'll be filtered anyway)
        pass
    
    # ========== HIGHLIGHT KEYWORDS (up to 15 points) ==========
    keyword_matches = 0
    for keyword in HIGHLIGHT_KEYWORDS:
        if keyword in combined:
            keyword_matches += 1
    score += min(keyword_matches * 3, 15)
    
    # Bonus for "extended highlights" (best content)
    if "extended" in title and "highlight" in title:
        score += 5
    
    # ========== DURATION APPROPRIATENESS (up to 10 points) ==========
    duration_seconds = _parse_duration_to_seconds(duration)
    if duration_seconds:
        # Ideal highlight length: 5-20 minutes
        if 300 <= duration_seconds <= 1200:
            score += 10
        elif 180 <= duration_seconds < 300:  # 3-5 min
            score += 7
        elif 1200 < duration_seconds <= 1800:  # 20-30 min
            score += 5
        elif duration_seconds > 2400:  # >40 min
            score -= 10  # Penalty for too long
    
    # ========== RECENCY BONUS/PENALTY (CRITICAL - up to 100 points) ==========
    # HEAVILY prioritize recent matches to avoid showing old highlights
    
    # Check for current season pattern
    if current_month >= 8:
        current_season_str = f"{current_year}-{str(current_year + 1)[2:]}"
        current_season_short = f"{current_year}/{str(current_year + 1)[2:]}"
    else:
        current_season_str = f"{current_year - 1}-{str(current_year)[2:]}"
        current_season_short = f"{current_year - 1}/{str(current_year)[2:]}"
    
    # Try to extract specific date from title (e.g., "11/30/2025" or "12/14/2024")
    import re
    date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', title_raw)
    if date_match:
        vid_month, vid_day, vid_year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
        video["has_specific_date"] = True
        video["video_date"] = f"{vid_year}-{vid_month:02d}-{vid_day:02d}"
        
        # Calculate how recent this is - HUGE bonus for specific dates
        if vid_year == current_year and vid_month == current_month:
            score += 100  # This month - MASSIVE bonus
        elif vid_year == current_year and vid_month >= current_month - 1:
            score += 80  # Last month - huge bonus
        elif vid_year == current_year:
            score += 50  # This year with specific date
        else:
            score -= 40  # Older year with specific date
    else:
        video["has_specific_date"] = False
        # No specific date - LOWER scores since we can't verify recency
        # Prefer videos with explicit dates
        if current_season_str in title_raw or current_season_short in title_raw:
            score += 30  # Season mention without specific date
        elif str(current_year) in title_raw:
            score += 20  # Just year, no specific date - could be any match from that year
        elif str(current_year - 1) in title_raw:
            score += 5
    
    # HEAVY PENALTY for old matches (2023, 2022, etc.)
    old_years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    for old_year in old_years:
        if old_year in title_raw and str(current_year) not in title_raw:
            score -= 60  # Heavy penalty for old content
            break
    
    # ========== LOW RELEVANCE PENALTIES (subtract up to 40 points) ==========
    for keyword in LOW_RELEVANCE_KEYWORDS:
        if keyword in combined:
            score -= 8
    
    # Penalty for live streams (not highlights)
    if "live" in title and "highlight" not in title:
        score -= 20
    
    # ========== SIMULATION PENALTY (heavy) ==========
    if _is_simulation(title, description, publisher):
        score -= 50
    
    # Clamp score to valid range
    return max(0.0, min(100.0, score))


def _filter_and_rank_by_relevance(results: list[dict], match_info: dict) -> list[dict]:
    """
    Filter and rank video results - PRIORITIZE NBC Sports and official clubs.
    
    Strategy:
        1. NBC Sports videos with matching date get TOP priority
        2. Official club channels second
        3. Other sources filtered out unless nothing else found
    
    Args:
        results: Raw video results.
        match_info: Parsed match info.
        
    Returns:
        Ranked list (max 2 results).
    """
    if not results:
        return []
    
    home_team = match_info.get("home_team", "")
    away_team = match_info.get("away_team", "")
    target_date = match_info.get("date", "")  # YYYY-MM-DD
    
    print(f"\n[YouTubeSearch] === FILTERING FOR BEST HIGHLIGHTS ===")
    print(f"[YouTubeSearch] Match: {home_team} vs {away_team}")
    if target_date:
        print(f"[YouTubeSearch] Target date: {target_date}")
    
    nbc_results = []
    club_results = []
    other_results = []
    
    # Parse target date for matching
    target_month, target_day, target_year = None, None, None
    if target_date:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            target_month, target_day, target_year = dt.month, dt.day, dt.year
        except ValueError:
            pass
    
    for video in results:
        publisher = video.get("publisher", "")
        title = video.get("title", "")
        title_lower = title.lower()
        description = video.get("description", "")
        
        # Skip simulations
        if _is_simulation(title, description, publisher):
            continue
        
        # Skip previews, reactions, talk shows (not actual highlights)
        if any(kw in title_lower for kw in LOW_RELEVANCE_KEYWORDS):
            print(f"[YouTubeSearch]   ⚠️ SKIP [Talk show/analysis]: {title[:40]}...")
            continue
        
        # MUST contain "highlights" or "goals" to be actual match highlights
        is_highlight_video = any(word in title_lower for word in ["highlight", "highlights", "goals", "all goals", "extended highlight"])
        if not is_highlight_video:
            print(f"[YouTubeSearch]   ⚠️ SKIP [Not highlights video]: {title[:40]}...")
            continue
        
        # Skip very short videos (likely clips/commentary, not full highlights)
        # Full highlights are typically 6-15 minutes
        duration = video.get("duration", "")
        duration_seconds = _parse_duration_to_seconds(duration)
        if duration_seconds and duration_seconds < 300:  # Less than 5 minutes
            if "highlight" not in title_lower and "goals" not in title_lower:
                print(f"[YouTubeSearch]   ⚠️ SKIP [Too short - {duration}]: {title[:35]}...")
                continue
        
        # Check for date in title (NBC Sports format: MM/DD/YYYY)
        import re
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', title)
        video_matches_date = False
        if date_match and target_year:
            vid_month, vid_day, vid_year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            video_matches_date = (vid_month == target_month and vid_day == target_day and vid_year == target_year)
            video["video_date"] = f"{vid_month}/{vid_day}/{vid_year}"
        
        # Categorize by source
        is_nbc = "nbc sports" in title_lower or "nbc sports" in publisher.lower()
        is_club = (home_team and home_team.lower() in publisher.lower()) or \
                  (away_team and away_team.lower() in publisher.lower())
        
        video["is_nbc_sports"] = is_nbc
        video["is_official_club"] = is_club
        video["matches_date"] = video_matches_date
        
        # Calculate relevance score
        score = _calculate_video_relevance(video, match_info)
        
        # BONUS for matching date
        if video_matches_date:
            score += 100
        
        video["_relevance_score"] = score
        
        if is_nbc:
            video["source_type"] = "NBC Sports"
            nbc_results.append((score, video))
            date_str = f" [{video.get('video_date', '')}]" if video.get('video_date') else ""
            match_str = " ✓DATE" if video_matches_date else ""
            print(f"[YouTubeSearch]   📺 NBC Sports{date_str}{match_str}: {title[:45]}...")
        elif is_club:
            video["source_type"] = "Official Club"
            club_results.append((score, video))
            print(f"[YouTubeSearch]   🏟️ Official Club: {title[:45]}...")
        else:
            video["source_type"] = "Other"
            other_results.append((score, video))
    
    # Sort each category by score
    nbc_results.sort(key=lambda x: x[0], reverse=True)
    club_results.sort(key=lambda x: x[0], reverse=True)
    other_results.sort(key=lambda x: x[0], reverse=True)
    
    # Build final results: prioritize NBC Sports, then clubs
    final_results = []
    
    # Add NBC Sports results first (prefer those with matching dates)
    for score, video in nbc_results:
        if len(final_results) < MAX_RESULTS:
            final_results.append(video)
    
    # Add official club results
    for score, video in club_results:
        if len(final_results) < MAX_RESULTS:
            final_results.append(video)
    
    # Only add others if we need more
    if not final_results:
        for score, video in other_results[:MAX_RESULTS]:
            final_results.append(video)
    
    print(f"\n[YouTubeSearch] Selected {len(final_results)} highlights")
    
    return final_results


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
                
                is_tier1 = _is_tier1_channel(publisher)
                is_top_priority = is_tier1 or _is_top_priority_channel(publisher)
                is_trusted = is_top_priority or _is_trusted_channel(publisher)
                
                results.append({
                    "title": title,
                    "url": video_url,
                    "duration": duration,
                    "thumbnail": video.get("images", {}).get("large", ""),
                    "description": description,
                    "publisher": publisher,
                    "is_youtube": is_youtube,
                    "is_tier1": is_tier1,  # Premier League, NBC Sports, Clubs
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
    
    STRICT: If user asks for Team A vs Team B, video must have BOTH teams.
    
    Args:
        video: Video result dict.
        match_info: Parsed match info.
        
    Returns:
        True if video matches the requested teams.
    """
    title_lower = video.get("title", "").lower()
    
    home = (match_info.get("home_team") or "").lower()
    away = (match_info.get("away_team") or "").lower()
    
    def team_in_title(team: str, title: str) -> bool:
        """Check if team name appears in title."""
        if not team:
            return False
        # Check full name
        if team in title:
            return True
        # Check first word (e.g., "Arsenal" from "Arsenal FC")
        first_word = team.split()[0]
        if len(first_word) >= 4 and first_word in title:
            return True
        return False
    
    home_in_title = team_in_title(home, title_lower)
    away_in_title = team_in_title(away, title_lower)
    
    # STRICT: If BOTH teams specified, BOTH must be in title
    if home and away:
        return home_in_title and away_in_title
    
    # If only one team specified
    if home:
        return home_in_title
    
    return True


def search_match_highlights(match_query: str, deep_validate: bool = False) -> list[dict]:
    """
    Search for match highlights on YouTube from trusted sports channels.
    
    Uses comprehensive relevance scoring to return ONLY highly relevant results.
    
    Smart features:
        - Understands home vs away team order (first team = home)
        - Defaults to current season if none specified
        - Validates videos actually match the requested teams
        - Prioritizes official club channels and broadcasters
        - Scores and filters by relevance (minimum threshold)
    
    Args:
        match_query: Match description (e.g., "Arsenal vs Chelsea 2024-03-15").
        deep_validate: If True, fetches video description/transcript for validation.
        
    Returns:
        List of video results (verified as real, highly relevant highlights).
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
    competition = match_info.get("competition", "Not specified")
    
    print(f"\n[YouTubeSearch] === Search Parameters ===")
    print(f"[YouTubeSearch] Home team: {home}")
    print(f"[YouTubeSearch] Away team: {away}")
    print(f"[YouTubeSearch] Date: {date}")
    print(f"[YouTubeSearch] Season: {season}")
    print(f"[YouTubeSearch] Competition: {competition}")
    
    # Build smart search queries
    queries = _build_highlight_queries(match_info)
    
    if not queries:
        print(f"[YouTubeSearch] ERROR: Could not build queries")
        return []
    
    all_results = []
    seen_urls = set()
    wrong_match_filtered = 0
    
    print(f"\n[YouTubeSearch] === Executing Searches ===")
    
    # Run more queries to get a larger pool for relevance scoring
    queries_to_run = queries[:6]  # Run up to 6 queries
    
    for query in queries_to_run:
        print(f"\n[YouTubeSearch] Searching: \"{query}\"")
        results = _search_youtube_highlights(query, max_results=6)
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # VALIDATE: Check if video matches the teams we're looking for
                if not _video_matches_query(result, match_info):
                    wrong_match_filtered += 1
                    print(f"[YouTubeSearch]   ✗ Wrong match: {title[:45]}...")
                    continue
                
                print(f"[YouTubeSearch]   ✓ Match found: {title[:45]}...")
                result["match_info"] = match_info
                all_results.append(result)
        
        # Stop if we have a good pool of results
        if len(all_results) >= MAX_SEARCH_RESULTS:
            break
    
    # Log raw results
    print(f"\n[YouTubeSearch] === Raw Results Summary ===")
    print(f"[YouTubeSearch] Total results collected: {len(all_results)}")
    print(f"[YouTubeSearch] Wrong match filtered: {wrong_match_filtered}")
    
    if not all_results:
        print(f"[YouTubeSearch] ❌ NO VALID HIGHLIGHTS FOUND")
        print(f"[YouTubeSearch] Could not find highlights for {home} vs {away}")
        print(f"{'='*60}\n")
        return []
    
    # ========== RELEVANCE SCORING & FILTERING ==========
    print(f"\n[YouTubeSearch] === Applying Relevance Scoring ===")
    
    # Score and filter by relevance
    ranked_results = _filter_and_rank_by_relevance(all_results, match_info)
    
    print(f"[YouTubeSearch] After relevance filtering: {len(ranked_results)} results")
    print(f"[YouTubeSearch] (Minimum score threshold: {MIN_RELEVANCE_THRESHOLD})")
    
    if not ranked_results:
        # Fall back to old scoring if nothing passes threshold
        print(f"[YouTubeSearch] No results passed threshold, using basic scoring...")
        ranked_results = _score_and_sort_results(all_results, match_info)[:MAX_RESULTS]
    
    # Count sources by type
    nbc_count = sum(1 for r in ranked_results if r.get("is_nbc_sports"))
    club_count = sum(1 for r in ranked_results if r.get("is_official_club"))
    
    # Log top results with scores
    print(f"\n[YouTubeSearch] === OFFICIAL SOURCES ONLY ===")
    print(f"[YouTubeSearch] Found {len(ranked_results)} from allowed sources:")
    if nbc_count > 0:
        print(f"[YouTubeSearch]   📺 {nbc_count} from NBC Sports")
    if club_count > 0:
        print(f"[YouTubeSearch]   🏟️ {club_count} from Official Club Channels")
    
    for i, r in enumerate(ranked_results[:MAX_RESULTS], 1):
        score = r.get("_relevance_score", 0)
        tier = "🥇" if score >= 60 else "🥈" if score >= 40 else "📄"
        
        # Show source type
        if r.get("is_nbc_sports"):
            source = " [NBC Sports]"
        elif r.get("is_official_club"):
            source = " [Official Club]"
        else:
            source = ""
        
        print(f"[YouTubeSearch]   {tier} {i}. [Score: {score:.0f}]{source} {r.get('title', '')[:40]}...")
    
    # Deep validation (optional - disabled by default for speed)
    if deep_validate and ranked_results:
        print(f"\n[YouTubeSearch] === Deep Validation ===")
        validated_results = []
        
        for video in ranked_results[:MAX_RESULTS + 3]:
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
    
    print(f"[YouTubeSearch] SEARCH COMPLETE - Returning {min(len(ranked_results), MAX_RESULTS)} from official sources only")
    print(f"{'='*60}\n")
    return ranked_results[:MAX_RESULTS]


def _score_and_sort_results(results: list[dict], match_info: dict) -> list[dict]:
    """
    Score and sort results by relevance to the specific match.
    
    This is the fallback scoring function (used when new relevance scoring
    doesn't produce results above threshold).
    
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
        
        # Store a normalized relevance score for display consistency
        # Convert internal score to 0-100 scale
        normalized_score = max(0, min(100, (score + 200) / 6))  # Rough normalization
        r["_relevance_score"] = normalized_score
        
        return (-score, not r.get("is_top_priority", False), not r.get("is_trusted", False))
    
    return sorted(filtered_results, key=score_result)


# =============================================================================
# Display and Formatting
# =============================================================================

def format_highlight_results(results: list[dict], match_metadata: dict = None) -> str:
    """
    Format highlight results for display with optional match context.
    
    Args:
        results: List of video result dicts.
        match_metadata: Optional match info to show context.
        
    Returns:
        Formatted string for CLI display.
    """
    if not results:
        return (
            "❌ No highlights found.\n\n"
            "The match highlights may not have been uploaded yet.\n"
            "Try again later or check YouTube directly."
        )
    
    lines = []
    
    # Show match context if available
    if match_metadata and match_metadata.get("score"):
        home = match_metadata.get("home_team", "Team A")
        away = match_metadata.get("away_team", "Team B")
        score = match_metadata.get("score", "?-?")
        lines.append(f"🎬 Highlights for: {home} {score} {away}\n")
    else:
        lines.append("🎬 Match Highlights:\n")
    
    for i, video in enumerate(results, 1):
        title = video.get("title", "Untitled")
        url = video.get("url", "")
        duration = video.get("duration", "")
        source_type = video.get("source_type", "")
        
        # RAG validation info
        rag_validation = video.get("rag_validation", {})
        confidence = rag_validation.get("confidence", 0)
        
        # Source badge
        if video.get("is_nbc_sports"):
            badge = "📺 NBC Sports"
        elif video.get("is_official_club"):
            badge = "🏟️ Official"
        else:
            badge = ""
        
        # Confidence indicator (if RAG validated)
        if confidence >= 0.7:
            conf_badge = "✓ Verified"
        elif confidence >= 0.5:
            conf_badge = "○ Likely match"
        else:
            conf_badge = ""
        
        lines.append(f"  {i}. {title}")
        if badge or conf_badge:
            badges = " | ".join(filter(None, [badge, conf_badge]))
            lines.append(f"     {badges}")
        if duration:
            lines.append(f"     ⏱️ {duration}")
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
# LLM-Based Video Validation
# =============================================================================

def _llm_validate_highlight_video(title: str, description: str, home_team: str, away_team: str) -> tuple[bool, str]:
    """
    Use LLM to intelligently validate if a video is actual match highlights.
    
    Args:
        title: Video title.
        description: Video description.
        home_team: Home team name.
        away_team: Away team name.
        
    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        client = _get_openai_client()
        
        prompt = f"""Analyze this YouTube video and determine if it's ACTUAL MATCH HIGHLIGHTS.

Video Title: "{title}"
Video Description: "{description[:200] if description else 'N/A'}"
Match we're looking for: {home_team} vs {away_team}

VALID (actual highlights):
- Official match highlights showing goals, key moments
- Extended highlights, full highlights
- "All Goals" compilations from the actual match
- Official broadcaster highlights (NBC Sports, Sky Sports, ESPN, etc.)

INVALID (not actual highlights):
- Talk shows, analysis, reactions (e.g., "Takeaways", "Reaction", "The 2 Robbies")
- Press conferences, interviews
- Previews, predictions
- Fan reactions, vlogs
- Compilations from multiple matches
- Video game footage (FIFA, EA FC)

Respond with ONLY one word: VALID or INVALID
Then on the next line, give a 2-3 word reason."""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You classify YouTube videos. Respond with VALID or INVALID, then a brief reason."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20,
        )
        
        answer = response.choices[0].message.content.strip()
        lines = answer.split('\n')
        
        is_valid = "VALID" in lines[0].upper() and "INVALID" not in lines[0].upper()
        reason = lines[1].strip() if len(lines) > 1 else ("Highlights" if is_valid else "Not highlights")
        
        return is_valid, reason
        
    except Exception as e:
        # Fallback to keyword matching if LLM fails
        title_lower = title.lower()
        
        # Check for talk show indicators
        talk_show_keywords = ["takeaway", "reaction", "preview", "prediction", "press conference", 
                             "interview", "analysis", "the 2 robbies", "thoughts on"]
        if any(kw in title_lower for kw in talk_show_keywords):
            return False, "Talk show/analysis"
        
        # Must have highlights or goals
        if not any(word in title_lower for word in ["highlight", "goals"]):
            return False, "No highlights keyword"
        
        return True, "Keyword match"


# =============================================================================
# Trusted Sports Channels for Highlights
# =============================================================================

TRUSTED_HIGHLIGHT_CHANNELS = [
    # US Broadcasters
    "nbc sports",
    "cbs sports",
    "cbs sports golazo",
    "golazo",
    "espn fc",
    "espn",
    # UK Broadcasters
    "sky sports",
    "sky sports football",
    "sky sports premier league",
    "bt sport",
    "tnt sports",
    # Official League Channels
    "premier league",  # Official PL highlights
    "bundesliga",
    "laliga",
    "la liga",
    "serie a",
    "ligue 1",
    "uefa",
    "champions league",
    # Official Club Channels (common names)
    "arsenal",
    "chelsea fc",
    "liverpool fc",
    "manchester city",
    "manchester united",
    "tottenham hotspur",
    "spurs",
]


# =============================================================================
# RAG-Based Video Validation Against Web Search Results
# =============================================================================

def _compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using embeddings.
    
    Args:
        text1: First text (e.g., video title + description)
        text2: Second text (e.g., web search context)
        
    Returns:
        Similarity score between 0 and 1.
    """
    if not RAG_EMBEDDINGS_AVAILABLE:
        return 0.5  # Neutral if embeddings not available
    
    try:
        model = _get_embedding_model()
        
        # Generate embeddings
        emb1 = model.encode(text1, show_progress_bar=False)
        emb2 = model.encode(text2, show_progress_bar=False)
        
        # Compute cosine similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
        
    except Exception as e:
        print(f"[RAGEmbeddings] Similarity error: {e}")
        return 0.5


def _validate_video_with_embeddings(
    video: dict,
    web_context: str,
    match_metadata: dict
) -> dict:
    """
    Validate video using semantic embeddings comparison.
    
    Computes similarity between video metadata and web search context.
    Higher similarity = more likely to be the correct match.
    
    Args:
        video: Video dict with title, description
        web_context: Web search context text
        match_metadata: Match info for context
        
    Returns:
        dict with similarity_score and is_likely_match
    """
    title = video.get("title", "")
    description = video.get("description", "")
    
    # Build video text for comparison
    video_text = f"{title}. {description[:300]}"
    
    # Build context text (key match facts)
    home = match_metadata.get("home_team", "")
    away = match_metadata.get("away_team", "")
    date = match_metadata.get("match_date", "")
    score = match_metadata.get("score", "")
    
    context_text = f"{home} vs {away}. Date: {date}. Score: {score}. {web_context[:500]}"
    
    # Compute similarity
    similarity = _compute_semantic_similarity(video_text, context_text)
    
    # Threshold for considering it a match
    is_likely_match = similarity >= 0.4  # Reasonable threshold for semantic similarity
    
    return {
        "similarity_score": similarity,
        "is_likely_match": is_likely_match,
    }


def _validate_video_against_web_context(
    video: dict,
    web_summary: str,
    match_metadata: dict
) -> dict:
    """
    Use LLM to validate if a YouTube video matches the web search results.
    
    This is the RAG validation step:
    1. Uses the web search summary as ground truth
    2. Compares video title/description against known match facts
    3. Returns validation result with confidence
    
    Args:
        video: Video dict with title, description, url
        web_summary: The LLM-generated summary from web search
        match_metadata: Dict with home_team, away_team, match_date, score
        
    Returns:
        dict with:
            - is_valid: bool (True if video matches the web context)
            - confidence: float (0-1)
            - reason: str (explanation)
    """
    try:
        client = _get_openai_client()
        
        title = video.get("title", "")
        description = video.get("description", "")
        publisher = video.get("publisher", "")
        
        home_team = match_metadata.get("home_team", "Unknown")
        away_team = match_metadata.get("away_team", "Unknown")
        match_date = match_metadata.get("match_date", "Unknown")
        score = match_metadata.get("score", "Unknown")
        
        prompt = f"""You are validating if a YouTube video shows the CORRECT match highlights based on verified web search results.

=== VERIFIED MATCH INFORMATION (from web search) ===
Home Team: {home_team}
Away Team: {away_team}
Match Date: {match_date}
Score: {score}

Summary from web search:
{web_summary[:1000]}

=== YOUTUBE VIDEO TO VALIDATE ===
Title: {title}
Publisher: {publisher}
Description: {description[:500] if description else "N/A"}

=== VALIDATION TASK ===
Determine if this YouTube video is showing highlights for the SAME match described above.

Check for:
1. Do the team names match? (both teams should be the same)
2. Does the date/time context match? (same match, not an older game)
3. Does the score match if shown in title? (e.g., "3-0" should match "{score}")
4. Is it from a DIFFERENT match between the same teams? (reject if wrong date)

CRITICAL: A video from a PREVIOUS match between these teams is INVALID.
For example, if we're looking for the Dec 2025 match, a video from Nov 2024 is INVALID.

Respond with EXACTLY this format:
VERDICT: VALID or INVALID
CONFIDENCE: HIGH, MEDIUM, or LOW
REASON: One sentence explanation"""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You validate YouTube videos against verified match data. Be strict - only validate videos that clearly match the specific match."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Parse response
        is_valid = "VALID" in answer.upper() and "INVALID" not in answer.split("VERDICT")[1].split("\n")[0].upper() if "VERDICT" in answer.upper() else False
        
        confidence = 0.7
        if "HIGH" in answer.upper():
            confidence = 0.95
        elif "LOW" in answer.upper():
            confidence = 0.5
        
        # Extract reason
        reason = "LLM validation"
        if "REASON:" in answer:
            reason = answer.split("REASON:")[-1].strip()
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "reason": reason,
        }
        
    except Exception as e:
        print(f"[RAGValidation] LLM validation error: {e}")
        # Fall back to basic validation
        return {
            "is_valid": True,  # Default to showing if can't verify
            "confidence": 0.5,
            "reason": f"Could not validate: {str(e)[:50]}",
        }


def _validate_videos_with_rag(
    videos: list[dict],
    web_summary: str,
    match_metadata: dict,
    max_results: int = 2
) -> list[dict]:
    """
    Validate a list of videos against web search results using RAG.
    
    Uses a two-step validation process:
    1. Semantic embeddings: Compute similarity between video and web context
    2. LLM validation: Use LLM to verify match details
    
    A video passes if it scores well on BOTH checks.
    
    Args:
        videos: List of video dicts to validate.
        web_summary: The web search summary as ground truth.
        match_metadata: Match info from web search.
        max_results: Maximum validated videos to return.
        
    Returns:
        List of validated video dicts (with validation info added).
    """
    if not videos or not web_summary:
        return videos[:max_results]
    
    print(f"\n[RAGValidation] === RAG Validation Pipeline ===")
    print(f"[RAGValidation] Match: {match_metadata.get('home_team')} vs {match_metadata.get('away_team')}")
    print(f"[RAGValidation] Date: {match_metadata.get('match_date')}")
    print(f"[RAGValidation] Score: {match_metadata.get('score')}")
    print(f"[RAGValidation] Videos to validate: {len(videos)}")
    
    if RAG_EMBEDDINGS_AVAILABLE:
        print(f"[RAGValidation] Using: Embeddings + LLM (full RAG)")
    else:
        print(f"[RAGValidation] Using: LLM only (embeddings not available)")
    
    validated = []
    
    for video in videos:
        title = video.get("title", "")
        
        # Step 1: Embeddings-based similarity (if available)
        embedding_result = {"similarity_score": 0.5, "is_likely_match": True}
        if RAG_EMBEDDINGS_AVAILABLE:
            embedding_result = _validate_video_with_embeddings(video, web_summary, match_metadata)
            similarity = embedding_result["similarity_score"]
            print(f"[RAGValidation] 📊 Embedding similarity: {similarity:.2f} - {title[:40]}...")
        
        # Step 2: LLM validation for detailed verification
        llm_validation = _validate_video_against_web_context(video, web_summary, match_metadata)
        
        # Combine results: Both must pass
        embedding_pass = embedding_result["is_likely_match"] or embedding_result["similarity_score"] >= 0.35
        llm_pass = llm_validation["is_valid"]
        
        # Final decision
        is_valid = embedding_pass and llm_pass
        
        # Calculate combined confidence
        if RAG_EMBEDDINGS_AVAILABLE:
            # Weight: 40% embeddings, 60% LLM
            combined_confidence = (0.4 * embedding_result["similarity_score"]) + (0.6 * llm_validation["confidence"])
        else:
            combined_confidence = llm_validation["confidence"]
        
        video["rag_validation"] = {
            "is_valid": is_valid,
            "confidence": combined_confidence,
            "embedding_similarity": embedding_result.get("similarity_score", 0.5),
            "llm_valid": llm_pass,
            "reason": llm_validation.get("reason", ""),
        }
        
        if is_valid:
            validated.append(video)
            conf_icon = "🟢" if combined_confidence >= 0.7 else "🟡" if combined_confidence >= 0.5 else "🔴"
            print(f"[RAGValidation] {conf_icon} VALID ({combined_confidence:.0%}): {title[:45]}...")
        else:
            # Show why it failed
            fail_reason = []
            if not embedding_pass:
                fail_reason.append("low similarity")
            if not llm_pass:
                fail_reason.append(f"LLM: {llm_validation.get('reason', 'rejected')[:30]}")
            print(f"[RAGValidation] ❌ INVALID: {title[:40]}... ({', '.join(fail_reason)})")
        
        if len(validated) >= max_results:
            break
    
    print(f"\n[RAGValidation] Pipeline complete: {len(validated)}/{len(videos)} videos validated")
    return validated


# =============================================================================
# New Main Entry Point - With Match Metadata and RAG Validation
# =============================================================================

def search_and_display_highlights_with_metadata(
    home_team: str,
    away_team: str,
    match_date: str = None,
    web_summary: str = None,
    match_metadata: dict = None
) -> str:
    """
    Search for match highlights using extracted match metadata.
    
    This is the SMART RAG version that:
    1. Uses exact match info from web search
    2. Searches specific trusted channels (NBC Sports, CBS, ESPN, etc.)
    3. Validates video upload dates are within ±5 days of match date
    4. Uses LLM to validate videos against web search summary (RAG validation)
    
    Args:
        home_team: Home team name.
        away_team: Away team name.
        match_date: Match date in YYYY-MM-DD format.
        web_summary: The summary text from web search (for RAG validation).
        match_metadata: Full match metadata from web search (teams, date, score).
        
    Returns:
        Formatted string with validated highlights.
    """
    print(f"\n{'='*60}")
    print(f"[YouTubeSearch] SMART RAG HIGHLIGHT SEARCH")
    print(f"{'='*60}")
    
    # Use provided metadata or build from args
    if match_metadata is None:
        match_metadata = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "score": None,
        }
    
    use_rag_validation = web_summary is not None and len(web_summary) > 50
    # Simplify team names for better YouTube search
    # "Liverpool Football Club" -> "Liverpool"
    # "Sunderland Association Football Club" -> "Sunderland"
    def simplify_team_name(name):
        if not name:
            return name
        # Remove common suffixes (order matters - longer first)
        for suffix in [" Association Football Club", " Football Club", " AFC", " FC", " Association"]:
            if suffix.lower() in name.lower():
                idx = name.lower().find(suffix.lower())
                name = name[:idx]
        return name.strip()
    
    home_team = simplify_team_name(home_team)
    away_team = simplify_team_name(away_team)
    
    print(f"[YouTubeSearch] Home: {home_team}")
    print(f"[YouTubeSearch] Away: {away_team}")
    print(f"[YouTubeSearch] Match Date: {match_date or 'Unknown'}")
    
    if not home_team:
        return "❌ Could not determine teams for highlight search."
    
    # Parse match date for validation
    match_dt = None
    if match_date:
        try:
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            # Format for NBC Sports style search (MM/DD/YYYY)
            nbc_date = match_dt.strftime("%m/%d/%Y")
            month_name = match_dt.strftime("%B")
            year = match_dt.year
        except ValueError:
            nbc_date = None
            month_name = datetime.now().strftime("%B")
            year = datetime.now().year
    else:
        nbc_date = None
        month_name = datetime.now().strftime("%B")
        year = datetime.now().year
    
    # Build targeted search queries for trusted channels
    match_str = f"{home_team} vs {away_team}" if away_team else home_team
    
    queries = []
    
    # NBC Sports with date (most reliable)
    if nbc_date:
        queries.append(f"{match_str} NBC Sports {nbc_date}")
        queries.append(f"{away_team} vs {home_team} NBC Sports {nbc_date}")
    
    # Trusted channels
    queries.append(f"{match_str} NBC Sports highlights {month_name} {year}")
    queries.append(f"{match_str} CBS Sports Golazo highlights {year}")
    queries.append(f"{match_str} ESPN FC highlights {year}")
    
    # Official club channels
    if home_team:
        queries.append(f"{home_team} official highlights vs {away_team} {month_name} {year}")
    if away_team:
        queries.append(f"{away_team} official highlights vs {home_team} {month_name} {year}")
    
    print(f"\n[YouTubeSearch] Searching trusted channels...")
    for q in queries[:4]:
        print(f"[YouTubeSearch]   • {q}")
    
    # Collect results
    all_results = []
    seen_urls = set()
    
    for query in queries[:5]:  # Limit queries
        results = _search_youtube_highlights(query, max_results=5)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Check if matches both teams
                if _video_matches_query(r, {"home_team": home_team, "away_team": away_team}):
                    all_results.append(r)
    
    print(f"[YouTubeSearch] Found {len(all_results)} matching videos")
    
    if not all_results:
        return _not_found_message(f"{home_team} vs {away_team}")
    
    # Filter and validate results (basic filtering)
    validated = _validate_and_filter_highlights(all_results, home_team, away_team, match_dt)
    
    if not validated:
        return _not_found_message(f"{home_team} vs {away_team}")
    
    # RAG Validation: If we have web search context, validate videos against it
    if use_rag_validation:
        print(f"\n[YouTubeSearch] Using RAG validation with web search context...")
        validated = _validate_videos_with_rag(
            videos=validated,
            web_summary=web_summary,
            match_metadata=match_metadata,
            max_results=MAX_RESULTS
        )
        
        if not validated:
            # Fall back to basic validation if RAG rejects all
            print(f"[YouTubeSearch] RAG rejected all videos, using basic validation...")
            validated = _validate_and_filter_highlights(all_results, home_team, away_team, match_dt)[:MAX_RESULTS]
    
    if not validated:
        return _not_found_message(f"{home_team} vs {away_team}")
    
    return format_highlight_results(validated, match_metadata)


def _validate_and_filter_highlights(
    results: list[dict],
    home_team: str,
    away_team: str,
    match_dt: datetime = None
) -> list[dict]:
    """
    Validate highlights:
    1. Must be from trusted channel OR official club
    2. If match_dt provided, video must be posted within ±5 days
    
    Args:
        results: Raw video results.
        home_team: Home team name.
        away_team: Away team name.
        match_dt: Match datetime for date validation.
        
    Returns:
        Validated and sorted results.
    """
    print(f"\n[YouTubeSearch] === VALIDATING HIGHLIGHTS ===")
    
    validated = []
    
    for video in results:
        title = video.get("title", "")
        title_lower = title.lower()
        publisher = video.get("publisher", "").lower()
        description = video.get("description", "")
        
        # Skip obvious simulations
        if _is_simulation(title, description, publisher):
            continue
        
        # Skip talk shows, analysis, reactions (not actual highlights)
        # This is faster and more reliable than LLM validation
        talk_show_keywords = [
            "takeaway", "takeaways", "reaction", "reactions", "preview", 
            "prediction", "predictions", "press conference", "interview",
            "analysis", "the 2 robbies", "2 robbies", "thoughts on",
            "what we learned", "talking points", "debate", "discussion",
            "pro soccer talk", "matchweek preview", "fan cam", "fan reaction",
            "watch along", "watchalong"
        ]
        if any(kw in title_lower for kw in talk_show_keywords):
            print(f"[YouTubeSearch]   ⚠️ SKIP [Talk show]: {title[:40]}...")
            continue
        
        # Must have "highlights" or "goals" in title to be actual highlights
        if not any(word in title_lower for word in ["highlight", "highlights", "goals", "all goals"]):
            print(f"[YouTubeSearch]   ⚠️ SKIP [No 'highlights' in title]: {title[:40]}...")
            continue
        
        # Use LLM to validate if this is actual match highlights (only if passed keyword checks)
        is_valid, reason = _llm_validate_highlight_video(title, description, home_team, away_team)
        if not is_valid:
            print(f"[YouTubeSearch]   ⚠️ SKIP [{reason}]: {title[:40]}...")
            continue
        
        # Skip very short videos
        duration = video.get("duration", "")
        duration_seconds = _parse_duration_to_seconds(duration)
        if duration_seconds and duration_seconds < 300:
            if "highlight" not in title_lower and "goals" not in title_lower:
                continue
        
        # Check if from trusted channel
        is_trusted = False
        source_type = "Other"
        
        # Check trusted highlight channels
        for channel in TRUSTED_HIGHLIGHT_CHANNELS:
            if channel in publisher or channel in title_lower:
                is_trusted = True
                source_type = channel.title()
                break
        
        # Check official club channels
        if not is_trusted:
            if home_team and home_team.lower() in publisher:
                is_trusted = True
                source_type = f"{home_team} Official"
            elif away_team and away_team.lower() in publisher:
                is_trusted = True
                source_type = f"{away_team} Official"
        
        if not is_trusted:
            print(f"[YouTubeSearch]   ⚠️ SKIP [Untrusted]: {title[:40]}...")
            continue
        
        # Validate date if we have match_dt
        date_valid = True
        if match_dt:
            # Try to extract date from title (NBC Sports format: MM/DD/YYYY)
            import re
            date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', title)
            if date_match:
                try:
                    vid_month = int(date_match.group(1))
                    vid_day = int(date_match.group(2))
                    vid_year = int(date_match.group(3))
                    video_dt = datetime(vid_year, vid_month, vid_day)
                    
                    # Check if within ±5 days
                    day_diff = abs((video_dt - match_dt).days)
                    if day_diff > 5:
                        print(f"[YouTubeSearch]   ⚠️ SKIP [Wrong date: {day_diff} days off]: {title[:35]}...")
                        date_valid = False
                    else:
                        video["date_match"] = True
                        print(f"[YouTubeSearch]   ✓ Date valid ({day_diff} days): {title[:40]}...")
                except ValueError:
                    pass
        
        if not date_valid:
            continue
        
        video["source_type"] = source_type
        video["is_trusted"] = True
        validated.append(video)
        
        print(f"[YouTubeSearch]   ✅ [{source_type}]: {title[:45]}...")
    
    # Sort: prefer videos with date match, then by duration (longer = better)
    def sort_key(v):
        has_date = 1 if v.get("date_match") else 0
        dur = _parse_duration_to_seconds(v.get("duration", "")) or 0
        return (-has_date, -dur)
    
    validated.sort(key=sort_key)
    
    print(f"\n[YouTubeSearch] Validated: {len(validated)} highlights")
    
    return validated[:MAX_RESULTS]


# =============================================================================
# Original Entry Point (kept for compatibility)
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

