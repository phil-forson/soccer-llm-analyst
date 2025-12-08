"""
REST API Backend for Soccer LLM Analyst.

Provides HTTP endpoints for querying sports information.
Can be integrated with any frontend (React, Vue, Angular, etc.).
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import uvicorn
import json
import sys
import io
import logging
from contextlib import redirect_stdout
from datetime import datetime
from urllib.parse import unquote

from .query_parser_agent import parse_query, should_fetch_highlights, QueryIntent
from .web_search_agent import search_with_rag
from .youtube_search_agent import search_and_display_highlights_with_metadata
from .game_analyst_agent import analyze_match_from_web_results, format_analysis
from .config import get_openai_key, DEFAULT_LLM_MODEL
from openai import OpenAI


# =============================================================================
# Configuration
# =============================================================================

# Default API URL (for game analyst to call back to this API)
DEFAULT_API_URL = "http://localhost:8000"

# LLM Client for thinking messages
_thinking_llm_client: Optional[OpenAI] = None

def _get_thinking_llm_client() -> OpenAI:
    """Get or initialize the OpenAI client for thinking messages."""
    global _thinking_llm_client
    if _thinking_llm_client is None:
        _thinking_llm_client = OpenAI(api_key=get_openai_key())
    return _thinking_llm_client


# =============================================================================
# Logging Setup
# =============================================================================

# Configure logging for thinking messages
thinking_logger = logging.getLogger("thinking")
thinking_logger.setLevel(logging.INFO)

# Create file handler for thinking logs
thinking_handler = logging.FileHandler('thinking.log', mode='a', encoding='utf-8')
thinking_handler.setLevel(logging.INFO)

# Create formatter
thinking_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
thinking_handler.setFormatter(thinking_formatter)

# Add handler to logger
if not thinking_logger.handlers:
    thinking_logger.addHandler(thinking_handler)

def _log_thinking(stage: str, message: str, status: str, data: Optional[Dict] = None):
    """Log thinking messages to file."""
    log_msg = f"[{stage.upper()}] [{status.upper()}] {message}"
    if data:
        log_msg += f" | Data: {json.dumps(data, default=str)}"
    thinking_logger.info(log_msg)


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Soccer LLM Analyst API",
    description="Intelligent RAG-based football/soccer information API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    include_highlights: Optional[bool] = None  # Auto-detect if not provided


class KeyMoment(BaseModel):
    """Model for a key match moment."""
    minute: Optional[str] = None
    event: str
    description: str
    team: Optional[str] = None


class MatchMetadata(BaseModel):
    """Model for match metadata."""
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    match_date: Optional[str] = None
    score: Optional[str] = None
    competition: Optional[str] = None
    key_moments: List[KeyMoment] = []
    man_of_the_match: Optional[str] = None
    match_summary: Optional[str] = None


class HighlightVideo(BaseModel):
    """Model for a highlight video."""
    title: str
    url: str
    duration: Optional[str] = None
    source_type: Optional[str] = None
    is_nbc_sports: bool = False
    is_official_club: bool = False
    confidence: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    intent: str
    summary: str
    match_metadata: Optional[MatchMetadata] = None
    highlights: Optional[List[HighlightVideo]] = []
    sources: List[str] = []
    # Optional game analysis (included for match_result queries)
    game_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ThinkingMessage(BaseModel):
    """Model for thinking/status messages."""
    stage: str  # "query_parser", "web_search", "game_analyst", "highlights"
    message: str
    status: str  # "starting", "processing", "complete", "error"
    data: Optional[Dict[str, Any]] = None


class MomentumShift(BaseModel):
    """Model for a momentum shift event."""
    minute: str
    event: str
    description: str
    team: Optional[str] = None
    momentum_impact: str  # "neutral", "high", "critical"
    reasoning: str


class GameAnalysisResponse(BaseModel):
    """Response model for game analysis endpoint."""
    success: bool
    match_info: Optional[Dict[str, Any]] = None
    deep_analysis: Optional[str] = None
    momentum_analysis: List[MomentumShift] = []
    tactical_analysis: Optional[Dict[str, Any]] = None
    key_moments: List[KeyMoment] = []
    highlights: List[HighlightVideo] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_highlights_output(highlights_text: str) -> List[HighlightVideo]:
    """
    Parse the formatted highlights text into structured video objects.
    
    Args:
        highlights_text: Formatted string from format_highlight_results()
        
    Returns:
        List of HighlightVideo objects.
    """
    videos = []
    
    if not highlights_text or "No highlights found" in highlights_text or "Could not find" in highlights_text:
        return videos
    
    lines = highlights_text.split("\n")
    current_video = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            # Empty line - save current video if we have one
            if current_video.get("title") and current_video.get("url"):
                try:
                    videos.append(HighlightVideo(**current_video))
                except Exception:
                    pass  # Skip invalid videos
                current_video = {}
            continue
        
        # Check if it's a numbered video entry (e.g., "1. Title" or "🥇 1. Title")
        import re
        number_match = re.match(r'^[🥇🥈📄\s]*(\d+)\.\s+(.+)', line)
        if number_match:
            # Save previous video if exists
            if current_video.get("title") and current_video.get("url"):
                try:
                    videos.append(HighlightVideo(**current_video))
                except Exception:
                    pass
            # Start new video
            title = number_match.group(2)
            current_video = {
                "title": title,
                "url": "",
                "duration": None,
                "source_type": None,
                "is_nbc_sports": False,
                "is_official_club": False,
                "confidence": None,
            }
        
        # Parse video details
        elif current_video:
            if "🔗" in line or "http" in line:
                # Extract URL
                url_match = re.search(r'https?://[^\s\)]+', line)
                if url_match:
                    current_video["url"] = url_match.group(0)
            elif "⏱️" in line or "Duration:" in line:
                # Extract duration
                dur_match = re.search(r'(\d+:\d+)', line)
                if dur_match:
                    current_video["duration"] = dur_match.group(1)
            elif "📺 NBC Sports" in line or "NBC Sports" in line:
                current_video["is_nbc_sports"] = True
                current_video["source_type"] = "NBC Sports"
            elif "🏟️ Official" in line or "Official Club" in line:
                current_video["is_official_club"] = True
                current_video["source_type"] = "Official Club"
            elif "✓ Verified" in line or "○ Likely match" in line:
                # Extract confidence if available
                if "Verified" in line:
                    current_video["confidence"] = 0.9
                elif "Likely" in line:
                    current_video["confidence"] = 0.7
    
    # Add last video
    if current_video.get("title") and current_video.get("url"):
        try:
            videos.append(HighlightVideo(**current_video))
        except Exception:
            pass
    
    return videos


def _normalise_highlight_results(raw: Any) -> List[HighlightVideo]:
    """
    Normalise whatever the highlights agent returns into a list[HighlightVideo].

    - If `raw` is a string: treat it as the old formatted text and parse it.
    - If `raw` is a list[dict]: treat each dict as YouTube metadata from youtube_search_agent.
    - If `raw` is already a list[HighlightVideo], just return it.
    """
    # Case 1: old behaviour – formatted text
    if isinstance(raw, str):
        return _parse_highlights_output(raw)

    videos: List[HighlightVideo] = []

    # Case 2: already our Pydantic models
    if isinstance(raw, list) and raw and isinstance(raw[0], HighlightVideo):
        return raw

    # Case 3: list of dicts from youtube_search_agent
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, HighlightVideo):
                videos.append(item)
                continue
            if not isinstance(item, dict):
                continue

            video_id = item.get("video_id") or item.get("videoId")
            url = item.get("url")
            if not url and video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"

            title = item.get("title") or ""
            channel_title = (
                item.get("channel_title")
                or item.get("channelTitle")
                or ""
            )
            source_type = item.get("source_type") or None

            # Simple heuristics
            ct_lower = channel_title.lower()
            is_nbc = "nbc sports" in ct_lower or "nbcsports" in ct_lower
            is_official = "official" in ct_lower or " fc" in ct_lower or " cf" in ct_lower

            try:
                videos.append(
                    HighlightVideo(
                        title=title,
                        url=url or "",
                        duration=None,          # Wire later if you add duration
                        source_type=source_type,
                        is_nbc_sports=is_nbc,
                        is_official_club=is_official,
                        confidence=None,        # Wire ranking scores later if needed
                    )
                )
            except Exception:
                # Defensive: skip broken entries
                continue

        return videos

    # Fallback
    return []


def _extract_sources_from_summary(summary: str) -> List[str]:
    """
    Extract source URLs from the summary text.
    
    Args:
        summary: Summary text that may contain source URLs
        
    Returns:
        List of source URLs.
    """
    sources = []
    lines = summary.split("\n")
    
    for line in lines:
        if "📚 Sources" in line or "•" in line or "http" in line:
            # Extract URLs
            import re
            urls = re.findall(r'https?://[^\s\)]+', line)
            sources.extend(urls)
    
    return list(set(sources))  # Remove duplicates


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


async def _generate_thinking_message(
    stage: str,
    action: str,
    context: Dict[str, Any],
    status: str = "processing"
) -> str:
    """
    Use LLM to generate detailed thinking messages explaining what's happening.
    
    Args:
        stage: Current stage (query_parser, web_search, game_analyst, highlights)
        action: What action is being performed
        context: Contextual information about what's happening
        status: Status of the stage
        
    Returns:
        Detailed thinking message
    """
    try:
        client = _get_thinking_llm_client()
        
        stage_descriptions = {
            "query_parser": "I'm analyzing the user's query to understand their intent and extract key information like team names, dates, and what they're looking for.",
            "web_search": "I'm searching the web using RAG (Retrieval Augmented Generation) to find relevant match information, then retrieving the most relevant chunks and generating a summary.",
            "game_analyst": "I'm analyzing the match data to identify momentum shifts, tactical patterns, and generate comprehensive insights about how the game unfolded.",
            "highlights": "I'm searching for highlight videos and validating them against the match context using RAG to ensure they match the actual game."
        }
        
        base_description = stage_descriptions.get(stage, "I'm processing the request.")
        
        # Make the prompt more flexible and natural
        prompt = f"""You are explaining what's happening in a RAG-based football information system.

Current Stage: {stage}
Action: {action}
Context: {json.dumps(context, indent=2)}
Status: {status}

Base Description: {base_description}

Generate a natural, conversational thinking message (2-3 sentences) that explains:
1. What task is being performed
2. Why it matters for answering the user's query
3. What information or insights we're seeking

Guidelines:
- Write in first person, as if thinking out loud
- Use natural, varied language (avoid repetitive phrases like "right now" multiple times)
- Be specific but conversational
- Vary your sentence structure
- Don't use markdown formatting
- Make it sound like a helpful assistant explaining their process"""

        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains what a RAG system is doing in natural, conversational language. Be specific and informative."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=150,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback to simple message if LLM fails
        return f"{action}..."


def _send_thinking(stage: str, message: str, status: str = "processing", data: Optional[Dict] = None) -> str:
    """Format a thinking message for SSE and log it."""
    # Log to file
    _log_thinking(stage, message, status, data)
    
    # Also print to console for debugging
    print(f"[THINKING] [{stage.upper()}] [{status.upper()}] {message}")
    if data:
        print(f"[THINKING] Data: {json.dumps(data, indent=2, default=str)}")
    
    msg = ThinkingMessage(
        stage=stage,
        message=message,
        status=status,
        data=data or {}
    )
    
    # Format as SSE message - ensure proper format
    msg_json = msg.model_dump_json()
    sse_message = f"data: {msg_json}\n\n"
    print(f"[THINKING] Sending SSE: {sse_message[:150]}...")  # Debug output
    return sse_message


async def _process_query_with_thinking(request: QueryRequest):
    """
    Process query and yield thinking messages as they happen.
    
    Yields SSE-formatted thinking messages, then final result.
    """
    try:
        # Step 1: Query Parser
        # Send immediate message first (non-blocking)
        yield _send_thinking("query_parser", "Analyzing your query to understand what you're looking for...", "starting", {"query": request.query})
        
        # Optionally enhance with LLM (but don't block on it)
        try:
            thinking_msg = await _generate_thinking_message(
                "query_parser",
                "Analyzing query intent",
                {"query": request.query, "stage": "starting"},
                "starting"
            )
            yield _send_thinking("query_parser", thinking_msg, "starting", {"query": request.query})
        except:
            pass  # If LLM fails, we already sent a message
        
        parsed = parse_query(request.query)
        
        # Check if query is relevant to football/soccer
        is_relevant = parsed.get("is_relevant", True)
        validation_error = parsed.get("validation_error")
        
        if not is_relevant and validation_error:
            # Query is not about football - return early to save credits
            error_reason = validation_error.get("reason", "This query is not about football/soccer.")
            suggestion = validation_error.get("suggestion", "Try asking about a match, like 'Barcelona vs Atletico Madrid'")
            
            yield _send_thinking(
                "query_parser",
                f"Query validation failed: {error_reason}",
                "error",
                {"reason": error_reason, "suggestion": suggestion}
            )
            
            # Return error response immediately
            error_response = {
                "success": False,
                "intent": "general",
                "summary": f"❌ {error_reason}\n\n💡 Suggestion: {suggestion}",
                "error": "query_not_relevant",
                "validation_error": {
                    "reason": error_reason,
                    "suggestion": suggestion
                },
                "match_metadata": None,
                "highlights": [],
                "sources": [],
                "game_analysis": None
            }
            yield f"data: {json.dumps({'type': 'result', 'data': error_response})}\n\n"
            return
        
        intent = parsed.get("intent", "general")
        search_query = parsed.get("search_query", request.query)
        teams = parsed.get("teams", [])
        
        # Send immediate completion message
        yield _send_thinking(
            "query_parser", 
            f"Query parsed successfully. Intent: {intent}. Found {len(teams)} team(s).",
            "complete",
            {"intent": intent, "search_query": search_query, "teams": teams}
        )
        
        # Optionally enhance with LLM
        try:
            thinking_msg = await _generate_thinking_message(
                "query_parser",
                "Query parsing complete",
                {
                    "intent": intent,
                    "search_query": search_query,
                    "teams": teams,
                    "extracted_info": {
                        "home_team": parsed.get("home_team"),
                        "away_team": parsed.get("away_team"),
                        "date": parsed.get("date")
                    }
                },
                "complete"
            )
            yield _send_thinking("query_parser", thinking_msg, "complete", {"intent": intent, "search_query": search_query, "teams": teams})
        except:
            pass
        
        # Determine if highlights should be fetched
        show_highlights = request.include_highlights
        if show_highlights is None:
            show_highlights = should_fetch_highlights(parsed)
        
        # Step 2: Web Search (RAG)
        # Send immediate messages first (non-blocking)
        yield _send_thinking("web_search", "Searching the web for match information using RAG...", "starting", {"search_query": search_query})
        
        match_metadata = {}
        web_summary = ""
        
        try:
            yield _send_thinking("web_search", "Executing search queries and retrieving relevant information...", "processing", {"action": "searching"})
            
            result, match_metadata = search_with_rag(
                query=search_query,
                intent=intent,
                original_query=request.query,
                parsed_query=parsed
            )
            web_summary = result
            
            # Send immediate completion message
            score_info = match_metadata.get('score', 'Not found')
            teams_info = f"{match_metadata.get('home_team', 'Unknown')} vs {match_metadata.get('away_team', 'Unknown')}"
            yield _send_thinking(
                "web_search",
                f"Web search complete. Found match: {teams_info}. Score: {score_info if score_info else 'Not found in search results'}.",
                "complete",
                {
                    "has_match": bool(match_metadata.get('score')),
                    "home_team": match_metadata.get('home_team'),
                    "away_team": match_metadata.get('away_team'),
                    "score": match_metadata.get('score')
                }
            )
            
            # Optionally enhance with LLM
            try:
                thinking_msg = await _generate_thinking_message(
                    "web_search",
                    "Web search complete, generating summary with RAG",
                    {
                        "home_team": match_metadata.get('home_team'),
                        "away_team": match_metadata.get('away_team'),
                        "score": match_metadata.get('score'),
                        "key_moments_count": len(match_metadata.get('key_moments', [])),
                        "has_match": bool(match_metadata.get('score'))
                    },
                    "complete"
                )
                yield _send_thinking("web_search", thinking_msg, "complete", {"has_match": bool(match_metadata.get('score'))})
            except:
                pass
        except Exception as e:
            yield _send_thinking("web_search", f"Web search error: {str(e)}", "error")
            yield f"data: {json.dumps({'type': 'result', 'data': {'success': False, 'intent': intent, 'summary': f'Error searching the web: {str(e)}', 'error': str(e)}})}\n\n"
            return
        
        # Extract sources from summary
        sources = _extract_sources_from_summary(web_summary)
        
        # Convert match_metadata to MatchMetadata model
        match_meta = None
        game_analysis_data = None
        
        if match_metadata:
            key_moments = [
                KeyMoment(**moment) for moment in match_metadata.get("key_moments", [])
            ]
            match_meta = MatchMetadata(
                home_team=match_metadata.get("home_team"),
                away_team=match_metadata.get("away_team"),
                match_date=match_metadata.get("match_date"),
                score=match_metadata.get("score"),
                competition=match_metadata.get("competition"),
                key_moments=key_moments,
                man_of_the_match=match_metadata.get("man_of_the_match"),
                match_summary=match_metadata.get("match_summary"),
            )
            
            # Step 3: Game Analyst (CHAINED)
            if intent in ["match_result", "match_highlights"] and match_metadata.get("score"):
                try:
                    # Send immediate messages first
                    yield _send_thinking("game_analyst", "Starting comprehensive match analysis...", "starting", {"action": "analysis_start"})
                    yield _send_thinking("game_analyst", "Analyzing momentum shifts and tactical patterns...", "processing", {"action": "momentum_analysis"})
                    
                    analysis = analyze_match_from_web_results(
                        web_summary=web_summary,
                        match_metadata=match_metadata,
                        original_query=request.query
                    )
                    
                    if analysis.get('success'):
                        yield _send_thinking("game_analyst", "Generating deep tactical and strategic insights...", "processing", {"action": "generating_insights"})
                        
                        momentum_shifts = [
                            {
                                "minute": m.get("minute"),
                                "event": m.get("event"),
                                "description": m.get("description"),
                                "team": m.get("team"),
                                "momentum_impact": m.get("momentum_impact"),
                                "reasoning": m.get("reasoning")
                            }
                            for m in analysis.get('momentum_analysis', [])
                        ]
                        
                        game_analysis_data = {
                            "deep_analysis": analysis.get('deep_analysis'),
                            "momentum_analysis": momentum_shifts,
                            "tactical_analysis": analysis.get('tactical_analysis'),
                        }
                        
                        # Send immediate completion
                        significant_shifts = len([m for m in momentum_shifts if m.get('momentum_impact') != 'neutral'])
                        yield _send_thinking(
                            "game_analyst",
                            f"Match analysis complete. Identified {len(momentum_shifts)} momentum shifts ({significant_shifts} significant).",
                            "complete",
                            {"momentum_shifts": len(momentum_shifts), "significant_shifts": significant_shifts}
                        )
                        
                        # Optionally enhance with LLM
                        try:
                            thinking_msg = await _generate_thinking_message(
                                "game_analyst",
                                "Match analysis complete",
                                {
                                    "momentum_shifts": len(momentum_shifts),
                                    "significant_shifts": significant_shifts,
                                    "tactical_insights": bool(analysis.get('tactical_analysis'))
                                },
                                "complete"
                            )
                            yield _send_thinking("game_analyst", thinking_msg, "complete", {"momentum_shifts": len(momentum_shifts)})
                        except:
                            pass
                except Exception as e:
                    yield _send_thinking("game_analyst", f"Game analysis error: {str(e)}", "error")
                    game_analysis_data = None
        
        # Step 4: Highlights (LAST in chain)
        highlights: List[HighlightVideo] = []
        if show_highlights:
            try:
                home_team = match_metadata.get("home_team") or (parsed.get("teams", [None])[0])
                away_team = match_metadata.get("away_team") or (parsed.get("teams", [None, None])[1] if len(parsed.get("teams", [])) > 1 else None)
                match_date = match_metadata.get("match_date")
                
                if home_team:
                    # Send immediate messages first
                    yield _send_thinking("highlights", f"Searching for highlight videos: {home_team} vs {away_team}...", "starting", {"home_team": home_team, "away_team": away_team})
                    yield _send_thinking("highlights", "Validating videos using RAG against match context...", "processing", {"action": "rag_validation"})
                    
                    raw_highlights = search_and_display_highlights_with_metadata(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        web_summary=web_summary,
                        match_metadata=match_metadata
                    )
                    highlights = _normalise_highlight_results(raw_highlights)
                    
                    # Extract source types safely (handles both dicts and HighlightVideo objects)
                    sources_list = []
                    for h in highlights[:3]:
                        if isinstance(h, dict):
                            sources_list.append(h.get('source_type', 'Unknown'))
                        elif hasattr(h, 'source_type'):
                            sources_list.append(h.source_type)
                        else:
                            sources_list.append('Unknown')
                    
                    yield _send_thinking(
                        "highlights",
                        f"Found {len(highlights)} validated highlight videos",
                        "complete",
                        {"count": len(highlights), "sources": sources_list}
                    )
                    
                    # Optionally enhance with LLM (non-blocking)
                    try:
                        thinking_msg = await _generate_thinking_message(
                            "highlights",
                            "Highlights search and validation complete",
                            {
                                "videos_found": len(highlights),
                                "validated": True,
                                "sources": sources_list
                            },
                            "complete"
                        )
                        yield _send_thinking("highlights", thinking_msg, "complete", {"count": len(highlights)})
                    except:
                        pass
            except Exception as e:
                yield _send_thinking("highlights", f"Highlights error: {str(e)}", "error")
        
        # Final result - convert Pydantic models to dicts
        result_data = {
            "success": True,
            "intent": intent,
            "summary": web_summary,
            "match_metadata": match_meta.model_dump() if match_meta else None,
            "highlights": [h.model_dump() if hasattr(h, 'model_dump') else h for h in highlights],
            "sources": sources,
            "game_analysis": game_analysis_data,
        }
        result_json = json.dumps({'type': 'result', 'data': result_data})
        yield f"data: {result_json}\n\n"
        
    except Exception as e:
        yield _send_thinking("error", f"An error occurred: {str(e)}", "error")
        yield f"data: {json.dumps({'type': 'result', 'data': {'success': False, 'intent': 'general', 'summary': '', 'error': f'An error occurred: {str(e)}'}})}\n\n"


@app.get("/query/stream")
async def query_stream_get_endpoint(
    query: str = Query(..., description="The query to process"),
    include_highlights: Optional[bool] = Query(None, description="Whether to include highlights")
):
    """
    GET endpoint for SSE streaming - compatible with native EventSource API.
    
    Uses Server-Sent Events (SSE) to stream thinking messages showing:
    - Query parsing progress
    - Web search (RAG) progress
    - Game analyst progress
    - Highlights search progress
    
    Final result is sent as a JSON object at the end.
    
    Args:
        query: The user's question (URL encoded)
        include_highlights: Whether to include highlights
        
    Returns:
        StreamingResponse with SSE-formatted thinking messages
    """
    # Decode query if needed
    query = unquote(query)
    
    # Create request object
    request = QueryRequest(query=query, include_highlights=include_highlights)
    
    import asyncio
    
    async def generate():
        try:
            async for message in _process_query_with_thinking(request):
                # Ensure each message is sent immediately
                if message:
                    # Ensure proper SSE format
                    if not message.startswith("data: "):
                        # If it's already formatted, use as-is
                        if message.startswith("data:"):
                            yield message
                        else:
                            yield f"data: {message}\n\n"
                    else:
                        yield message
                    # Force immediate flush by yielding control to event loop
                    await asyncio.sleep(0.001)  # Small delay to allow flush
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[SSE ERROR] {error_trace}")
            # Send error as thinking message
            error_msg = ThinkingMessage(
                stage="error",
                message=f"An error occurred: {str(e)}",
                status="error",
                data={"error": str(e)}
            )
            yield f"data: {error_msg.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )


@app.post("/query/stream")
async def query_stream_post_endpoint(request: QueryRequest):
    """
    POST endpoint for SSE streaming - for clients that prefer POST.
    
    Uses Server-Sent Events (SSE) to stream thinking messages showing:
    - Query parsing progress
    - Web search (RAG) progress
    - Game analyst progress
    - Highlights search progress
    
    Final result is sent as a JSON object at the end.
    
    Args:
        request: QueryRequest with the user's question
        
    Returns:
        StreamingResponse with SSE-formatted thinking messages
    """
    import asyncio
    
    async def generate():
        try:
            async for message in _process_query_with_thinking(request):
                # Ensure each message is sent immediately
                if message:
                    # Ensure proper SSE format
                    if not message.startswith("data: "):
                        # If it's already formatted, use as-is
                        if message.startswith("data:"):
                            yield message
                        else:
                            yield f"data: {message}\n\n"
                    else:
                        yield message
                    # Force immediate flush by yielding control to event loop
                    await asyncio.sleep(0.001)  # Small delay to allow flush
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[SSE ERROR] {error_trace}")
            # Send error as thinking message
            error_msg = ThinkingMessage(
                stage="error",
                message=f"An error occurred: {str(e)}",
                status="error",
                data={"error": str(e)}
            )
            yield f"data: {error_msg.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint - processes natural language football queries.
    
    This is the primary endpoint that:
    1. Parses the query to understand intent
    2. Searches the web using RAG
    3. Optionally searches for match highlights
    4. Returns structured response
    
    Args:
        request: QueryRequest with the user's question
        
    Returns:
        QueryResponse with summary, metadata, and highlights
    """
    try:
        # Step 1: Parse the query
        parsed = parse_query(request.query)
        
        # Check if query is relevant to football/soccer
        is_relevant = parsed.get("is_relevant", True)
        validation_error = parsed.get("validation_error")
        
        if not is_relevant and validation_error:
            # Query is not about football - return early to save credits
            error_reason = validation_error.get("reason", "This query is not about football/soccer.")
            suggestion = validation_error.get("suggestion", "Try asking about a match, like 'Barcelona vs Atletico Madrid'")
            
            return QueryResponse(
                success=False,
                intent="general",
                summary=f"❌ {error_reason}\n\n💡 Suggestion: {suggestion}",
                error="query_not_relevant",
                match_metadata=None,
                highlights=[],
                sources=[],
                game_analysis=None
            )
        
        intent = parsed.get("intent", "general")
        search_query = parsed.get("search_query", request.query)
        summary_focus = parsed.get("summary_focus", "key information")
        
        # Determine if highlights should be fetched
        show_highlights = request.include_highlights
        if show_highlights is None:
            show_highlights = should_fetch_highlights(parsed)
        
        # Step 2: Search the web using RAG
        match_metadata = {}
        web_summary = ""
        
        try:
            result, match_metadata = search_with_rag(
                query=search_query,
                intent=intent,
                original_query=request.query,
                parsed_query=parsed
            )
            web_summary = result
        except Exception as e:
            return QueryResponse(
                success=False,
                intent=intent,
                summary=f"Error searching the web: {str(e)}",
                error=str(e)
            )
        
        # Extract sources from summary
        sources = _extract_sources_from_summary(web_summary)
        
        # Convert match_metadata to MatchMetadata model
        match_meta = None
        game_analysis_data = None
        
        if match_metadata:
            key_moments = [
                KeyMoment(**moment) for moment in match_metadata.get("key_moments", [])
            ]
            
            match_meta = MatchMetadata(
                home_team=match_metadata.get("home_team"),
                away_team=match_metadata.get("away_team"),
                match_date=match_metadata.get("match_date"),
                score=match_metadata.get("score"),
                competition=match_metadata.get("competition"),
                key_moments=key_moments,
                man_of_the_match=match_metadata.get("man_of_the_match"),
                match_summary=match_metadata.get("match_summary"),
            )
            
            # Step 3: Game Analyst (CHAINED)
            # Chain: Web Search → Game Analyst (passes results directly)
            if intent in ["match_result", "match_highlights"] and match_metadata.get("score"):
                try:
                    print(f"\n[API] Step 3: Chaining to Game Analyst Agent...")
                    print(f"[API] Passing web search results to game analyst...")
                    
                    # Pass web search results directly to game analyst (CHAIN)
                    analysis = analyze_match_from_web_results(
                        web_summary=web_summary,
                        match_metadata=match_metadata,
                        original_query=request.query
                    )
                    
                    if analysis.get('success'):
                        # Convert momentum shifts
                        momentum_shifts = [
                            {
                                "minute": m.get("minute"),
                                "event": m.get("event"),
                                "description": m.get("description"),
                                "team": m.get("team"),
                                "momentum_impact": m.get("momentum_impact"),
                                "reasoning": m.get("reasoning")
                            }
                            for m in analysis.get('momentum_analysis', [])
                        ]
                        
                        game_analysis_data = {
                            "deep_analysis": analysis.get('deep_analysis'),
                            "momentum_analysis": momentum_shifts,
                            "tactical_analysis": analysis.get('tactical_analysis'),
                        }
                        print(f"[API] ✓ Game analysis complete (chained from web search)")
                except Exception as e:
                    print(f"[API] ⚠️ Game analysis failed: {e}")
                    # Don't fail the whole request if analysis fails
                    game_analysis_data = None
        
        # Step 4: Highlights (LAST in chain - uses all previous results)
        # Chain: Query Parser → Web Search → Game Analyst → Highlights
        highlights: List[HighlightVideo] = []
        if show_highlights:
            try:
                print(f"\n[API] Step 4: Chaining to Highlights Agent (last step)...")
                home_team = match_metadata.get("home_team") or (parsed.get("teams", [None])[0])
                away_team = match_metadata.get("away_team") or (parsed.get("teams", [None, None])[1] if len(parsed.get("teams", [])) > 1 else None)
                match_date = match_metadata.get("match_date")
                
                if home_team:
                    # Pass web_summary and match_metadata for RAG validation
                    raw_highlights = search_and_display_highlights_with_metadata(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        web_summary=web_summary,  # From web search agent
                        match_metadata=match_metadata  # From web search agent
                    )
                    highlights = _normalise_highlight_results(raw_highlights)
                    print(f"[API] ✓ Highlights complete (chained from web search + game analysis)")
            except Exception as e:
                # Don't fail the whole request if highlights fail
                print(f"[API] ⚠️ Highlights failed: {e}")
        
        return QueryResponse(
            success=True,
            intent=intent,
            # Prefer comprehensive game analysis summary when available, otherwise fall back to web summary
            summary=game_analysis_data.get("deep_analysis") if game_analysis_data and game_analysis_data.get("deep_analysis") else web_summary,
            match_metadata=match_meta,
            highlights=highlights,
            sources=sources,
            game_analysis=game_analysis_data,
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            intent="general",
            summary="",
            error=f"An error occurred: {str(e)}"
        )


@app.get("/intents")
async def list_intents():
    """List all available query intents."""
    return {
        "intents": [
            {"value": QueryIntent.MATCH_RESULT, "description": "Get match score/result"},
            {"value": QueryIntent.MATCH_HIGHLIGHTS, "description": "Watch match highlights"},
            {"value": QueryIntent.LINEUP, "description": "Get starting XI/lineup"},
            {"value": QueryIntent.PLAYER_INFO, "description": "Get player information"},
            {"value": QueryIntent.TRANSFER_NEWS, "description": "Get transfer news"},
            {"value": QueryIntent.TEAM_NEWS, "description": "Get team news"},
            {"value": QueryIntent.STANDINGS, "description": "Get league standings"},
            {"value": QueryIntent.FIXTURES, "description": "Get upcoming fixtures"},
            {"value": QueryIntent.STATS, "description": "Get statistics"},
            {"value": QueryIntent.GENERAL, "description": "General sports query"},
        ]
    }


@app.post("/analyze", response_model=GameAnalysisResponse)
async def analyze_match_endpoint(request: QueryRequest):
    """
    Comprehensive game analysis endpoint.
    
    Provides sophisticated match analysis including:
    - Momentum shift analysis
    - Tactical breakdown
    - Deep strategic insights
    - Match implications
    
    Args:
        request: QueryRequest with the match query
        
    Returns:
        GameAnalysisResponse with comprehensive analysis
    """
    try:
        # Step 1: Get web search results first (CHAIN)
        print(f"[API] Step 1: Getting web search results...")
        parsed = parse_query(request.query)
        
        # Check if query is relevant to football/soccer
        is_relevant = parsed.get("is_relevant", True)
        validation_error = parsed.get("validation_error")
        
        if not is_relevant and validation_error:
            # Query is not about football - return early to save credits
            error_reason = validation_error.get("reason", "This query is not about football/soccer.")
            suggestion = validation_error.get("suggestion", "Try asking about a match, like 'Barcelona vs Atletico Madrid'")
            
            return GameAnalysisResponse(
                success=False,
                error=f"{error_reason} {suggestion}"
            )
        
        search_query = parsed.get("search_query", request.query)
        intent = parsed.get("intent", "general")
        
        try:
            web_summary, match_metadata = search_with_rag(
                query=search_query,
                intent=intent,
                original_query=request.query,
                parsed_query=parsed
            )
        except Exception as e:
            return GameAnalysisResponse(
                success=False,
                error=f"Web search failed: {str(e)}"
            )
        
        if not match_metadata or not match_metadata.get('score'):
            return GameAnalysisResponse(
                success=False,
                error="Could not extract match information. The query may not be about a specific match."
            )
        
        # Step 2: Pass web search results to game analyst (CHAIN)
        print(f"[API] Step 2: Passing results to Game Analyst Agent...")
        analysis = analyze_match_from_web_results(
            web_summary=web_summary,
            match_metadata=match_metadata,
            original_query=request.query
        )
        
        # Step 3: Highlights (LAST in chain - uses all previous results)
        # Chain: Query Parser → Web Search → Game Analyst → Highlights
        highlights: List[HighlightVideo] = []
        if request.include_highlights if request.include_highlights is not None else True:
            try:
                print(f"\n[API] Step 3: Chaining to Highlights Agent (last step)...")
                home_team = match_metadata.get("home_team")
                away_team = match_metadata.get("away_team")
                match_date = match_metadata.get("match_date")
                
                if home_team:
                    # Pass web_summary and match_metadata for RAG validation
                    raw_highlights = search_and_display_highlights_with_metadata(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        web_summary=web_summary,  # From web search agent
                        match_metadata=match_metadata  # From web search agent
                    )
                    highlights = _normalise_highlight_results(raw_highlights)
                    print(f"[API] ✓ Highlights complete (chained from web search + game analysis)")
            except Exception as e:
                print(f"[API] ⚠️ Highlights failed: {e}")
                highlights = []
        
        if not analysis.get('success'):
            return GameAnalysisResponse(
                success=False,
                error=analysis.get('error', 'Unknown error')
            )
        
        # Convert momentum shifts to models
        momentum_shifts = [
            MomentumShift(**shift) for shift in analysis.get('momentum_analysis', [])
        ]
        
        # Convert key moments to models
        key_moments = [
            KeyMoment(**moment) for moment in analysis.get('key_moments', [])
        ]
        
        return GameAnalysisResponse(
            success=True,
            match_info=analysis.get('match_info'),
            deep_analysis=analysis.get('deep_analysis'),
            momentum_analysis=momentum_shifts,
            tactical_analysis=analysis.get('tactical_analysis'),
            key_moments=key_moments,
            highlights=highlights,
        )
        
    except Exception as e:
        return GameAnalysisResponse(
            success=False,
            error=f"An error occurred: {str(e)}"
        )


# =============================================================================
# Run Server
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    uvicorn.run(
        app,  # Use app object directly
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    # Run server when executed directly
    run_server(host="0.0.0.0", port=8000, reload=True)