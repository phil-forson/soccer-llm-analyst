"""
REST API Backend for Soccer LLM Analyst.

Provides HTTP endpoints for querying sports information.
- /query - Answer user questions (web search + summary)
- /analyze - Deep game analysis (only on request)
"""

import asyncio
import json
import logging
import re
from typing import Optional, List, Dict, Any, AsyncGenerator
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    QueryRequest, QueryResponse, GameAnalysisResponse, HealthResponse,
    ThinkingMessage, KeyMoment, MatchMetadata, HighlightVideo, MomentumShift
)
from .utils import get_openai_client, is_quota_error, extract_urls_from_text
from .config import DEFAULT_LLM_MODEL
from .query_parser_agent import parse_query, should_fetch_highlights, QueryIntent
from .web_search_agent import search_with_rag
from .youtube_search_agent import search_and_display_highlights_with_metadata
from .game_analyst_agent import analyze_match_from_web_results


# =============================================================================
# Configuration
# =============================================================================

API_VERSION = "1.0.0"


# =============================================================================
# Logging Setup
# =============================================================================

thinking_logger = logging.getLogger("thinking")
thinking_logger.setLevel(logging.INFO)

if not thinking_logger.handlers:
    _handler = logging.FileHandler('thinking.log', mode='a', encoding='utf-8')
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    _handler.setFormatter(_formatter)
    thinking_logger.addHandler(_handler)


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Soccer LLM Analyst API",
    description="Intelligent football/soccer information API with semantic search",
    version=API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def _log_thinking(stage: str, message: str, status: str, data: Optional[Dict] = None):
    """Log thinking messages to file."""
    log_msg = f"[{stage.upper()}] [{status.upper()}] {message}"
    if data:
        log_msg += f" | Data: {json.dumps(data, default=str)}"
    thinking_logger.info(log_msg)


def _format_sse_message(stage: str, message: str, status: str, data: Optional[Dict] = None) -> str:
    """Format a thinking message for SSE and log it."""
    _log_thinking(stage, message, status, data)
    msg = ThinkingMessage(stage=stage, message=message, status=status, data=data or {})
    return f"data: {msg.model_dump_json()}\n\n"


def _normalise_highlight_results(raw: Any) -> List[HighlightVideo]:
    """Normalise highlight results into a list of HighlightVideo objects."""
    if isinstance(raw, str):
        return _parse_highlights_from_text(raw)
    
    if isinstance(raw, list) and raw and isinstance(raw[0], HighlightVideo):
        return raw
    
    videos: List[HighlightVideo] = []
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
            channel_title = item.get("channel_title") or item.get("channelTitle") or ""
            source_type = item.get("source_type")
            
            ct_lower = channel_title.lower()
            is_nbc = "nbc sports" in ct_lower or "nbcsports" in ct_lower
            is_official = "official" in ct_lower or " fc" in ct_lower
            
            try:
                videos.append(HighlightVideo(
                    title=title,
                    url=url or "",
                    duration=None,
                    source_type=source_type,
                    is_nbc_sports=is_nbc,
                    is_official_club=is_official,
                    confidence=None,
                ))
            except Exception:
                continue
    
    return videos


def _parse_highlights_from_text(highlights_text: str) -> List[HighlightVideo]:
    """Parse formatted highlights text into structured video objects."""
    videos = []
    if not highlights_text or "No highlights found" in highlights_text:
        return videos
    
    lines = highlights_text.split("\n")
    current_video = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_video.get("title") and current_video.get("url"):
                try:
                    videos.append(HighlightVideo(**current_video))
                except Exception:
                    pass
                current_video = {}
            continue
        
        number_match = re.match(r'^[🥇🥈📄\s]*(\d+)\.\s+(.+)', line)
        if number_match:
            if current_video.get("title") and current_video.get("url"):
                try:
                    videos.append(HighlightVideo(**current_video))
                except Exception:
                    pass
            current_video = {
                "title": number_match.group(2),
                "url": "",
                "duration": None,
                "source_type": None,
                "is_nbc_sports": False,
                "is_official_club": False,
                "confidence": None,
            }
        elif current_video:
            if "http" in line:
                url_match = re.search(r'https?://[^\s\)]+', line)
                if url_match:
                    current_video["url"] = url_match.group(0)
    
    if current_video.get("title") and current_video.get("url"):
        try:
            videos.append(HighlightVideo(**current_video))
        except Exception:
            pass
    
    return videos


def _check_match_found(match_metadata: dict, web_summary: str) -> bool:
    """Check if we actually found a match."""
    return not (
        not match_metadata
        or match_metadata.get("no_match_found")
        or (
            not match_metadata.get("score")
            and not match_metadata.get("match_date")
            and not match_metadata.get("key_moments")
        )
        or (web_summary.strip().startswith("❌"))
    )


def _build_match_metadata_model(match_metadata: dict) -> Optional[MatchMetadata]:
    """Build MatchMetadata model from dict."""
    if not match_metadata:
        return None
    
    key_moments = [
        KeyMoment(**moment) for moment in match_metadata.get("key_moments", [])
    ]
    
    return MatchMetadata(
        home_team=match_metadata.get("home_team"),
        away_team=match_metadata.get("away_team"),
        match_date=match_metadata.get("match_date"),
        score=match_metadata.get("score"),
        competition=match_metadata.get("competition"),
        key_moments=key_moments,
        man_of_the_match=match_metadata.get("man_of_the_match"),
        match_summary=match_metadata.get("match_summary"),
    )


# =============================================================================
# Core Query Processing (Web Search + Summary - NO game analysis)
# =============================================================================

async def _process_query_core(
    request: QueryRequest,
    stream_thinking: bool = False
) -> AsyncGenerator:
    """
    Core query processing: web search and answer generation.
    
    NOTE: This does NOT call the game analyst - that's only via /analyze endpoint.
    """
    
    def send(stage: str, message: str, status: str, data: Optional[Dict] = None):
        if stream_thinking:
            return _format_sse_message(stage, message, status, data)
        return None
    
    try:
        # Step 1: Query Parser
        msg = send("query_parser", "Analyzing your query...", "starting", {"query": request.query})
        if msg:
            yield msg
        
        try:
            parsed = parse_query(
                request.query,
                emphasize_order=request.emphasize_order or False
            )
        except Exception as e:
            if is_quota_error(e):
                error_msg = "Our AI engine has run out of credits. Please try again later."
                if stream_thinking:
                    yield send("query_parser", error_msg, "error", {"error": str(e)})
                yield {"type": "error", "error": "openai_quota_exceeded", "message": error_msg}
                return
            raise
        
        # Check relevance
        if not parsed.get("is_relevant", True) and parsed.get("validation_error"):
            error_reason = parsed["validation_error"].get("reason", "Query not about football.")
            suggestion = parsed["validation_error"].get("suggestion", "Try asking about a match.")
            if stream_thinking:
                yield send("query_parser", f"Validation failed: {error_reason}", "error")
            yield {
                "type": "error",
                "error": "query_not_relevant",
                "message": f"❌ {error_reason}\n\n💡 Suggestion: {suggestion}"
            }
            return
        
        intent = parsed.get("intent", "general")
        search_query = parsed.get("search_query", request.query)
        teams = parsed.get("teams", [])
        
        msg = send("query_parser", f"Query parsed. Intent: {intent}.", "complete",
                   {"intent": intent, "teams": teams})
        if msg:
            yield msg
        
        show_highlights = request.include_highlights
        if show_highlights is None:
            show_highlights = should_fetch_highlights(parsed)
        
        # Step 2: Web Search
        msg = send("web_search", "Searching for match information...", "starting")
        if msg:
            yield msg
        
        try:
            web_summary, match_metadata = search_with_rag(
                query=search_query,
                intent=intent,
                original_query=request.query,
                parsed_query=parsed,
                gender=request.gender or "men",
            )
        except Exception as e:
            if is_quota_error(e):
                error_msg = "Our AI engine has run out of credits. Please try again later."
                if stream_thinking:
                    yield send("web_search", error_msg, "error")
                yield {"type": "error", "error": "openai_quota_exceeded", "message": error_msg}
                return
            raise
        
        # Check if match was found
        if not _check_match_found(match_metadata, web_summary):
            sources = extract_urls_from_text(web_summary or "")
            msg = send("web_search", "Could not find reliable match information.", "complete")
            if msg:
                yield msg
            yield {
                "type": "result",
                "data": {
                    "success": False,
                    "intent": intent,
                    "summary": web_summary or "❌ Could not find match information.",
                    "error": "no_match_found",
                    "match_metadata": None,
                    "highlights": [],
                    "sources": sources,
                    "game_analysis": None,
                }
            }
            return

        score_info = match_metadata.get('score', 'Not found')
        teams_info = f"{match_metadata.get('home_team', 'Unknown')} vs {match_metadata.get('away_team', 'Unknown')}"
        msg = send("web_search", f"Found match: {teams_info}. Score: {score_info}.", "complete")
        if msg:
            yield msg
        
        sources = extract_urls_from_text(web_summary)
        match_meta = _build_match_metadata_model(match_metadata)
        
        # Step 3: Highlights (if requested)
        highlights: List[HighlightVideo] = []
        if show_highlights:
            msg = send("highlights", "Searching for highlight videos...", "starting")
            if msg:
                yield msg
            
            try:
                home_team = match_metadata.get("home_team") or (teams[0] if teams else None)
                away_team = match_metadata.get("away_team") or (teams[1] if len(teams) > 1 else None)
                match_date = match_metadata.get("match_date")
                
                if home_team:
                    raw_highlights = search_and_display_highlights_with_metadata(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        web_summary=web_summary,
                        match_metadata=match_metadata,
                        parsed_query=parsed,  # Pass parsed_query for team order filtering
                    )
                    highlights = _normalise_highlight_results(raw_highlights)
                    msg = send("highlights", f"Found {len(highlights)} highlight videos.", "complete")
                    if msg:
                        yield msg
            except Exception as e:
                msg = send("highlights", f"Highlights error: {str(e)}", "error")
                if msg:
                    yield msg
        
        # Final result - NO game analysis (use /analyze endpoint for that)
        yield {
            "type": "result",
            "data": {
            "success": True,
            "intent": intent,
            "summary": web_summary,
            "match_metadata": match_meta.model_dump() if match_meta else None,
            "highlights": [h.model_dump() if hasattr(h, 'model_dump') else h for h in highlights],
            "sources": sources,
                "game_analysis": None,  # Not included - use /analyze endpoint
        }
        }
        
    except Exception as e:
        if is_quota_error(e):
            error_msg = "Our AI engine has run out of credits. Please try again later."
            if stream_thinking:
                yield send("error", error_msg, "error")
            yield {"type": "error", "error": "openai_quota_exceeded", "message": error_msg}
        else:
            if stream_thinking:
                yield send("error", f"Error: {str(e)}", "error")
            yield {"type": "error", "error": str(e), "message": f"An error occurred: {str(e)}"}


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": API_VERSION}


@app.get("/query/stream")
@app.post("/query/stream")
async def query_stream_endpoint(
    request: Optional[QueryRequest] = None,
    query: Optional[str] = Query(None, description="The query to process"),
    include_highlights: Optional[bool] = Query(None, description="Whether to include highlights")
):
    """
    SSE streaming endpoint for query processing.
    
    Returns: Web search results + summary + highlights (if requested).
    For deep game analysis, use the /analyze endpoint separately.
    """
    if request is None:
        if query is None:
            error_response = {
                "type": "result",
                "data": {
                    "success": False,
                    "intent": "general",
                    "summary": "❌ Query is required",
                    "error": "missing_query",
                    "match_metadata": None,
                    "highlights": [],
                    "sources": [],
                    "game_analysis": None,
                }
            }
            return StreamingResponse(
                f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n",
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Content-Type": "text/event-stream; charset=utf-8",
                }
            )
        query = unquote(query) if query else ""
    else:
        query = request.query
    
    if not query:
        error_response = {
            "type": "result",
            "data": {
                "success": False,
                "intent": "general",
                "summary": "❌ Query is required",
                "error": "missing_query",
                "match_metadata": None,
                "highlights": [],
                "sources": [],
                "game_analysis": None,
            }
        }
        return StreamingResponse(
            f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n",
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8",
            }
        )
    
    if request is None:
        request = QueryRequest(query=query, include_highlights=include_highlights)
    
    async def generate():
        try:
            async for item in _process_query_core(request, stream_thinking=True):
                if isinstance(item, str):
                    yield item
                    await asyncio.sleep(0.001)
                elif isinstance(item, dict):
                    if item.get("type") == "result":
                        yield f"data: {json.dumps(item)}\n\n"
                    elif item.get("type") == "error":
                        error_response = {
                            "success": False,
                            "intent": "general",
                            "summary": f"❌ {item.get('message', 'Unknown error')}",
                            "error": item.get("error"),
                            "match_metadata": None,
                            "highlights": [],
                            "sources": [],
                            "game_analysis": None,
                        }
                        yield f"data: {json.dumps({'type': 'result', 'data': error_response})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
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
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint - answers user questions.
    
    Returns: Web search results + summary + highlights.
    For deep game analysis, use the /analyze endpoint separately.
    """
    result = None
    async for item in _process_query_core(request, stream_thinking=False):
        if isinstance(item, dict):
            if item.get("type") == "result":
                result = item["data"]
            elif item.get("type") == "error":
                return QueryResponse(
                    success=False,
                    intent="general",
                    summary=f"❌ {item.get('message', 'Unknown error')}",
                    error=item.get("error"),
                match_metadata=None,
                highlights=[],
                sources=[],
                game_analysis=None,
            )
        
    if result:
        return QueryResponse(**result)
    
    return QueryResponse(
        success=False,
        intent="general",
        summary="❌ No result generated",
        error="no_result"
    )


@app.get("/intents")
async def list_intents():
    """List all available query intents."""
    return {
        "intents": [
            {"value": QueryIntent.MATCH_RESULT, "description": "Get match score/result"},
            {"value": QueryIntent.MATCH_HIGHLIGHTS, "description": "Watch match highlights"},
            {"value": QueryIntent.COMPETITION_LATEST, "description": "Latest game in a competition"},
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
    Deep game analysis endpoint - ONLY called on explicit request.
    
    Provides sophisticated match analysis including:
    - Momentum shift analysis
    - Tactical breakdown
    - Deep strategic insights
    
    This is separate from /query to save API costs - only call when user
    explicitly wants deep analysis.
    """
    try:
        # Step 1: Parse query
        try:
            parsed = parse_query(
                request.query,
                emphasize_order=request.emphasize_order or False
            )
        except Exception as e:
            if is_quota_error(e):
                return GameAnalysisResponse(
                    success=False,
                    error="Our AI engine has run out of credits. Please try again later."
                )
            return GameAnalysisResponse(success=False, error=f"Error parsing query: {str(e)}")
        
        if not parsed.get("is_relevant", True) and parsed.get("validation_error"):
            error = parsed["validation_error"]
            return GameAnalysisResponse(
                success=False,
                error=f"{error.get('reason', 'Not about football.')} {error.get('suggestion', '')}"
            )
        
        search_query = parsed.get("search_query", request.query)
        intent = parsed.get("intent", "general")
        
        # Step 2: Web search
        try:
            web_summary, match_metadata = search_with_rag(
                query=search_query,
                intent=intent,
                original_query=request.query,
                parsed_query=parsed,
                gender=request.gender or "men",
            )
        except Exception as e:
            if is_quota_error(e):
                return GameAnalysisResponse(
                    success=False,
                    error="Our AI engine has run out of credits. Please try again later."
                )
            return GameAnalysisResponse(success=False, error=f"Web search failed: {str(e)}")
        
        if not _check_match_found(match_metadata, web_summary):
            return GameAnalysisResponse(
                success=False,
                error="Could not find match information for this query."
            )
        
        # Step 3: Deep game analysis
        try:
            analysis = analyze_match_from_web_results(
                web_summary=web_summary,
                match_metadata=match_metadata,
                original_query=request.query
            )
        except Exception as e:
            if is_quota_error(e):
                return GameAnalysisResponse(
                    success=False,
                    error="Our AI engine has run out of credits during analysis."
                )
            return GameAnalysisResponse(success=False, error=f"Game analysis failed: {str(e)}")
        
        if not analysis.get('success'):
            return GameAnalysisResponse(
                success=False,
                error=analysis.get('error', 'Unknown error')
            )
        
        # Step 4: Highlights (optional)
        highlights: List[HighlightVideo] = []
        include_highlights = request.include_highlights if request.include_highlights is not None else True
        if include_highlights:
            try:
                home_team = match_metadata.get("home_team")
                away_team = match_metadata.get("away_team")
                match_date = match_metadata.get("match_date")
                
                if home_team:
                    raw_highlights = search_and_display_highlights_with_metadata(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        web_summary=web_summary,
                        match_metadata=match_metadata,
                        parsed_query=parsed,  # Pass parsed_query for team order filtering
                    )
                    highlights = _normalise_highlight_results(raw_highlights)
            except Exception:
                pass
        
        momentum_shifts = [
            MomentumShift(**shift) for shift in analysis.get('momentum_analysis', [])
        ]
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
        if is_quota_error(e):
            return GameAnalysisResponse(
                success=False,
                error="Our AI engine has run out of credits. Please try again later."
            )
        return GameAnalysisResponse(success=False, error=f"An error occurred: {str(e)}")


# =============================================================================
# Run Server
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000, reload=True)
