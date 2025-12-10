"""
Pydantic models for Soccer LLM Analyst API.

Separates data models from API logic for better organization.
"""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel


# =============================================================================
# Request Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    include_highlights: Optional[bool] = None  # Auto-detect if not provided
    
    # When True, the order teams appear in the query matters
    # First team mentioned = home team, second = away team
    emphasize_order: Optional[bool] = False
    
    # Gender preference - defaults to "men" for men's football
    # Options: "men", "women", "any"
    gender: Optional[str] = "men"


# =============================================================================
# Match Data Models
# =============================================================================

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


class MomentumShift(BaseModel):
    """Model for a momentum shift event."""
    minute: str
    event: str
    description: str
    team: Optional[str] = None
    momentum_impact: str  # "neutral", "high", "critical"
    reasoning: str


# =============================================================================
# Response Models
# =============================================================================

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    intent: str
    summary: str
    match_metadata: Optional[MatchMetadata] = None
    highlights: Optional[List[HighlightVideo]] = []
    sources: List[str] = []
    game_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ThinkingMessage(BaseModel):
    """Model for thinking/status messages in SSE streams."""
    stage: str  # "query_parser", "web_search", "game_analyst", "highlights"
    message: str
    status: str  # "starting", "processing", "complete", "error"
    data: Optional[Dict[str, Any]] = None


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

