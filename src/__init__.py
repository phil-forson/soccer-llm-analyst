"""
Soccer LLM Analyst - Source Package

A web search-based football/soccer information system with intelligent query parsing,
game analysis, and highlight video discovery.
"""

from .config import get_openai_key, get_youtube_api_key, DEFAULT_LLM_MODEL
from .utils import get_openai_client, safe_lower, is_quota_error, extract_urls_from_text
from .models import (
    QueryRequest, QueryResponse, GameAnalysisResponse, HealthResponse,
    ThinkingMessage, KeyMoment, MatchMetadata, HighlightVideo, MomentumShift
)
from .query_parser_agent import parse_query, should_fetch_highlights, QueryIntent
from .web_search_agent import search_with_rag
from .youtube_search_agent import search_and_display_highlights_with_metadata
from .game_analyst_agent import analyze_match_from_web_results, format_analysis

__version__ = "1.0.0"

__all__ = [
    # Config
    "get_openai_key",
    "get_youtube_api_key", 
    "DEFAULT_LLM_MODEL",
    # Utils
    "get_openai_client",
    "safe_lower",
    "is_quota_error",
    "extract_urls_from_text",
    # Models
    "QueryRequest",
    "QueryResponse", 
    "GameAnalysisResponse",
    "HealthResponse",
    "ThinkingMessage",
    "KeyMoment",
    "MatchMetadata",
    "HighlightVideo",
    "MomentumShift",
    # Query parsing
    "parse_query",
    "should_fetch_highlights",
    "QueryIntent",
    # Web search
    "search_with_rag",
    # YouTube
    "search_and_display_highlights_with_metadata",
    # Game analysis
    "analyze_match_from_web_results",
    "format_analysis",
]
