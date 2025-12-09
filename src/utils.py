"""
Shared utilities for Soccer LLM Analyst.

Consolidates common functionality to avoid duplication and memory leaks.
"""

import re
from typing import Any, Optional

from openai import OpenAI

from .config import get_openai_key


# =============================================================================
# Singleton OpenAI Client
# =============================================================================

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Get or initialize the shared OpenAI client.
    
    This is a singleton to prevent memory leaks from creating
    multiple client instances across modules.
    
    Returns:
        OpenAI: The shared OpenAI client instance.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# =============================================================================
# String Helpers
# =============================================================================

def safe_lower(value: Any) -> str:
    """
    Safely convert any value to lowercase string.
    
    Args:
        value: Any value to convert.
        
    Returns:
        Lowercase string representation, or empty string if None.
    """
    return str(value).lower() if value is not None else ""


def extract_urls_from_text(text: str) -> list[str]:
    """
    Extract all HTTP/HTTPS URLs from text.
    
    Args:
        text: Text to search for URLs.
        
    Returns:
        List of unique URLs found.
    """
    if not text:
        return []
    urls = re.findall(r'https?://[^\s\)]+', text)
    return list(set(urls))  # Remove duplicates


# =============================================================================
# Error Detection Helpers
# =============================================================================

def is_quota_error(exc: Exception) -> bool:
    """
    Check if an exception is an OpenAI quota/credit error.
    
    Args:
        exc: The exception to check.
        
    Returns:
        True if it appears to be a quota-related error.
    """
    msg = str(exc).lower()
    quota_keywords = [
        "insufficient_quota",
        "you exceeded your current quota",
        "billing hard limit",
        "rate limit",
        "quota",
    ]
    return any(key in msg for key in quota_keywords)


# =============================================================================
# Score/Date Extraction Helpers
# =============================================================================

def extract_score_pattern(text: str) -> list[tuple[int, int, int, int]]:
    """
    Find all score-like patterns (X-Y) in text.
    
    Args:
        text: Text to search.
        
    Returns:
        List of tuples: (left_score, right_score, start_pos, end_pos)
    """
    if not text:
        return []
    
    text = text.replace("–", "-")  # Normalize dashes
    results = []
    
    for match in re.finditer(r'\b(\d{1,2})\s*-\s*(\d{1,2})\b', text):
        left = int(match.group(1))
        right = int(match.group(2))
        # Filter out obviously non-score values
        if left <= 15 and right <= 15:
            results.append((left, right, match.start(), match.end()))
    
    return results


def normalize_team_name(name: Optional[str]) -> str:
    """
    Normalize a team name for comparison.
    
    Args:
        name: Team name to normalize.
        
    Returns:
        Lowercase, stripped team name.
    """
    if not name:
        return ""
    return name.lower().strip()


# =============================================================================
# Domain/URL Helpers
# =============================================================================

def domain_from_url(url: str) -> str:
    """
    Extract domain from a URL.
    
    Args:
        url: URL to parse.
        
    Returns:
        Domain without www prefix, or empty string on error.
    """
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

