"""
Wikipedia fallback search module.

Provides a fallback search when Sportmonks cannot resolve a match.
Uses Wikipedia's public API (no authentication required).
"""

import requests

# Wikipedia API endpoints
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Maximum summary length to return (characters)
MAX_SUMMARY_LENGTH = 1500


def fallback_search(query: str) -> str:
    """
    Search Wikipedia for information about a football match.
    
    This is used as a fallback when Sportmonks cannot resolve the match.
    
    Args:
        query: The user's original match query (e.g., "Celtic vs Rangers 2024-04-07").
        
    Returns:
        str: A summary paragraph about the match if found,
             or a friendly message if nothing useful is found.
    """
    # Enhance query for better football match results
    search_query = f"{query} football match"
    
    try:
        # Step 1: Search for relevant Wikipedia pages
        page_title = _search_wikipedia(search_query)
        
        if not page_title:
            # Try alternative search without "football match"
            page_title = _search_wikipedia(query)
        
        if not page_title:
            return _not_found_message(query)
        
        # Step 2: Fetch the summary for the found page
        summary = _fetch_page_summary(page_title)
        
        if summary:
            return summary
        else:
            return _not_found_message(query)
            
    except requests.RequestException as e:
        print(f"[Wikipedia] Network error: {e}")
        return _not_found_message(query)
    except Exception as e:
        print(f"[Wikipedia] Unexpected error: {e}")
        return _not_found_message(query)


def _search_wikipedia(query: str) -> str | None:
    """
    Search Wikipedia and return the title of the best matching page.
    
    Args:
        query: Search query string.
        
    Returns:
        str: Title of the best matching page, or None if no results.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,  # Get top 5 results
        "format": "json",
    }
    
    response = requests.get(WIKIPEDIA_SEARCH_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    search_results = data.get("query", {}).get("search", [])
    
    if not search_results:
        return None
    
    # Return the title of the first (most relevant) result
    return search_results[0].get("title")


def _fetch_page_summary(title: str) -> str | None:
    """
    Fetch the summary extract for a Wikipedia page.
    
    Args:
        title: The Wikipedia page title.
        
    Returns:
        str: The page summary (trimmed to MAX_SUMMARY_LENGTH), or None if not found.
    """
    # URL-encode the title for the REST API
    encoded_title = requests.utils.quote(title, safe="")
    url = f"{WIKIPEDIA_SUMMARY_URL}/{encoded_title}"
    
    response = requests.get(url, timeout=10)
    
    if response.status_code == 404:
        return None
    
    response.raise_for_status()
    data = response.json()
    
    # Get the extract (summary text)
    extract = data.get("extract", "")
    
    if not extract:
        return None
    
    # Trim if too long
    if len(extract) > MAX_SUMMARY_LENGTH:
        # Try to cut at a sentence boundary
        truncated = extract[:MAX_SUMMARY_LENGTH]
        last_period = truncated.rfind(".")
        if last_period > MAX_SUMMARY_LENGTH // 2:
            extract = truncated[:last_period + 1]
        else:
            extract = truncated + "..."
    
    # Add source attribution
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    if page_url:
        extract += f"\n\n📖 Source: {page_url}"
    
    return extract


def _not_found_message(query: str) -> str:
    """
    Return a friendly message when no information is found.
    
    Args:
        query: The original search query.
        
    Returns:
        str: A friendly "not found" message.
    """
    return (
        f"I couldn't find reliable web information for \"{query}\".\n"
        "This match may not have a Wikipedia page, or the query might need adjustment.\n"
        "Try searching with different team names or date format."
    )
