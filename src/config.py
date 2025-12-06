"""
Configuration module for Soccer Match LLM Analyst.

Contains LLM settings and environment variable helpers.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# LLM Configuration
# =============================================================================

# Default OpenAI model (can be changed to gpt-4, gpt-4-turbo, etc.)
DEFAULT_LLM_MODEL = "gpt-4.1-mini"


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def get_openai_key() -> str:
    """
    Retrieve the OpenAI API key from environment variables.
    
    Returns:
        str: The API key.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    return key
