"""
Q&A module for answering questions about matches.

Uses the embeddings store for retrieval and an LLM for generation.
"""

from openai import OpenAI

from .config import get_openai_key, DEFAULT_LLM_MODEL
from .embeddings_store import retrieve_match_context


# =============================================================================
# LLM Client
# =============================================================================

_openai_client = None


def _get_openai_client() -> OpenAI:
    """
    Get or initialize the OpenAI client.
    
    Returns:
        OpenAI: The OpenAI client instance.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are a knowledgeable football (soccer) match analyst assistant.

Your task is to answer questions about a specific match based ONLY on the provided match transcript context.

Rules:
1. Use ONLY the information in the provided context chunks to answer questions.
2. When referencing events, mention the minute they occurred (e.g., "In the 45th minute...").
3. If the answer is not clearly supported by the context, say "I'm not sure based on the available match data."
4. Be concise but informative. Focus on the facts from the match.
5. Do not make up events or statistics that are not in the context.
6. If asked about something outside the match context, politely explain you can only answer about events in the provided transcript."""


def _build_context_prompt(context_chunks: list[dict]) -> str:
    """
    Build the context portion of the prompt from retrieved chunks.
    
    Args:
        context_chunks: List of chunks with text and minute ranges.
        
    Returns:
        str: Formatted context string for the prompt.
    """
    if not context_chunks:
        return "No match context available."
    
    lines = ["=== MATCH TRANSCRIPT CONTEXT ===\n"]
    
    for i, chunk in enumerate(context_chunks, 1):
        start = chunk.get("start_minute", "?")
        end = chunk.get("end_minute", "?")
        text = chunk.get("text", "")
        
        lines.append(f"[Chunk {i} | Minutes {start}'-{end}']")
        lines.append(text)
        lines.append("")  # Blank line between chunks
    
    lines.append("=== END OF CONTEXT ===")
    
    return "\n".join(lines)


# =============================================================================
# Generation Functions
# =============================================================================

def generate_answer_from_context(
    question: str,
    context_chunks: list[dict],
    model: str = None
) -> str:
    """
    Generate an answer to a question using the provided context chunks.
    
    Args:
        question: The user's question about the match.
        context_chunks: List of relevant context chunks from the embeddings store.
        model: Optional LLM model override (defaults to DEFAULT_LLM_MODEL).
        
    Returns:
        str: The generated answer.
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    
    client = _get_openai_client()
    
    # Build the context portion of the prompt
    context_text = _build_context_prompt(context_chunks)
    
    # Build the user message
    user_message = f"""{context_text}

QUESTION: {question}

Please answer the question based only on the match transcript context above."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=500,
        )
        
        answer = response.choices[0].message.content
        return answer.strip() if answer else "I couldn't generate an answer."
        
    except Exception as e:
        print(f"[QA] Error calling LLM: {e}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"


# =============================================================================
# High-Level Q&A Function
# =============================================================================

def answer_match_question(
    match_id: str,
    question: str,
    k: int = 6,
    model: str = None
) -> str:
    """
    Answer a question about a specific match.
    
    This is the main entry point for the Q&A system. It:
    1. Retrieves relevant context chunks from the embeddings store
    2. Generates an answer using the LLM
    
    Args:
        match_id: The match identifier (from Sportmonks).
        question: The user's question about the match.
        k: Number of context chunks to retrieve (default: 6).
        model: Optional LLM model override.
        
    Returns:
        str: The generated answer.
    """
    match_id = str(match_id)
    
    # Retrieve relevant context chunks
    print(f"[QA] Retrieving context for match {match_id}...")
    context_chunks = retrieve_match_context(match_id, question, k=k)
    
    if not context_chunks:
        return (
            "I don't have any match data indexed for this game. "
            "Please make sure the match events were loaded correctly."
        )
    
    print(f"[QA] Found {len(context_chunks)} relevant chunks. Generating answer...")
    
    # Generate and return the answer
    return generate_answer_from_context(question, context_chunks, model=model)


def format_stats_summary(stats: dict, home_team: str, away_team: str) -> str:
    """
    Format match statistics into a readable summary.
    
    Args:
        stats: Statistics dict from fetch_match_stats().
        home_team: Name of home team.
        away_team: Name of away team.
        
    Returns:
        str: Formatted statistics summary.
    """
    lines = [f"📊 Match Statistics: {home_team} vs {away_team}\n"]
    
    # Score
    home_goals = stats.get("home_goals", 0)
    away_goals = stats.get("away_goals", 0)
    lines.append(f"⚽ Final Score: {home_team} {home_goals} - {away_goals} {away_team}")
    
    # Possession
    home_poss = stats.get("home_possession")
    away_poss = stats.get("away_possession")
    if home_poss is not None and away_poss is not None:
        lines.append(f"🔄 Possession: {home_team} {home_poss}% - {away_poss}% {away_team}")
    
    # Shots
    home_shots = stats.get("home_shots")
    away_shots = stats.get("away_shots")
    if home_shots is not None and away_shots is not None:
        lines.append(f"🎯 Total Shots: {home_team} {home_shots} - {away_shots} {away_team}")
    
    # Shots on target
    home_sot = stats.get("home_shots_on_target")
    away_sot = stats.get("away_shots_on_target")
    if home_sot is not None and away_sot is not None:
        lines.append(f"🥅 Shots on Target: {home_team} {home_sot} - {away_sot} {away_team}")
    
    # Corners
    home_corners = stats.get("home_corners")
    away_corners = stats.get("away_corners")
    if home_corners is not None and away_corners is not None:
        lines.append(f"📐 Corners: {home_team} {home_corners} - {away_corners} {away_team}")
    
    # Fouls
    home_fouls = stats.get("home_fouls")
    away_fouls = stats.get("away_fouls")
    if home_fouls is not None and away_fouls is not None:
        lines.append(f"⚠️ Fouls: {home_team} {home_fouls} - {away_fouls} {away_team}")
    
    return "\n".join(lines)
