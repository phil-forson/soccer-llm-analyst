"""
Game Analyst Agent - Provides sophisticated match analysis.

This agent receives web search results and generates deep tactical, 
momentum, and strategic analysis. It's part of the agent chain:
Query Parser → Web Search (RAG) → Game Analyst
"""

from datetime import datetime
from typing import Optional, Dict, List, Any

from .config import get_openai_key, DEFAULT_LLM_MODEL
from openai import OpenAI


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
# Main Game Analyst Function (Chain-based)
# =============================================================================


# =============================================================================
# Momentum Analysis
# =============================================================================

def _analyze_momentum_shifts(key_moments: List[Dict]) -> List[Dict]:
    """
    Analyze momentum shifts in the match based on key moments.
    
    Args:
        key_moments: List of key moments with minute, event, description, team.
        
    Returns:
        List of momentum shift events with analysis.
    """
    if not key_moments:
        return []
    
    momentum_shifts = []
    
    # Sort by minute
    sorted_moments = sorted(
        [m for m in key_moments if m.get('minute')],
        key=lambda x: int(x.get('minute', '0').replace("'", "").split("+")[0] or 0)
    )
    
    # Analyze patterns
    goals = [m for m in sorted_moments if 'GOAL' in m.get('event', '').upper()]
    cards = [m for m in sorted_moments if 'CARD' in m.get('event', '').upper()]
    penalties = [m for m in sorted_moments if 'PENALTY' in m.get('event', '').upper()]
    
    # Identify momentum shifts
    for i, moment in enumerate(sorted_moments):
        shift = {
            "minute": moment.get('minute', '?'),
            "event": moment.get('event', ''),
            "description": moment.get('description', ''),
            "team": moment.get('team', ''),
            "momentum_impact": "neutral",
            "reasoning": ""
        }
        
        event = moment.get('event', '').upper()
        
        # Goals create major momentum shifts
        if 'GOAL' in event:
            # Early goals (0-30 min) set tone
            minute = int(moment.get('minute', '0').replace("'", "").split("+")[0] or 0)
            if minute <= 30:
                shift["momentum_impact"] = "high"
                shift["reasoning"] = "Early goal sets the tone and puts pressure on the opponent"
            # Late goals (75+ min) are decisive
            elif minute >= 75:
                shift["momentum_impact"] = "critical"
                shift["reasoning"] = "Late goal can be decisive and demoralizing"
            else:
                shift["momentum_impact"] = "high"
                shift["reasoning"] = "Goal shifts momentum significantly"
        
        # Red cards are major momentum shifts
        elif 'RED_CARD' in event:
            shift["momentum_impact"] = "critical"
            shift["reasoning"] = "Red card creates numerical advantage and tactical shift"
        
        # Penalties missed can shift momentum
        elif 'PENALTY' in event and 'missed' in moment.get('description', '').lower():
            shift["momentum_impact"] = "high"
            shift["reasoning"] = "Missed penalty can shift momentum to the defending team"
        
        momentum_shifts.append(shift)
    
    return momentum_shifts


# =============================================================================
# Tactical Analysis
# =============================================================================

def _analyze_tactical_patterns(key_moments: List[Dict], match_metadata: Dict) -> Dict[str, Any]:
    """
    Analyze tactical patterns from key moments.
    
    Args:
        key_moments: List of key moments.
        match_metadata: Match metadata with teams and score.
        
    Returns:
        Tactical analysis dict.
    """
    if not key_moments:
        return {}
    
    home_team = match_metadata.get('home_team', 'Home')
    away_team = match_metadata.get('away_team', 'Away')
    score = match_metadata.get('score', '0-0')
    
    # Parse score
    try:
        home_score, away_score = map(int, score.split('-'))
    except:
        home_score, away_score = 0, 0
    
    # Count events by team
    home_events = [m for m in key_moments if m.get('team') == 'home']
    away_events = [m for m in key_moments if m.get('team') == 'away']
    
    home_goals = len([m for m in home_events if 'GOAL' in m.get('event', '').upper()])
    away_goals = len([m for m in away_events if 'GOAL' in m.get('event', '').upper()])
    
    # Analyze match phases
    first_half = [m for m in key_moments if int(m.get('minute', '0').replace("'", "").split("+")[0] or 0) <= 45]
    second_half = [m for m in key_moments if int(m.get('minute', '0').replace("'", "").split("+")[0] or 0) > 45]
    
    analysis = {
        "match_phases": {
            "first_half_events": len(first_half),
            "second_half_events": len(second_half),
            "more_active_half": "first" if len(first_half) > len(second_half) else "second"
        },
        "goal_distribution": {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "total_goals": home_goals + away_goals
        },
        "team_activity": {
            "home_events": len(home_events),
            "away_events": len(away_events),
            "more_active_team": home_team if len(home_events) > len(away_events) else away_team
        }
    }
    
    return analysis


# =============================================================================
# Deep Match Analysis with LLM
# =============================================================================

def _generate_deep_analysis(
    match_metadata: Dict,
    key_moments: List[Dict],
    momentum_shifts: List[Dict],
    tactical_analysis: Dict,
    web_summary: str
) -> str:
    """
    Use LLM to generate sophisticated match analysis.
    
    Args:
        match_metadata: Match metadata.
        key_moments: List of key moments.
        momentum_shifts: Momentum shift analysis.
        tactical_analysis: Tactical pattern analysis.
        web_summary: Summary from web search.
        
    Returns:
        Detailed analysis text.
    """
    client = _get_openai_client()
    
    home_team = match_metadata.get('home_team', 'Home')
    away_team = match_metadata.get('away_team', 'Away')
    score = match_metadata.get('score', '0-0')
    match_date = match_metadata.get('match_date', '')
    competition = match_metadata.get('competition', '')
    man_of_match = match_metadata.get('man_of_the_match', '')
    
    # Build context
    moments_text = "\n".join([
        f"{m.get('minute', '?')}' - {m.get('event', 'EVENT')}: {m.get('description', '')} ({m.get('team', 'unknown')} team)"
        for m in key_moments
    ])
    
    momentum_text = "\n".join([
        f"{m['minute']}' - {m['event']}: {m['momentum_impact'].upper()} impact - {m['reasoning']}"
        for m in momentum_shifts if m.get('momentum_impact') != 'neutral'
    ])
    
    prompt = f"""You are an elite football analyst providing deep tactical and strategic analysis of a match.

=== MATCH INFORMATION ===
Teams: {home_team} vs {away_team}
Score: {score}
Date: {match_date}
Competition: {competition}
Man of the Match: {man_of_match or 'Not specified'}

=== KEY MOMENTS (Chronological) ===
{moments_text if moments_text else "No key moments recorded"}

=== MOMENTUM SHIFTS ===
{momentum_text if momentum_text else "No significant momentum shifts identified"}

=== TACTICAL ANALYSIS ===
Match Phases: {tactical_analysis.get('match_phases', {})}
Goal Distribution: {tactical_analysis.get('goal_distribution', {})}
Team Activity: {tactical_analysis.get('team_activity', {})}

=== WEB SEARCH SUMMARY ===
{web_summary[:1000]}

=== YOUR ANALYSIS TASK ===

Provide a comprehensive, sophisticated match analysis covering:

1. **Match Narrative** (2-3 paragraphs)
   - How the match unfolded
   - Key storylines and themes
   - Overall match quality and intensity

2. **Momentum Analysis** (detailed)
   - When did momentum shift and why?
   - Which team controlled which phases?
   - Critical turning points and their impact
   - How momentum changes affected the outcome

3. **Tactical Breakdown**
   - Tactical patterns observed
   - Key tactical decisions that influenced the match
   - Formation and style of play analysis
   - Set pieces and transitions

4. **Key Performances**
   - Standout individual performances
   - Man of the match analysis (if provided)
   - Key substitutions and their impact

5. **Implications & Context**
   - What this result means for both teams
   - League/competition implications
   - Historical context if relevant
   - What to watch for next

6. **Statistical Insights**
   - Notable patterns in goal timing
   - Match phases analysis
   - Team activity levels

Write in an engaging, analytical style suitable for serious football analysis. Be specific, cite moments by minute, and provide tactical insights.

Format your response with clear sections and use emojis sparingly for visual breaks."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite football analyst with deep tactical knowledge. Provide sophisticated, detailed match analysis with tactical insights, momentum analysis, and strategic implications."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more factual analysis
            max_tokens=2000,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


# =============================================================================
# Main Game Analyst Function (Chain-based - receives web search results)
# =============================================================================

def analyze_match_from_web_results(
    web_summary: str,
    match_metadata: Dict[str, Any],
    original_query: str = ""
) -> Dict[str, Any]:
    """
    Comprehensive match analysis using web search results directly.
    
    This is the CHAINED version that receives web search results:
    Query Parser → Web Search (RAG) → Game Analyst (this function)
    
    This function:
    1. Receives web search summary and match metadata
    2. Analyzes momentum shifts from key moments
    3. Provides tactical analysis
    4. Generates deep LLM-based analysis
    
    Args:
        web_summary: Summary text from web search (RAG-generated)
        match_metadata: Match metadata dict with:
            - home_team, away_team, score, match_date, competition
            - key_moments: List of key moments
            - man_of_the_match, match_summary
        original_query: Original user query (for context)
        
    Returns:
        Dict with comprehensive analysis:
            - success: bool
            - match_info: dict (basic match info)
            - momentum_analysis: list (momentum shifts)
            - tactical_analysis: dict (tactical patterns)
            - deep_analysis: str (LLM-generated comprehensive analysis)
            - key_moments: list (chronological events)
            - error: str or None
    """
    print(f"\n{'='*70}")
    print(f"[GameAnalyst] STARTING COMPREHENSIVE MATCH ANALYSIS")
    print(f"{'='*70}")
    print(f"[GameAnalyst] Match: {match_metadata.get('home_team', 'Unknown')} vs {match_metadata.get('away_team', 'Unknown')}")
    
    if not match_metadata or not match_metadata.get('score'):
        return {
            "success": False,
            "error": "Could not extract match information. Missing score or match data.",
            "match_info": None,
            "momentum_analysis": [],
            "tactical_analysis": {},
            "deep_analysis": "",
            "key_moments": [],
        }
    
    key_moments = match_metadata.get('key_moments', [])
    
    print(f"[GameAnalyst] ✓ Match: {match_metadata.get('home_team')} {match_metadata.get('score')} {match_metadata.get('away_team')}")
    print(f"[GameAnalyst] ✓ Found {len(key_moments)} key moments")
    print(f"[GameAnalyst] ✓ Using web search summary (RAG-validated)")
    
    # Step 1: Analyze momentum shifts
    print(f"\n[GameAnalyst] Step 1: Analyzing momentum shifts...")
    momentum_shifts = _analyze_momentum_shifts(key_moments)
    significant_shifts = [m for m in momentum_shifts if m.get('momentum_impact') != 'neutral']
    print(f"[GameAnalyst] ✓ Identified {len(significant_shifts)} significant momentum shifts")
    
    # Step 2: Tactical analysis
    print(f"\n[GameAnalyst] Step 2: Analyzing tactical patterns...")
    tactical_analysis = _analyze_tactical_patterns(key_moments, match_metadata)
    print(f"[GameAnalyst] ✓ Tactical analysis complete")
    
    # Step 3: Generate deep analysis
    print(f"\n[GameAnalyst] Step 3: Generating comprehensive analysis with LLM...")
    deep_analysis = _generate_deep_analysis(
        match_metadata,
        key_moments,
        momentum_shifts,
        tactical_analysis,
        web_summary
    )
    print(f"[GameAnalyst] ✓ Deep analysis generated")
    
    # Print comprehensive match analysis to console
    print(f"\n{'='*70}")
    print(f"📊 COMPREHENSIVE MATCH ANALYSIS")
    print(f"{'='*70}")
    print(f"\n{deep_analysis}\n")
    print(f"{'='*70}\n")
    
    print(f"[GameAnalyst] ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return {
        "success": True,
        "match_info": {
            "home_team": match_metadata.get('home_team'),
            "away_team": match_metadata.get('away_team'),
            "score": match_metadata.get('score'),
            "match_date": match_metadata.get('match_date'),
            "competition": match_metadata.get('competition'),
            "man_of_the_match": match_metadata.get('man_of_the_match'),
            "match_summary": match_metadata.get('match_summary'),
        },
        "momentum_analysis": momentum_shifts,
        "tactical_analysis": tactical_analysis,
        "deep_analysis": deep_analysis,
        "key_moments": key_moments,
        "error": None
    }


# =============================================================================
# Legacy Function (for backward compatibility - calls API)
# =============================================================================

def analyze_match(
    query: str,
    api_url: str = "http://localhost:8000",
    include_highlights: bool = True
) -> Dict[str, Any]:
    """
    Legacy function that calls the API.
    
    For chained usage, use analyze_match_from_web_results() instead.
    This function is kept for backward compatibility.
    """
    import requests
    
    try:
        response = requests.post(
            f"{api_url}/query",
            json={
                "query": query,
                "include_highlights": include_highlights
            },
            timeout=60
        )
        response.raise_for_status()
        api_result = response.json()
        
        if not api_result.get('success'):
            return {
                "success": False,
                "error": api_result.get('error', 'Unknown error'),
                "match_info": None,
                "momentum_analysis": [],
                "tactical_analysis": {},
                "deep_analysis": "",
                "key_moments": [],
            }
        
        # Extract data and use the chained function
        web_summary = api_result.get('summary', '')
        match_metadata = api_result.get('match_metadata', {})
        
        return analyze_match_from_web_results(web_summary, match_metadata, query)
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "match_info": None,
            "momentum_analysis": [],
            "tactical_analysis": {},
            "deep_analysis": "",
            "key_moments": [],
        }


# =============================================================================
# Formatted Output
# =============================================================================

def format_analysis(analysis: Dict[str, Any]) -> str:
    """
    Format the analysis result for display.
    
    Args:
        analysis: Analysis result from analyze_match().
        
    Returns:
        Formatted string.
    """
    if not analysis.get('success'):
        return f"❌ Error: {analysis.get('error', 'Unknown error')}"
    
    lines = []
    
    # Header
    match_info = analysis.get('match_info', {})
    home = match_info.get('home_team', 'Home')
    away = match_info.get('away_team', 'Away')
    score = match_info.get('score', '0-0')
    date = match_info.get('match_date', '')
    competition = match_info.get('competition', '')
    
    lines.append("")
    lines.append("╔" + "═" * 68 + "╗")
    lines.append("║" + " " * 20 + "📊 COMPREHENSIVE MATCH ANALYSIS" + " " * 18 + "║")
    lines.append("╠" + "═" * 68 + "╣")
    lines.append(f"║  {home} {score} {away}" + " " * (66 - len(f"{home} {score} {away}")) + "║")
    if date:
        lines.append(f"║  📅 {date}" + " " * (66 - len(f"📅 {date}")) + "║")
    if competition:
        lines.append(f"║  🏆 {competition}" + " " * (66 - len(f"🏆 {competition}")) + "║")
    lines.append("╠" + "═" * 68 + "╣")
    
    # Deep Analysis
    deep_analysis = analysis.get('deep_analysis', '')
    if deep_analysis:
        lines.append("║" + " " * 68 + "║")
        # Word wrap the analysis
        words = deep_analysis.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= 64:
                current_line += (" " if current_line else "") + word
            else:
                lines.append(f"║  {current_line}" + " " * (68 - len(current_line) - 2) + "║")
                current_line = word
        if current_line:
            lines.append(f"║  {current_line}" + " " * (68 - len(current_line) - 2) + "║")
        lines.append("║" + " " * 68 + "║")
    
    # Momentum Shifts
    momentum_shifts = analysis.get('momentum_analysis', [])
    significant = [m for m in momentum_shifts if m.get('momentum_impact') != 'neutral']
    if significant:
        lines.append("╠" + "═" * 68 + "╣")
        lines.append("║  ⚡ MOMENTUM SHIFTS" + " " * 48 + "║")
        lines.append("║" + " " * 68 + "║")
        for shift in significant[:5]:  # Top 5
            impact = shift.get('momentum_impact', 'neutral').upper()
            minute = shift.get('minute', '?')
            event = shift.get('event', 'EVENT')
            reasoning = shift.get('reasoning', '')
            
            moment_line = f"║  {minute}' - {event} ({impact} IMPACT)"
            lines.append(moment_line + " " * (68 - len(moment_line)) + "║")
            if reasoning:
                reason_line = f"║    → {reasoning[:62]}"
                lines.append(reason_line + " " * (68 - len(reason_line)) + "║")
        lines.append("║" + " " * 68 + "║")
    
    # Tactical Analysis Summary
    tactical = analysis.get('tactical_analysis', {})
    if tactical:
        lines.append("╠" + "═" * 68 + "╣")
        lines.append("║  🎯 TACTICAL INSIGHTS" + " " * 46 + "║")
        lines.append("║" + " " * 68 + "║")
        
        phases = tactical.get('match_phases', {})
        if phases:
            active_half = phases.get('more_active_half', '')
            lines.append(f"║  More active half: {active_half.title()}" + " " * (68 - len(f"More active half: {active_half.title()}") - 2) + "║")
        
        activity = tactical.get('team_activity', {})
        if activity:
            more_active = activity.get('more_active_team', '')
            lines.append(f"║  More active team: {more_active}" + " " * (68 - len(f"More active team: {more_active}") - 2) + "║")
        
        lines.append("║" + " " * 68 + "║")
    
    # Highlights
    highlights = analysis.get('highlights', [])
    if highlights:
        lines.append("╠" + "═" * 68 + "╣")
        lines.append("║  🎬 HIGHLIGHTS" + " " * 53 + "║")
        lines.append("║" + " " * 68 + "║")
        for i, video in enumerate(highlights[:3], 1):
            title = video.get('title', '')[:60]
            url = video.get('url', '')
            lines.append(f"║  {i}. {title}" + " " * (68 - len(f"{i}. {title}") - 2) + "║")
            if url:
                lines.append(f"║    🔗 {url[:62]}" + " " * (68 - len(f"🔗 {url[:62]}") - 2) + "║")
        lines.append("║" + " " * 68 + "║")
    
    lines.append("╚" + "═" * 68 + "╝")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# CLI Interface for Game Analyst
# =============================================================================

def main():
    """CLI interface for the game analyst."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ⚽ Game Analyst Agent ⚽                               ║
║                                                                          ║
║  Provides sophisticated match analysis including:                        ║
║  • Momentum shift analysis                                               ║
║  • Tactical breakdown                                                    ║
║  • Deep strategic insights                                               ║
║  • Match implications                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    while True:
        query = input("\n⚽ Enter match query (or 'q' to quit): ").strip()
        
        if not query or query.lower() in ('q', 'quit', 'exit'):
            print("\n👋 Goodbye!")
            break
        
        try:
            analysis = analyze_match(query)
            formatted = format_analysis(analysis)
            print(formatted)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Returning to main menu...")
            continue
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()

