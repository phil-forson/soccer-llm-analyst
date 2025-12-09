"""
Game Analyst Agent - Provides sophisticated match analysis.

This agent receives web search results and generates deep tactical, 
momentum, and strategic analysis. Part of the agent chain:
Query Parser → Web Search (RAG) → Game Analyst
"""

import re
from typing import Optional, Dict, List, Any

from .utils import get_openai_client
from .config import DEFAULT_LLM_MODEL


# =============================================================================
# Momentum Analysis
# =============================================================================

def _analyze_momentum_shifts(key_moments: List[Dict]) -> List[Dict]:
    """Analyze momentum shifts in the match based on key moments."""
    if not key_moments:
        return []
    
    momentum_shifts = []
    
    # Sort by minute
    sorted_moments = sorted(
        [m for m in key_moments if m.get('minute')],
        key=lambda x: int(x.get('minute', '0').replace("'", "").split("+")[0] or 0)
    )
    
    for moment in sorted_moments:
        shift = {
            "minute": moment.get('minute', '?'),
            "event": moment.get('event', ''),
            "description": moment.get('description', ''),
            "team": moment.get('team', ''),
            "momentum_impact": "neutral",
            "reasoning": ""
        }
        
        event = moment.get('event', '').upper()
        
        if 'GOAL' in event:
            minute = int(moment.get('minute', '0').replace("'", "").split("+")[0] or 0)
            if minute <= 30:
                shift["momentum_impact"] = "high"
                shift["reasoning"] = "Early goal sets the tone and puts pressure on the opponent"
            elif minute >= 75:
                shift["momentum_impact"] = "critical"
                shift["reasoning"] = "Late goal can be decisive and demoralizing"
            else:
                shift["momentum_impact"] = "high"
                shift["reasoning"] = "Goal shifts momentum significantly"
        elif 'RED_CARD' in event:
            shift["momentum_impact"] = "critical"
            shift["reasoning"] = "Red card creates numerical advantage and tactical shift"
        elif 'PENALTY' in event and 'missed' in moment.get('description', '').lower():
            shift["momentum_impact"] = "high"
            shift["reasoning"] = "Missed penalty can shift momentum to the defending team"
        
        momentum_shifts.append(shift)
    
    return momentum_shifts


# =============================================================================
# Tactical Analysis
# =============================================================================

def _analyze_tactical_patterns(key_moments: List[Dict], match_metadata: Dict) -> Dict[str, Any]:
    """Analyze tactical patterns from key moments."""
    if not key_moments:
        return {}
    
    home_team = match_metadata.get('home_team', 'Home')
    away_team = match_metadata.get('away_team', 'Away')
    score = match_metadata.get('score', '0-0')
    
    try:
        home_score, away_score = map(int, score.split('-'))
    except (ValueError, AttributeError):
        home_score, away_score = 0, 0
    
    home_events = [m for m in key_moments if m.get('team') == 'home']
    away_events = [m for m in key_moments if m.get('team') == 'away']
    
    home_goals = len([m for m in home_events if 'GOAL' in m.get('event', '').upper()])
    away_goals = len([m for m in away_events if 'GOAL' in m.get('event', '').upper()])
    
    first_half = [m for m in key_moments if int(m.get('minute', '0').replace("'", "").split("+")[0] or 0) <= 45]
    second_half = [m for m in key_moments if int(m.get('minute', '0').replace("'", "").split("+")[0] or 0) > 45]
    
    return {
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


# =============================================================================
# Deep Analysis with LLM
# =============================================================================

def _generate_deep_analysis(
    match_metadata: Dict,
    key_moments: List[Dict],
    momentum_shifts: List[Dict],
    tactical_analysis: Dict,
    web_summary: str
) -> str:
    """Use LLM to generate sophisticated match analysis."""
    client = get_openai_client()
    
    home_team = match_metadata.get('home_team', 'Home')
    away_team = match_metadata.get('away_team', 'Away')
    score = match_metadata.get('score', '0-0')
    match_date = match_metadata.get('match_date', '')
    competition = match_metadata.get('competition', '')
    man_of_match = match_metadata.get('man_of_the_match', '')
    
    moments_text = "\n".join([
        f"{m.get('minute', '?')}' - {m.get('event', 'EVENT')}: {m.get('description', '')} ({m.get('team', 'unknown')} team)"
        for m in key_moments
    ])
    
    momentum_text = "\n".join([
        f"{m['minute']}' - {m['event']}: {m['momentum_impact'].upper()} impact - {m['reasoning']}"
        for m in momentum_shifts if m.get('momentum_impact') != 'neutral'
    ])
    
    prompt = f"""You are an elite football analyst providing deep tactical and strategic analysis.

=== MATCH INFORMATION ===
Teams: {home_team} vs {away_team}
Score: {score}
Date: {match_date}
Competition: {competition}
Man of the Match: {man_of_match or 'Not specified'}

=== KEY MOMENTS ===
{moments_text if moments_text else "No key moments recorded"}

=== MOMENTUM SHIFTS ===
{momentum_text if momentum_text else "No significant momentum shifts identified"}

=== TACTICAL ANALYSIS ===
Match Phases: {tactical_analysis.get('match_phases', {})}
Goal Distribution: {tactical_analysis.get('goal_distribution', {})}
Team Activity: {tactical_analysis.get('team_activity', {})}

=== WEB SUMMARY ===
{web_summary[:1000]}

Provide comprehensive match analysis covering:
1. **Match Narrative** - How the match unfolded
2. **Momentum Analysis** - When and why momentum shifted
3. **Tactical Breakdown** - Key tactical decisions and patterns
4. **Key Performances** - Standout players
5. **Implications** - What this result means

Write in an engaging, analytical style. Be specific, cite moments by minute."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite football analyst. Provide sophisticated, detailed match analysis."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


# =============================================================================
# Main Analysis Function
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
    """
    print(f"\n{'='*70}")
    print(f"[GameAnalyst] STARTING COMPREHENSIVE MATCH ANALYSIS")
    print(f"{'='*70}")
    
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
    
    print(f"[GameAnalyst] Match: {match_metadata.get('home_team')} {match_metadata.get('score')} {match_metadata.get('away_team')}")
    print(f"[GameAnalyst] Found {len(key_moments)} key moments")
    
    # Step 1: Momentum analysis
    print(f"[GameAnalyst] Analyzing momentum shifts...")
    momentum_shifts = _analyze_momentum_shifts(key_moments)
    significant_shifts = [m for m in momentum_shifts if m.get('momentum_impact') != 'neutral']
    print(f"[GameAnalyst] Identified {len(significant_shifts)} significant momentum shifts")
    
    # Step 2: Tactical analysis
    print(f"[GameAnalyst] Analyzing tactical patterns...")
    tactical_analysis = _analyze_tactical_patterns(key_moments, match_metadata)
    
    # Step 3: Deep LLM analysis
    print(f"[GameAnalyst] Generating comprehensive analysis...")
    deep_analysis = _generate_deep_analysis(
        match_metadata,
        key_moments,
        momentum_shifts,
        tactical_analysis,
        web_summary
    )
    
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
# Legacy Function (backward compatibility)
# =============================================================================

def analyze_match(
    query: str,
    api_url: str = "http://localhost:8000",
    include_highlights: bool = True
) -> Dict[str, Any]:
    """
    Legacy function that calls the API.
    For chained usage, use analyze_match_from_web_results() instead.
    """
    import requests
    
    try:
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "include_highlights": include_highlights},
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
    """Format the analysis result for display."""
    if not analysis.get('success'):
        return f"❌ Error: {analysis.get('error', 'Unknown error')}"
    
    lines = []
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
    lines.append(f"║  {home} {score} {away}" + " " * max(0, 66 - len(f"{home} {score} {away}")) + "║")
    if date:
        lines.append(f"║  📅 {date}" + " " * max(0, 66 - len(f"📅 {date}")) + "║")
    if competition:
        lines.append(f"║  🏆 {competition}" + " " * max(0, 66 - len(f"🏆 {competition}")) + "║")
    lines.append("╚" + "═" * 68 + "╝")
    
    deep_analysis = analysis.get('deep_analysis', '')
    if deep_analysis:
        lines.append("")
        lines.append(deep_analysis)
    
    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for the game analyst."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ⚽ Game Analyst Agent ⚽                               ║
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
            print("\n\n⚠️  Interrupted.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
