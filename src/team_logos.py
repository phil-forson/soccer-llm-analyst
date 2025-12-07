"""
Team Logo Service - Provides logo URLs for football teams.

Uses multiple logo sources for reliability.
"""

from typing import Optional, Dict

# Team name to logo URL mappings
# Using reliable logo services
TEAM_LOGO_MAP: Dict[str, str] = {
    # Premier League
    "Arsenal": "https://logos-world.net/wp-content/uploads/2020/06/Arsenal-Logo.png",
    "Chelsea": "https://logos-world.net/wp-content/uploads/2020/06/Chelsea-Logo.png",
    "Liverpool": "https://logos-world.net/wp-content/uploads/2020/06/Liverpool-Logo.png",
    "Manchester United": "https://logos-world.net/wp-content/uploads/2020/06/Manchester-United-Logo.png",
    "Manchester City": "https://logos-world.net/wp-content/uploads/2020/06/Manchester-City-Logo.png",
    "Tottenham": "https://logos-world.net/wp-content/uploads/2020/06/Tottenham-Hotspur-Logo.png",
    "Tottenham Hotspur": "https://logos-world.net/wp-content/uploads/2020/06/Tottenham-Hotspur-Logo.png",
    "Newcastle United": "https://logos-world.net/wp-content/uploads/2020/06/Newcastle-United-Logo.png",
    "Aston Villa": "https://logos-world.net/wp-content/uploads/2020/06/Aston-Villa-Logo.png",
    "West Ham United": "https://logos-world.net/wp-content/uploads/2020/06/West-Ham-United-Logo.png",
    "Brighton": "https://logos-world.net/wp-content/uploads/2020/06/Brighton-Hove-Albion-Logo.png",
    "Brighton & Hove Albion": "https://logos-world.net/wp-content/uploads/2020/06/Brighton-Hove-Albion-Logo.png",
    "Wolves": "https://logos-world.net/wp-content/uploads/2020/06/Wolverhampton-Wanderers-Logo.png",
    "Wolverhampton Wanderers": "https://logos-world.net/wp-content/uploads/2020/06/Wolverhampton-Wanderers-Logo.png",
    "Bournemouth": "https://logos-world.net/wp-content/uploads/2020/06/Bournemouth-Logo.png",
    "Fulham": "https://logos-world.net/wp-content/uploads/2020/06/Fulham-Logo.png",
    "Crystal Palace": "https://logos-world.net/wp-content/uploads/2020/06/Crystal-Palace-Logo.png",
    "Brentford": "https://logos-world.net/wp-content/uploads/2020/06/Brentford-Logo.png",
    "Everton": "https://logos-world.net/wp-content/uploads/2020/06/Everton-Logo.png",
    "Nottingham Forest": "https://logos-world.net/wp-content/uploads/2020/06/Nottingham-Forest-Logo.png",
    "Ipswich": "https://logos-world.net/wp-content/uploads/2020/06/Ipswich-Town-Logo.png",
    "Ipswich Town": "https://logos-world.net/wp-content/uploads/2020/06/Ipswich-Town-Logo.png",
    "Leicester City": "https://logos-world.net/wp-content/uploads/2020/06/Leicester-City-Logo.png",
    "Leicester": "https://logos-world.net/wp-content/uploads/2020/06/Leicester-City-Logo.png",
    "Southampton": "https://logos-world.net/wp-content/uploads/2020/06/Southampton-Logo.png",
    "Sunderland": "https://logos-world.net/wp-content/uploads/2020/06/Sunderland-Logo.png",
    "Leeds United": "https://logos-world.net/wp-content/uploads/2020/06/Leeds-United-Logo.png",
    
    # La Liga
    "Barcelona": "https://logos-world.net/wp-content/uploads/2020/06/Barcelona-Logo.png",
    "FC Barcelona": "https://logos-world.net/wp-content/uploads/2020/06/Barcelona-Logo.png",
    "Real Madrid": "https://logos-world.net/wp-content/uploads/2020/06/Real-Madrid-Logo.png",
    "Atletico Madrid": "https://logos-world.net/wp-content/uploads/2020/06/Atletico-Madrid-Logo.png",
    "Atlético Madrid": "https://logos-world.net/wp-content/uploads/2020/06/Atletico-Madrid-Logo.png",
    "Sevilla": "https://logos-world.net/wp-content/uploads/2020/06/Sevilla-Logo.png",
    "Valencia": "https://logos-world.net/wp-content/uploads/2020/06/Valencia-Logo.png",
    "Villarreal": "https://logos-world.net/wp-content/uploads/2020/06/Villarreal-Logo.png",
    
    # Serie A
    "Juventus": "https://logos-world.net/wp-content/uploads/2020/06/Juventus-Logo.png",
    "AC Milan": "https://logos-world.net/wp-content/uploads/2020/06/AC-Milan-Logo.png",
    "Inter Milan": "https://logos-world.net/wp-content/uploads/2020/06/Inter-Milan-Logo.png",
    "Inter": "https://logos-world.net/wp-content/uploads/2020/06/Inter-Milan-Logo.png",
    "Roma": "https://logos-world.net/wp-content/uploads/2020/06/AS-Roma-Logo.png",
    "Napoli": "https://logos-world.net/wp-content/uploads/2020/06/Napoli-Logo.png",
    
    # Bundesliga
    "Bayern Munich": "https://logos-world.net/wp-content/uploads/2020/06/Bayern-Munich-Logo.png",
    "Bayern": "https://logos-world.net/wp-content/uploads/2020/06/Bayern-Munich-Logo.png",
    "Borussia Dortmund": "https://logos-world.net/wp-content/uploads/2020/06/Borussia-Dortmund-Logo.png",
    "Dortmund": "https://logos-world.net/wp-content/uploads/2020/06/Borussia-Dortmund-Logo.png",
    "RB Leipzig": "https://logos-world.net/wp-content/uploads/2020/06/RB-Leipzig-Logo.png",
    
    # Ligue 1
    "PSG": "https://logos-world.net/wp-content/uploads/2020/06/Paris-Saint-Germain-Logo.png",
    "Paris Saint-Germain": "https://logos-world.net/wp-content/uploads/2020/06/Paris-Saint-Germain-Logo.png",
    "Lyon": "https://logos-world.net/wp-content/uploads/2020/06/Lyon-Logo.png",
    "Marseille": "https://logos-world.net/wp-content/uploads/2020/06/Marseille-Logo.png",
    
    # Champions League / Other
    "Porto": "https://logos-world.net/wp-content/uploads/2020/06/Porto-Logo.png",
    "Benfica": "https://logos-world.net/wp-content/uploads/2020/06/Benfica-Logo.png",
}


def get_team_logo_url(team_name: Optional[str]) -> Optional[str]:
    """
    Get team logo URL.
    
    Args:
        team_name: Full team name
        
    Returns:
        Logo URL or None if not found
    """
    if not team_name:
        return None
    
    # Try exact match first
    if team_name in TEAM_LOGO_MAP:
        return TEAM_LOGO_MAP[team_name]
    
    # Try case-insensitive match
    team_lower = team_name.lower()
    for key, url in TEAM_LOGO_MAP.items():
        if key.lower() == team_lower:
            return url
    
    # Try partial match (e.g., "Manchester United" matches "Man United")
    for key, url in TEAM_LOGO_MAP.items():
        key_lower = key.lower()
        if team_lower in key_lower or key_lower in team_lower:
            return url
    
    # Try removing common suffixes
    team_clean = team_name
    for suffix in [" FC", " AFC", " Football Club", " Association Football Club"]:
        if team_clean.endswith(suffix):
            team_clean = team_clean[:-len(suffix)].strip()
            break
    
    if team_clean in TEAM_LOGO_MAP:
        return TEAM_LOGO_MAP[team_clean]
    
    # Try case-insensitive match on cleaned name
    team_clean_lower = team_clean.lower()
    for key, url in TEAM_LOGO_MAP.items():
        if key.lower() == team_clean_lower:
            return url
    
    # Fallback: try to construct a URL (may not work for all teams)
    # Using a generic logo service
    team_slug = team_clean.lower().replace(" ", "-").replace("&", "and")
    return f"https://logo.clearbit.com/{team_slug.replace('-', '')}.com"

