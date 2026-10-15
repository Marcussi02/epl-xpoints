"""One canonical name per club, whatever the source calls it.

football-data.co.uk names are the canonical form (short and stable since the 1990s).
Every other spelling maps onto them here, and an unknown name raises instead of
silently creating a "new" club: a typo would otherwise split one team's history in two.
"""

# Alternative spelling -> canonical (football-data.co.uk) name.
ALIASES: dict[str, str] = {
    # Fantasy Premier League API short names
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
    # Full names used by other sources
    "Arsenal FC": "Arsenal",
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Huddersfield Town": "Huddersfield",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
}

# Every club that has played in the Premier League since 2000-01.
CANONICAL: frozenset[str] = frozenset({
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham", "Blackburn", "Blackpool", "Bolton",
    "Bournemouth", "Bradford", "Brentford", "Brighton", "Burnley", "Cardiff", "Charlton",
    "Chelsea", "Coventry", "Crystal Palace", "Derby", "Everton", "Fulham", "Huddersfield",
    "Hull", "Ipswich", "Leeds", "Leicester", "Liverpool", "Luton", "Man City", "Man United",
    "Middlesbrough", "Newcastle", "Norwich", "Nott'm Forest", "Portsmouth", "QPR", "Reading",
    "Sheffield United", "Southampton", "Stoke", "Sunderland", "Swansea", "Tottenham", "Watford",
    "West Brom", "West Ham", "Wigan", "Wolves",
})


class UnknownTeamError(KeyError):
    """A team name that is neither canonical nor a known alias."""


def canonical(name: str) -> str:
    """Canonical club name for any known spelling. Raises UnknownTeamError otherwise."""
    cleaned = " ".join(str(name).split())
    if cleaned in CANONICAL:
        return cleaned
    if cleaned in ALIASES:
        return ALIASES[cleaned]
    raise UnknownTeamError(f"unknown team name {name!r}; add it to epl/data/teams.py")
