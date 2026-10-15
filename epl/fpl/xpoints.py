"""Expected FPL points for every player in an upcoming gameweek.

Team level comes from Dixon-Coles: each fixture's expected goals for (lam) and
against (mu), and the clean-sheet probability. Player level comes from the player's
own recent rows, using only gameweeks before the one being predicted:

    minutes     p_play, p60 and expected minutes from recent appearances, scaled by the
                injury flag (chance of playing) when one is known
    goals       E[goals] = lam * goal_share_per90 * E[minutes]/90, where the share is the
                player's fraction of his team's xG while on the pitch (goals before
                2022-23), shrunk towards his position's typical share
    assists     the same with xA
    clean sheet p60 * P(clean sheet) * points for his position
    conceded    GK and DEF lose a point per two goals conceded: p60 * E[floor(C/2)], C ~ Pois(mu)
    the rest    bonus, saves, cards, penalties and defensive contributions are not modelled
                one by one: a per-90 residual (actual points minus the points explained by
                the items above) carries them, shrunk towards the position's residual

A double gameweek sums the player's fixtures; a blank gameweek scores zero.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..models.dixon_coles import DixonColesModel

POSITIONS = ("GK", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class Rules:
    goal: dict = field(default_factory=lambda: {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4})
    assist: int = 3
    clean_sheet: dict = field(default_factory=lambda: {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0})


RULES = Rules()

# Typical share of a team's xG / xA per 90 minutes by position.
PRIOR_GOAL_SHARE = {"GK": 0.0, "DEF": 0.04, "MID": 0.10, "FWD": 0.26}
PRIOR_ASSIST_SHARE = {"GK": 0.01, "DEF": 0.06, "MID": 0.12, "FWD": 0.11}


@dataclass(frozen=True)
class XPConfig:
    """Shrinkage strengths are pseudo-counts in matches: how much of a player's own
    evidence it takes to move away from his position's typical value."""

    share_strength: float = 3.0
    residual_strength: float = 4.0
    minutes_window: int = 4  # 4 beat 6 and 10 on the 2023-24 validation season
    share_window: int = 20


CONFIG = XPConfig()


def explained_points(rows: pd.DataFrame, rules: Rules = RULES) -> pd.Series:
    """Points from appearance, goals, assists, clean sheets and goals conceded."""
    pos = rows["position"]
    pts = (rows["minutes"] > 0).astype(int) + (rows["minutes"] >= 60).astype(int)
    pts += rows["goals"] * pos.map(rules.goal) + rows["assists"] * rules.assist
    pts += rows["clean_sheets"] * pos.map(rules.clean_sheet)
    pts -= np.where(pos.isin(["GK", "DEF"]), rows["goals_conceded"] // 2, 0)
    return pts


def player_profiles(history: pd.DataFrame, rules: Rules = RULES,
                    config: XPConfig = CONFIG) -> pd.DataFrame:
    """One row per player: minutes expectations, attacking shares and residual rate.
    `history` holds player-gameweek rows from before the gameweek being predicted."""
    h = history.sort_values("kickoff").copy()
    use_xg = h["xg"].notna()
    h["g_stat"] = np.where(use_xg, h["xg"], h["goals"])
    h["a_stat"] = np.where(h["xa"].notna(), h["xa"], h["assists"])
    team = h.groupby(["team", "kickoff"])[["g_stat", "a_stat"]].transform("sum")
    h["team_g"], h["team_a"] = team["g_stat"], team["a_stat"]
    h["resid"] = h["total_points"] - explained_points(h, rules)

    recent = h.groupby("element").tail(config.minutes_window)
    minutes = recent.groupby("element").agg(
        p_play=("minutes", lambda m: (m > 0).mean()),
        p60=("minutes", lambda m: (m >= 60).mean()),
        exp_minutes=("minutes", "mean"),
    )

    played = h[h["minutes"] > 0].groupby("element").tail(config.share_window).copy()
    played["n90"] = played["minutes"] / 90
    played["team_g90"] = played["team_g"] * played["n90"]
    played["team_a90"] = played["team_a"] * played["n90"]
    sums = played.groupby("element")[["g_stat", "a_stat", "team_g90", "team_a90", "n90",
                                      "resid"]].sum()
    last = h.groupby("element").tail(1).set_index("element")[["name", "position", "team"]]
    prof = last.join(minutes).join(sums).fillna(0.0)

    pos = prof["position"]
    team_g_per_match = max(h.groupby(["team", "kickoff"])["g_stat"].sum().mean(), 0.5)
    k_g = config.share_strength * team_g_per_match
    prof["goal_share"] = (prof["g_stat"] + k_g * pos.map(PRIOR_GOAL_SHARE)) / (
        prof["team_g90"] + k_g)
    prof["assist_share"] = (prof["a_stat"] + k_g * pos.map(PRIOR_ASSIST_SHARE)) / (
        prof["team_a90"] + k_g)
    pos_resid = (prof.groupby("position")["resid"].sum()
                 / prof.groupby("position")["n90"].sum().clip(lower=1))
    k_r = config.residual_strength
    prof["resid90"] = (prof["resid"] + k_r * pos.map(pos_resid)) / (prof["n90"] + k_r)
    return prof[["name", "position", "team", "p_play", "p60", "exp_minutes", "goal_share",
                 "assist_share", "resid90"]]


def team_fixture_rates(model: DixonColesModel, fixtures: pd.DataFrame) -> pd.DataFrame:
    """fixtures (gw, home, away) -> one row per team per fixture with lam, mu, P(clean sheet)."""
    lam, mu = model.rates(fixtures)
    cs_home, cs_away = model.clean_sheet_probs(fixtures)
    home = pd.DataFrame({"gw": fixtures["gw"].to_numpy(), "team": fixtures["home"].to_numpy(),
                         "opponent": fixtures["away"].to_numpy(), "was_home": True,
                         "lam": lam, "mu": mu, "p_cs": cs_home})
    away = pd.DataFrame({"gw": fixtures["gw"].to_numpy(), "team": fixtures["away"].to_numpy(),
                         "opponent": fixtures["home"].to_numpy(), "was_home": False,
                         "lam": mu, "mu": lam, "p_cs": cs_away})
    return pd.concat([home, away], ignore_index=True)


def expected_conceded_penalty(mu: np.ndarray, max_goals: int = 15) -> np.ndarray:
    """E[floor(C / 2)] for C ~ Poisson(mu)."""
    c = np.arange(max_goals + 1)
    return (poisson.pmf(c[None, :], np.asarray(mu)[:, None]) * (c // 2)).sum(axis=1)


def expected_points(profiles: pd.DataFrame, team_rates: pd.DataFrame,
                    availability: pd.Series | None = None, rules: Rules = RULES) -> pd.DataFrame:
    """Expected points per player per gameweek. `availability` (0-1 per element, e.g. the
    FPL injury flag) scales the minutes expectations; missing means fully available."""
    p = profiles.copy()
    if availability is not None:
        avail = availability.reindex(p.index).fillna(1.0).clip(0, 1)
        p[["p_play", "p60", "exp_minutes"]] = p[["p_play", "p60", "exp_minutes"]].mul(avail,
                                                                                      axis=0)
    rows = p.reset_index().merge(team_rates, on="team", how="inner")
    frac = rows["exp_minutes"] / 90
    pos = rows["position"]
    xp = rows["p_play"] + rows["p60"]
    xp += frac * rows["lam"] * rows["goal_share"] * pos.map(rules.goal)
    xp += frac * rows["lam"] * rows["assist_share"] * rules.assist
    xp += rows["p60"] * rows["p_cs"] * pos.map(rules.clean_sheet)
    xp -= np.where(pos.isin(["GK", "DEF"]), rows["p60"] * expected_conceded_penalty(rows["mu"]),
                   0.0)
    xp += frac * rows["resid90"]
    rows["xp"] = xp
    return (rows.groupby(["element", "gw"], as_index=False)
            .agg(name=("name", "first"), position=("position", "first"),
                 team=("team", "first"), xp=("xp", "sum"), fixtures=("opponent", "size")))
