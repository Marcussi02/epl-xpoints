"""Expected-points model: scoring arithmetic, minutes, doubles, blanks and flags."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from epl.fpl import xpoints


def rows(**cols):
    base = {"position": "MID", "minutes": 90, "goals": 0, "assists": 0, "clean_sheets": 0,
            "goals_conceded": 0}
    n = max(len(v) if isinstance(v, list) else 1 for v in cols.values())
    return pd.DataFrame({k: cols.get(k, v) if isinstance(cols.get(k, v), list)
                         else [cols.get(k, v)] * n for k, v in base.items()})


def test_explained_points_follow_fpl_rules():
    df = rows(position=["FWD", "DEF", "GK", "MID"], minutes=[90, 90, 45, 0],
              goals=[2, 1, 0, 0], assists=[1, 0, 0, 0], clean_sheets=[0, 1, 0, 0],
              goals_conceded=[3, 0, 5, 0])
    # FWD: 2 + 2*4 + 3 = 13 | DEF: 2 + 6 + 4 = 12 | GK sub: 1 - floor(5/2) = -1 | unused: 0
    assert xpoints.explained_points(df).tolist() == [13, 12, -1, 0]


def test_expected_conceded_penalty_matches_direct_sum():
    mu = np.array([0.5, 1.4, 3.0])
    c = np.arange(30)
    direct = [(poisson.pmf(c, m) * (c // 2)).sum() for m in mu]
    assert xpoints.expected_conceded_penalty(mu) == pytest.approx(direct, rel=1e-6)


def profile(position="MID", **kw):
    base = {"name": "P", "position": position, "team": "A", "p_play": 1.0, "p60": 1.0,
            "exp_minutes": 90.0, "goal_share": 0.2, "assist_share": 0.1, "resid90": 0.5}
    base.update(kw)
    return pd.DataFrame([base], index=pd.Index([7], name="element"))


def rates(gws=(1,), lam=1.5, mu=1.0, p_cs=0.3):
    return pd.DataFrame({"gw": list(gws), "team": "A", "opponent": "B", "was_home": True,
                         "lam": lam, "mu": mu, "p_cs": p_cs})


def test_expected_points_arithmetic_for_a_midfielder():
    xp = xpoints.expected_points(profile(), rates()).iloc[0]
    # 2 (appearance) + 1.5*0.2*5 + 1.5*0.1*3 + 0.3*1 + 0.5 = 2 + 1.5 + 0.45 + 0.3 + 0.5
    assert xp.xp == pytest.approx(4.75)


def test_defenders_lose_points_for_expected_goals_conceded():
    mid = xpoints.expected_points(profile("MID", goal_share=0, assist_share=0, resid90=0),
                                  rates(p_cs=0)).iloc[0].xp
    dfn = xpoints.expected_points(profile("DEF", goal_share=0, assist_share=0, resid90=0),
                                  rates(p_cs=0)).iloc[0].xp
    assert mid - dfn == pytest.approx(xpoints.expected_conceded_penalty(np.array([1.0]))[0])


def test_double_gameweek_sums_and_blank_scores_nothing():
    single = xpoints.expected_points(profile(), rates((1,))).iloc[0].xp
    double = xpoints.expected_points(profile(), rates((1, 1)))
    assert double.iloc[0].xp == pytest.approx(2 * single) and double.iloc[0].fixtures == 2
    blank = xpoints.expected_points(profile(team="Z"), rates())
    assert blank.empty


def test_injury_flag_scales_minutes():
    flagged = xpoints.expected_points(profile(), rates(), availability=pd.Series({7: 0.0}))
    assert flagged.iloc[0].xp == pytest.approx(0.0)
    half = xpoints.expected_points(profile(), rates(), availability=pd.Series({7: 0.5}))
    assert half.iloc[0].xp == pytest.approx(4.75 / 2)


def test_profiles_use_only_history_and_shrink_newcomers():
    k = pd.Timestamp("2024-08-17")
    history = pd.DataFrame({
        "element": [1, 1, 1, 2], "name": ["Star", "Star", "Star", "New"],
        "position": ["FWD"] * 4, "team": ["A", "A", "A", "A"],
        "kickoff": [k, k + pd.Timedelta(weeks=1), k + pd.Timedelta(weeks=2),
                    k + pd.Timedelta(weeks=2)],
        "minutes": [90, 90, 90, 10], "total_points": [8, 2, 9, 1], "goals": [1, 0, 1, 0],
        "assists": [0, 0, 1, 0], "clean_sheets": [0, 0, 0, 0], "goals_conceded": [1, 1, 0, 0],
        "xg": [0.9, 0.2, 1.1, 0.0], "xa": [0.1, 0.0, 0.6, 0.0],
    })
    prof = xpoints.player_profiles(history)
    assert prof.loc[1, "p60"] == 1.0 and prof.loc[2, "p60"] == 0.0
    # The regular starter's share is pulled up by his output; the newcomer stays near prior.
    assert prof.loc[1, "goal_share"] > prof.loc[2, "goal_share"]
    assert prof.loc[2, "goal_share"] == pytest.approx(xpoints.PRIOR_GOAL_SHARE["FWD"], abs=0.05)
