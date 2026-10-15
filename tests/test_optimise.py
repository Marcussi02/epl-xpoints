"""Optimiser: every plan obeys the FPL rules (property-tested), transfers behave sensibly."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from epl.fpl import optimise

POSITION_MIX = ["GK"] * 3 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5


def pool(seed: int, teams: int = 20, horizon: int = 1):
    rng = np.random.default_rng(seed)
    rows, xp = [], []
    element = 0
    for t in range(teams):
        for pos in POSITION_MIX:
            element += 1
            base = {"GK": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}[pos]
            quality = rng.gamma(2.0, 1.0)
            rows.append({"element": element, "position": pos, "team": f"Club{t}",
                         "price": round(base + quality * 1.5 + rng.normal(0, 0.3), 1)})
            xp.append(np.clip(quality * 1.8 + rng.normal(0, 0.8, horizon), 0, None))
    players = pd.DataFrame(rows).set_index("element")
    players["price"] = players["price"].clip(lower=3.9)
    xp = pd.DataFrame(np.array(xp), index=players.index,
                      columns=[f"gw{k}" for k in range(horizon)])
    return players, xp


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), budget=st.sampled_from([83.0, 90.0, 100.0]),
       teams=st.integers(6, 20))
def test_squad_plans_always_obey_the_rules(seed, budget, teams):
    players, xp = pool(seed, teams)
    plan = optimise.plan(players, xp, budget=budget)
    assert optimise.validate_plan(plan, players, budget) == []
    # captain's points count twice
    scored = xp.loc[plan.xi, "gw0"].sum() + xp.loc[plan.captain, "gw0"]
    assert plan.expected_points == pytest.approx(scored)
    assert plan.captain == xp.loc[plan.xi, "gw0"].idxmax()


@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), free=st.integers(1, 3))
def test_transfer_plans_obey_rules_and_budget(seed, free):
    players, xp = pool(seed, horizon=3)
    start = optimise.plan(players, xp[["gw0"]], budget=100.0)
    rng = np.random.default_rng(seed)
    shocked = xp.copy()
    shocked.loc[rng.choice(start.xi, 3, replace=False)] = 0.0  # three starters get injured
    plan = optimise.plan(players, shocked, squad=start.squad, bank=start.bank,
                         free_transfers=free)
    assert optimise.validate_plan(plan, players) == []
    assert len(plan.buy) == len(plan.sell)
    assert set(plan.squad) == (set(start.squad) - set(plan.sell)) | set(plan.buy)
    assert plan.bank >= -1e-9
    assert plan.hits == max(0, len(plan.buy) - free)


def test_planner_makes_an_obvious_transfer_and_avoids_pointless_hits():
    players, xp = pool(1, horizon=2)
    start = optimise.plan(players, xp, budget=100.0)
    # A cheap midfielder outside the squad suddenly projects 15 points a week.
    outsider = players.index.difference(start.squad)
    star = players.loc[outsider][players.loc[outsider, "position"] == "MID"]["price"].idxmin()
    boosted = xp.copy()
    boosted.loc[star] = 15.0
    plan = optimise.plan(players, boosted, squad=start.squad, bank=start.bank, free_transfers=1)
    assert plan.buy == [star] and plan.hits == 0 and plan.captain == star
    # With nothing to gain, it rolls the free transfer instead of spending it.
    hold = optimise.plan(players, xp, squad=start.squad, bank=start.bank, free_transfers=1)
    assert hold.buy == [] and hold.hits == 0


def test_selling_price_limits_what_can_be_afforded():
    players, xp = pool(2)
    start = optimise.plan(players, xp, budget=100.0)
    # Selling everyone for 0.1 leaves no money: the squad cannot change.
    stuck = optimise.plan(players, xp * 0 + 1, squad=start.squad, bank=0.0,
                          sell_prices={e: 0.1 for e in start.squad}, free_transfers=5)
    assert stuck.buy == []


# ---------- season replay mechanics ----------


def test_selling_price_follows_fpl_rule():
    from epl.fpl.season import selling_price

    assert selling_price(5.0, 5.3) == 5.1  # half of 0.3 rounds down to 0.1
    assert selling_price(5.0, 5.4) == 5.2
    assert selling_price(5.0, 4.7) == 4.7  # falls are passed on in full


def test_auto_substitution_keeps_a_valid_formation():
    from epl.fpl.season import auto_substitute

    pos = {1: "GK", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID",
           9: "FWD", 10: "FWD", 11: "MID", 12: "GK", 13: "DEF", 14: "FWD", 15: "MID"}
    xi = list(range(1, 12))
    bench = [14, 15, 13, 12]
    # A defender misses out: the first bench player (a forward) would leave only two
    # defenders, so the planner skips him and brings on the bench defender.
    minutes = {e: 90 for e in pos} | {2: 0}
    final = auto_substitute(xi, bench, minutes, pos)
    assert 2 not in final and 13 in final and 14 not in final
    # The keeper misses out: only the reserve keeper may replace him.
    minutes = {e: 90 for e in pos} | {1: 0}
    assert 12 in auto_substitute(xi, bench, minutes, pos)
