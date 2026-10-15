"""Fantasy Premier League squad selection and transfer planning as integer programs.

Every decision is a 0/1 variable per player (in squad, in starting XI, captain, bought,
sold), solved exactly by HiGHS through scipy.optimize.milp. The rules are constraints:

    squad      15 players: 2 GK, 5 DEF, 5 MID, 3 FWD, at most 3 from one club, within budget
    XI         11 starters: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, all from the squad
    captain    one starter, whose points count twice

plan() looks `horizon` gameweeks ahead and maximises the discounted expected points
of the XI plus captain, minus 4 points for every transfer beyond the free ones. Free
transfers roll over up to `max_free` (one more each gameweek, minus those used), so the
planner can choose to bank a transfer. Only the first gameweek's moves are acted on:
next week it plans again with new information (a rolling horizon).

Bench players earn a small weight (bench_weight x their expected points), which keeps
cover sensible when starters do not play.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
HIT_COST = 4.0
# The planner charges itself 8 per hit, not 4: chosen on the 2023-24 season replay, where
# it cut hits from 37 to 15 and raised the season total (see REPORT.md).
HIT_PENALTY = 8.0


@dataclass
class Plan:
    squad: list[int]
    xi: list[int]
    captain: int
    vice: int
    bench: list[int]  # outfield bench in priority order, then the reserve keeper
    buy: list[int] = field(default_factory=list)
    sell: list[int] = field(default_factory=list)
    hits: int = 0
    expected_points: float = 0.0  # first gameweek, after any hit cost
    bank: float = 0.0


class _Builder:
    """Collects sparse constraint rows for milp."""

    def __init__(self, n_vars: int):
        self.n = n_vars
        self.rows: list[tuple[dict[int, float], float, float]] = []

    def add(self, coefs: dict[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        self.rows.append((coefs, lb, ub))

    def constraint(self) -> LinearConstraint:
        a = lil_matrix((len(self.rows), self.n))
        lb, ub = np.empty(len(self.rows)), np.empty(len(self.rows))
        for r, (coefs, lo, hi) in enumerate(self.rows):
            for j, v in coefs.items():
                a[r, j] = v
            lb[r], ub[r] = lo, hi
        return LinearConstraint(a.tocsr(), lb, ub)


def shortlist(players: pd.DataFrame, xp: pd.DataFrame, keep: set[int], per_position: int = 40
              ) -> pd.DataFrame:
    """Shrink the pool without cutting off good squads: per position, the best players by
    expected points, the best value per million, and the cheapest (budget enablers),
    plus everyone in `keep` (the current squad)."""
    p = players.join(xp.sum(axis=1).rename("total"), how="inner")
    p["value"] = p["total"] / p["price"]
    chosen = set(keep & set(players.index))
    for _, g in p.groupby("position"):
        chosen |= set(g.nlargest(per_position, "total").index)
        chosen |= set(g.nlargest(per_position // 2, "value").index)
        chosen |= set(g.nsmallest(per_position // 4, "price").index)
    return players.loc[sorted(chosen)]


def plan(players: pd.DataFrame, xp: pd.DataFrame, budget: float = 100.0,
         squad: list[int] | None = None, sell_prices: dict[int, float] | None = None,
         bank: float = 0.0, free_transfers: int = 1, max_free: int = 5,
         discount: float = 0.9, bench_weight: float = 0.1, per_position: int = 40,
         hit_penalty: float = HIT_PENALTY, time_limit: float = 60.0) -> Plan:
    """Choose a squad (squad=None) or transfers for an existing squad.

    players: index element, columns position, team, price (buying price now).
    xp: index element, one column per upcoming gameweek in order (missing = 0).
    hit_penalty: what the planner charges itself per extra transfer. Above the real 4
    points it demands a margin before taking a hit, since forecasts are noisy.
    """
    owned = set(squad or [])
    pool = shortlist(players, xp.reindex(players.index).fillna(0.0), owned, per_position)
    ids = pool.index.to_numpy()
    n, horizon = len(ids), xp.shape[1]
    xp_m = xp.reindex(ids).fillna(0.0).to_numpy()
    price = pool["price"].to_numpy(dtype=float)
    sell = np.array([(sell_prices or {}).get(i, p) for i, p in zip(ids, price, strict=True)])
    held = np.array([i in owned for i in ids], dtype=float)
    transfers = squad is not None

    # variable layout, per gameweek t: s, x, c (and b, o when planning transfers), then
    # h (paid transfers) and f (free transfers available) per gameweek
    blocks = 5 if transfers else 3
    per_t = blocks * n

    def var(kind: int, t: int, i: int) -> int:
        return t * per_t + kind * n + i

    h0 = horizon * per_t
    n_vars = h0 + (2 * horizon if transfers else 0)

    def hv(t: int) -> int:
        return h0 + t

    def fv(t: int) -> int:
        return h0 + horizon + t

    S, X, C, B, O = range(5)  # noqa: E741 - squad, xi, captain, bought, sold
    cons = _Builder(n_vars)
    objective = np.zeros(n_vars)
    weights = discount ** np.arange(horizon)

    for t in range(horizon):
        w = weights[t]
        for i in range(n):
            objective[var(X, t, i)] -= w * xp_m[i, t]
            objective[var(C, t, i)] -= w * xp_m[i, t]
            objective[var(S, t, i)] -= w * bench_weight * xp_m[i, t]
            cons.add({var(X, t, i): 1, var(S, t, i): -1}, ub=0)  # starters are in the squad
            cons.add({var(C, t, i): 1, var(X, t, i): -1}, ub=0)  # captain starts
        for pos, quota in SQUAD_QUOTA.items():
            members = np.flatnonzero(pool["position"].to_numpy() == pos)
            cons.add({var(S, t, i): 1 for i in members}, quota, quota)
            lo, hi = XI_LIMITS[pos]
            cons.add({var(X, t, i): 1 for i in members}, lo, hi)
        for _, members in pool.reset_index().groupby("team").groups.items():
            cons.add({var(S, t, i): 1 for i in members}, ub=MAX_PER_CLUB)
        cons.add({var(X, t, i): 1 for i in range(n)}, 11, 11)
        cons.add({var(C, t, i): 1 for i in range(n)}, 1, 1)

        if not transfers:
            if t == 0:
                cons.add({var(S, 0, i): price[i] for i in range(n)}, ub=budget)
            else:  # without transfers the squad is fixed across the horizon
                for i in range(n):
                    cons.add({var(S, t, i): 1, var(S, 0, i): -1}, 0, 0)
            continue

        for i in range(n):
            # squad evolves: s_t = s_{t-1} + bought - sold
            row = {var(S, t, i): 1, var(B, t, i): -1, var(O, t, i): 1}
            if t == 0:
                cons.add(row, held[i], held[i])
            else:
                row[var(S, t - 1, i)] = -1
                cons.add(row, 0, 0)
            cons.add({var(B, t, i): 1, var(O, t, i): 1}, ub=1)
        # money in the bank never goes negative
        spend = {}
        for tau in range(t + 1):
            for i in range(n):
                spend[var(B, tau, i)] = price[i]
                spend[var(O, tau, i)] = -sell[i]
        cons.add(spend, ub=bank)
        # paid transfers h_t >= transfers made - free transfers available
        made = {var(B, t, i): 1 for i in range(n)}
        cons.add({**made, hv(t): -1, fv(t): -1}, ub=0)
        objective[hv(t)] += w * hit_penalty
        if t == 0:
            cons.add({fv(0): 1}, free_transfers, free_transfers)
        if t + 1 < horizon:  # roll-over: f_{t+1} <= f_t - free used + 1, capped
            cons.add({fv(t + 1): 1, fv(t): -1, **made, hv(t): -1}, ub=1)

    integrality = np.ones(n_vars)
    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    if transfers:
        upper[h0:h0 + horizon] = 15
        upper[h0 + horizon:] = max_free
    res = milp(objective, constraints=cons.constraint(), integrality=integrality,
               bounds=Bounds(lower, upper), options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"no feasible plan: {res.message}")
    sol = np.round(res.x).astype(int)

    def chosen(kind: int, t: int = 0) -> list[int]:
        return [int(ids[i]) for i in range(n) if sol[var(kind, t, i)]]

    xp0 = dict(zip(ids, xp_m[:, 0], strict=True))
    new_squad = chosen(S)
    xi = sorted(chosen(X), key=lambda e: -xp0[e])
    captain = chosen(C)[0]
    vice = next(e for e in xi if e != captain)
    positions = pool["position"].to_dict()
    bench_out = sorted((e for e in new_squad if e not in xi and positions[e] != "GK"),
                       key=lambda e: -xp0[e])
    bench_gk = [e for e in new_squad if e not in xi and positions[e] == "GK"]
    hits = int(sol[hv(0)]) if transfers else 0
    buys, sells = (chosen(B), chosen(O)) if transfers else ([], [])
    sale = sell_prices or {}
    cost = sum(players.loc[buys, "price"]) - sum(sale.get(e, players.loc[e, "price"])
                                                 for e in sells)
    return Plan(
        squad=new_squad, xi=xi, captain=captain, vice=vice, bench=bench_out + bench_gk,
        buy=buys, sell=sells, hits=hits,
        expected_points=float(sum(xp0[e] for e in xi) + xp0[captain] - HIT_COST * hits),
        bank=round(bank - cost if transfers else budget - sum(players.loc[new_squad, "price"]), 1),
    )


def validate_plan(p: Plan, players: pd.DataFrame, budget: float | None = None) -> list[str]:
    """Rule violations in a plan (empty when valid). Used by tests and the backtest."""
    errors = []
    squad = players.loc[p.squad]
    counts = squad["position"].value_counts().to_dict()
    if len(p.squad) != 15 or counts != SQUAD_QUOTA:
        errors.append(f"squad composition {counts}")
    if squad["team"].value_counts().max() > MAX_PER_CLUB:
        errors.append("more than 3 players from one club")
    if budget is not None and squad["price"].sum() > budget + 1e-9:
        errors.append(f"over budget: {squad['price'].sum():.1f}")
    xi = players.loc[p.xi]
    if len(p.xi) != 11 or not set(p.xi) <= set(p.squad):
        errors.append("XI must be 11 squad players")
    for pos, (lo, hi) in XI_LIMITS.items():
        k = int((xi["position"] == pos).sum())
        if not lo <= k <= hi:
            errors.append(f"{k} {pos} in XI")
    if p.captain not in p.xi or p.vice not in p.xi or p.captain == p.vice:
        errors.append("captain and vice must be different starters")
    return errors
