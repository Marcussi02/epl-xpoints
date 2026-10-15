"""Elo and Dixon-Coles: invariants, gradient correctness and parameter recovery."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from epl.models.dixon_coles import DixonColesModel
from epl.models.elo import EloModel, margin_multiplier

TEAMS = [f"T{i:02d}" for i in range(20)]


def round_robin(teams, season, start, rng, score):
    """Double round robin; score(home, away) -> (hg, ag). One matchday per week."""
    n = len(teams)
    order = list(teams)
    rows, day = [], pd.Timestamp(start)
    rounds = []
    for _ in range(n - 1):  # circle method
        rounds.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    for leg in range(2):
        for pairs in rounds:
            for h, a in pairs:
                home, away = (h, a) if leg == 0 else (a, h)
                hg, ag = score(home, away)
                rows.append({"match_id": f"{season}-{home}-{away}", "season": season,
                             "date": day, "home": home, "away": away, "hg": hg, "ag": ag,
                             "result": "H" if hg > ag else "D" if hg == ag else "A"})
            day += pd.Timedelta(weeks=1)
    return rows


def simulate_dc(att, dfn, home, rho, seasons=3, seed=7):
    """Seasons of matches drawn from a known Dixon-Coles model."""
    rng = np.random.default_rng(seed)
    truth = DixonColesModel()
    truth.teams_, truth.attack_, truth.defence_ = TEAMS, att, dfn
    truth.home_, truth.rho_ = home, rho

    keys = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    pairs = pd.DataFrame(keys, columns=["home", "away"])
    matrices = dict(zip(keys, truth.score_matrix(pairs), strict=True))

    def score(h, a):
        m = matrices[h, a]
        k = rng.choice(m.size, p=m.ravel())
        return divmod(int(k), m.shape[1])

    rows = []
    for s in range(seasons):
        y = 2000 + s
        rows += round_robin(TEAMS, f"{y}-{(y + 1) % 100:02d}", f"{y}-08-01", rng, score)
    return pd.DataFrame(rows)


# ---------- Dixon-Coles ----------


@pytest.fixture(scope="module")
def dc_truth():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.25, 20)  # roughly the spread of real Premier League sides
    att = dict(zip(TEAMS, a - a.mean(), strict=True))
    dfn = dict(zip(TEAMS, rng.normal(0, 0.25, 20), strict=True))
    return att, dfn, 0.27, -0.12


@pytest.fixture(scope="module")
def dc_fit(dc_truth):
    data = simulate_dc(*dc_truth, seasons=6)
    return DixonColesModel(xi=0.0).fit(data), data


def test_dixon_coles_recovers_known_parameters(dc_truth, dc_fit):
    att, dfn, home, rho = dc_truth
    model, _ = dc_fit
    assert model.converged_
    fitted_att = np.array([model.attack_[t] for t in TEAMS])
    fitted_def = np.array([model.defence_[t] for t in TEAMS])
    # defence is identified only up to a shift shared with home advantage's baseline
    true_def = np.array([dfn[t] for t in TEAMS])
    assert np.corrcoef(fitted_att, [att[t] for t in TEAMS])[0, 1] > 0.9
    assert np.corrcoef(fitted_def, true_def)[0, 1] > 0.9
    assert model.home_ - fitted_def.mean() == pytest.approx(home - true_def.mean(), abs=0.08)


def test_dixon_coles_rho_is_unbiased_across_seeds(dc_truth):
    # rho only moves four scorelines, so any single fit is noisy (sd ~0.02 at 12 seasons,
    # and the odd seed lands further out). Averaging over seeds tests the estimator itself.
    rho = dc_truth[3]
    estimates = [DixonColesModel(xi=0.0).fit(simulate_dc(*dc_truth, seasons=6, seed=s)).rho_
                 for s in range(20, 25)]
    assert np.mean(estimates) == pytest.approx(rho, abs=0.03)


def test_dixon_coles_analytic_gradient_matches_numerical(dc_fit):
    model, _ = dc_fit
    theta = model._theta + np.random.default_rng(0).normal(0, 0.05, model._theta.size)

    def f(t):
        return model._objective(t)[0]

    def g(t):
        return model._objective(t)[1]

    err = check_grad(f, g, theta, epsilon=1e-6)
    assert err / np.linalg.norm(g(theta)) < 1e-4


def test_dixon_coles_probabilities_are_coherent(dc_fit):
    model, _ = dc_fit
    fixtures = pd.DataFrame({"home": ["T00", "T05", "Newcomer"], "away": ["T01", "T00", "T03"]})
    probs = model.predict(fixtures)
    assert np.allclose(probs.sum(axis=1), 1) and (probs > 0).all()
    m = model.score_matrix(fixtures)
    assert np.allclose(m.sum(axis=(1, 2)), 1)
    cs_home, cs_away = model.clean_sheet_probs(fixtures)
    lam, mu = model.rates(fixtures)
    assert (cs_home > 0).all() and (cs_away > 0).all()
    # a newly promoted side is rated like the weakest teams in the training data
    # (weak sides can still attack well, so compare win chances, not expected goals)
    others = [t for t in TEAMS if t != "T03"]
    vs_t03 = model.predict(pd.DataFrame({"home": others, "away": "T03"}))[:, 0]
    newcomer = model.predict(pd.DataFrame({"home": ["Newcomer"], "away": ["T03"]}))[0, 0]
    assert newcomer < np.median(vs_t03)


def test_time_decay_weights_recent_matches_more():
    rng = np.random.default_rng(11)
    # T00 is strong in the old season and weak in the recent one.
    def score(h, a):
        if current[0] == "2020-21":
            return (3, 0) if h == "T00" else (0, 3) if a == "T00" else (1, 1)
        return (0, 2) if h == "T00" else (2, 0) if a == "T00" else (1, 1)

    current = ["2020-21"]
    rows = round_robin(TEAMS[:6], "2020-21", "2020-08-01", rng, score)
    current[0] = "2021-22"
    rows += round_robin(TEAMS[:6], "2021-22", "2021-08-01", rng, score)
    data = pd.DataFrame(rows)
    flat = DixonColesModel(xi=0.0).fit(data)
    decayed = DixonColesModel(xi=0.01).fit(data)
    assert decayed.attack_["T00"] < flat.attack_["T00"]


# ---------- Elo ----------


def test_margin_multiplier():
    assert [margin_multiplier(g) for g in (0, 1, -1, 2, 3, 5)] == [1, 1, 1, 1.5, 1.75, 2.0]


def test_elo_updates_are_zero_sum_within_a_season():
    rng = np.random.default_rng(5)
    data = pd.DataFrame(round_robin(TEAMS, "2022-23", "2022-08-01", rng,
                                    lambda h, a: (rng.poisson(1.5), rng.poisson(1.1))))
    model = EloModel().fit(data)
    assert sum(model.ratings_.values()) == pytest.approx(1500 * len(TEAMS))


def test_elo_season_regression_preserves_the_mean_and_shrinks_spread():
    model = EloModel(regress=0.5)
    ratings = {"A": 1600.0, "B": 1400.0, "C": 1500.0}
    model._new_season(ratings, {"A", "B", "C"})
    assert ratings == {"A": 1550.0, "B": 1450.0, "C": 1500.0}


def test_elo_is_symmetric_without_home_advantage():
    rng = np.random.default_rng(2)
    data = pd.DataFrame(round_robin(TEAMS[:8], "2022-23", "2022-08-01", rng,
                                    lambda h, a: (rng.poisson(1.3), rng.poisson(1.3))))
    model = EloModel(home_adv=0.0, burn_in=0).fit(data)
    model.link_ = (model.link_[0], -0.7, 0.7)  # a symmetric link isolates the ratings
    p = model.predict(pd.DataFrame({"season": "2022-23", "home": ["T01", "T02"],
                                    "away": ["T02", "T01"]}))
    assert p[0] == pytest.approx(p[1][::-1])


def test_elo_probabilities_favour_the_stronger_team():
    rng = np.random.default_rng(9)

    def score(h, a):  # T00 always wins
        return (2, 0) if h == "T00" else (0, 2) if a == "T00" else (rng.poisson(1.3),
                                                                     rng.poisson(1.1))

    data = pd.DataFrame(round_robin(TEAMS[:8], "2022-23", "2022-08-01", rng, score)
                        + round_robin(TEAMS[:8], "2023-24", "2023-08-01", rng, score))
    model = EloModel(burn_in=20).fit(data)
    p = model.predict(pd.DataFrame({"season": "2024-25", "home": ["T00", "T03", "Newbie"],
                                    "away": ["T03", "T00", "T04"]}))
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()
    assert p[0, 0] > 0.6 and p[1, 2] > 0.5  # T00 wins home or away


def test_score_matrix_stays_a_distribution_for_extreme_rates():
    model = DixonColesModel()
    model.teams_, model.home_, model.rho_ = ["A", "B"], 0.3, -0.2
    model.attack_, model.defence_ = {"A": 1.0, "B": -1.0}, {"A": 1.0, "B": -1.0}
    m = model.score_matrix(pd.DataFrame({"home": ["A"], "away": ["B"]}))  # lam ~ 9.97
    assert (m >= 0).all() and m.sum() == pytest.approx(1)
