"""Scoring rules, calibration, baselines and the walk-forward harness."""

import math

import numpy as np
import pandas as pd
import pytest

from epl.data.load import PREMATCH_COLUMNS
from epl.eval import backtest, calibration, metrics
from epl.models.baselines import FrequencyModel, MarketModel, demargin

# ---------- scoring rules (hand-computed) ----------


def test_rps_known_values():
    # Perfect forecast scores 0; (0.8, 0.1, 0.1) on a home win: ((0.8-1)^2 + (0.9-1)^2)/2.
    assert metrics.rps([[1, 0, 0]], ["H"]) == 0
    assert metrics.rps([[0.8, 0.1, 0.1]], ["H"]) == pytest.approx(0.025)
    # RPS respects order: on a home win, wrongly backing the draw beats backing the away win.
    assert metrics.rps([[0, 1, 0]], ["H"]) < metrics.rps([[0, 0, 1]], ["H"])


def test_log_loss_and_brier_known_values():
    p, y = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], ["H", "D"]
    assert metrics.log_loss(p, y) == pytest.approx(-(math.log(0.5) + math.log(0.3)) / 2)
    # (0.25 + 0.09 + 0.04) and (0.04 + 0.49 + 0.25)
    assert metrics.brier(p, y) == pytest.approx((0.38 + 0.78) / 2)
    assert metrics.accuracy(p, y) == 0.5


def test_metrics_reject_non_distributions():
    with pytest.raises(ValueError):
        metrics.rps([[0.5, 0.5, 0.5]], ["H"])
    with pytest.raises(ValueError):
        metrics.brier([[0.5, 0.5]], ["H"])


# ---------- calibration ----------


def test_perfectly_calibrated_forecasts_have_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([4, 2, 3], size=20_000)
    y = [metrics.OUTCOMES[rng.choice(3, p=row)] for row in p]
    assert calibration.ece(p, y) < 0.01
    table = calibration.reliability(p, y)
    assert (table["ci_low"] <= table["observed"]).all()
    assert (table["observed"] <= table["ci_high"]).all()


def test_overconfident_forecasts_show_up_in_ece():
    rng = np.random.default_rng(1)
    truth = rng.dirichlet([4, 2, 3], size=20_000)
    y = [metrics.OUTCOMES[rng.choice(3, p=row)] for row in truth]
    sharp = truth**3 / (truth**3).sum(axis=1, keepdims=True)
    assert calibration.ece(sharp, y) > 0.05


# ---------- baselines ----------


def test_demargin_methods_sum_to_one_and_power_shrinks_longshots_more():
    odds = np.array([[1.25, 6.5, 11.0], [2.4, 3.3, 3.0]])
    prop, power = demargin(odds, "proportional"), demargin(odds, "power")
    assert np.allclose(prop.sum(axis=1), 1) and np.allclose(power.sum(axis=1), 1)
    assert power[0, 2] < prop[0, 2]  # the 11.0 longshot
    assert power[0, 0] > prop[0, 0]  # the favourite


def test_frequency_model_uses_training_rates():
    train = pd.DataFrame({"result": ["H"] * 46 + ["D"] * 24 + ["A"] * 30})
    probs = FrequencyModel().fit(train).predict(pd.DataFrame(index=range(2)))
    assert probs.shape == (2, 3)
    assert probs[0] == pytest.approx(np.array([47, 25, 31]) / 103)


def test_market_model_requires_odds():
    with pytest.raises(ValueError):
        MarketModel().predict(pd.DataFrame({"odds_h": [np.nan], "odds_d": [3.0], "odds_a": [3.0]}))


# ---------- walk-forward harness ----------


def synthetic_matches(seasons=("2021-22", "2022-23"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    for s_idx, season in enumerate(seasons):
        start = pd.Timestamp(f"{2021 + s_idx}-08-07")  # a Saturday
        for week in range(10):
            date = start + pd.Timedelta(weeks=week)
            order = rng.permutation(teams)
            for h, a in zip(order[::2], order[1::2], strict=True):
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                rows.append({"match_id": f"{date:%Y%m%d}-{h}-{a}", "season": season,
                             "date": date, "home": h, "away": a, "hg": hg, "ag": ag,
                             "result": "H" if hg > ag else "D" if hg == ag else "A",
                             "odds_h": 2.2, "odds_d": 3.4, "odds_a": 3.4})
    return pd.DataFrame(rows)


class SpyModel:
    """Records what the harness shows it, and tries to cheat by reading results."""

    name = "spy"
    seen: list = []

    def fit(self, train):
        self.train_max = train["date"].max() if len(train) else None
        return self

    def predict(self, fixtures):
        SpyModel.seen.append((self.train_max, fixtures["date"].min(), set(fixtures.columns)))
        assert "result" not in fixtures and "hg" not in fixtures
        return np.full((len(fixtures), 3), 1 / 3)


def test_walk_forward_only_ever_shows_the_past():
    SpyModel.seen = []
    preds = backtest.walk_forward(synthetic_matches(), SpyModel, ["2022-23"])
    assert len(preds) == 30 and set(preds["season"]) == {"2022-23"}
    for train_max, first_fixture, columns in SpyModel.seen:
        assert train_max < first_fixture
        assert columns == set(PREMATCH_COLUMNS)


def test_leakage_guard_raises():
    m = synthetic_matches()
    with pytest.raises(backtest.LeakageError):
        backtest.check_no_leakage(m, m[m["season"] == "2022-23"])


def test_train_seasons_limits_history():
    m = synthetic_matches(("2020-21", "2021-22", "2022-23"))

    class FirstSeen:
        name, firsts = "first", []

        def fit(self, train):
            FirstSeen.firsts.append(train["season"].min())
            return self

        def predict(self, fixtures):
            return np.full((len(fixtures), 3), 1 / 3)

    backtest.walk_forward(m, FirstSeen, ["2022-23"], train_seasons=1)
    assert set(FirstSeen.firsts) == {"2021-22"}


def test_summaries_and_comparison_tables():
    m = synthetic_matches()
    preds = {"frequency": backtest.walk_forward(m, FrequencyModel, ["2022-23"]),
             "market": backtest.walk_forward(m, MarketModel, ["2022-23"])}
    table = backtest.compare(preds)
    assert set(table["model"]) == {"frequency", "market"}
    assert {"rps", "log_loss", "brier", "accuracy", "matches"} <= set(table.columns)
    assert (table["matches"] == 30).all()
    overall = backtest.summarise(preds["market"], by=None)
    assert len(overall) == 1
