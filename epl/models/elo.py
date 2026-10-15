"""Elo ratings with an ordered-logistic link to home/draw/away probabilities.

Ratings (World Football Elo style):
    expected  E = 1 / (1 + 10^(-(R_home + home_adv - R_away) / 400))
    update    R += k * G * (S - E), zero-sum between the two teams, where S is 1/0.5/0
              and G grows with the goal margin: 1, 1.5, then (11 + margin) / 8
    seasons   optionally, every rating regresses part-way to the mean at each new season
              (off by default: tuning found no benefit); a newly promoted club starts at
              the average of the three lowest-rated clubs (roughly, the teams it replaced)

Probabilities: Elo only gives an expected score, not a draw probability, so an ordered
logit is fitted on the training matches' pre-match rating differences d:
    P(away) = s(t1 - b*d),  P(away or draw) = s(t2 - b*d),  t1 < t2
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from ..eval.metrics import OUTCOMES


def margin_multiplier(goal_diff: int) -> float:
    m = abs(goal_diff)
    return 1.0 if m <= 1 else 1.5 if m == 2 else (11 + m) / 8


class EloModel:
    name = "elo"

    # Defaults chosen by grid search on six validation seasons, 2016-17 to 2021-22 (see
    # REPORT.md).
    def __init__(self, k: float = 10.0, home_adv: float = 60.0, regress: float = 0.1,
                 initial: float = 1500.0, burn_in: int = 380):
        self.k, self.home_adv, self.regress = k, home_adv, regress
        self.initial, self.burn_in = initial, burn_in

    # ---------- ratings ----------

    def _new_season(self, ratings: dict[str, float], teams: set[str]) -> None:
        if ratings:
            mean = np.mean(list(ratings.values()))
            for team in ratings:
                ratings[team] = mean + (ratings[team] - mean) * (1 - self.regress)
        for team in sorted(teams - set(ratings)):
            ratings[team] = self._promoted_rating(ratings)

    def _promoted_rating(self, ratings: dict[str, float]) -> float:
        if len(ratings) < 3:
            return self.initial
        return float(np.mean(sorted(ratings.values())[:3]))

    def diff(self, ratings: dict[str, float], home: str, away: str) -> float:
        return ratings[home] + self.home_adv - ratings[away]

    def fit(self, train: pd.DataFrame) -> "EloModel":
        train = train.sort_values("date")
        ratings: dict[str, float] = {}
        diffs, results = [], []
        season = None
        for row in train.itertuples(index=False):
            if row.season != season:
                season = row.season
                in_season = train.loc[train["season"] == season, ["home", "away"]]
                self._new_season(ratings, set(in_season.values.ravel()))
            d = self.diff(ratings, row.home, row.away)
            diffs.append(d)
            results.append(row.result)
            expected = 1 / (1 + 10 ** (-d / 400))
            actual = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
            delta = self.k * margin_multiplier(row.hg - row.ag) * (actual - expected)
            ratings[row.home] += delta
            ratings[row.away] -= delta
        self.ratings_ = ratings
        self.season_ = season
        start = min(self.burn_in, max(0, len(diffs) - 50))  # ratings need time to settle
        self._fit_link(np.array(diffs[start:]), results[start:])
        return self

    # ---------- link ----------

    def _fit_link(self, d: np.ndarray, results: list[str]) -> None:
        y = np.array([OUTCOMES.index(r) for r in results])
        x = d / 400

        def nll(params):
            b, t1, log_gap = params
            p = self._probs(x, b, t1, t1 + np.exp(log_gap))
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum()

        res = minimize(nll, x0=[1.5, -0.6, -0.5], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
        b, t1, log_gap = res.x
        self.link_ = (b, t1, t1 + np.exp(log_gap))

    @staticmethod
    def _probs(x: np.ndarray, b: float, t1: float, t2: float) -> np.ndarray:
        away = expit(t1 - b * x)
        away_or_draw = expit(t2 - b * x)
        return np.column_stack([1 - away_or_draw, away_or_draw - away, away])

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        ratings = dict(self.ratings_)
        new_seasons = sorted(set(fixtures["season"]) - {self.season_})
        if new_seasons:  # predicting a season that has not started in the training data
            teams = set(fixtures[["home", "away"]].values.ravel())
            self._new_season(ratings, teams)
        for team in set(fixtures[["home", "away"]].values.ravel()) - set(ratings):
            ratings[team] = self._promoted_rating(ratings)
        d = np.array([self.diff(ratings, h, a)
                      for h, a in zip(fixtures["home"], fixtures["away"], strict=True)])
        return self._probs(d / 400, *self.link_)
