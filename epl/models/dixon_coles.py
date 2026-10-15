"""Dixon & Coles (1997): Poisson goals with a low-score correction and time decay.

    home goals  X ~ Poisson(lam),  lam = exp(home + attack[h] - defence[a])
    away goals  Y ~ Poisson(mu),   mu  = exp(attack[a] - defence[h])

Independent Poissons under-predict 0-0 and 1-1 and over-predict 1-0 and 0-1, so the
joint probability is multiplied by tau(x, y), which only touches those four scores:
    tau(0,0) = 1 - lam*mu*rho    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho        tau(1,1) = 1 - rho

Recent form matters more than last season's, so each match's log-likelihood is
weighted by exp(-xi * days_ago); xi = 0.0006/day is a half-life of about 3 years.
attack sums to zero (identifiability); higher defence means conceding fewer.

Parameters are fitted by maximising the weighted log-likelihood with L-BFGS-B, using
the analytic gradient (about 45 parameters for 20 teams, refitted every week of a
backtest). A small optional ridge penalty shrinks clubs with little data towards
average. A club absent from training (newly promoted) gets the mean parameters of
the three weakest fitted clubs.

The fitted model gives a full scoreline matrix, so win/draw/loss, clean-sheet and
over/under probabilities and the expected goals used by the FPL model all come from it.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    return np.where((x == 1) & (y == 1), 1 - rho, t)


class DixonColesModel:
    name = "dixon_coles"

    # Defaults chosen by grid search on the 2021-22 validation season (see REPORT.md):
    # a ~3-year half-life and light shrinkage beat the paper's xi=0.0019 with no ridge.
    def __init__(self, xi: float = 0.0006, l2: float = 2.0, max_goals: int = MAX_GOALS):
        self.xi, self.l2, self.max_goals = xi, l2, max_goals

    # ---------- fitting ----------

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        teams = sorted(set(train["home"]) | set(train["away"]))
        index = {t: i for i, t in enumerate(teams)}
        h = train["home"].map(index).to_numpy()
        a = train["away"].map(index).to_numpy()
        x = train["hg"].to_numpy(dtype=float)
        y = train["ag"].to_numpy(dtype=float)
        as_of = as_of or train["date"].max() + pd.Timedelta(days=1)
        days = (as_of - train["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)
        n = len(teams)

        def unpack(theta):
            att_free, dfn = theta[:n], theta[n:2 * n]
            return att_free - att_free.mean(), dfn, theta[2 * n], theta[2 * n + 1]

        def objective(theta):
            att, dfn, home, rho = unpack(theta)
            log_lam = home + att[h] - dfn[a]
            log_mu = att[a] - dfn[h]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            t = np.clip(tau(x, y, lam, mu, rho), 1e-10, None)
            ll = w * (np.log(t) + x * log_lam - lam + y * log_mu - mu)

            # d log tau / d(log lam, log mu, rho), non-zero only for the four low scores
            g_lam = w * (x - lam)
            g_mu = w * (y - mu)
            g_rho = np.zeros_like(lam)
            s00, s01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
            s10, s11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
            lm = lam * mu
            g_lam[s00] += w[s00] * (-lm[s00] * rho / t[s00])
            g_mu[s00] += w[s00] * (-lm[s00] * rho / t[s00])
            g_rho[s00] = w[s00] * (-lm[s00] / t[s00])
            g_lam[s01] += w[s01] * (lam[s01] * rho / t[s01])
            g_rho[s01] = w[s01] * (lam[s01] / t[s01])
            g_mu[s10] += w[s10] * (mu[s10] * rho / t[s10])
            g_rho[s10] = w[s10] * (mu[s10] / t[s10])
            g_rho[s11] = w[s11] * (-1 / t[s11])

            g_att = np.zeros(n)
            g_def = np.zeros(n)
            np.add.at(g_att, h, g_lam)
            np.add.at(g_att, a, g_mu)
            np.add.at(g_def, a, -g_lam)
            np.add.at(g_def, h, -g_mu)
            g_att -= g_att.mean()  # through the sum-to-zero centring

            penalty = self.l2 * (att @ att + dfn @ dfn)
            grad = np.concatenate([
                g_att - 2 * self.l2 * (att - att.mean()),
                g_def - 2 * self.l2 * dfn,
                [g_lam.sum(), g_rho.sum()],
            ])
            return -(ll.sum() - penalty), -grad

        theta0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n) + [(None, None), (-0.2, 0.2)]
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        att, dfn, home, rho = unpack(res.x)
        self.teams_ = teams
        self.attack_ = dict(zip(teams, att, strict=True))
        self.defence_ = dict(zip(teams, dfn, strict=True))
        self.home_, self.rho_ = float(home), float(rho)
        self.converged_ = bool(res.success)
        self._objective = objective  # kept for gradient checks in tests
        self._theta = res.x
        return self

    def _params(self, team: str) -> tuple[float, float]:
        if team in self.attack_:
            return self.attack_[team], self.defence_[team]
        weakest = sorted(self.teams_, key=lambda t: self.attack_[t] + self.defence_[t])[:3]
        return (float(np.mean([self.attack_[t] for t in weakest])),
                float(np.mean([self.defence_[t] for t in weakest])))

    # ---------- predictions ----------

    def rates(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals (lam, mu) for home and away sides."""
        hp = [self._params(t) for t in fixtures["home"]]
        ap = [self._params(t) for t in fixtures["away"]]
        lam = np.exp([self.home_ + ha - ad for (ha, _), (_, ad) in zip(hp, ap, strict=True)])
        mu = np.exp([aa - hd for (_, hd), (aa, _) in zip(hp, ap, strict=True)])
        return np.asarray(lam), np.asarray(mu)

    def score_matrix(self, fixtures: pd.DataFrame) -> np.ndarray:
        """(n, G+1, G+1) array: P(home scores i, away scores j)."""
        lam, mu = self.rates(fixtures)
        goals = np.arange(self.max_goals + 1)
        px = poisson.pmf(goals[None, :], lam[:, None])
        py = poisson.pmf(goals[None, :], mu[:, None])
        m = px[:, :, None] * py[:, None, :]
        m[:, 0, 0] *= 1 - lam * mu * self.rho_
        m[:, 0, 1] *= 1 + lam * self.rho_
        m[:, 1, 0] *= 1 + mu * self.rho_
        m[:, 1, 1] *= 1 - self.rho_
        # tau can dip below zero for extreme rates (e.g. 1 + lam*rho with lam > 1/|rho|)
        np.clip(m, 0, None, out=m)
        return m / m.sum(axis=(1, 2), keepdims=True)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        m = self.score_matrix(fixtures)
        home = np.tril(np.ones(m.shape[1:]), -1)  # i > j
        draw = np.eye(m.shape[1])
        h = (m * home).sum(axis=(1, 2))
        d = (m * draw).sum(axis=(1, 2))
        return np.column_stack([h, d, 1 - h - d])

    def clean_sheet_probs(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """P(home keeps a clean sheet), P(away keeps a clean sheet)."""
        m = self.score_matrix(fixtures)
        return m[:, :, 0].sum(axis=1), m[:, 0, :].sum(axis=1)
