"""Proper scoring rules for home/draw/away forecasts.

Probabilities are an (n, 3) array in OUTCOMES order (H, D, A); outcomes are the
matching labels. All three rules are "proper": a forecaster minimises them in
expectation only by reporting what it actually believes, so none can be gamed.

    log_loss   -mean log p(observed)             punishes confident misses hardest
    brier      mean sum_k (p_k - o_k)^2          range 0..2
    rps        ranked probability score          respects the order H < D < A, so
                                                 predicting a draw when the home side
                                                 wins costs less than predicting an away
                                                 win. The standard metric for football.
"""

import numpy as np

OUTCOMES = ("H", "D", "A")
_INDEX = {o: i for i, o in enumerate(OUTCOMES)}


def one_hot(outcomes) -> np.ndarray:
    idx = np.array([_INDEX[o] for o in outcomes])
    out = np.zeros((len(idx), 3))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def check_probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (n, 3) probabilities, got shape {p.shape}")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6) or (p < 0).any():
        raise ValueError("each row must be a probability distribution over H, D, A")
    return p


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    p = check_probs(probs)
    return float(-np.mean(np.log(np.clip((p * one_hot(outcomes)).sum(axis=1), eps, 1))))


def brier(probs, outcomes) -> float:
    p = check_probs(probs)
    return float(np.mean(((p - one_hot(outcomes)) ** 2).sum(axis=1)))


def rps(probs, outcomes) -> float:
    p = check_probs(probs)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(one_hot(outcomes), axis=1)[:, :-1]
    return float(np.mean(((cum_p - cum_o) ** 2).sum(axis=1) / 2))


def accuracy(probs, outcomes) -> float:
    p = check_probs(probs)
    return float(np.mean(p.argmax(axis=1) == one_hot(outcomes).argmax(axis=1)))


def score_all(probs, outcomes) -> dict[str, float]:
    return {
        "log_loss": log_loss(probs, outcomes),
        "brier": brier(probs, outcomes),
        "rps": rps(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }
