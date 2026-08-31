from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable

import numpy as np


SignSets = Dict[int, tuple[int, ...]]
Pair = tuple[int, int, int, int]


@dataclass
class HillEstimate:
    xi: float
    threshold: float
    n_positive: int
    defined: bool


def estimate_alpha_ecf_1d(
    x: np.ndarray,
    frequencies: Iterable[float] = (0.40, 0.60, 0.90, 1.30, 1.80),
    alpha_bounds: tuple[float, float] = (0.20, 2.00),
) -> float:
    r"""
    Fast stable-law alpha estimator from the empirical characteristic function.

    For every nondegenerate univariate alpha-stable law,
        |phi(t)| = exp(-sigma^alpha |t|^alpha),
    so log(-log|phi(t)|) is affine in log|t| with slope alpha.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan

    # Robustly scale before evaluating the ECF. This keeps chosen frequencies
    # informative without changing alpha.
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75.0, 25.0]))
    scale = iqr / 1.349 if iqr > 0.0 else np.median(np.abs(x - med))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.std(x)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.nan
    y = (x - med) / scale

    t = np.asarray(tuple(frequencies), dtype=float)
    phi = np.mean(np.exp(1j * np.outer(t, y)), axis=1)
    mod = np.abs(phi)
    keep = (mod > 1e-5) & (mod < 1.0 - 1e-5)
    if np.sum(keep) < 2:
        return np.nan

    xx = np.log(t[keep])
    yy = np.log(-np.log(mod[keep]))
    slope = np.polyfit(xx, yy, 1)[0]
    return float(np.clip(slope, *alpha_bounds))


def estimate_common_alpha(
    x: np.ndarray,
    frequencies: Iterable[float] = (0.40, 0.60, 0.90, 1.30, 1.80),
    pool: str = "median",
) -> tuple[float, np.ndarray]:
    alpha_i = np.array(
        [estimate_alpha_ecf_1d(x[:, j], frequencies) for j in range(x.shape[1])]
    )
    good = alpha_i[np.isfinite(alpha_i)]
    if good.size == 0:
        return np.nan, alpha_i

    if pool == "median":
        pooled = np.median(good)
    elif pool == "mean":
        pooled = np.mean(good)
    elif pool == "trimmed_mean":
        s = np.sort(good)
        trim = int(np.floor(0.1 * s.size))
        pooled = np.mean(s[trim:s.size-trim]) if 2 * trim < s.size else np.mean(s)
    else:
        raise ValueError("unknown alpha pooling rule.")
    return float(pooled), alpha_i


def all_two_sided_signs(d: int) -> SignSets:
    return {i: (-1, 1) for i in range(d)}


def positive_only_signs(d: int) -> SignSets:
    return {i: (1,) for i in range(d)}


def estimate_active_signs(
    x: np.ndarray,
    alpha_hat: float,
    m_gamma: float = 0.60,
    b_power: float = 1.0 / 3.0,
) -> tuple[SignSets, dict[tuple[int, int], float]]:
    n, d = x.shape
    m = int(np.floor(n ** m_gamma))
    m = min(max(m, 1), n - 1)
    b_n = n ** (-b_power)
    idx = n - m - 1  # 0-based index of Y_(n-m)

    sets: SignSets = {}
    constants: dict[tuple[int, int], float] = {}
    for i in range(d):
        active: list[int] = []
        for eps in (-1, 1):
            y = np.maximum(eps * x[:, i], 0.0)
            threshold = float(np.partition(y, idx)[idx])
            c_hat = 0.0 if threshold <= 0.0 else (m / n) * threshold ** alpha_hat
            constants[(i, eps)] = float(c_hat)
            if c_hat > b_n:
                active.append(eps)
        sets[i] = tuple(active)
    return sets, constants


def active_pairs(active_sets: SignSets, upper_tail_only: bool = False) -> list[Pair]:
    pairs: list[Pair] = []
    d = len(active_sets)
    for i, j in combinations(range(d), 2):
        for eps in active_sets.get(i, ()):
            for eta in active_sets.get(j, ()):
                if upper_tail_only and (eps, eta) != (1, 1):
                    continue
                pairs.append((i, j, int(eps), int(eta)))
    return pairs


def signed_minimum(x: np.ndarray, pair: Pair) -> np.ndarray:
    i, j, eps, eta = pair
    return np.minimum(
        np.maximum(eps * x[:, i], 0.0),
        np.maximum(eta * x[:, j], 0.0),
    )


def hill_estimate(w: np.ndarray, k: int) -> HillEstimate:
    w = np.asarray(w, dtype=float)
    n = w.size
    n_positive = int(np.count_nonzero(w > 0.0))
    if k < 1 or k >= n or n_positive < k + 1:
        return HillEstimate(np.nan, np.nan, n_positive, False)

    idx = n - k - 1
    part = np.partition(w, idx)
    threshold = float(part[idx])
    if not (threshold > 0.0 and np.isfinite(threshold)):
        return HillEstimate(np.nan, threshold, n_positive, False)

    top = part[idx + 1:]
    xi = float(np.mean(np.log(top / threshold)))
    if not np.isfinite(xi) or xi <= 0.0:
        return HillEstimate(np.nan, threshold, n_positive, False)
    return HillEstimate(xi, threshold, n_positive, True)
