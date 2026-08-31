from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import norm

from .estimators import Pair, SignSets, active_pairs, hill_estimate, signed_minimum


@dataclass
class PairStat:
    pair: Pair
    xi_hat: float
    delta_hat: float
    t_stat: float
    threshold: float
    n_positive: int
    defined: bool


@dataclass
class STGResult:
    alpha_hat: float
    xi_hat: float
    k: int
    tau: float
    active_sets: SignSets
    pair_stats: list[PairStat]
    delta_global: float
    t_global: float
    p_value: float
    reject: bool


def stg_test(
    x: np.ndarray,
    alpha_hat: float,
    k: int,
    active_sets: SignSets,
    tau: float = 0.05,
    upper_tail_only: bool = False,
) -> STGResult:
    xi_hat = 1.0 / alpha_hat
    pair_stats: list[PairStat] = []
    pairs = active_pairs(active_sets, upper_tail_only=upper_tail_only)

    for pair in pairs:
        h = hill_estimate(signed_minimum(x, pair), k)
        if h.defined:
            delta = xi_hat / h.xi - 1.0
            t = sqrt(k) * delta
        else:
            # Exactly the manuscript convention: insufficient positive
            # observations => T_{a,n}=-infinity.
            delta = np.nan
            t = -np.inf
        pair_stats.append(
            PairStat(pair, h.xi, delta, t, h.threshold, h.n_positive, h.defined)
        )

    if not pair_stats:
        t_global = -np.inf
        delta_global = np.nan
    else:
        t_global = float(min(p.t_stat for p in pair_stats))
        if all(p.defined for p in pair_stats):
            delta_global = float(min(p.delta_hat for p in pair_stats))
        else:
            delta_global = np.nan

    if np.isposinf(t_global):
        p_value = 0.0
    elif np.isneginf(t_global) or np.isnan(t_global):
        p_value = 1.0
    else:
        p_value = float(norm.sf(t_global))
    crit = float(norm.ppf(1.0 - tau))
    return STGResult(
        alpha_hat=float(alpha_hat),
        xi_hat=float(xi_hat),
        k=int(k),
        tau=float(tau),
        active_sets=active_sets,
        pair_stats=pair_stats,
        delta_global=delta_global,
        t_global=t_global,
        p_value=p_value,
        reject=bool(t_global > crit),
    )
