from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.linalg import toeplitz

from .stable import (
    StableNormalScoreTransform,
    positive_stable_rvs,
    sas_rvs,
)


def equicorrelation_matrix(d: int, rho: float) -> np.ndarray:
    lower = -1.0 / (d - 1)
    if not (lower < rho < 1.0):
        raise ValueError(
            f"equicorrelation requires rho in ({lower:.6g},1) for d={d}."
        )
    return (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))


def toeplitz_correlation_matrix(d: int, rho: float) -> np.ndarray:
    if not (-1.0 < rho < 1.0):
        raise ValueError("Toeplitz AR(1) correlation requires |rho|<1.")
    return toeplitz(rho ** np.arange(d))


def correlation_matrix(d: int, rho: float, structure: str) -> np.ndarray:
    if structure == "equicorr":
        return equicorrelation_matrix(d, rho)
    if structure == "toeplitz":
        return toeplitz_correlation_matrix(d, rho)
    raise ValueError("structure must be 'equicorr' or 'toeplitz'.")


def stable_factor_sample(
    n: int,
    d: int,
    alpha: float,
    lam: float,
    rng: Generator,
    sign_structure: str = "aligned",
    margin: str = "symmetric",
) -> np.ndarray:
    if not (0.0 < lam < 1.0):
        raise ValueError("lambda must be in (0,1).")

    a_idio = (1.0 - lam) ** (1.0 / alpha)
    a_common = lam ** (1.0 / alpha)

    if margin == "symmetric":
        v0 = sas_rvs(alpha, n, rng)
        vi = sas_rvs(alpha, (n, d), rng)
        if sign_structure == "aligned":
            s = np.ones(d)
        elif sign_structure == "alternating":
            s = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        else:
            raise ValueError("unknown symmetric sign structure.")
        return a_idio * vi + a_common * v0[:, None] * s[None, :]

    if margin == "positive":
        v0 = positive_stable_rvs(alpha, n, rng)
        vi = positive_stable_rvs(alpha, (n, d), rng)
        return a_idio * vi + a_common * v0[:, None]

    raise ValueError("margin must be 'symmetric' or 'positive'.")


def gaussian_copula_stable_sample(
    n: int,
    d: int,
    alpha: float,
    rho: float,
    rng: Generator,
    corr_structure: str,
    transform: StableNormalScoreTransform,
) -> tuple[np.ndarray, np.ndarray]:
    r = correlation_matrix(d, rho, corr_structure)
    chol = np.linalg.cholesky(r)
    z = rng.standard_normal((n, d)) @ chol.T
    return transform(z), r
