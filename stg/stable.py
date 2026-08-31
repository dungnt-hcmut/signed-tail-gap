from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma
from scipy.stats import levy_stable, norm

# The manuscript and formulas below use the S1 characteristic-function
# parameterization. SciPy supports S0 and S1; S1 is SciPy's default.
levy_stable.parameterization = "S1"


def stable_tail_constant(alpha: float) -> float:
    """C_alpha = Gamma(alpha) sin(pi alpha/2) / pi, with C_1=1/pi."""
    if np.isclose(alpha, 1.0):
        return 1.0 / np.pi
    if np.isclose(alpha, 2.0):
        return 0.0
    return float(gamma(alpha) * np.sin(np.pi * alpha / 2.0) / np.pi)


def signed_tail_constants_s1(
    alpha: float,
    beta: float,
    scale: float,
) -> tuple[float, float]:
    """
    First-order S1 stable tail constants:
      P(X>x)  ~ c_plus  x^{-alpha},
      P(X<-x) ~ c_minus x^{-alpha}.

    Location does not affect first-order constants.
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must lie in (0,2].")
    if not (-1.0 <= beta <= 1.0):
        raise ValueError("beta must lie in [-1,1].")
    if not (scale > 0.0):
        raise ValueError("scale must be positive.")
    c = stable_tail_constant(alpha) * scale ** alpha
    return c * (1.0 + beta), c * (1.0 - beta)


def sas_rvs(alpha: float, size, rng: Generator) -> np.ndarray:
    """
    Standard symmetric alpha-stable random variables with CF exp(-|t|^alpha).
    Chambers-Mallows-Stuck generation.
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (0,2].")
    if np.isclose(alpha, 2.0):
        return np.sqrt(2.0) * rng.standard_normal(size=size)

    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    w = np.maximum(rng.exponential(scale=1.0, size=size), np.finfo(float).tiny)

    if np.isclose(alpha, 1.0):
        return np.tan(u)

    first = np.sin(alpha * u) / np.power(np.cos(u), 1.0 / alpha)
    second = np.power(
        np.cos((1.0 - alpha) * u) / w,
        (1.0 - alpha) / alpha,
    )
    return first * second


def positive_stable_rvs(alpha: float, size, rng: Generator) -> np.ndarray:
    """
    Positive alpha-stable variables normalized by E exp(-tX)=exp(-t^alpha),
    valid for 0<alpha<1. Kanter representation.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("positive stable law requires alpha in (0,1).")

    # Avoid exact endpoints, where the analytic representation is singular.
    eps = np.finfo(float).eps
    v = rng.uniform(eps, np.pi - eps, size=size)
    e = np.maximum(rng.exponential(scale=1.0, size=size), np.finfo(float).tiny)
    first = np.sin(alpha * v) / np.power(np.sin(v), 1.0 / alpha)
    second = np.power(
        np.sin((1.0 - alpha) * v) / e,
        (1.0 - alpha) / alpha,
    )
    return first * second


def positive_stable_s1_scale(alpha: float) -> float:
    """S1 scale giving Laplace transform exp(-t^alpha) for beta=1, 0<alpha<1."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    return float(np.cos(np.pi * alpha / 2.0) ** (1.0 / alpha))


def _safe_ppf_switch(alpha: float, margin: str, requested: float) -> float:
    """
    Choose the largest safe normal-score switch not exceeding ``requested``.

    In deep tails, numerical stable PPF evaluation can plateau for some alpha.
    We detect this by comparing a few candidate PPF values with the known
    first-order stable tail quantile. The comparison is used only to choose the
    numerical switch; the outer branch itself preserves the analytic tail
    constant and is not defined by the numerical PPF.
    """
    candidates = [z for z in (3.75, 3.50, 3.25, 3.00, 2.75, 2.50) if z <= requested + 1e-12]
    if not candidates:
        return float(requested)

    for z in candidates:
        p = float(norm.sf(z))
        try:
            if margin == "positive":
                scale = positive_stable_s1_scale(alpha)
                q = float(levy_stable.ppf(1.0 - p, alpha, 1.0, loc=0.0, scale=scale))
                c = 1.0 / gamma(1.0 - alpha)
            else:
                q = float(levy_stable.ppf(1.0 - p, alpha, 0.0, loc=0.0, scale=1.0))
                c = stable_tail_constant(alpha)
            q_asym = (c / p) ** (1.0 / alpha)
            ratio = q / q_asym
        except Exception:
            continue
        if np.isfinite(q) and q > 0.0 and 0.97 <= ratio <= 1.08:
            return float(z)

    # Conservative fallback. The continuity correction makes the tail branch
    # continuous and decays to one, so the first-order tail constant remains exact.
    return float(min(requested, 2.50))


@dataclass
class StableNormalScoreTransform:
    """
    Fast monotone numerical map z -> F_alpha^{-1}(Phi(z)).

    SciPy's ``levy_stable.ppf`` is used only on a validated central normal-score
    region. A shape-preserving PCHIP interpolant handles that region. Beyond a
    conservative switch point, the transform uses the first-order stable tail
    quantile with a *decaying continuity correction*. The correction equals the
    central/tail ratio at the switch and tends to one as |z| grows, so the map is
    continuous while retaining the correct first-order stable tail constant.

    For positive stable margins the extreme lower tail is numerically irrelevant
    to the STG upper-tail statistic; a positive monotone continuation is used
    there to preserve the Gaussian copula construction.
    """

    alpha: float
    margin: str = "symmetric"
    zmax: float = 3.75
    grid_size: int = 301
    cache_dir: Path = Path("cache")

    def __post_init__(self):
        if self.margin not in {"symmetric", "positive"}:
            raise ValueError("margin must be 'symmetric' or 'positive'.")
        if self.margin == "positive" and not (0.0 < self.alpha < 1.0):
            raise ValueError("positive stable margin requires alpha in (0,1).")
        if self.margin == "symmetric" and not (0.0 < self.alpha < 2.0):
            raise ValueError("symmetric stable margin requires alpha in (0,2).")
        if self.grid_size < 51:
            raise ValueError("grid_size should be at least 51.")

        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._zswitch = _safe_ppf_switch(self.alpha, self.margin, self.zmax)
        tag = (
            f"a{self.alpha:.6f}_{self.margin}_"
            f"z{self._zswitch:.2f}_m{self.grid_size}_v2"
        )
        path = self.cache_dir / f"stable_normal_score_{tag}.npz"

        if path.exists():
            dat = np.load(path)
            z_grid = dat["z_grid"]
            q_grid = dat["q_grid"]
        else:
            z_grid = np.linspace(-self._zswitch, self._zswitch, self.grid_size)
            u = norm.cdf(z_grid)
            if self.margin == "symmetric":
                q_grid = levy_stable.ppf(u, self.alpha, 0.0, loc=0.0, scale=1.0)
            else:
                c = positive_stable_s1_scale(self.alpha)
                q_grid = levy_stable.ppf(u, self.alpha, 1.0, loc=0.0, scale=c)
            if not np.all(np.isfinite(q_grid)) or not np.all(np.diff(q_grid) > 0.0):
                raise RuntimeError(
                    "stable PPF grid is not finite/strictly increasing; reduce zmax."
                )
            np.savez_compressed(path, z_grid=z_grid, q_grid=q_grid)

        self._z_grid = z_grid
        self._q_grid = q_grid
        self._interp = PchipInterpolator(z_grid, q_grid, extrapolate=False)

        # Tail matching ratios at the switch. These ratios are close to one.
        if self.margin == "symmetric":
            c_plus = stable_tail_constant(self.alpha)
            asym_u = (c_plus / norm.sf(self._zswitch)) ** (1.0 / self.alpha)
            asym_l = (c_plus / norm.cdf(-self._zswitch)) ** (1.0 / self.alpha)
            self._ratio_u = float(self._q_grid[-1] / asym_u)
            self._ratio_l = float(abs(self._q_grid[0]) / asym_l)
        else:
            c_plus = 1.0 / gamma(1.0 - self.alpha)
            asym_u = (c_plus / norm.sf(self._zswitch)) ** (1.0 / self.alpha)
            self._ratio_u = float(self._q_grid[-1] / asym_u)
            self._ratio_l = np.nan

    @staticmethod
    def _decaying_match(ratio0: float, distance: np.ndarray) -> np.ndarray:
        # Equals ratio0 at the switch and converges exponentially to 1.
        return 1.0 + (ratio0 - 1.0) * np.exp(-distance)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        central = np.abs(z) <= self._zswitch
        out[central] = self._interp(z[central])

        upper = z > self._zswitch
        lower = z < -self._zswitch

        if self.margin == "symmetric":
            c = stable_tail_constant(self.alpha)
            if np.any(upper):
                zu = z[upper]
                p = np.maximum(norm.sf(zu), np.finfo(float).tiny)
                base = np.power(c / p, 1.0 / self.alpha)
                match = self._decaying_match(self._ratio_u, zu - self._zswitch)
                out[upper] = match * base
            if np.any(lower):
                zl = z[lower]
                p = np.maximum(norm.cdf(zl), np.finfo(float).tiny)
                base = np.power(c / p, 1.0 / self.alpha)
                match = self._decaying_match(self._ratio_l, -zl - self._zswitch)
                out[lower] = -match * base

        else:  # positive
            c_plus = 1.0 / gamma(1.0 - self.alpha)
            if np.any(upper):
                zu = z[upper]
                p = np.maximum(norm.sf(zu), np.finfo(float).tiny)
                base = np.power(c_plus / p, 1.0 / self.alpha)
                match = self._decaying_match(self._ratio_u, zu - self._zswitch)
                out[upper] = match * base
            if np.any(lower):
                # Keep positivity and monotonicity. Extreme lower-score values
                # correspond to values near zero and do not enter upper-tail Hill
                # statistics. The continuation is continuous at the switch.
                q0 = max(float(self._q_grid[0]), np.finfo(float).tiny)
                out[lower] = q0 * np.exp(z[lower] + self._zswitch)

        return out
