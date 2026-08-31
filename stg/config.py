from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Prespecified baseline used in the reported finite-sample benchmark.
# The full threshold-sensitivity experiment is reported separately, so this
# value is not selected after looking at an empirical Section 7 p-value.
BASELINE_K_GAMMA = 0.40
THRESHOLD_GAMMAS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
BENCHMARK_LAMBDAS = (0.30, 0.60)
BENCHMARK_RHOS = (0.30, 0.50, 0.70)


@dataclass(frozen=True)
class SimulationConfig:
    """One data-generating configuration in Section 6."""

    model: Literal["stable", "gaussian"]
    n: int
    d: int
    alpha: float
    dependence: float  # lambda for stable, rho for Gaussian copula

    margin: Literal["symmetric", "positive"] = "symmetric"
    sign_structure: Literal["aligned", "alternating", "positive"] = "aligned"
    corr_structure: Literal["equicorr", "toeplitz"] = "equicorr"

    k_gamma: float = BASELINE_K_GAMMA
    tau: float = 0.05
    active_set_mode: Literal["known", "estimated"] = "known"
    upper_tail_only: bool = False

    # Numerical transform for Gaussian-copula stable margins.
    quantile_zmax: float = 3.75
    quantile_grid_size: int = 301

    label: str = ""
    experiment: str = "main"

    @property
    def k(self) -> int:
        return max(1, min(self.n - 1, int(self.n ** self.k_gamma)))


@dataclass(frozen=True)
class MonteCarloConfig:
    B: int = 2000
    seed: int = 20260815
    workers: int = 1
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")


@dataclass(frozen=True)
class EmpiricalConfig:
    """Settings for the Section 7 pipeline."""

    date_column: str = "Date"
    price_columns: tuple[str, ...] = ()
    return_scale: float = 100.0

    # AR(1)-GARCH(1,1) prefilter.
    filter_mean: Literal["AR"] = "AR"
    filter_lags: int = 1
    arch_dist: Literal["normal", "t"] = "t"

    # Stable margins.
    stable_fit_backend: Literal["scipy_mle", "ecf"] = "ecf"
    bootstrap_B: int = 299
    bootstrap_seed: int = 20260816
    bootstrap_fit_backend: Literal["same", "scipy_mle", "ecf"] = "same"
    common_alpha_pool: Literal["median", "mean", "trimmed_mean"] = "median"

    # Active signs.
    active_m_gamma: float = 0.60
    active_threshold_power: float = 1.0 / 3.0

    # STG thresholds. The empirical baseline mirrors Section 6 and neighboring
    # values are prespecified sensitivity checks, not post-hoc choices.
    k_gammas: tuple[float, ...] = (0.35, 0.40, 0.45)
    baseline_gamma: float = BASELINE_K_GAMMA
    diagnostic_k_gammas: tuple[float, ...] = THRESHOLD_GAMMAS
    tau: float = 0.05

    # Independence screen.
    independence_permutations: int = 199
    independence_seed: int = 20260817
    copula_chunk_size: int = 128

    # Risk extrapolation multipliers relative to the Hill threshold.
    risk_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

    # Vanishing-level model selector.
    use_model_selection: bool = True
