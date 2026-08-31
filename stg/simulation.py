from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
from scipy.stats import norm

from .config import MonteCarloConfig, SimulationConfig
from .dgp import gaussian_copula_stable_sample, stable_factor_sample
from .estimators import (
    all_two_sided_signs,
    estimate_active_signs,
    estimate_common_alpha,
    positive_only_signs,
)
from .stable import StableNormalScoreTransform
from .stg_test import stg_test


def wilson_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(0.5 + level / 2.0))
    phat = successes / total
    den = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / den
    half = z * np.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def population_gap(cfg: SimulationConfig) -> float:
    if cfg.model == "stable":
        return 0.0
    rho = cfg.dependence
    if cfg.margin == "symmetric":
        if cfg.corr_structure == "equicorr":
            rho_star = abs(rho)
        else:
            # All sign combinations are active; maximize absolute off-diagonal.
            rho_star = abs(rho)
        return (1.0 - rho_star) / (1.0 + rho_star)
    # Positive-only signs: rho_* is the largest raw off-diagonal correlation.
    if cfg.corr_structure == "equicorr":
        rho_star = rho
    else:
        # Toeplitz rho^{|i-j|}: for negative rho and d>=3 the largest raw
        # off-diagonal correlation may be rho^2 > 0.
        vals = [rho ** abs(i - j) for i in range(cfg.d) for j in range(i + 1, cfg.d)]
        rho_star = max(vals)
    return (1.0 - rho_star) / (1.0 + rho_star)


def _one_replication(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    transform: StableNormalScoreTransform | None,
) -> dict:
    if cfg.model == "stable":
        x = stable_factor_sample(
            cfg.n, cfg.d, cfg.alpha, cfg.dependence, rng,
            sign_structure=cfg.sign_structure,
            margin=cfg.margin,
        )
    elif cfg.model == "gaussian":
        assert transform is not None
        x, _ = gaussian_copula_stable_sample(
            cfg.n, cfg.d, cfg.alpha, cfg.dependence, rng,
            corr_structure=cfg.corr_structure,
            transform=transform,
        )
    else:
        raise ValueError("unknown model.")

    alpha_practical, _ = estimate_common_alpha(x, pool="median")

    if cfg.margin == "symmetric":
        active_oracle = all_two_sided_signs(cfg.d)
        active_practical = active_oracle
        active_recovered = True
    else:
        active_oracle = positive_only_signs(cfg.d)
        if cfg.active_set_mode == "estimated":
            active_practical, _ = estimate_active_signs(x, alpha_practical)
            active_recovered = active_practical == active_oracle
        else:
            active_practical = active_oracle
            active_recovered = True

    practical = stg_test(
        x,
        alpha_hat=alpha_practical,
        k=cfg.k,
        active_sets=active_practical,
        tau=cfg.tau,
        upper_tail_only=cfg.upper_tail_only,
    )
    oracle = stg_test(
        x,
        alpha_hat=cfg.alpha,
        k=cfg.k,
        active_sets=active_oracle,
        tau=cfg.tau,
        upper_tail_only=cfg.upper_tail_only,
    )

    tau_n = cfg.n ** (-0.5)
    z_model = norm.ppf(1.0 - tau_n)
    return {
        "reject_practical": practical.reject,
        "reject_oracle": oracle.reject,
        "T_practical": practical.t_global,
        "T_oracle": oracle.t_global,
        "delta_hat_practical": practical.delta_global,
        "delta_hat_oracle": oracle.delta_global,
        "alpha_hat": practical.alpha_hat,
        "active_recovered": active_recovered,
        "select_G_practical": practical.t_global > z_model,
        "select_G_oracle": oracle.t_global > z_model,
        "all_pairs_defined_practical": bool(practical.pair_stats) and all(p.defined for p in practical.pair_stats),
        "all_pairs_defined_oracle": bool(oracle.pair_stats) and all(p.defined for p in oracle.pair_stats),
    }


def run_configuration(
    cfg: SimulationConfig,
    mc: MonteCarloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary row, replication-level results)."""
    transform = None
    if cfg.model == "gaussian":
        transform = StableNormalScoreTransform(
            alpha=cfg.alpha,
            margin=cfg.margin,
            zmax=cfg.quantile_zmax,
            grid_size=cfg.quantile_grid_size,
            cache_dir=mc.cache_dir,
        )

    ss = SeedSequence(mc.seed)
    child_seeds = ss.spawn(mc.B)
    rows = []
    for b, s in enumerate(child_seeds, start=1):
        rng = default_rng(s)
        row = _one_replication(cfg, rng, transform)
        row["replication"] = b
        rows.append(row)
    reps = pd.DataFrame(rows)

    correct_practical = (
        1.0 - reps["select_G_practical"].mean()
        if cfg.model == "stable"
        else reps["select_G_practical"].mean()
    )
    correct_oracle = (
        1.0 - reps["select_G_oracle"].mean()
        if cfg.model == "stable"
        else reps["select_G_oracle"].mean()
    )

    p_pr = float(reps["reject_practical"].mean())
    p_or = float(reps["reject_oracle"].mean())
    lo_pr, hi_pr = wilson_interval(int(reps["reject_practical"].sum()), mc.B)
    lo_or, hi_or = wilson_interval(int(reps["reject_oracle"].sum()), mc.B)

    summary = {
        **asdict(cfg),
        "k": cfg.k,
        "B": mc.B,
        "population_gap": population_gap(cfg),
        "rejection_practical": p_pr,
        "rejection_oracle": p_or,
        "mcse_practical": np.sqrt(p_pr * (1.0 - p_pr) / mc.B),
        "mcse_oracle": np.sqrt(p_or * (1.0 - p_or) / mc.B),
        "wilson95_low_practical": lo_pr,
        "wilson95_high_practical": hi_pr,
        "wilson95_low_oracle": lo_or,
        "wilson95_high_oracle": hi_or,
        "alpha_hat_mean": reps["alpha_hat"].mean(),
        "alpha_hat_median": reps["alpha_hat"].median(),
        "active_recovery": reps["active_recovered"].mean(),
        "all_pairs_defined_practical": reps["all_pairs_defined_practical"].mean(),
        "all_pairs_defined_oracle": reps["all_pairs_defined_oracle"].mean(),
        "median_T_practical": reps["T_practical"].replace([np.inf, -np.inf], np.nan).median(),
        "median_delta_practical": reps["delta_hat_practical"].replace([np.inf, -np.inf], np.nan).median(),
        "model_selection_accuracy_practical": correct_practical,
        "model_selection_accuracy_oracle": correct_oracle,
    }
    return pd.DataFrame([summary]), reps


def _dgp_seed(cfg: SimulationConfig, base_seed: int) -> int:
    """Order-independent seed shared by configurations with the same DGP.

    Tuning changes such as k_gamma, active-set mode, or upper-tail-only do not
    alter the generated sample. This gives common random numbers for threshold
    sensitivity/calibration and makes comparisons across gamma substantially
    less noisy.
    """
    key = (
        cfg.model, cfg.n, cfg.d, cfg.alpha, cfg.dependence, cfg.margin,
        cfg.sign_structure, cfg.corr_structure, cfg.quantile_zmax,
        cfg.quantile_grid_size,
    )
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(digest, "little") % 2_000_000_000
    return int((base_seed + offset) % (2**32 - 1))


def _worker_run(args):
    idx, cfg, mc, save_replications = args
    mc_i = replace(mc, seed=_dgp_seed(cfg, mc.seed))
    summary, reps = run_configuration(cfg, mc_i)
    if save_replications:
        label = cfg.label or f"cfg_{idx:04d}"
        reps.to_csv(mc.output_dir / f"replications_{label}.csv", index=False)
    return idx, summary


def run_configurations(
    configs: Iterable[SimulationConfig],
    mc: MonteCarloConfig,
    save_replications: bool = False,
) -> pd.DataFrame:
    configs = list(configs)
    mc.output_dir.mkdir(parents=True, exist_ok=True)
    mc.cache_dir.mkdir(parents=True, exist_ok=True)

    # Prewarm Gaussian stable-quantile caches before multiprocessing so that
    # several workers never try to create the same cache file simultaneously.
    seen = set()
    for cfg in configs:
        if cfg.model != "gaussian":
            continue
        key = (cfg.alpha, cfg.margin, cfg.quantile_zmax, cfg.quantile_grid_size)
        if key not in seen:
            StableNormalScoreTransform(
                cfg.alpha, cfg.margin, cfg.quantile_zmax, cfg.quantile_grid_size, mc.cache_dir
            )
            seen.add(key)

    jobs = [(idx, cfg, mc, save_replications) for idx, cfg in enumerate(configs, start=1)]
    summaries = []
    if mc.workers <= 1:
        for idx, cfg, _, _ in jobs:
            print(f"[{idx}/{len(configs)}] {cfg.model} n={cfg.n} d={cfg.d} alpha={cfg.alpha} dep={cfg.dependence}")
            _, summary = _worker_run((idx, cfg, mc, save_replications))
            summaries.append((idx, summary))
    else:
        with ProcessPoolExecutor(max_workers=mc.workers) as ex:
            futs = {ex.submit(_worker_run, job): job[0] for job in jobs}
            for fut in as_completed(futs):
                idx, summary = fut.result()
                print(f"completed {idx}/{len(configs)}")
                summaries.append((idx, summary))

    summaries.sort(key=lambda x: x[0])
    ans = pd.concat([s for _, s in summaries], ignore_index=True)
    ans.to_csv(mc.output_dir / "simulation_summary.csv", index=False)
    return ans
