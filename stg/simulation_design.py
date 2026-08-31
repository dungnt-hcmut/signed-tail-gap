from __future__ import annotations

from itertools import product

from .config import (
    BASELINE_K_GAMMA,
    BENCHMARK_LAMBDAS,
    BENCHMARK_RHOS,
    THRESHOLD_GAMMAS,
    SimulationConfig,
)


def smoke_design() -> list[SimulationConfig]:
    """Very small pipeline check; not for inferential interpretation."""
    return [
        SimulationConfig("stable", 300, 2, 1.5, 0.30, label="smoke_stable", experiment="smoke"),
        SimulationConfig("gaussian", 300, 2, 1.5, 0.50, label="smoke_gaussian", experiment="smoke"),
        SimulationConfig(
            "stable", 300, 2, 0.8, 0.30,
            margin="positive", sign_structure="positive", active_set_mode="estimated",
            label="smoke_positive_stable", experiment="smoke",
        ),
    ]


def benchmark_design() -> list[SimulationConfig]:
    """Main finite-sample benchmark reported in the body of Section 6.

    It deliberately stays in the central bivariate, two-sided regime. The aim
    is to assess finite-sample level and power where the pointwise asymptotic
    theory is expected to become visible at realistic n. Boundary regimes are
    kept out of the benchmark and reported separately as stress diagnostics.
    """
    out: list[SimulationConfig] = []

    # Stable-null size benchmark.
    for n, lam in product((500, 1000, 2500), BENCHMARK_LAMBDAS):
        out.append(SimulationConfig(
            "stable", n, 2, 1.5, lam,
            label=f"bench_size_n{n}_l{lam}", experiment="benchmark_size",
        ))

    # Gaussian-copula power benchmark. rho=.30,.50,.70 correspond to
    # population gaps about .538,.333,.176 and give a clean separation axis.
    for n, rho in product((500, 1000, 2500), BENCHMARK_RHOS):
        out.append(SimulationConfig(
            "gaussian", n, 2, 1.5, rho,
            label=f"bench_power_n{n}_r{rho}", experiment="benchmark_power",
        ))

    # Full signed procedure in an opposite-sign stable configuration. The
    # upper-tail-only ablation is placed in the stress design.
    out.append(SimulationConfig(
        "stable", 2500, 2, 1.5, 0.30,
        sign_structure="alternating", label="signed_full_stable", experiment="signed_full",
    ))
    out.append(SimulationConfig(
        "gaussian", 2500, 2, 1.5, -0.80,
        corr_structure="toeplitz", label="signed_full_gaussian", experiment="signed_full",
    ))
    return out


def core_design() -> list[SimulationConfig]:
    """Backward-compatible alias for the final benchmark profile."""
    return benchmark_design()


def threshold_design() -> list[SimulationConfig]:
    """Prespecified bias--variance sensitivity around the central benchmark."""
    out: list[SimulationConfig] = []
    for gamma in THRESHOLD_GAMMAS:
        out.append(SimulationConfig(
            "stable", 2500, 2, 1.5, 0.30, k_gamma=gamma,
            label=f"threshold_S_g{gamma}", experiment="threshold",
        ))
        out.append(SimulationConfig(
            "gaussian", 2500, 2, 1.5, 0.70, k_gamma=gamma,
            label=f"threshold_G_g{gamma}", experiment="threshold",
        ))
    return out


def tuning_design() -> list[SimulationConfig]:
    """Development/audit profile for the baseline threshold exponent.

    Configurations that differ only in k_gamma share the same generated Monte
    Carlo samples through the common-random-number seeding in simulation.py.
    This profile is not needed for the final paper run once gamma=.40 is fixed.
    """
    out: list[SimulationConfig] = []
    for n in (500, 1000, 2500):
        for gamma in THRESHOLD_GAMMAS:
            out.append(SimulationConfig(
                "stable", n, 2, 1.5, 0.30, k_gamma=gamma,
                label=f"tune_S_n{n}_g{gamma}", experiment="tuning",
            ))
            for rho in BENCHMARK_RHOS:
                out.append(SimulationConfig(
                    "gaussian", n, 2, 1.5, rho, k_gamma=gamma,
                    label=f"tune_G_n{n}_r{rho}_g{gamma}", experiment="tuning",
                ))
    return out


def calibration_design() -> list[SimulationConfig]:
    """Backward-compatible alias for threshold tuning/auditing."""
    return tuning_design()


def stress_design() -> list[SimulationConfig]:
    """Targeted regimes where slow finite-sample convergence is expected."""
    out: list[SimulationConfig] = []

    # Weak stable spectral dependence: Delta(P)=0 is unchanged, but the stable
    # witness mass becomes small and the first-order tail is difficult to see.
    for n in (1000, 2500, 5000):
        for lam in (0.02, 0.05, 0.10):
            out.append(SimulationConfig(
                "stable", n, 2, 1.5, lam,
                label=f"weakS_n{n}_l{lam}", experiment="weak_stable",
            ))

    # Gaussian alternatives with increasingly small population gap.
    for n in (1000, 2500, 5000):
        for rho in (0.90, 0.95, 0.98):
            out.append(SimulationConfig(
                "gaussian", n, 2, 1.5, rho,
                label=f"nearG_n{n}_r{rho}", experiment="near_boundary",
            ))

    # Tail-index sensitivity, including the non-Gaussian/near-Gaussian boundary.
    for alpha in (1.20, 1.80):
        out.append(SimulationConfig(
            "stable", 1000, 2, alpha, 0.30,
            label=f"alphaS_{alpha}", experiment="tail_index",
        ))
        out.append(SimulationConfig(
            "gaussian", 1000, 2, alpha, 0.70,
            label=f"alphaG_{alpha}", experiment="tail_index",
        ))
    out.append(SimulationConfig(
        "stable", 2500, 2, 1.95, 0.30,
        label="near2_stable", experiment="near_alpha2",
    ))
    out.append(SimulationConfig(
        "gaussian", 2500, 2, 1.95, 0.70,
        label="near2_gaussian", experiment="near_alpha2",
    ))

    # Dimension stress: fixed-d theory does not imply uniform performance as
    # the number of signed pairwise constraints increases.
    for d in (5, 10):
        out.append(SimulationConfig(
            "stable", 2500, d, 1.5, 0.30,
            label=f"dimS_d{d}", experiment="dimension",
        ))
        out.append(SimulationConfig(
            "gaussian", 2500, d, 1.5, 0.70,
            label=f"dimG_d{d}", experiment="dimension",
        ))

    # One-sided active-sign recovery.
    for model, dep in (("stable", 0.30), ("gaussian", 0.70)):
        out.append(SimulationConfig(
            model, 2500, 2, 0.8, dep,
            margin="positive", sign_structure="positive", active_set_mode="estimated",
            label=f"onesided_{model}", experiment="one_sided",
        ))

    # Threshold sensitivity and signed ablation.
    out.extend(threshold_design())
    out.append(SimulationConfig(
        "stable", 2500, 2, 1.5, 0.30,
        sign_structure="alternating", upper_tail_only=True,
        label="signed_upper_only_stable", experiment="ablation",
    ))
    out.append(SimulationConfig(
        "gaussian", 2500, 2, 1.5, -0.80,
        corr_structure="toeplitz", upper_tail_only=True,
        label="signed_upper_only_gaussian", experiment="ablation",
    ))
    # Full signed counterparts in the same DGPs, useful when stress is run by itself.
    out.append(SimulationConfig(
        "stable", 2500, 2, 1.5, 0.30,
        sign_structure="alternating", label="signed_full_stable_stress", experiment="signed_full",
    ))
    out.append(SimulationConfig(
        "gaussian", 2500, 2, 1.5, -0.80,
        corr_structure="toeplitz", label="signed_full_gaussian_stress", experiment="signed_full",
    ))
    return out


def exhaustive_design() -> list[SimulationConfig]:
    """Optional large Cartesian grid retained for supplementary exploration."""
    out: list[SimulationConfig] = []
    ns = (500, 1000, 2500)
    ds = (2, 5, 10)
    alphas = (1.2, 1.5, 1.8)
    for n, d, alpha, lam, sign in product(ns, ds, alphas, (0.10, 0.30, 0.60), ("aligned", "alternating")):
        out.append(SimulationConfig(
            "stable", n, d, alpha, lam, sign_structure=sign,
            label=f"XS_n{n}_d{d}_a{alpha}_l{lam}_{sign}", experiment="exhaustive",
        ))
    for n, d, alpha, rho in product(ns, ds, alphas, (0.30, 0.50, 0.70, 0.90)):
        out.append(SimulationConfig(
            "gaussian", n, d, alpha, rho,
            label=f"XG_n{n}_d{d}_a{alpha}_r{rho}", experiment="exhaustive",
        ))
    return out


def full_design() -> list[SimulationConfig]:
    # Deduplicate configurations that are present in both benchmark and stress
    # solely to make the standalone stress profile self-contained.
    configs = benchmark_design() + stress_design()
    seen = set()
    out = []
    for c in configs:
        key = (
            c.model, c.n, c.d, c.alpha, c.dependence, c.margin, c.sign_structure,
            c.corr_structure, c.k_gamma, c.tau, c.active_set_mode, c.upper_tail_only,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
