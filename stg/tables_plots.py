from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BASELINE_K_GAMMA


def _save(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    if df.empty:
        return
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(path, index=False)

def _save_figure(fig: plt.Figure, outdir: Path, stem: str, png_dpi: int = 250) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=png_dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.eps", format="eps", bbox_inches="tight")



def section6_tables(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ci = [
        "wilson95_low_practical", "wilson95_high_practical",
        "wilson95_low_oracle", "wilson95_high_oracle",
    ]

    size = summary[summary.experiment == "benchmark_size"].copy()
    power = summary[summary.experiment == "benchmark_power"].copy()
    _save(size, outdir / "table_benchmark_size.csv", [
        "n", "d", "alpha", "dependence", "k_gamma", "k",
        "rejection_practical", "rejection_oracle", *ci,
        "alpha_hat_mean", "model_selection_accuracy_practical",
    ])
    _save(power, outdir / "table_benchmark_power.csv", [
        "n", "d", "alpha", "dependence", "population_gap", "k_gamma", "k",
        "rejection_practical", "rejection_oracle", *ci,
        "alpha_hat_mean", "model_selection_accuracy_practical",
    ])

    # Backward-compatible manuscript file names.
    _save(size, outdir / "table_stable_size.csv", [
        "n", "d", "alpha", "dependence", "k_gamma", "k",
        "rejection_practical", "rejection_oracle", *ci,
    ])
    _save(power, outdir / "table_gaussian_power.csv", [
        "n", "d", "alpha", "dependence", "population_gap", "k_gamma", "k",
        "rejection_practical", "rejection_oracle", *ci,
    ])

    threshold = summary[summary.experiment == "threshold"].copy()
    _save(threshold, outdir / "table_threshold_sensitivity.csv", [
        "model", "n", "alpha", "dependence", "k_gamma", "k",
        "rejection_practical", "rejection_oracle", *ci,
    ])

    signed = summary[summary.experiment.isin(["signed_full", "ablation"])].copy()
    _save(signed, outdir / "table_signed_ablation.csv", [
        "label", "model", "n", "alpha", "dependence", "sign_structure", "corr_structure",
        "upper_tail_only", "k_gamma", "k", "rejection_practical", "rejection_oracle",
        "model_selection_accuracy_practical",
    ])

    stress = summary[summary.experiment.isin([
        "weak_stable", "near_boundary", "tail_index", "near_alpha2", "dimension", "one_sided",
    ])].copy()
    _save(stress, outdir / "table_stress_diagnostics.csv", [
        "label", "experiment", "model", "n", "d", "alpha", "dependence", "population_gap",
        "k_gamma", "k", "active_set_mode", "rejection_practical", "rejection_oracle", *ci,
        "active_recovery", "all_pairs_defined_practical",
    ])

    # Full compact robustness export for audit/reproducibility.
    robust_cols = [
        "label", "experiment", "model", "n", "d", "alpha", "dependence", "population_gap",
        "k_gamma", "k", "upper_tail_only", "active_set_mode",
        "rejection_practical", "rejection_oracle", "wilson95_low_practical", "wilson95_high_practical",
        "active_recovery", "all_pairs_defined_practical", "model_selection_accuracy_practical",
        "model_selection_accuracy_oracle",
    ]
    _save(summary, outdir / "table_robustness.csv", robust_cols)

    tune = summary[summary.experiment == "tuning"].copy()
    if not tune.empty:
        _save(tune, outdir / "table_threshold_tuning.csv", [
            "model", "n", "alpha", "dependence", "population_gap", "k_gamma", "k",
            "rejection_practical", "rejection_oracle", *ci,
        ])


def plot_power_gap(summary: pd.DataFrame, outdir: Path) -> None:
    g = summary[
        (summary.model == "gaussian")
        & (summary.experiment == "benchmark_power")
        & np.isclose(summary.k_gamma, BASELINE_K_GAMMA)
        & (summary.d == 2)
        & np.isclose(summary.alpha, 1.5)
        & (~summary.upper_tail_only)
    ].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n, sub in g.groupby("n"):
        q = sub.sort_values("population_gap")
        ax.plot(q.population_gap, q.rejection_practical, marker="o", label=f"n={n}")
    ax.set_xlabel(r"Population tail gap $\Delta(P)$")
    ax.set_ylabel("Empirical rejection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, outdir, "fig_power_gap", png_dpi=250)
    plt.close(fig)


def plot_threshold_sensitivity(summary: pd.DataFrame, outdir: Path) -> None:
    sub = summary[summary.experiment == "threshold"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model, q in sub.groupby("model"):
        q = q.sort_values("k_gamma")
        ax.plot(q.k_gamma, q.rejection_practical, marker="o", label=model)
    ax.axvline(BASELINE_K_GAMMA, linestyle="--", linewidth=1.0, label="baseline")
    ax.axhline(0.05, linestyle=":", linewidth=1.0, label="nominal level")
    ax.set_xlabel(r"Threshold exponent $\gamma$ in $k_n=\lfloor n^\gamma\rfloor$")
    ax.set_ylabel("Empirical rejection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, outdir, "fig_threshold_sensitivity", png_dpi=250)
    plt.close(fig)
