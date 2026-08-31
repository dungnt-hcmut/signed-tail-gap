from __future__ import annotations

import argparse
from pathlib import Path

from stg.config import MonteCarloConfig
from stg.simulation import run_configurations
from stg.simulation_design import (
    benchmark_design,
    calibration_design,
    core_design,
    exhaustive_design,
    full_design,
    smoke_design,
    stress_design,
    tuning_design,
)
from stg.tables_plots import plot_power_gap, plot_threshold_sensitivity, section6_tables


def main():
    p = argparse.ArgumentParser(description="Run Section 6 Monte Carlo study.")
    p.add_argument(
        "--profile",
        choices=["smoke", "benchmark", "core", "tuning", "calibration", "stress", "full", "exhaustive"],
        default="smoke",
    )
    p.add_argument("--B", type=int, default=None, help="Override Monte Carlo replications.")
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output directory. Default: outputs/section6_<profile>, preventing benchmark/stress overwrite.",
    )
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--save-replications", action="store_true")
    args = p.parse_args()

    designs = {
        "smoke": smoke_design,
        "benchmark": benchmark_design,
        "core": core_design,  # legacy alias
        "tuning": tuning_design,
        "calibration": calibration_design,  # legacy alias
        "stress": stress_design,
        "full": full_design,
        "exhaustive": exhaustive_design,
    }
    default_B = 20 if args.profile == "smoke" else (300 if args.profile in {"tuning", "calibration"} else 2000)
    B = args.B if args.B is not None else default_B
    outdir = args.output if args.output is not None else Path(f"outputs/section6_{args.profile}")

    mc = MonteCarloConfig(B=B, seed=args.seed, workers=args.workers, output_dir=outdir, cache_dir=args.cache)
    summary = run_configurations(designs[args.profile](), mc, save_replications=args.save_replications)
    section6_tables(summary, outdir)
    plot_power_gap(summary, outdir)
    plot_threshold_sensitivity(summary, outdir)
    print(f"Saved Section 6 outputs to {outdir}")


if __name__ == "__main__":
    main()
