from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import levy_stable, norm

from stg.estimators import estimate_alpha_ecf_1d
from stg.stable import StableNormalScoreTransform

levy_stable.parameterization = "S1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--margin", choices=["symmetric", "positive"], default="symmetric")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    args = p.parse_args()

    trans = StableNormalScoreTransform(args.alpha, args.margin, cache_dir=args.cache)
    # Test central interpolation only; tail branch is an explicit asymptotic formula.
    z = np.array([-3.413, -2.731, -1.917, -1.083, -0.337, 0.417, 1.237, 2.031, 2.817, 3.367])
    u = norm.cdf(z)
    if args.margin == "symmetric":
        exact = levy_stable.ppf(u, args.alpha, 0.0)
    else:
        from stg.stable import positive_stable_s1_scale
        exact = levy_stable.ppf(u, args.alpha, 1.0, scale=positive_stable_s1_scale(args.alpha))
    approx = trans(z)
    rel = np.abs(approx - exact) / np.maximum(1.0, np.abs(exact))
    print("central transform validation")
    print("max relative/scaled error:", float(rel.max()))
    print("median error:", float(np.median(rel)))

    rng = np.random.default_rng(123)
    zsim = rng.standard_normal(50000)
    xsim = trans(zsim)
    ahat = estimate_alpha_ecf_1d(xsim)
    print("alpha target:", args.alpha)
    print("alpha ECF estimate from transformed sample:", ahat)


if __name__ == "__main__":
    main()
