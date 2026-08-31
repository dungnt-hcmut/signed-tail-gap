from pathlib import Path
import numpy as np
from numpy.random import default_rng

from stg.config import SimulationConfig, MonteCarloConfig
from stg.dgp import stable_factor_sample
from stg.estimators import all_two_sided_signs, estimate_common_alpha
from stg.simulation import population_gap, run_configuration
from stg.stg_test import stg_test


def main():
    rng = default_rng(12345)
    x = stable_factor_sample(800, 2, 1.5, 0.30, rng, sign_structure="aligned", margin="symmetric")
    alpha_hat, _ = estimate_common_alpha(x)
    assert np.isfinite(alpha_hat) and 0.2 <= alpha_hat <= 2.0

    result = stg_test(x, alpha_hat, k=int(800 ** 0.40), active_sets=all_two_sided_signs(2), tau=0.05)
    assert len(result.pair_stats) == 4
    assert np.isfinite(result.t_global)
    assert 0.0 <= result.p_value <= 1.0

    gcfg = SimulationConfig("gaussian", 500, 2, 1.5, 0.5)
    assert abs(population_gap(gcfg) - 1.0 / 3.0) < 1e-12

    scfg = SimulationConfig("stable", 300, 2, 1.5, 0.30)
    mc = MonteCarloConfig(B=3, seed=20260815, workers=1, output_dir=Path("outputs/validation"), cache_dir=Path("cache"))
    summary, reps = run_configuration(scfg, mc)
    assert len(summary) == 1 and len(reps) == 3

    print("ALL_SECTION6_VALIDATION_CHECKS_PASSED")


if __name__ == "__main__":
    main()
