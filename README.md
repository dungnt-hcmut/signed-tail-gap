# Signed Tail-Gap (STG) Test

Reproducibility code for the paper

**Multivariate Stable Dependence and Gaussian Copulas: Signed Tail-Gap Testing**  
Thien Nguyen and Dung Nguyen

This repository contains the Python implementation used for the numerical study in Section 6 of the manuscript. It implements the signed tail-gap (STG) statistic for discriminating genuine multivariate stable dependence from Gaussian-copula dependence with the same stable margins.

## Repository contents

- `run_section6.py` — main Monte Carlo runner.
- `stg/` — simulation, estimators, STG statistic, DGPs, stable transforms, and table/figure routines.
- `validate_numerics.py` — numerical validation checks.
- `test_section6_validation.py` — Section 6 implementation checks.
- `test_smoke.py` — lightweight smoke test.
- `reference_results/` — retained benchmark results for the manuscript (`B=2000`).
- `cache/` — validated stable normal-score transform caches used to accelerate reproducibility.
- `figures/` — final EPS vector figures used by the manuscript.
- `WORKFLOW.tex` — detailed copy-and-paste computational workflow.

The archived empirical Section 7 code is intentionally **not** included in this repository because it is not part of the submitted manuscript.

## Python environment

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Validation

Before running the simulations:

```bash
python -m compileall -q .
python test_section6_validation.py
python validate_numerics.py
```

A lightweight execution check is:

```bash
python run_section6.py \
    --profile smoke \
    --B 20 \
    --workers 1 \
    --output outputs/section6_smoke
```

## Reproduce the manuscript benchmark

The manuscript benchmark uses `B=2000`, nominal level `0.05`, seed `20260815`, and baseline threshold exponent `gamma=0.40`.

```bash
python run_section6.py \
    --profile benchmark \
    --B 2000 \
    --seed 20260815 \
    --workers 4 \
    --output outputs/section6_benchmark
```

Run the stress diagnostics separately:

```bash
python run_section6.py \
    --profile stress \
    --B 2000 \
    --seed 20260815 \
    --workers 4 \
    --output outputs/section6_stress
```

For Apple Silicon, if BLAS oversubscription causes instability or excessive CPU usage:

```bash
VECLIB_MAXIMUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python run_section6.py \
    --profile benchmark \
    --B 2000 \
    --seed 20260815 \
    --workers 4 \
    --output outputs/section6_benchmark
```

## Figure formats

The plotting routines export each manuscript figure in vector PDF/EPS and high-resolution PNG formats. Final EPS copies are also retained in `figures/`.

## Reference results

`reference_results/benchmark_B2000_reference.csv` contains the retained `B=2000` benchmark used to audit reproduced output. Monte Carlo reruns on different software stacks may differ slightly within simulation error, but gross discrepancies indicate a configuration or implementation mismatch.

## Citation

If this code is useful, please cite the associated manuscript. A machine-readable citation template is provided in `CITATION.cff`.

## License

No open-source license has been assigned in this repository. Unless and until the authors add a license, normal copyright restrictions apply.
