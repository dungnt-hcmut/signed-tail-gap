# signed-tail-gap

Reproducibility code for the numerical experiments accompanying the manuscript

**Multivariate Stable Dependence and Gaussian Copulas: Signed Tail-Gap Testing**

by **Thien Nguyen and Dung Nguyen**.

This repository contains the computational implementation used for the numerical study in Section 6 of the manuscript. Its purpose is to make the simulation design, calibration procedures, diagnostic experiments, and reported numerical results reproducible.

## Repository structure

```text
signed-tail-gap/
├── stg/                         # Core implementation
├── run_section6.py              # Main experiment driver
├── test_section6_validation.py  # Section 6 implementation checks
├── validate_numerics.py         # Numerical validation checks
├── requirements.txt             # Python dependencies
├── reference_results/           # Retained benchmark results
├── CITATION.cff                 # Citation metadata
└── README.md
```

The directory `stg/` contains the computational routines used by the simulation study, while `run_section6.py` provides a unified interface for reproducing the experiments reported in the manuscript.

The directory `reference_results/` contains retained benchmark outputs corresponding to the reference numerical configuration used in the manuscript.

## Computational environment

Python 3.11 or later is recommended.

Install the required packages with

```bash
python -m pip install -r requirements.txt
```

The numerical results accompanying the manuscript were generated and checked using a Python 3.11 environment.

Because numerical libraries may evolve over time, small machine-dependent differences in floating-point output are possible. Such differences should not materially affect the conclusions of the experiments.

## Quick validation

Before running the full simulation study, the implementation can be checked using

```bash
python test_section6_validation.py
python validate_numerics.py
```

These scripts perform the implementation and numerical consistency checks provided with the repository.

A short computational run can also be executed using

```bash
python run_section6.py \
    --profile smoke \
    --seed 20260815 \
    --output outputs/section6_smoke
```

The smoke profile is intended only to verify that the computational pipeline runs correctly. It is not intended to reproduce the final Monte Carlo accuracy reported in the manuscript.

## Reproducing the benchmark experiment

The principal benchmark configuration can be run with

```bash
python run_section6.py \
    --profile benchmark \
    --B 2000 \
    --seed 20260815 \
    --workers 4 \
    --output outputs/section6_benchmark
```

Here:

- `B` denotes the number of Monte Carlo replications;
- `seed` controls the pseudorandom-number initialization;
- `workers` specifies the number of parallel worker processes;
- `output` specifies the directory in which tables, diagnostics, and figures are stored.

The value of `workers` may be changed according to the available hardware without changing the statistical design of the experiment.

## Experiment profiles

The driver `run_section6.py` provides several experiment profiles, including

```text
smoke
benchmark
tuning
stress
full
exhaustive
```

These profiles are designed for different computational purposes.

- `smoke` provides a rapid implementation check.
- `benchmark` reproduces the main reference experiment.
- `tuning` examines numerical calibration choices.
- `stress` performs additional robustness and diagnostic experiments.
- `full` executes the principal collection of experiments.
- `exhaustive` runs the most computationally extensive collection of available experiments.

For reproduction of the results reported in the manuscript, the benchmark or corresponding manuscript-specific profile should be used rather than the smoke configuration.

## Reference results

The directory

```text
reference_results/
```

contains retained numerical outputs from the benchmark computations used for checking reproducibility.

These files provide reference tables and figures against which newly generated output can be compared. Exact floating-point equality is not required across different operating systems, processors, or numerical-library versions, but the reported statistical conclusions and numerical patterns should remain stable.

## Randomness and reproducibility

The principal numerical experiments use the fixed seed

```text
20260815
```

unless otherwise specified.

All stochastic calculations are generated algorithmically from the stated pseudorandom initialization. Repeated computations under the same software environment should therefore produce the same or numerically equivalent results.

For Monte Carlo experiments, minor variation can occur if a user intentionally changes the random seed, number of replications, or numerical environment.

## Relation to the manuscript

The code in this repository supports the numerical investigation in Section 6 of the manuscript **Multivariate Stable Dependence and Gaussian Copulas: Signed Tail-Gap Testing**.

In particular, it is intended to reproduce the computational evidence used to examine the finite-sample behavior of the signed tail-gap methodology and the associated diagnostic and robustness experiments.

The mathematical statements, assumptions, and theoretical results are given in the manuscript. This repository should therefore be read as computational supplementary material rather than as a replacement for the mathematical definitions in the paper.

## Version corresponding to peer review

The version associated with the submitted manuscript is archived as the GitHub release

```text
v1.0-review
```

Reviewers wishing to reproduce the submitted numerical results are encouraged to use that release rather than a later development version of the `main` branch.

## Citation

Citation metadata are provided in

```text
CITATION.cff
```

When referring specifically to the computational implementation, please cite the accompanying manuscript and, where appropriate, this repository release.

## License and use

No open-source license is assigned to this repository at the peer-review stage.

The source code is made publicly accessible primarily for inspection, verification, and reproducibility of the accompanying research manuscript. Unless otherwise stated, normal copyright restrictions therefore apply.

## Contact

Questions concerning the mathematical methodology should be addressed with reference to the accompanying manuscript.

Technical issues concerning reproducibility may be reported through the GitHub issue tracker.
