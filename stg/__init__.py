"""Signed tail-gap code for Sections 6 and 7."""

from .config import SimulationConfig, MonteCarloConfig, EmpiricalConfig
from .stg_test import STGResult, stg_test

__all__ = [
    "SimulationConfig",
    "MonteCarloConfig",
    "EmpiricalConfig",
    "STGResult",
    "stg_test",
]
