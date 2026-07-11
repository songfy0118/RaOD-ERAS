"""Research code for RaOD-ERAS road-obstacle anomaly segmentation."""

from .config import ExperimentConfig, MethodConfig
from .experiment import run_experiment

__all__ = ["ExperimentConfig", "MethodConfig", "run_experiment"]
