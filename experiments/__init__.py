"""Experiment and visualization code for the step-by-step tests."""

from .runners import (
    run_adapter_ablation_test,
    run_dinov3_cost_volume_comparison,
    run_geometry_adapter_test,
)

__all__ = [
    "run_adapter_ablation_test",
    "run_dinov3_cost_volume_comparison",
    "run_geometry_adapter_test",
]
