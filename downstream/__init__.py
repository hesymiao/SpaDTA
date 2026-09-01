"""Downstream analysis entrypoints for spaDTA."""

from .workflow import DownstreamConfig, run_downstream, run_downstream_for_sample, run_downstream_for_samples

__all__ = [
    "DownstreamConfig",
    "run_downstream",
    "run_downstream_for_sample",
    "run_downstream_for_samples",
]
