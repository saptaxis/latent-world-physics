"""Probing framework for mechanistic analysis of RL agent representations.

Key components:
    targets: Behavioral target computation (TWR, descent rate, etc.)
    hooks: Forward hook activation capture on policy networks
    collection: Probe data collection (activations + targets)
    training: Ridge regression probe training with episode-level CV
"""
from lwp.probing.targets import (
    compute_behavioral_targets,
    PARAMETRIC_TARGET_NAMES,
    BEHAVIORAL_TARGET_NAMES,
    ALL_TARGET_NAMES,
)
from lwp.probing.hooks import ActivationCollector
from lwp.probing.collection import collect_probe_data
from lwp.probing.training import train_single_probe, train_all_probes
