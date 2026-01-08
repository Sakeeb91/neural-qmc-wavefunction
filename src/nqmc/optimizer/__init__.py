"""VMC optimization algorithms.

This module provides optimizers for variational Monte Carlo,
combining MCMC sampling with gradient-based parameter updates.

Example:
    >>> from nqmc.optimizer import VMCOptimizer
    >>> optimizer = VMCOptimizer(learning_rate=1e-3)
    >>> params, history = optimizer.train(wf, params, molecule, n_steps=1000)
"""
from nqmc.optimizer.vmc import (
    VMCOptimizer,
    VMCState,
    compute_gradient,
    estimate_energy,
)

__all__ = [
    "VMCOptimizer",
    "VMCState",
    "compute_gradient",
    "estimate_energy",
]
