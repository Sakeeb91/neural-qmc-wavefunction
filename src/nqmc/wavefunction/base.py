"""Abstract wavefunction interface for neural QMC.

This module defines the base interface that all wavefunction ansätze must
implement. The interface is designed to work with JAX for automatic
differentiation and JIT compilation.

Example:
    >>> class MyWavefunction(Wavefunction):
    ...     def __call__(self, params, r):
    ...         return jnp.sum(r)  # Trivial example
    ...
    >>> wf = MyWavefunction()
    >>> params = wf.init(jax.random.PRNGKey(0), jnp.zeros((2, 3)))
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import jax.numpy as jnp
from jax import random

# Type alias for parameter pytrees
Params = Dict[str, Any]


class Wavefunction(ABC):
    """Abstract base class for neural network wavefunctions.

    A wavefunction ansatz must implement methods for:
    - Computing log|ψ(r)| given electron positions
    - Initializing network parameters
    - Computing log probability for MCMC sampling

    The wavefunction is represented in log-space for numerical stability,
    as |ψ|² can span many orders of magnitude.
    """

    @abstractmethod
    def __call__(self, params: Params, r: jnp.ndarray) -> jnp.ndarray:
        """Compute log|ψ(r)| for given electron configuration.

        Args:
            params: Network parameters (pytree)
            r: Electron positions, shape (n_electrons, 3)

        Returns:
            Scalar log|ψ(r)|
        """
        pass

    @abstractmethod
    def init(self, key: random.PRNGKey, r_sample: jnp.ndarray) -> Params:
        """Initialize network parameters.

        Args:
            key: JAX random key for initialization
            r_sample: Sample electron configuration for shape inference,
                      shape (n_electrons, 3)

        Returns:
            Initial parameters as pytree
        """
        pass

    def log_prob(self, params: Params, r: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability density log|ψ(r)|² = 2*log|ψ(r)|.

        Used for MCMC sampling where we sample from |ψ|².

        Args:
            params: Network parameters
            r: Electron positions, shape (n_electrons, 3)

        Returns:
            Scalar log|ψ(r)|²
        """
        return 2.0 * self(params, r)
