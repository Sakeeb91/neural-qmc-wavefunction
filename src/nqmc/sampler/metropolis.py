"""Metropolis-Hastings MCMC sampler for VMC.

This module implements the Metropolis-Hastings algorithm for sampling
electron configurations from |ψ(r)|². The samples are used to estimate
the variational energy and its gradient.

Example:
    >>> from nqmc.sampler import MetropolisSampler
    >>> sampler = MetropolisSampler(step_size=0.1)
    >>> key = jax.random.PRNGKey(0)
    >>> samples, acceptance_rate = sampler.sample(
    ...     key, log_prob_fn, r_init, n_samples=1000
    ... )
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import random


class SamplerState(NamedTuple):
    """State for the MCMC sampler.

    Attributes:
        positions: Current electron positions, shape (n_chains, n_electrons, 3)
        log_prob: Current log probability values, shape (n_chains,)
        n_accepted: Number of accepted moves per chain, shape (n_chains,)
        n_steps: Total number of steps taken
    """

    positions: jnp.ndarray
    log_prob: jnp.ndarray
    n_accepted: jnp.ndarray
    n_steps: int


@dataclass
class MetropolisSampler:
    """Metropolis-Hastings sampler for electron configurations.

    Samples from |ψ(r)|² using symmetric Gaussian proposals.
    Supports multiple independent chains for parallel sampling.

    Attributes:
        step_size: Standard deviation of Gaussian proposal (in Bohr)
        target_acceptance: Target acceptance rate for adaptive step size
        adapt_step: Whether to adapt step size during burn-in
    """

    step_size: float = 0.1
    target_acceptance: float = 0.5
    adapt_step: bool = True

    def init_state(
        self,
        key: random.PRNGKey,
        log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
        n_chains: int,
        n_electrons: int,
        init_width: float = 1.0,
    ) -> SamplerState:
        """Initialize sampler state with random positions.

        Args:
            key: JAX random key
            log_prob_fn: Function r -> log|ψ(r)|²
            n_chains: Number of independent Markov chains
            n_electrons: Number of electrons
            init_width: Width of initial position distribution (in Bohr)

        Returns:
            Initial SamplerState
        """
        # Initialize positions randomly around origin
        positions = init_width * random.normal(key, (n_chains, n_electrons, 3))

        # Compute initial log probabilities
        log_prob = jax.vmap(log_prob_fn)(positions)

        return SamplerState(
            positions=positions,
            log_prob=log_prob,
            n_accepted=jnp.zeros(n_chains),
            n_steps=0,
        )
