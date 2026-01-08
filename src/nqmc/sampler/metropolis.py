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
        init_width: float = 0.5,
        init_positions: jnp.ndarray = None,
    ) -> SamplerState:
        """Initialize sampler state with positions near molecule.

        Args:
            key: JAX random key
            log_prob_fn: Function r -> log|ψ(r)|²
            n_chains: Number of independent Markov chains
            n_electrons: Number of electrons
            init_width: Width of Gaussian noise around init positions (in Bohr)
            init_positions: Optional (n_electrons, 3) array of mean positions.
                           If None, uses origin.

        Returns:
            Initial SamplerState
        """
        if init_positions is None:
            # Default: electrons around origin
            init_positions = jnp.zeros((n_electrons, 3))

        # Replicate for all chains and add Gaussian noise
        # init_positions: (n_el, 3) -> (n_chains, n_el, 3)
        positions = init_positions[None, :, :] + init_width * random.normal(
            key, (n_chains, n_electrons, 3)
        )

        # Compute initial log probabilities
        log_prob = jax.vmap(log_prob_fn)(positions)

        return SamplerState(
            positions=positions,
            log_prob=log_prob,
            n_accepted=jnp.zeros(n_chains),
            n_steps=0,
        )

    def step(
        self,
        key: random.PRNGKey,
        state: SamplerState,
        log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> SamplerState:
        """Perform one Metropolis-Hastings step.

        Args:
            key: JAX random key
            state: Current sampler state
            log_prob_fn: Function r -> log|ψ(r)|²

        Returns:
            Updated SamplerState after one MH step
        """
        key_proposal, key_accept = random.split(key)

        # Generate proposals: r' = r + δ, where δ ~ N(0, step_size²)
        n_chains = state.positions.shape[0]
        proposals = state.positions + self.step_size * random.normal(
            key_proposal, state.positions.shape
        )

        # Compute log probability of proposals
        log_prob_proposed = jax.vmap(log_prob_fn)(proposals)

        # Compute acceptance probability: min(1, |ψ(r')|²/|ψ(r)|²)
        # In log space: log_accept = log_prob_proposed - log_prob_current
        log_accept_ratio = log_prob_proposed - state.log_prob

        # Accept with probability min(1, exp(log_accept_ratio))
        # Use log-uniform for numerical stability
        log_uniform = jnp.log(random.uniform(key_accept, (n_chains,)))
        accept = log_uniform < log_accept_ratio  # shape (n_chains,)

        # Update positions and log_prob where accepted
        new_positions = jnp.where(
            accept[:, None, None], proposals, state.positions
        )
        new_log_prob = jnp.where(accept, log_prob_proposed, state.log_prob)
        new_n_accepted = state.n_accepted + accept.astype(jnp.float32)

        return SamplerState(
            positions=new_positions,
            log_prob=new_log_prob,
            n_accepted=new_n_accepted,
            n_steps=state.n_steps + 1,
        )

    def sample(
        self,
        key: random.PRNGKey,
        log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
        state: SamplerState,
        n_samples: int,
        n_burn: int = 100,
        n_thin: int = 1,
    ) -> Tuple[jnp.ndarray, SamplerState, float]:
        """Generate samples from |ψ|² with burn-in and thinning.

        Args:
            key: JAX random key
            log_prob_fn: Function r -> log|ψ(r)|²
            state: Initial sampler state
            n_samples: Number of samples to collect per chain
            n_burn: Number of burn-in steps (discarded)
            n_thin: Thinning interval (keep every n_thin-th sample)

        Returns:
            Tuple of:
            - samples: shape (n_samples, n_chains, n_electrons, 3)
            - final_state: SamplerState after sampling
            - acceptance_rate: average acceptance rate during sampling
        """
        n_chains = state.positions.shape[0]

        # Reset acceptance counter for this sampling run
        state = SamplerState(
            positions=state.positions,
            log_prob=state.log_prob,
            n_accepted=jnp.zeros(n_chains),
            n_steps=0,
        )

        # Burn-in phase
        key, subkey = random.split(key)
        keys_burn = random.split(subkey, n_burn)

        def burn_step(state, key):
            return self.step(key, state, log_prob_fn), None

        state, _ = jax.lax.scan(burn_step, state, keys_burn)

        # Reset acceptance counter after burn-in
        state = SamplerState(
            positions=state.positions,
            log_prob=state.log_prob,
            n_accepted=jnp.zeros(n_chains),
            n_steps=0,
        )

        # Sampling phase with thinning
        n_total_steps = n_samples * n_thin
        key, subkey = random.split(key)
        keys_sample = random.split(subkey, n_total_steps)

        def sample_step(state, key):
            new_state = self.step(key, state, log_prob_fn)
            return new_state, new_state.positions

        final_state, all_positions = jax.lax.scan(sample_step, state, keys_sample)

        # Apply thinning: keep every n_thin-th sample
        samples = all_positions[::n_thin]  # (n_samples, n_chains, n_el, 3)

        # Compute acceptance rate
        acceptance_rate = jnp.mean(final_state.n_accepted) / final_state.n_steps

        return samples, final_state, acceptance_rate

    def get_acceptance_rate(self, state: SamplerState) -> float:
        """Compute current acceptance rate.

        Args:
            state: Current sampler state

        Returns:
            Acceptance rate (0 to 1)
        """
        if state.n_steps == 0:
            return 0.0
        return float(jnp.mean(state.n_accepted) / state.n_steps)
