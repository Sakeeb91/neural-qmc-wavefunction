"""Variational Monte Carlo optimizer.

This module implements the VMC optimization loop, computing the variational
energy and its gradient using Monte Carlo sampling.

The variational energy is:
    E[θ] = ⟨ψ_θ|H|ψ_θ⟩/⟨ψ_θ|ψ_θ⟩ = ⟨E_L⟩_{|ψ_θ|²}

The gradient uses the log-derivative trick:
    ∇_θ E = 2⟨(E_L - ⟨E_L⟩) ∇_θ log|ψ|⟩

Example:
    >>> from nqmc.optimizer import VMCOptimizer
    >>> optimizer = VMCOptimizer(learning_rate=1e-3)
    >>> params, energies = optimizer.train(
    ...     wf, params, molecule, n_steps=1000
    ... )
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import random, vmap

from nqmc.hamiltonian.molecular import local_energy
from nqmc.sampler.metropolis import MetropolisSampler, SamplerState
from nqmc.systems.molecule import Molecule
from nqmc.wavefunction.base import Params


class VMCState(NamedTuple):
    """State for VMC optimization.

    Attributes:
        params: Wavefunction parameters
        opt_state: Optax optimizer state
        sampler_state: MCMC sampler state
        step: Current optimization step
    """

    params: Params
    opt_state: Any
    sampler_state: SamplerState
    step: int


def estimate_energy(
    wavefunction,
    params: Params,
    samples: jnp.ndarray,
    molecule: Molecule,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate variational energy from samples.

    Args:
        wavefunction: Wavefunction ansatz
        params: Current parameters
        samples: Electron configurations, shape (n_samples, n_chains, n_el, 3)
        molecule: Molecular system

    Returns:
        Tuple of (mean_energy, energy_std)
    """
    # Flatten samples: (n_samples, n_chains, n_el, 3) -> (n_total, n_el, 3)
    n_samples, n_chains = samples.shape[:2]
    samples_flat = samples.reshape(-1, *samples.shape[2:])

    # Compute local energies for all samples
    local_energies = vmap(
        lambda r: local_energy(wavefunction, params, r, molecule)
    )(samples_flat)

    # Statistics
    mean_energy = jnp.mean(local_energies)
    energy_std = jnp.std(local_energies) / jnp.sqrt(len(local_energies))

    return mean_energy, energy_std


def compute_gradient(
    wavefunction,
    params: Params,
    samples: jnp.ndarray,
    molecule: Molecule,
) -> Tuple[Params, jnp.ndarray, jnp.ndarray]:
    """Compute VMC energy gradient using log-derivative trick.

    The gradient is:
        ∇_θ E = 2⟨(E_L - ⟨E_L⟩) ∇_θ log|ψ|⟩

    This is an unbiased estimator of the true gradient.

    Args:
        wavefunction: Wavefunction ansatz
        params: Current parameters
        samples: Electron configurations, shape (n_samples, n_chains, n_el, 3)
        molecule: Molecular system

    Returns:
        Tuple of (gradient, mean_energy, energy_std)
    """
    # Flatten samples
    samples_flat = samples.reshape(-1, *samples.shape[2:])
    n_total = samples_flat.shape[0]

    # Compute local energies
    local_energies = vmap(
        lambda r: local_energy(wavefunction, params, r, molecule)
    )(samples_flat)

    mean_energy = jnp.mean(local_energies)
    energy_std = jnp.std(local_energies) / jnp.sqrt(n_total)

    # Centered energies for variance reduction
    centered_energies = local_energies - mean_energy

    # Compute gradient of log|ψ| w.r.t. params for each sample
    def grad_log_psi(r):
        return jax.grad(lambda p: wavefunction(p, r))(params)

    grad_log_psis = vmap(grad_log_psi)(samples_flat)

    # Gradient: 2 * mean((E_L - <E_L>) * ∇log|ψ|)
    def weighted_sum(grad_tree):
        return jax.tree.map(
            lambda g: 2.0 * jnp.mean(centered_energies[:, None] * g.reshape(n_total, -1), axis=0).reshape(g.shape[1:])
            if g.ndim > 1 else 2.0 * jnp.mean(centered_energies * g, axis=0),
            grad_tree
        )

    gradient = weighted_sum(grad_log_psis)

    return gradient, mean_energy, energy_std


@dataclass
class VMCOptimizer:
    """Variational Monte Carlo optimizer.

    Combines MCMC sampling with gradient descent to optimize
    the wavefunction parameters.

    Attributes:
        learning_rate: Base learning rate for Adam optimizer
        n_samples: Number of MCMC samples per optimization step
        n_chains: Number of parallel MCMC chains
        n_burn: Number of burn-in steps for MCMC
        step_size: MCMC proposal step size (in Bohr)
        clip_grad: Maximum gradient norm (for stability)
    """

    learning_rate: float = 1e-2
    n_samples: int = 200
    n_chains: int = 32
    n_burn: int = 200
    step_size: float = 0.5
    clip_grad: float = 1.0

    def init(
        self,
        key: random.PRNGKey,
        wavefunction,
        params: Params,
        molecule: Molecule,
    ) -> VMCState:
        """Initialize VMC optimization state.

        Args:
            key: JAX random key
            wavefunction: Wavefunction ansatz
            params: Initial parameters
            molecule: Molecular system

        Returns:
            Initial VMCState
        """
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_grad),
            optax.adam(self.learning_rate),
        )
        opt_state = optimizer.init(params)

        # Create sampler and initialize MCMC chains
        sampler = MetropolisSampler(step_size=self.step_size)

        def log_prob_fn(r):
            return wavefunction.log_prob(params, r)

        sampler_state = sampler.init_state(
            key, log_prob_fn, self.n_chains, molecule.n_electrons
        )

        return VMCState(
            params=params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            step=0,
        )

    def step(
        self,
        key: random.PRNGKey,
        state: VMCState,
        wavefunction,
        molecule: Molecule,
    ) -> Tuple[VMCState, Dict[str, float]]:
        """Perform one VMC optimization step.

        Args:
            key: JAX random key
            state: Current VMCState
            wavefunction: Wavefunction ansatz
            molecule: Molecular system

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        params = state.params

        # Sample from |ψ|²
        def log_prob_fn(r):
            return wavefunction.log_prob(params, r)

        sampler = MetropolisSampler(step_size=self.step_size)
        samples, sampler_state, acceptance_rate = sampler.sample(
            key,
            log_prob_fn,
            state.sampler_state,
            n_samples=self.n_samples,
            n_burn=self.n_burn if state.step == 0 else 10,  # Less burn-in after first step
        )

        # Compute gradient
        gradient, mean_energy, energy_std = compute_gradient(
            wavefunction, params, samples, molecule
        )

        # Apply optimizer update
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_grad),
            optax.adam(self.learning_rate),
        )
        updates, opt_state = optimizer.update(gradient, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree.leaves(gradient))
        )

        new_state = VMCState(
            params=new_params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            step=state.step + 1,
        )

        metrics = {
            "energy": float(mean_energy),
            "energy_std": float(energy_std),
            "acceptance_rate": float(acceptance_rate),
            "grad_norm": float(grad_norm),
        }

        return new_state, metrics

    def train(
        self,
        key: random.PRNGKey,
        wavefunction,
        params: Params,
        molecule: Molecule,
        n_steps: int,
        log_every: int = 10,
        callback: Optional[Callable] = None,
    ) -> Tuple[Params, List[Dict[str, float]]]:
        """Run full VMC optimization.

        Args:
            key: JAX random key
            wavefunction: Wavefunction ansatz
            params: Initial parameters
            molecule: Molecular system
            n_steps: Number of optimization steps
            log_every: Logging frequency
            callback: Optional callback(step, state, metrics)

        Returns:
            Tuple of (final_params, metrics_history)
        """
        key, init_key = random.split(key)
        state = self.init(init_key, wavefunction, params, molecule)

        history = []

        for i in range(n_steps):
            key, step_key = random.split(key)
            state, metrics = self.step(step_key, state, wavefunction, molecule)
            history.append(metrics)

            if (i + 1) % log_every == 0:
                print(
                    f"Step {i+1:4d}: E = {metrics['energy']:.6f} ± {metrics['energy_std']:.6f} Ha, "
                    f"acc = {metrics['acceptance_rate']:.2%}, |∇| = {metrics['grad_norm']:.4f}"
                )

            if callback is not None:
                callback(i, state, metrics)

        return state.params, history
