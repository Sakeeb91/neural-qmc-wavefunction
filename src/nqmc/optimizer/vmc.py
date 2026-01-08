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
