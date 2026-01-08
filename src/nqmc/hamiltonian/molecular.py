"""Molecular Hamiltonian operators for VMC.

This module implements the quantum mechanical Hamiltonian for molecular
systems, computing the local energy E_L(r) = Hψ(r)/ψ(r).

The Hamiltonian consists of:
- Kinetic energy: T = -½∇²
- Electron-nuclear attraction: V_en = -∑_{i,A} Z_A/|r_i - R_A|
- Electron-electron repulsion: V_ee = ∑_{i<j} 1/|r_i - r_j|
- Nuclear-nuclear repulsion: V_nn = ∑_{A<B} Z_A*Z_B/|R_A - R_B|

Example:
    >>> from nqmc.hamiltonian.molecular import local_energy
    >>> from nqmc.wavefunction import SimpleWavefunction
    >>> from nqmc.systems.molecule import hydrogen_molecule
    >>>
    >>> mol = hydrogen_molecule()
    >>> wf = SimpleWavefunction()
    >>> params = wf.init(jax.random.PRNGKey(0), jnp.zeros((2, 3)))
    >>> r = jnp.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> e_local = local_energy(wf, params, r, mol)
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, hessian, vmap

from nqmc.systems.molecule import Molecule
from nqmc.wavefunction.base import Params

# Small regularization to avoid 1/0 singularities
EPSILON = 1e-10


def kinetic_energy(
    log_psi_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Compute kinetic energy from log|ψ|.

    Uses the identity for kinetic energy in terms of log|ψ|:
        T = -½∇²ψ/ψ = -½(∇²log|ψ| + |∇log|ψ||²)

    Args:
        log_psi_fn: Function r -> log|ψ(r)| (scalar output)
        r: Electron positions, shape (n_electrons, 3)

    Returns:
        Kinetic energy (scalar)
    """
    # Flatten r for gradient computation: (n_el, 3) -> (n_el * 3,)
    r_flat = r.flatten()

    # Define function on flattened coordinates
    def log_psi_flat(r_f):
        return log_psi_fn(r_f.reshape(r.shape))

    # Gradient: ∇log|ψ|
    grad_log_psi = grad(log_psi_flat)(r_flat)

    # Hessian diagonal gives Laplacian: ∇²log|ψ| = tr(H)
    hess = hessian(log_psi_flat)(r_flat)
    laplacian_log_psi = jnp.trace(hess)

    # T = -½(∇²log|ψ| + |∇log|ψ||²)
    kinetic = -0.5 * (laplacian_log_psi + jnp.sum(grad_log_psi**2))

    return kinetic


def electron_nuclear_potential(
    r: jnp.ndarray,
    molecule: Molecule,
) -> jnp.ndarray:
    """Compute electron-nuclear attraction energy.

    V_en = -∑_{i,A} Z_A / |r_i - R_A|

    Args:
        r: Electron positions, shape (n_electrons, 3)
        molecule: Molecular system with nuclear positions and charges

    Returns:
        Electron-nuclear potential energy (scalar, negative)
    """
    # Get nuclear positions as JAX array
    nuclear_positions = jnp.array(molecule.positions)
    charges = jnp.array(molecule.charges)

    # Compute all electron-nuclear distances
    # r[:, None, :] - nuclear_positions[None, :, :] -> (n_el, n_atoms, 3)
    diff = r[:, None, :] - nuclear_positions[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)  # (n_el, n_atoms)

    # Regularize to avoid division by zero at coalescence
    distances = jnp.maximum(distances, EPSILON)

    # V_en = -∑_{i,A} Z_A / r_{iA}
    v_en = -jnp.sum(charges[None, :] / distances)

    return v_en


def electron_electron_potential(r: jnp.ndarray) -> jnp.ndarray:
    """Compute electron-electron repulsion energy.

    V_ee = ∑_{i<j} 1 / |r_i - r_j|

    Args:
        r: Electron positions, shape (n_electrons, 3)

    Returns:
        Electron-electron potential energy (scalar, positive)
    """
    n_electrons = r.shape[0]

    if n_electrons < 2:
        return jnp.array(0.0)

    # Compute all pairwise distances
    # r[:, None, :] - r[None, :, :] -> (n_el, n_el, 3)
    diff = r[:, None, :] - r[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)  # (n_el, n_el)

    # Regularize diagonal (self-interaction) to avoid 0/0
    # We'll mask these out anyway, but this prevents NaN in gradients
    distances = distances + jnp.eye(n_electrons) * 1e10

    # Sum upper triangle only (i < j pairs)
    # Use triu with k=1 to exclude diagonal
    upper_mask = jnp.triu(jnp.ones((n_electrons, n_electrons)), k=1)
    v_ee = jnp.sum(upper_mask / distances)

    return v_ee
