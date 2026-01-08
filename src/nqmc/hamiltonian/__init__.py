"""Molecular Hamiltonian operators.

This module provides Hamiltonian operators for computing local energy
in variational Monte Carlo calculations.

Example:
    >>> from nqmc.hamiltonian import local_energy
    >>> e_local = local_energy(wf, params, r, molecule)
"""
from nqmc.hamiltonian.molecular import (
    EPSILON,
    electron_electron_potential,
    electron_nuclear_potential,
    kinetic_energy,
    local_energy,
    local_energy_batched,
)

__all__ = [
    "local_energy",
    "local_energy_batched",
    "kinetic_energy",
    "electron_nuclear_potential",
    "electron_electron_potential",
    "EPSILON",
]
