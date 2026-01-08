"""Molecular system definitions.

This module provides molecular system dataclasses for quantum Monte Carlo.

Example:
    >>> from nqmc.systems import hydrogen_molecule, hydrogen_chain
    >>> h2 = hydrogen_molecule(bond_length=1.4)
    >>> h4 = hydrogen_chain(n_atoms=4)
"""
from nqmc.systems.molecule import (
    BOHR_TO_ANGSTROM,
    ELEMENT_CHARGES,
    Molecule,
    hydrogen_atom,
    hydrogen_chain,
    hydrogen_molecule,
)

__all__ = [
    "Molecule",
    "hydrogen_molecule",
    "hydrogen_chain",
    "hydrogen_atom",
    "ELEMENT_CHARGES",
    "BOHR_TO_ANGSTROM",
]
