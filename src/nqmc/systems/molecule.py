"""Molecular system definitions for neural QMC.

This module provides the Molecule dataclass and helper functions for
defining molecular systems to be studied with neural network wavefunctions.

Example:
    >>> from nqmc.systems.molecule import hydrogen_molecule, hydrogen_chain
    >>> h2 = hydrogen_molecule(bond_length=1.4)
    >>> print(f"H2: {h2.n_electrons} electrons, spin {h2.spin}")
    H2: 2 electrons, spin (1, 1)
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Element to nuclear charge mapping
ELEMENT_CHARGES = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8
}

# Conversion factors
BOHR_TO_ANGSTROM = 0.529177


@dataclass
class Molecule:
    """Represents a molecular system for quantum Monte Carlo.

    This dataclass stores the molecular geometry and electron configuration
    needed for VMC calculations.

    Attributes:
        atoms: List of (element, position) tuples. Element is str,
               position is numpy array of shape (3,) in Bohr.
        charges: Nuclear charges for each atom as numpy array.
        n_electrons: Total number of electrons.
        spin: Tuple of (n_up, n_down) electrons.

    Example:
        >>> mol = Molecule.from_atoms(["H", "H"], np.array([[0, 0, 0], [1.4, 0, 0]]))
        >>> mol.n_electrons
        2
    """
    atoms: List[Tuple[str, np.ndarray]]
    charges: np.ndarray
    n_electrons: int
    spin: Tuple[int, int]

    @classmethod
    def from_atoms(
        cls,
        elements: List[str],
        positions: np.ndarray,
        charge: int = 0,
        spin_polarization: int = 0
    ) -> "Molecule":
        """Create molecule from element list and positions.

        Args:
            elements: List of element symbols (e.g., ["H", "H"])
            positions: Array of shape (n_atoms, 3) with atomic positions in Bohr
            charge: Net molecular charge (default 0)
            spin_polarization: n_up - n_down (default 0, singlet)

        Returns:
            Molecule instance

        Raises:
            ValueError: If elements and positions have mismatched lengths
            KeyError: If an element symbol is not recognized
        """
        if len(elements) != len(positions):
            raise ValueError(
                f"Length mismatch: {len(elements)} elements vs {len(positions)} positions"
            )

        atoms = [(el, np.array(pos, dtype=np.float64)) for el, pos in zip(elements, positions)]
        charges = np.array([ELEMENT_CHARGES[el] for el in elements], dtype=np.float64)
        n_electrons = int(np.sum(charges)) - charge

        # Determine spin configuration
        # n_up + n_down = n_electrons
        # n_up - n_down = spin_polarization
        n_up = (n_electrons + spin_polarization) // 2
        n_down = n_electrons - n_up

        if n_up < 0 or n_down < 0:
            raise ValueError(
                f"Invalid spin configuration: n_up={n_up}, n_down={n_down}"
            )

        return cls(
            atoms=atoms,
            charges=charges,
            n_electrons=n_electrons,
            spin=(n_up, n_down)
        )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions as (n_atoms, 3) array in Bohr."""
        return np.array([pos for _, pos in self.atoms])

    @property
    def elements(self) -> List[str]:
        """List of element symbols."""
        return [el for el, _ in self.atoms]

    def electron_nuclear_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute distances from each electron to each nucleus.

        Args:
            electron_positions: Shape (n_electrons, 3) array of electron positions

        Returns:
            Shape (n_electrons, n_atoms) distance matrix where
            result[i, A] = |r_i - R_A|
        """
        # Broadcasting: (n_el, 1, 3) - (1, n_atoms, 3) -> (n_el, n_atoms, 3)
        diff = electron_positions[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def electron_electron_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between electrons.

        Args:
            electron_positions: Shape (n_electrons, 3) array of electron positions

        Returns:
            Shape (n_electrons, n_electrons) symmetric distance matrix where
            result[i, j] = |r_i - r_j|. Diagonal is zero.
        """
        # Broadcasting: (n_el, 1, 3) - (1, n_el, 3) -> (n_el, n_el, 3)
        diff = electron_positions[:, None, :] - electron_positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def nuclear_repulsion_energy(self) -> float:
        """Compute nuclear-nuclear repulsion energy.

        Returns:
            Nuclear repulsion energy in Hartree
        """
        energy = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                r_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                energy += self.charges[i] * self.charges[j] / r_ij
        return energy

    def to_xyz_string(self) -> str:
        """Convert to XYZ format string (positions in Angstrom).

        Returns:
            XYZ format string suitable for visualization software
        """
        lines = [str(self.n_atoms), ""]
        for el, pos in self.atoms:
            pos_ang = pos * BOHR_TO_ANGSTROM
            lines.append(f"{el} {pos_ang[0]:.6f} {pos_ang[1]:.6f} {pos_ang[2]:.6f}")
        return "\n".join(lines)

    def get_spin_mask(self) -> np.ndarray:
        """Get boolean mask for same-spin electron pairs.

        Returns:
            Shape (n_electrons, n_electrons) boolean array where
            result[i, j] = True if electrons i and j have the same spin
        """
        n_up, n_down = self.spin
        n = self.n_electrons

        # First n_up are spin-up, rest are spin-down
        mask = np.zeros((n, n), dtype=bool)
        mask[:n_up, :n_up] = True      # up-up pairs
        mask[n_up:, n_up:] = True      # down-down pairs

        return mask


def hydrogen_molecule(bond_length: float = 1.4) -> Molecule:
    """Create H2 molecule at specified bond length.

    Args:
        bond_length: H-H distance in Bohr (default 1.4, near equilibrium ~1.4 Bohr)

    Returns:
        H2 Molecule instance with 2 electrons (1 up, 1 down)
    """
    positions = np.array([
        [0.0, 0.0, 0.0],
        [bond_length, 0.0, 0.0]
    ])
    return Molecule.from_atoms(["H", "H"], positions)


def hydrogen_chain(n_atoms: int, bond_length: float = 1.8) -> Molecule:
    """Create linear hydrogen chain.

    Creates a chain of hydrogen atoms along the x-axis with equal spacing.
    The number of atoms must be even for a closed-shell (singlet) ground state.

    Args:
        n_atoms: Number of hydrogen atoms (must be even)
        bond_length: H-H distance in Bohr (default 1.8)

    Returns:
        Hydrogen chain Molecule instance

    Raises:
        ValueError: If n_atoms is odd
    """
    if n_atoms % 2 != 0:
        raise ValueError(
            f"n_atoms must be even for closed-shell hydrogen chain, got {n_atoms}"
        )

    positions = np.array([[i * bond_length, 0.0, 0.0] for i in range(n_atoms)])
    elements = ["H"] * n_atoms

    return Molecule.from_atoms(elements, positions)


def hydrogen_atom() -> Molecule:
    """Create single hydrogen atom.

    Useful for testing exact solutions (E = -0.5 Ha).

    Returns:
        Hydrogen atom Molecule instance with 1 electron (1 up, 0 down)
    """
    positions = np.array([[0.0, 0.0, 0.0]])
    return Molecule.from_atoms(["H"], positions, spin_polarization=1)


if __name__ == "__main__":
    # Quick test
    print("Testing Molecule class...")

    # Test H2
    h2 = hydrogen_molecule(bond_length=1.4)
    print(f"H2: {h2.n_electrons} electrons, spin {h2.spin}")
    print(f"Nuclear repulsion: {h2.nuclear_repulsion_energy():.6f} Ha")

    # Test distance computation
    r = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 2 electrons
    r_en = h2.electron_nuclear_distances(r)
    print(f"Electron-nuclear distances shape: {r_en.shape}")  # Should be (2, 2)

    r_ee = h2.electron_electron_distances(r)
    print(f"Electron-electron distances shape: {r_ee.shape}")  # Should be (2, 2)
    print(f"e-e distance [0,1]: {r_ee[0, 1]:.4f}")  # Should be 0.5

    # Test H4
    h4 = hydrogen_chain(n_atoms=4)
    print(f"\nH4: {h4.n_electrons} electrons, spin {h4.spin}")
    print(f"H4 nuclear repulsion: {h4.nuclear_repulsion_energy():.6f} Ha")

    # Test H6
    h6 = hydrogen_chain(n_atoms=6)
    print(f"H6: {h6.n_electrons} electrons, spin {h6.spin}")

    print("\nAll tests passed!")
