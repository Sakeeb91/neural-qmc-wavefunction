"""Neural Quantum Monte Carlo Wavefunction.

A research-grade implementation of neural network wavefunctions for
Variational Monte Carlo, targeting hydrogen chain molecules.

Quick Start:
    >>> from nqmc import SimpleWavefunction, VMCOptimizer, hydrogen_molecule
    >>> molecule = hydrogen_molecule()
    >>> wf = SimpleWavefunction()
    >>> optimizer = VMCOptimizer(learning_rate=1e-3)
"""

__version__ = "0.1.0"

# Convenience imports for common usage
from nqmc.hamiltonian import local_energy
from nqmc.optimizer import VMCOptimizer
from nqmc.sampler import MetropolisSampler
from nqmc.systems import Molecule, hydrogen_atom, hydrogen_chain, hydrogen_molecule
from nqmc.wavefunction import SimpleWavefunction, Wavefunction

__all__ = [
    # Version
    "__version__",
    # Systems
    "Molecule",
    "hydrogen_molecule",
    "hydrogen_chain",
    "hydrogen_atom",
    # Wavefunction
    "Wavefunction",
    "SimpleWavefunction",
    # Hamiltonian
    "local_energy",
    # Sampler
    "MetropolisSampler",
    # Optimizer
    "VMCOptimizer",
]
