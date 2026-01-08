"""Neural network wavefunction architectures.

This module provides wavefunction ansätze for variational Monte Carlo.

Available wavefunctions:
- SimpleWavefunction: Basic MLP for testing (no antisymmetry)

Example:
    >>> from nqmc.wavefunction import SimpleWavefunction
    >>> wf = SimpleWavefunction(hidden_dims=(32, 32))
"""
from nqmc.wavefunction.base import Params, Wavefunction
from nqmc.wavefunction.simple import SimpleMLP, SimpleWavefunction

__all__ = [
    "Wavefunction",
    "Params",
    "SimpleWavefunction",
    "SimpleMLP",
]
