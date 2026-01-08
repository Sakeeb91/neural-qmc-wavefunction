"""Simple feedforward neural network wavefunction.

This module implements a basic MLP-based wavefunction ansatz for
initial testing. It does not enforce fermionic antisymmetry - that
comes in Phase 2 with Slater determinants.

Example:
    >>> from nqmc.wavefunction.simple import SimpleWavefunction
    >>> import jax.numpy as jnp
    >>> import jax
    >>>
    >>> wf = SimpleWavefunction(hidden_dims=(32, 32))
    >>> key = jax.random.PRNGKey(0)
    >>> r = jnp.zeros((2, 3))  # 2 electrons in 3D
    >>> params = wf.init(key, r)
    >>> log_psi = wf(params, r)
"""
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from nqmc.wavefunction.base import Params, Wavefunction


class SimpleMLP(nn.Module):
    """Multi-layer perceptron for wavefunction evaluation.

    Takes flattened electron positions and outputs a scalar log|ψ|.

    Attributes:
        hidden_dims: Tuple of hidden layer dimensions
        activation: Activation function name ('tanh' or 'silu')
    """

    hidden_dims: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network.

        Args:
            r: Electron positions, shape (n_electrons, 3)

        Returns:
            Scalar log|ψ(r)|
        """
        # Flatten electron positions: (n_electrons, 3) -> (n_electrons * 3,)
        x = r.flatten()

        # Select activation function
        if self.activation == "tanh":
            act_fn = nn.tanh
        elif self.activation == "silu":
            act_fn = nn.silu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act_fn(x)

        # Output layer: scalar log|ψ|
        log_psi = nn.Dense(1)(x).squeeze(-1)

        return log_psi


class SimpleWavefunction(Wavefunction):
    """Simple feedforward wavefunction with exponential envelope.

    This wavefunction combines a neural network with a physics-inspired
    exponential envelope that decays away from nuclei:
        ψ(r) = exp(MLP(r)) * exp(-α * Σ_i min_A |r_i - R_A|)

    The envelope ensures the wavefunction has the correct asymptotic
    behavior even at initialization.

    This does NOT enforce antisymmetry, so it's only suitable for:
    - Testing the VMC infrastructure
    - 2-electron systems where antisymmetry is less critical

    For proper fermionic systems, use SlaterWavefunction from Phase 2.

    Attributes:
        hidden_dims: Hidden layer sizes (default: (64, 64))
        activation: Activation function (default: 'tanh')
        envelope_decay: Decay rate for exponential envelope (default: 1.0)
        nuclear_positions: Array of nuclear positions for envelope
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        envelope_decay: float = 1.0,
        nuclear_positions: jnp.ndarray = None,
    ):
        """Initialize SimpleWavefunction.

        Args:
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function ('tanh' or 'silu')
            envelope_decay: Exponential decay rate (in 1/Bohr)
            nuclear_positions: (n_atoms, 3) array of nuclear positions.
                              If None, assumes H2 at equilibrium.
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.envelope_decay = envelope_decay

        if nuclear_positions is None:
            # Default: H2 at equilibrium
            self.nuclear_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        else:
            self.nuclear_positions = jnp.array(nuclear_positions)

        self._network = SimpleMLP(hidden_dims=hidden_dims, activation=activation)

    def _envelope(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compute log of exponential envelope.

        Uses sum of exponentials from all nuclei:
        envelope = Π_i Σ_A exp(-α |r_i - R_A|)

        This is similar to a molecular orbital formed from atomic orbitals.

        Args:
            r: Electron positions, shape (n_electrons, 3)

        Returns:
            Scalar log(envelope)
        """
        # r: (n_el, 3), nuclear_positions: (n_atoms, 3)
        # diff: (n_el, n_atoms, 3)
        diff = r[:, None, :] - self.nuclear_positions[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)  # (n_el, n_atoms)

        # For each electron, sum of exponentials from all nuclei
        # This gives a bonding-like orbital shape
        exp_terms = jnp.exp(-self.envelope_decay * distances)  # (n_el, n_atoms)
        sum_exp = jnp.sum(exp_terms, axis=1)  # (n_el,)

        # Return log(Π_i sum_exp_i) = Σ_i log(sum_exp_i)
        return jnp.sum(jnp.log(sum_exp))

    def __call__(self, params: Params, r: jnp.ndarray) -> jnp.ndarray:
        """Compute log|ψ(r)| for electron configuration.

        Args:
            params: Network parameters from init()
            r: Electron positions, shape (n_electrons, 3)

        Returns:
            Scalar log|ψ(r)|
        """
        # MLP output + exponential envelope
        mlp_output = self._network.apply(params, r)
        envelope = self._envelope(r)
        return mlp_output + envelope

    def init(self, key: random.PRNGKey, r_sample: jnp.ndarray) -> Params:
        """Initialize network parameters.

        Args:
            key: JAX random key
            r_sample: Sample configuration for shape inference

        Returns:
            Initialized parameters
        """
        return self._network.init(key, r_sample)

    @property
    def n_params(self) -> int:
        """Estimate number of parameters (requires init first)."""
        # This is just for documentation - actual count from params
        n_input = 6  # 2 electrons * 3 dims (for H2)
        total = 0
        prev_dim = n_input
        for dim in self.hidden_dims:
            total += prev_dim * dim + dim  # weights + bias
            prev_dim = dim
        total += prev_dim + 1  # output layer
        return total
