# Neural Quantum Monte Carlo Wavefunction

A research-grade implementation of neural network wavefunctions for Variational Monte Carlo (VMC), targeting hydrogen chain molecules (H₄, H₆) with antisymmetric neural architectures.

## Overview

This project implements neural network-parameterized wavefunctions for quantum Monte Carlo simulations. The approach replaces traditional Slater-Jastrow ansätze with expressive neural networks that can capture complex electron correlations while enforcing fermionic antisymmetry.

### Key Features

- Neural network wavefunction with enforced antisymmetry for fermions
- Variational Monte Carlo with MCMC sampling (Metropolis-Hastings)
- Electron-nucleus and electron-electron cusp conditions
- Energy optimization via the log-derivative trick
- Comparison benchmarks against Hartree-Fock and CCSD (via PySCF)

### Target Systems

- **Primary:** Hydrogen chains (H₄, H₆) at various bond lengths
- **Validation:** H₂ molecule for initial testing

## Theoretical Background

### Variational Principle

The ground state energy satisfies:

```
E₀ ≤ E[ψ_θ] = ⟨ψ_θ|H|ψ_θ⟩ / ⟨ψ_θ|ψ_θ⟩
```

### Monte Carlo Estimation

The energy expectation is computed as:

```
E[H] = ∫|ψ|²(Hψ/ψ)dr ≈ (1/N) Σᵢ E_L(rᵢ)
```

where `E_L(r) = Hψ(r)/ψ(r)` is the local energy and samples are drawn from `|ψ|²`.

### Gradient Estimation

Parameter gradients use the log-derivative trick:

```
∂E/∂θ = 2⟨(E_L - ⟨E_L⟩) ∂log|ψ|/∂θ⟩
```

## Installation

```bash
git clone https://github.com/Sakeeb91/neural-qmc-wavefunction.git
cd neural-qmc-wavefunction
pip install -e .
```

### Dependencies

- Python 3.9+
- JAX (for autodiff and JIT compilation)
- PySCF (for classical QC baselines)
- NumPy, SciPy

## Project Structure

```
neural-qmc-wavefunction/
├── src/
│   └── nqmc/
│       ├── wavefunction/     # Neural network architectures
│       ├── hamiltonian/      # Molecular Hamiltonians
│       ├── sampler/          # MCMC samplers
│       ├── optimizer/        # VMC optimization
│       └── systems/          # Molecular system definitions
├── scripts/                  # Training and evaluation scripts
├── tests/                    # Unit and integration tests
├── notebooks/                # Jupyter notebooks for analysis
└── docs/                     # Documentation
```

## Usage

```python
from nqmc import NeuralWavefunction, VMCOptimizer, HydrogenChain

# Define molecular system
molecule = HydrogenChain(n_atoms=4, bond_length=1.4)

# Create neural wavefunction
wavefunction = NeuralWavefunction(
    n_electrons=molecule.n_electrons,
    n_layers=2,
    hidden_dim=64
)

# Run VMC optimization
optimizer = VMCOptimizer(wavefunction, molecule)
energy, params = optimizer.optimize(n_steps=10000)

print(f"Ground state energy: {energy:.6f} Ha")
```

## References

- [FermiNet: Quantum Physics and Chemistry from First Principles](https://deepmind.google/discover/blog/ferminet-quantum-physics-and-chemistry-from-first-principles/)
- [PauliNet: Deep neural network solution of the electronic Schrödinger equation](https://www.nature.com/articles/s41557-020-0544-y)
- Foulkes et al., "Quantum Monte Carlo simulations of solids", Rev. Mod. Phys. 73, 33 (2001)

## License

MIT License

## Contributing

Contributions are welcome. Please see the implementation plan in `docs/IMPLEMENTATION_PLAN.md` for current development priorities.
