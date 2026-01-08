# Neural Quantum Monte Carlo Wavefunction

A research-grade implementation of neural network wavefunctions for Variational Monte Carlo (VMC), targeting hydrogen chain molecules (H₄, H₆) with antisymmetric neural architectures. Includes an interactive web application for real-time visualization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements neural network-parameterized wavefunctions for quantum Monte Carlo simulations. The approach replaces traditional Slater-Jastrow ansätze with expressive neural networks that can capture complex electron correlations while enforcing fermionic antisymmetry.

### Key Features

- Neural network wavefunction with enforced antisymmetry for fermions
- Variational Monte Carlo with MCMC sampling (Metropolis-Hastings)
- Electron-nucleus and electron-electron cusp conditions
- Backflow transformation for improved correlation capture
- Energy optimization via the log-derivative trick
- Comparison benchmarks against Hartree-Fock, MP2, and CCSD (via PySCF)
- **Interactive web application** with real-time training visualization

### Target Systems

| System | Electrons | Configuration | Status |
|--------|-----------|---------------|--------|
| H₂ | 2 | 1 up, 1 down | Validation |
| H₄ | 4 | 2 up, 2 down | Primary |
| H₆ | 6 | 3 up, 3 down | Stretch Goal |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEURAL QMC SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      WEB APPLICATION                                 │    │
│  │     React Frontend  ◄──── WebSocket ────►  FastAPI Backend           │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                         NQMC CORE LIBRARY                              │  │
│  │                                                                         │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │  │
│  │  │  Molecular   │───►│    Neural    │◄───│  Hamiltonian │              │  │
│  │  │  System      │    │  Wavefunction│    │   Operator   │              │  │
│  │  └──────────────┘    └──────┬───────┘    └──────────────┘              │  │
│  │                             │                                           │  │
│  │                             ▼                                           │  │
│  │                  ┌──────────────────────┐                               │  │
│  │                  │    MCMC Sampler      │                               │  │
│  │                  │  (Metropolis-Hastings)│                               │  │
│  │                  └──────────┬───────────┘                               │  │
│  │                             │                                           │  │
│  │                             ▼                                           │  │
│  │  ┌──────────────┐  ┌──────────────────────┐                             │  │
│  │  │    PySCF     │◄─│    VMC Optimizer     │                             │  │
│  │  │   Baselines  │  │  (log-deriv trick)   │                             │  │
│  │  └──────────────┘  └──────────────────────┘                             │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Theoretical Background

### Variational Principle

The ground state energy satisfies:

```
E₀ ≤ E[ψ_θ] = ⟨ψ_θ|H|ψ_θ⟩ / ⟨ψ_θ|ψ_θ⟩
```

### Neural Wavefunction Ansatz

The wavefunction is parameterized as:

```
ψ_θ(r) = det[Φ_up(r)] · det[Φ_down(r)] · exp(J(r))
```

where:
- `det[Φ]` are Slater determinants of neural network orbitals (antisymmetry)
- `J(r)` is a Jastrow correlation factor (electron-electron correlation)
- Cusp conditions enforce correct behavior at particle coalescence

### Monte Carlo Energy Estimation

```
E[H] = ∫|ψ|²(Hψ/ψ)dr ≈ (1/N) Σᵢ E_L(rᵢ)
```

where `E_L(r) = Hψ(r)/ψ(r)` is the local energy and samples are drawn from `|ψ|²` via MCMC.

### Gradient Estimation (Log-Derivative Trick)

```
∂E/∂θ = 2⟨(E_L - ⟨E_L⟩) · ∂log|ψ|/∂θ⟩
```

## Installation

### Core Library

```bash
git clone https://github.com/Sakeeb91/neural-qmc-wavefunction.git
cd neural-qmc-wavefunction
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Web Application

```bash
cd webapp
docker-compose up
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| JAX | >= 0.4.20 | Autodiff, JIT compilation, GPU support |
| Flax | >= 0.8.0 | Neural network layers |
| Optax | >= 0.1.7 | Optimizers |
| PySCF | >= 2.4.0 | Classical QC baselines |
| NumPy | >= 1.24.0 | Numerical operations |
| Matplotlib | >= 3.7.0 | Visualization |

## Project Structure

```
neural-qmc-wavefunction/
├── src/nqmc/
│   ├── systems/           # Molecular system definitions
│   │   └── molecule.py    # Molecule class, hydrogen chains
│   ├── wavefunction/      # Neural network architectures
│   │   ├── slater.py      # Slater determinant
│   │   ├── orbitals.py    # Neural orbitals
│   │   ├── jastrow.py     # Jastrow factor
│   │   ├── cusp.py        # Cusp conditions
│   │   └── backflow.py    # Backflow transformation
│   ├── hamiltonian/       # Molecular Hamiltonians
│   ├── sampler/           # MCMC samplers
│   ├── optimizer/         # VMC optimization
│   ├── baselines/         # PySCF wrapper
│   └── visualization/     # Plotting utilities
├── webapp/
│   ├── backend/           # FastAPI server
│   └── frontend/          # React application
├── scripts/               # Training scripts
├── tests/                 # Unit and integration tests
├── notebooks/             # Analysis notebooks
├── visualizations/        # Generated plots
│   ├── energy_curves/
│   ├── wavefunction_plots/
│   ├── sampling_diagnostics/
│   └── comparison_charts/
└── docs/
    └── IMPLEMENTATION_PLAN.md
```

## Usage

### Quick Start

```python
from nqmc.systems.molecule import hydrogen_chain
from nqmc.wavefunction import AntisymmetricWavefunction
from nqmc.optimizer import VMCOptimizer

# Define molecular system (H₄ chain)
molecule = hydrogen_chain(n_atoms=4, bond_length=1.8)
print(f"System: {molecule.n_electrons} electrons, spin {molecule.spin}")

# Create neural wavefunction
wavefunction = AntisymmetricWavefunction(
    n_up=molecule.spin[0],
    n_down=molecule.spin[1],
    orbital_hidden_dims=(64, 64),
    use_cusp=True,
    use_backflow=True
)

# Run VMC optimization
optimizer = VMCOptimizer(
    wavefunction=wavefunction,
    molecule=molecule,
    n_walkers=1024,
    learning_rate=1e-3
)

energy, params = optimizer.optimize(n_steps=10000)
print(f"Ground state energy: {energy:.6f} Ha")
```

### Compare with Classical Methods

```python
from nqmc.baselines import PySCFRunner

# Run classical calculations
runner = PySCFRunner(molecule, basis="cc-pVTZ")
results = runner.run_all()

print("Method Comparison:")
print(f"  HF:      {results['HF']:.6f} Ha")
print(f"  MP2:     {results['MP2']:.6f} Ha")
print(f"  CCSD:    {results['CCSD']:.6f} Ha")
print(f"  CCSD(T): {results['CCSD(T)']:.6f} Ha")
print(f"  Neural QMC: {energy:.6f} Ha")
```

### Generate Visualizations

```python
from nqmc.visualization import (
    plot_energy_convergence,
    plot_electron_density,
    plot_method_comparison
)

# Energy convergence during training
plot_energy_convergence(
    energies=optimizer.energy_history,
    target_energy=-2.05,
    save_path="visualizations/energy_curves/h4_convergence.png"
)

# Electron density visualization
plot_electron_density(
    wavefunction=wavefunction,
    params=params,
    molecule=molecule,
    save_path="visualizations/wavefunction_plots/h4_density.png"
)
```

## Web Application

The interactive web application provides real-time visualization of Neural QMC simulations.

### Features

| Feature | Description |
|---------|-------------|
| **Molecule Builder** | Visual constructor with presets (H₂, H₄, H₆) |
| **Training Dashboard** | Real-time energy convergence via WebSocket |
| **3D Visualizer** | Interactive electron density isosurfaces |
| **Method Comparison** | Side-by-side Neural QMC vs HF/MP2/CCSD |
| **Export** | PNG, SVG, CSV, LaTeX tables |

### Running the Web App

```bash
cd webapp
docker-compose up
```

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/molecules` | Create molecular system |
| POST | `/api/training/start` | Start VMC training |
| WS | `/ws/training/{id}` | Real-time training updates |
| GET | `/api/visualization/density/{id}` | Get density grid for 3D viz |
| POST | `/api/compare` | Compare methods |

## Development Roadmap

| Phase | Description | Status | Issue |
|-------|-------------|--------|-------|
| 1 | Foundation and H₂ Validation | Not Started | [#2](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/2) |
| 2 | Antisymmetric Wavefunction | Not Started | [#3](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/3) |
| 3 | Cusp Conditions and Backflow | Not Started | [#4](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/4) |
| 4 | Hydrogen Chain (H₄) | Not Started | [#5](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/5) |
| 5 | PySCF Benchmarking | Not Started | [#6](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/6) |
| 6 | H₆ + Advanced Optimization | Not Started | [#7](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/7) |
| Web | Interactive Web Application | Not Started | [#8](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/8) |

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for detailed implementation specifications.

## Expected Results

### H₂ Molecule (Validation)

| Method | Energy (Ha) | Error vs Exact |
|--------|-------------|----------------|
| Hartree-Fock | -1.110 | 64 mHa |
| MP2 | -1.144 | 30 mHa |
| CCSD | -1.166 | 8 mHa |
| CCSD(T) | -1.172 | 2 mHa |
| **Neural QMC** | **-1.17** | **< 5 mHa** |
| Exact | -1.174 | - |

### H₄ Chain (Primary Target)

| Method | Energy (Ha) | Correlation % |
|--------|-------------|---------------|
| Hartree-Fock | -1.95 | 0% |
| MP2 | -2.00 | ~60% |
| CCSD | -2.05 | ~95% |
| **Neural QMC** | **-2.0+** | **> 50%** |
| Exact (FCI) | -2.08 | 100% |

## Visualization Gallery

### Energy Convergence
Training loss showing VMC optimization converging to ground state.

### Electron Density
2D slice through molecular plane showing |ψ|² distribution.

### Method Comparison
Bar chart comparing Neural QMC against classical quantum chemistry methods.

### Potential Energy Surface
Energy vs bond length curves for H₄ dissociation.

*(Visualizations will be added as implementation progresses)*

## References

### Neural Network Wavefunctions
- [FermiNet: Quantum Physics and Chemistry from First Principles](https://deepmind.google/discover/blog/ferminet-quantum-physics-and-chemistry-from-first-principles/) (DeepMind, 2020)
- [PauliNet: Deep neural network solution of the electronic Schrödinger equation](https://www.nature.com/articles/s41557-020-0544-y) (Hermann et al., Nature Chemistry 2020)

### Quantum Monte Carlo
- Foulkes et al., "Quantum Monte Carlo simulations of solids", Rev. Mod. Phys. 73, 33 (2001)
- Needs et al., "Continuum variational and diffusion quantum Monte Carlo calculations", J. Phys.: Condens. Matter 22, 023201 (2010)

### Cusp Conditions
- Kato, T., "On the eigenfunctions of many-particle systems in quantum mechanics", Commun. Pure Appl. Math. 10, 151 (1957)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see:
1. [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) for development priorities
2. [GitHub Issues](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues) for open tasks
3. [Meta Issue #1](https://github.com/Sakeeb91/neural-qmc-wavefunction/issues/1) for project tracking

### Development Setup

```bash
# Clone and install
git clone https://github.com/Sakeeb91/neural-qmc-wavefunction.git
cd neural-qmc-wavefunction
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/
black src/ --check
```

## Acknowledgments

This project draws inspiration from:
- [FermiNet](https://github.com/deepmind/ferminet) by DeepMind
- [PauliNet](https://github.com/deepqmc/deepqmc) by the DeepQMC team
- The broader quantum Monte Carlo and machine learning for quantum chemistry communities
