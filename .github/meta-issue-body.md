## Summary

Master tracking issue for implementing a research-grade Neural Quantum Monte Carlo (Neural QMC) wavefunction for variational Monte Carlo simulations of hydrogen chain molecules (H₄, H₆).

This project implements neural network-parameterized wavefunctions that can capture complex electron correlations while enforcing fermionic antisymmetry, with comparison against classical quantum chemistry methods (HF, CCSD via PySCF).

## Project Overview

| Attribute | Value |
|-----------|-------|
| **Target Systems** | H₂ (validation), H₄, H₆ (primary) |
| **Approach** | Neural network wavefunction + VMC |
| **Baseline Methods** | Hartree-Fock, MP2, CCSD (via PySCF) |
| **Framework** | JAX + Flax |
| **Quality Level** | Research-grade |

## Phase Tracker

| Phase | Issue | Status | Description |
|-------|-------|--------|-------------|
| 1 | #2 | 🔴 Not Started | Foundation and H₂ Validation |
| 2 | #3 | 🔴 Not Started | Antisymmetric Wavefunction Architecture |
| 3 | #4 | 🔴 Not Started | Cusp Conditions and Backflow |
| 4 | #5 | 🔴 Not Started | Hydrogen Chain (H₄) Extension |
| 5 | #6 | 🔴 Not Started | PySCF Integration and Benchmarking |
| 6 | #7 | 🔴 Not Started | H₆ and Optimization Improvements (Stretch) |

## Key Milestones

- [ ] Phase 1 Complete: H₂ VMC working with < -1.10 Ha energy
- [ ] Phase 2 Complete: Antisymmetric wavefunction with determinant
- [ ] Phase 3 Complete: Cusp conditions enforced, < -1.17 Ha for H₂
- [ ] Phase 4 Complete: H₄ training with PES curve
- [ ] Phase 5 Complete: Full benchmark suite with publication figures
- [ ] Phase 6 Complete: H₆ results and scaling analysis

## Visualization Deliverables

- [ ] Energy convergence curves
- [ ] Electron density plots
- [ ] MCMC sampling diagnostics
- [ ] Method comparison charts
- [ ] Potential energy surfaces

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  MOLECULAR       │───▶│  NEURAL          │◀───│  HAMILTONIAN     │
│  SYSTEM          │    │  WAVEFUNCTION    │    │  OPERATOR        │
└──────────────────┘    └────────┬─────────┘    └──────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      MCMC SAMPLER            │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     VMC OPTIMIZER            │
                  └──────────────────────────────┘
```

## References

- [FermiNet (DeepMind)](https://deepmind.google/discover/blog/ferminet-quantum-physics-and-chemistry-from-first-principles/)
- [PauliNet](https://www.nature.com/articles/s41557-020-0544-y)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
