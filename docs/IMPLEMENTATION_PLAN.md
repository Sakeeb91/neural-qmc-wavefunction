# Implementation Plan: Neural Quantum Monte Carlo Wavefunction

## Expert Role

**Computational Physics ML Engineer** — This project sits at the intersection of quantum chemistry, variational methods, Monte Carlo sampling, and deep learning. The role requires understanding of:
- Quantum mechanics fundamentals (wavefunctions, antisymmetry, Hamiltonians)
- Variational Monte Carlo methodology
- Neural network architecture design with physical constraints
- Automatic differentiation and gradient estimation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEURAL QMC SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  MOLECULAR       │    │  NEURAL          │    │  HAMILTONIAN     │       │
│  │  SYSTEM          │───▶│  WAVEFUNCTION    │◀───│  OPERATOR        │       │
│  │                  │    │                  │    │                  │       │
│  │  - Atom coords   │    │  - Antisymmetric │    │  - Kinetic T     │       │
│  │  - Charges       │    │  - Cusp enforce  │    │  - Potential V   │       │
│  │  - n_electrons   │    │  - Backflow      │    │  - Local energy  │       │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌──────────────────────────────┐                         │
│                    │      MCMC SAMPLER            │                         │
│                    │                              │                         │
│                    │  - Metropolis-Hastings       │                         │
│                    │  - Proposal distribution     │                         │
│                    │  - Burn-in / Decorrelation   │                         │
│                    └──────────────┬───────────────┘                         │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌──────────────────────────────┐                         │
│                    │     VMC OPTIMIZER            │                         │
│                    │                              │                         │
│                    │  - Energy estimation         │                         │
│                    │  - Gradient (log-deriv)      │                         │
│                    │  - KFAC / Adam / SR          │                         │
│                    └──────────────┬───────────────┘                         │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────┐    ┌──────────────────────────────┐                   │
│  │  PYSCF BASELINE  │    │     TRAINING LOOP            │                   │
│  │                  │    │                              │                   │
│  │  - HF energy     │◀───│  - Checkpoint saving         │                   │
│  │  - CCSD energy   │    │  - Metrics logging           │                   │
│  │  - Comparison    │    │  - Convergence monitoring    │                   │
│  └──────────────────┘    └──────────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Define molecular system (atom positions, charges)
                    │
                    ▼
2. Initialize neural wavefunction ψ_θ(r₁,...,rₙ)
                    │
                    ▼
3. Sample electron configurations from |ψ|² via MCMC
                    │
                    ▼
4. Compute local energy E_L(r) = Hψ(r)/ψ(r) for each sample
                    │
                    ▼
5. Estimate ⟨E⟩ = mean(E_L) and gradient via log-derivative trick
                    │
                    ▼
6. Update parameters θ ← θ - η∇E
                    │
                    ▼
7. Repeat until convergence
```

---

## Technology Stack

| Component | Choice | Rationale | Fallback |
|-----------|--------|-----------|----------|
| **Core Framework** | JAX | Composable autodiff, vmap for batching, JIT compilation, GPU support | PyTorch (heavier, less functional) |
| **Neural Networks** | Flax/Haiku | Clean JAX-native NN libraries | Equinox (simpler but less mature) |
| **Classical QC** | PySCF | Standard, well-documented, Python-native | Psi4 (heavier install) |
| **Optimization** | Optax | JAX-native optimizers, easy custom schedules | Manual implementation |
| **Testing** | pytest | Standard, good fixtures | unittest |
| **Visualization** | Matplotlib | Standard, sufficient for energy curves | Plotly (overkill) |
| **Config** | Hydra | Flexible experiment configs | YAML + argparse |
| **Logging** | Weights & Biases (free tier) | Experiment tracking, free for individuals | TensorBoard |

### Free/Open-Source Commitment

All tools are free and open-source. Cloud compute: Google Colab (free tier) for initial experiments, with optional upgrade path to Colab Pro or Lambda Labs for larger systems.

---

## Phased Implementation Plan

### Phase 1: Foundation and H₂ Validation

**Objective:** Implement minimal VMC loop for H₂ molecule to validate the core algorithm.

**Scope:**
- `src/nqmc/systems/molecule.py` — Molecular system dataclass
- `src/nqmc/wavefunction/base.py` — Abstract wavefunction interface
- `src/nqmc/wavefunction/simple.py` — Simple feedforward NN (no antisymmetry yet)
- `src/nqmc/hamiltonian/molecular.py` — Kinetic + Coulomb operators
- `src/nqmc/sampler/metropolis.py` — Basic Metropolis-Hastings
- `src/nqmc/optimizer/vmc.py` — VMC energy + gradient estimation
- `scripts/train_h2.py` — Training script for H₂

**Deliverables:**
- Working VMC loop that trains on H₂
- Energy curve vs bond length plot
- Comparison against exact H₂ ground state (-1.174 Ha at equilibrium)

**Verification:**
- Unit tests for each component
- H₂ energy within 10 mHa of exact value

**Technical Challenges:**
1. Numerical stability in log-probability computation (use log-sum-exp tricks)
2. Local energy singularities at coalescence points (regularization needed)
3. MCMC convergence diagnostics (effective sample size, autocorrelation)

**Definition of Done:**
- [ ] All unit tests pass
- [ ] H₂ energy converges to < -1.10 Ha (within 70 mHa of exact)
- [ ] Training completes in < 30 minutes on laptop CPU
- [ ] Energy vs bond length plot generated

**Time Estimate:** 2-3 weeks

**Contingency:** If full H₂ is slow, test with a 1D harmonic oscillator analogue first.

---

### Phase 2: Antisymmetric Wavefunction Architecture

**Objective:** Implement fermionic antisymmetry via determinant-based architecture.

**Scope:**
- `src/nqmc/wavefunction/slater.py` — Slater determinant computation
- `src/nqmc/wavefunction/orbitals.py` — Neural network orbitals
- `src/nqmc/wavefunction/antisymmetric.py` — Full antisymmetric wavefunction
- `src/nqmc/wavefunction/jastrow.py` — Jastrow correlation factor

**Deliverables:**
- Slater determinant layer with stable gradient computation
- Neural orbitals that respect electron-nucleus structure
- Jastrow factor for electron-electron correlation

**Verification:**
- Antisymmetry test: ψ(..., rᵢ, ..., rⱼ, ...) = -ψ(..., rⱼ, ..., rᵢ, ...)
- H₂ energy improvement over Phase 1 (target: < -1.15 Ha)

**Technical Challenges:**
1. Determinant gradient computation (adjugate method vs Jacobi's formula)
2. Numerical stability for near-singular determinants
3. Spin handling (spin-up vs spin-down blocks)

**Definition of Done:**
- [ ] Antisymmetry property verified in tests
- [ ] H₂ energy improves by > 20 mHa over Phase 1
- [ ] No NaN/Inf in training for 10k steps
- [ ] Determinant gradients verified against finite differences

**Time Estimate:** 2 weeks

**Contingency:** If determinant is unstable, use log-det + sign decomposition.

---

### Phase 3: Cusp Conditions and Backflow

**Objective:** Enforce electron-nucleus and electron-electron cusp conditions for correct wavefunction behavior at coalescence.

**Scope:**
- `src/nqmc/wavefunction/cusp.py` — Cusp correction layers
- `src/nqmc/wavefunction/backflow.py` — Backflow transformation
- Update `src/nqmc/wavefunction/antisymmetric.py` to integrate cusp/backflow

**Deliverables:**
- Cusp conditions that satisfy Kato's theorem
- Backflow coordinates that improve correlation capture
- Reduced variance in local energy estimates

**Verification:**
- Check that local energy variance decreases by > 20%
- Verify cusp behavior analytically near nuclei

**Technical Challenges:**
1. Cusp implementation that doesn't break antisymmetry
2. Backflow Jacobian computation
3. Balancing cusp corrections with neural network flexibility

**Definition of Done:**
- [ ] Local energy variance reduced by > 20%
- [ ] Cusp conditions verified in tests
- [ ] H₂ energy reaches < -1.17 Ha (within 5 mHa of exact)

**Time Estimate:** 2 weeks

**Contingency:** Implement cusp without backflow first if time-constrained.

---

### Phase 4: Hydrogen Chain (H₄) Extension

**Objective:** Scale the implementation to H₄ linear chain.

**Scope:**
- `src/nqmc/systems/hydrogen_chain.py` — H₄/H₆ system generator
- Update sampler for higher-dimensional configuration space
- Implement spin handling for 4 electrons (2 up, 2 down)
- `scripts/train_h4.py` — H₄ training script

**Deliverables:**
- Working VMC for H₄ at various bond lengths
- Potential energy surface plot
- Comparison against PySCF HF and CCSD

**Verification:**
- Energy within 50 mHa of CCSD for equilibrium geometry
- Training converges within reasonable time (< 2 hours on laptop)

**Technical Challenges:**
1. Increased configuration space dimensionality (12D vs 6D)
2. Longer MCMC decorrelation times
3. Harder optimization landscape

**Definition of Done:**
- [ ] H₄ training completes without failure
- [ ] Energy vs bond length curve generated
- [ ] Comparison table: Neural QMC vs HF vs CCSD
- [ ] Training time < 2 hours on laptop CPU

**Time Estimate:** 2-3 weeks

**Contingency:** If H₄ is too slow, use GPU on Colab.

---

### Phase 5: PySCF Integration and Benchmarking

**Objective:** Implement comprehensive comparison framework against classical methods.

**Scope:**
- `src/nqmc/baselines/pyscf_runner.py` — PySCF wrapper for HF, MP2, CCSD
- `src/nqmc/analysis/comparison.py` — Comparison metrics and plotting
- `notebooks/benchmark_analysis.ipynb` — Analysis notebook
- `scripts/run_benchmark.py` — Full benchmark suite

**Deliverables:**
- Automated benchmark pipeline
- Publication-ready comparison figures
- Error analysis (statistical, basis set, correlation)

**Verification:**
- PySCF wrapper reproduces known values
- All energy values have error bars

**Technical Challenges:**
1. Consistent geometry definitions between neural QMC and PySCF
2. Basis set effects in classical methods vs basis-free neural approach
3. Statistical error estimation for VMC energies

**Definition of Done:**
- [ ] PySCF benchmarks for H₂, H₄ complete
- [ ] Comparison table with statistical errors
- [ ] Analysis notebook with publication-ready figures
- [ ] README updated with benchmark results

**Time Estimate:** 1-2 weeks

---

### Phase 6: H₆ and Optimization Improvements (Stretch Goal)

**Objective:** Scale to H₆ and implement advanced optimization.

**Scope:**
- Extend to H₆ (6 electrons)
- Implement KFAC or Stochastic Reconfiguration optimizer
- Parallel sampling across multiple walkers
- `scripts/train_h6.py`

**Deliverables:**
- H₆ results
- Comparison of optimization methods
- Analysis of scaling behavior

**Verification:**
- H₆ training completes
- Scaling analysis documented

**Time Estimate:** 2-3 weeks (stretch)

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning Signs | Mitigation |
|------|------------|--------|---------------------|------------|
| Numerical instability (NaN/Inf) | High | High | Training loss spikes, determinant near-singular | Log-det decomposition, gradient clipping, careful initialization |
| Slow MCMC convergence | High | Medium | Low acceptance rate, high autocorrelation | Tune proposal width, implement adaptive MCMC |
| VMC not reaching chemical accuracy | Medium | High | Energy plateau far from target | Add more network capacity, implement cusp/backflow |
| JAX/Flax complexity | Medium | Medium | Slow development, debugging difficulty | Start with pure JAX, add Flax incrementally |
| Compute limitations | Medium | Medium | Long training times on laptop | Use Colab GPU, optimize batch sizes |
| PySCF integration issues | Low | Medium | Import errors, geometry mismatches | Test integration early, use well-documented APIs |

---

## Testing Strategy

### Testing Levels

| Level | Scope | Tools | Coverage Target |
|-------|-------|-------|-----------------|
| Unit | Individual functions | pytest | 80% |
| Integration | Component interactions | pytest | Key pathways |
| System | Full VMC loop | pytest + manual | End-to-end |
| Validation | Physics correctness | Manual + known values | Critical properties |

### First Three Tests to Write

**1. `tests/test_wavefunction.py::test_antisymmetry`**
```python
def test_antisymmetry():
    """Verify wavefunction changes sign under electron exchange."""
    wf = AntisymmetricWavefunction(n_electrons=4, ...)
    params = wf.init(key, sample_config)

    r = random_electron_config(n_electrons=4)
    r_swapped = swap_electrons(r, i=0, j=1)

    psi_original = wf.apply(params, r)
    psi_swapped = wf.apply(params, r_swapped)

    assert jnp.allclose(psi_original, -psi_swapped, rtol=1e-5)
```

**2. `tests/test_hamiltonian.py::test_hydrogen_atom_energy`**
```python
def test_hydrogen_atom_energy():
    """Verify local energy for hydrogen atom ground state."""
    # For ψ = exp(-r), E_L = -0.5 Ha exactly
    hamiltonian = MolecularHamiltonian(atoms=[("H", [0, 0, 0])])

    def exact_wavefunction(r):
        return jnp.exp(-jnp.linalg.norm(r))

    r_samples = sample_from_exact_distribution(n_samples=1000)
    local_energies = hamiltonian.local_energy(exact_wavefunction, r_samples)

    assert jnp.abs(jnp.mean(local_energies) - (-0.5)) < 0.01
```

**3. `tests/test_sampler.py::test_metropolis_distribution`**
```python
def test_metropolis_distribution():
    """Verify Metropolis sampler produces correct distribution."""
    # Sample from |ψ|² where ψ = exp(-r²), should match Gaussian
    def log_prob(r):
        return -2 * jnp.sum(r**2)

    sampler = MetropolisSampler(log_prob_fn=log_prob, step_size=0.5)
    samples = sampler.sample(n_samples=10000, n_chains=4)

    # Check variance matches expected (should be 0.25 for this ψ)
    empirical_var = jnp.var(samples)
    expected_var = 0.25

    assert jnp.abs(empirical_var - expected_var) < 0.05
```

### Testing Framework

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/nqmc --cov-report=html

# Run specific test file
pytest tests/test_wavefunction.py -v
```

---

## First Concrete Task

### File: `src/nqmc/systems/molecule.py`

### Function Signature

```python
@dataclass
class Molecule:
    """Represents a molecular system for quantum Monte Carlo.

    Attributes:
        atoms: List of (element, position) tuples. Element is str, position is 3D array.
        charges: Nuclear charges for each atom.
        n_electrons: Total number of electrons.
        spin: Tuple of (n_up, n_down) electrons.
    """
    atoms: List[Tuple[str, np.ndarray]]
    charges: np.ndarray
    n_electrons: int
    spin: Tuple[int, int]

    @classmethod
    def from_xyz(cls, xyz_string: str) -> "Molecule":
        """Parse XYZ format string into Molecule."""
        ...

    def electron_nuclear_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute distances from each electron to each nucleus.

        Args:
            electron_positions: Shape (n_electrons, 3)

        Returns:
            Shape (n_electrons, n_atoms) distance matrix
        """
        ...

    def electron_electron_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between electrons.

        Args:
            electron_positions: Shape (n_electrons, 3)

        Returns:
            Shape (n_electrons, n_electrons) distance matrix (symmetric, zero diagonal)
        """
        ...
```

### Starter Code

```python
"""Molecular system definitions for neural QMC."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Element to nuclear charge mapping
ELEMENT_CHARGES = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8
}


@dataclass
class Molecule:
    """Represents a molecular system for quantum Monte Carlo.

    Attributes:
        atoms: List of (element, position) tuples.
        charges: Nuclear charges for each atom.
        n_electrons: Total number of electrons.
        spin: Tuple of (n_up, n_down) electrons.
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
        """
        atoms = [(el, np.array(pos)) for el, pos in zip(elements, positions)]
        charges = np.array([ELEMENT_CHARGES[el] for el in elements])
        n_electrons = int(np.sum(charges)) - charge

        # Determine spin configuration
        n_up = (n_electrons + spin_polarization) // 2
        n_down = n_electrons - n_up

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
        """Atomic positions as (n_atoms, 3) array."""
        return np.array([pos for _, pos in self.atoms])

    def electron_nuclear_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute distances from each electron to each nucleus.

        Args:
            electron_positions: Shape (n_electrons, 3)

        Returns:
            Shape (n_electrons, n_atoms) distance matrix
        """
        # TODO: Implement this
        # Hint: Use broadcasting - electron_positions[:, None, :] - self.positions[None, :, :]
        raise NotImplementedError

    def electron_electron_distances(self, electron_positions: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between electrons.

        Args:
            electron_positions: Shape (n_electrons, 3)

        Returns:
            Shape (n_electrons, n_electrons) distance matrix
        """
        # TODO: Implement this
        # Hint: diff = electron_positions[:, None, :] - electron_positions[None, :, :]
        raise NotImplementedError


def hydrogen_molecule(bond_length: float = 1.4) -> Molecule:
    """Create H2 molecule at specified bond length (in Bohr).

    Args:
        bond_length: H-H distance in Bohr (default 1.4, near equilibrium)

    Returns:
        H2 Molecule instance
    """
    positions = np.array([
        [0.0, 0.0, 0.0],
        [bond_length, 0.0, 0.0]
    ])
    return Molecule.from_atoms(["H", "H"], positions)


def hydrogen_chain(n_atoms: int, bond_length: float = 1.4) -> Molecule:
    """Create linear hydrogen chain.

    Args:
        n_atoms: Number of hydrogen atoms (must be even for closed shell)
        bond_length: H-H distance in Bohr

    Returns:
        Hydrogen chain Molecule instance
    """
    if n_atoms % 2 != 0:
        raise ValueError("n_atoms must be even for closed-shell hydrogen chain")

    positions = np.array([[i * bond_length, 0.0, 0.0] for i in range(n_atoms)])
    elements = ["H"] * n_atoms

    return Molecule.from_atoms(elements, positions)
```

### Verification Method

```bash
# After implementing the TODO methods, run:
python -c "
from src.nqmc.systems.molecule import hydrogen_molecule, hydrogen_chain
import numpy as np

# Test H2
h2 = hydrogen_molecule(bond_length=1.4)
print(f'H2: {h2.n_electrons} electrons, spin {h2.spin}')

# Test distance computation
r = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 2 electrons
r_en = h2.electron_nuclear_distances(r)
print(f'Electron-nuclear distances shape: {r_en.shape}')  # Should be (2, 2)

r_ee = h2.electron_electron_distances(r)
print(f'Electron-electron distances shape: {r_ee.shape}')  # Should be (2, 2)
print(f'e-e distance [0,1]: {r_ee[0,1]:.4f}')  # Should be 0.5

# Test H4
h4 = hydrogen_chain(n_atoms=4)
print(f'H4: {h4.n_electrons} electrons, spin {h4.spin}')
"
```

### First Commit Message

```
feat(systems): Add Molecule class with distance computations

Implement core molecular system dataclass with:
- Atom/charge/electron configuration
- Factory methods for H2 and hydrogen chains
- Electron-nuclear and electron-electron distance matrices

This provides the foundation for Hamiltonian and wavefunction evaluations.
```

---

## Appendix: Concepts to Study Before Coding

For a junior developer, these concepts require deeper understanding before implementation:

1. **Variational Principle** — Why minimizing ⟨H⟩ gives the ground state
2. **Slater Determinants** — How determinants enforce antisymmetry
3. **Cusp Conditions** — Kato's theorem and why wavefunctions have cusps
4. **MCMC Basics** — Metropolis-Hastings algorithm and detailed balance
5. **Log-derivative Trick** — Why gradients involve log ψ
6. **JAX Fundamentals** — vmap, grad, jit, and functional programming style

Recommended reading order: 2 → 1 → 4 → 5 → 3 → 6

---

## Visualization Strategy

### Visualization Directory Structure

```
visualizations/
├── energy_curves/          # Energy vs bond length, convergence plots
├── wavefunction_plots/     # Electron density, orbital visualizations
├── sampling_diagnostics/   # MCMC traces, autocorrelation, acceptance
├── architecture_diagrams/  # Neural network architecture visualizations
└── comparison_charts/      # Method comparison bar charts, scatter plots
```

### Phase-Specific Visualizations

#### Phase 1: Foundation Visualizations

| Visualization | File | Purpose |
|---------------|------|---------|
| Energy Convergence | `energy_curves/h2_convergence.png` | Training loss over optimization steps |
| Local Energy Histogram | `sampling_diagnostics/local_energy_dist.png` | Distribution of E_L to check for outliers |
| MCMC Trace | `sampling_diagnostics/mcmc_trace.png` | Electron position samples over time |
| Acceptance Rate | `sampling_diagnostics/acceptance_rate.png` | MCMC acceptance probability vs step size |

**Sample Code for Energy Convergence:**
```python
def plot_energy_convergence(energies, target_energy=None, save_path=None):
    """Plot energy vs optimization step with running average."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(len(energies))
    ax.plot(steps, energies, alpha=0.3, color='blue', label='Raw')

    # Running average
    window = min(100, len(energies) // 10)
    running_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], running_avg, color='blue', linewidth=2, label='Running avg')

    if target_energy:
        ax.axhline(y=target_energy, color='red', linestyle='--',
                   label=f'Target: {target_energy:.4f} Ha')

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title('VMC Energy Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

#### Phase 2: Wavefunction Visualizations

| Visualization | File | Purpose |
|---------------|------|---------|
| Orbital Contours | `wavefunction_plots/orbitals_2d.png` | 2D slices of neural orbitals |
| Electron Density | `wavefunction_plots/electron_density.png` | |ψ|² integrated over electron pairs |
| Determinant Magnitude | `wavefunction_plots/det_magnitude.png` | Slater determinant value landscape |

**Sample Code for Electron Density:**
```python
def plot_electron_density(wavefunction, params, molecule, resolution=50, save_path=None):
    """Plot 2D electron density slice through molecular plane."""
    # Create grid in x-y plane (z=0)
    x = np.linspace(-3, molecule.positions[:, 0].max() + 3, resolution)
    y = np.linspace(-3, 3, resolution)
    X, Y = np.meshgrid(x, y)

    # Compute |ψ|² at grid points (fix one electron, scan other)
    density = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            r1 = np.array([X[i, j], Y[i, j], 0.0])
            # Average over second electron positions
            density[i, j] = compute_marginal_density(wavefunction, params, r1)

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(X, Y, density, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Electron density |ψ|²')

    # Mark nuclei
    for el, pos in molecule.atoms:
        ax.scatter(pos[0], pos[1], c='red', s=100, marker='o', edgecolors='white')
        ax.annotate(el, (pos[0], pos[1] + 0.3), ha='center', fontsize=12, color='white')

    ax.set_xlabel('x (Bohr)', fontsize=12)
    ax.set_ylabel('y (Bohr)', fontsize=12)
    ax.set_title('Electron Density in Molecular Plane', fontsize=14)
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

#### Phase 3: Cusp Condition Visualizations

| Visualization | File | Purpose |
|---------------|------|---------|
| Radial Wavefunction | `wavefunction_plots/radial_cusp.png` | ψ(r) near nucleus showing cusp |
| Cusp Derivative | `wavefunction_plots/cusp_derivative.png` | dψ/dr at coalescence |
| Variance Reduction | `sampling_diagnostics/variance_comparison.png` | Local energy variance before/after cusp |

**Sample Code for Cusp Visualization:**
```python
def plot_cusp_condition(wavefunction, params, nucleus_pos, save_path=None):
    """Visualize wavefunction behavior near nucleus (cusp condition)."""
    # Radial distances from nucleus
    r = np.linspace(0.01, 2.0, 200)
    psi_values = []

    for ri in r:
        electron_pos = nucleus_pos + np.array([ri, 0, 0])
        # Fix other electrons far away
        psi = evaluate_wavefunction_at_point(wavefunction, params, electron_pos)
        psi_values.append(psi)

    psi_values = np.array(psi_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Wavefunction
    axes[0].plot(r, psi_values, 'b-', linewidth=2)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Distance from nucleus (Bohr)', fontsize=12)
    axes[0].set_ylabel('ψ(r)', fontsize=12)
    axes[0].set_title('Wavefunction Near Nucleus', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Right: Log derivative (should approach -Z at r→0)
    dpsi_dr = np.gradient(psi_values, r)
    log_deriv = dpsi_dr / (psi_values + 1e-10)

    axes[1].plot(r, log_deriv, 'b-', linewidth=2, label='Neural ψ')
    axes[1].axhline(y=-1, color='red', linestyle='--', label='Exact cusp (-Z)')
    axes[1].set_xlabel('Distance from nucleus (Bohr)', fontsize=12)
    axes[1].set_ylabel('(1/ψ) dψ/dr', fontsize=12)
    axes[1].set_title('Cusp Condition Check', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-5, 2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

#### Phase 4-5: Comparison Visualizations

| Visualization | File | Purpose |
|---------------|------|---------|
| Potential Energy Surface | `energy_curves/h4_pes.png` | Energy vs bond length for H₄ |
| Method Comparison | `comparison_charts/method_comparison.png` | Bar chart: Neural QMC vs HF vs CCSD |
| Correlation Energy | `comparison_charts/correlation_energy.png` | E_corr = E_exact - E_HF |
| Error Analysis | `comparison_charts/error_bars.png` | Energy with statistical error bars |
| Scaling Plot | `comparison_charts/scaling.png` | Computation time vs system size |

**Sample Code for Method Comparison:**
```python
def plot_method_comparison(results_dict, molecule_name, save_path=None):
    """Create publication-ready method comparison bar chart."""
    methods = list(results_dict.keys())
    energies = [results_dict[m]['energy'] for m in methods]
    errors = [results_dict[m].get('error', 0) for m in methods]

    # Color coding
    colors = {
        'HF': '#2ecc71',
        'MP2': '#3498db',
        'CCSD': '#9b59b6',
        'CCSD(T)': '#e74c3c',
        'Neural QMC': '#f39c12'
    }
    bar_colors = [colors.get(m, '#95a5a6') for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    bars = ax.bar(x, energies, yerr=errors, capsize=5, color=bar_colors,
                  edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, energy, error in zip(bars, energies, errors):
        height = bar.get_height()
        ax.annotate(f'{energy:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title(f'{molecule_name} Ground State Energy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add reference line for exact if available
    if 'Exact' in results_dict:
        ax.axhline(y=results_dict['Exact']['energy'], color='black',
                   linestyle='--', linewidth=2, label='Exact')
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

**Sample Code for PES Plot:**
```python
def plot_potential_energy_surface(bond_lengths, energies_dict, save_path=None):
    """Plot potential energy surface for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {'HF': 's', 'CCSD': '^', 'Neural QMC': 'o'}
    colors = {'HF': '#2ecc71', 'CCSD': '#9b59b6', 'Neural QMC': '#f39c12'}

    for method, energies in energies_dict.items():
        ax.plot(bond_lengths, energies,
                marker=markers.get(method, 'o'),
                color=colors.get(method, '#95a5a6'),
                linewidth=2, markersize=8,
                label=method)

        # Add error bands if available
        if f'{method}_std' in energies_dict:
            std = energies_dict[f'{method}_std']
            ax.fill_between(bond_lengths,
                            np.array(energies) - std,
                            np.array(energies) + std,
                            alpha=0.2, color=colors.get(method))

    ax.set_xlabel('Bond Length (Bohr)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title('H₄ Potential Energy Surface', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

### Publication-Quality Figure Guidelines

| Aspect | Specification |
|--------|---------------|
| **Resolution** | 150 DPI for draft, 300+ DPI for publication |
| **Figure Size** | 10×6 inches (single column) or 14×5 (double) |
| **Font Size** | 12pt for labels, 14pt for titles |
| **Line Width** | 2pt for data, 1.5pt for grid |
| **Color Scheme** | Colorblind-friendly (viridis, plasma, or custom) |
| **Format** | PNG for draft, PDF/SVG for publication |

### Visualization Module Structure

```python
# src/nqmc/visualization/
├── __init__.py           # Public API
├── energy_plots.py       # Energy convergence, PES curves
├── wavefunction_viz.py   # Electron density, orbitals
├── sampling_viz.py       # MCMC diagnostics
├── comparison_viz.py     # Method comparison charts
├── style.py              # Matplotlib style configuration
└── utils.py              # Common plotting utilities
```

### Required Visualizations Checklist

**Phase 1:**
- [ ] Energy convergence curve (training loss)
- [ ] Local energy histogram
- [ ] MCMC trace plot
- [ ] Acceptance rate vs step size

**Phase 2:**
- [ ] 2D orbital contour plots
- [ ] Electron density visualization
- [ ] Antisymmetry verification plot (ψ vs -ψ after swap)

**Phase 3:**
- [ ] Cusp condition near nucleus
- [ ] Variance reduction comparison
- [ ] Backflow coordinate visualization

**Phase 4:**
- [ ] H₄ potential energy surface
- [ ] Bond dissociation curve
- [ ] Training time scaling

**Phase 5:**
- [ ] Method comparison bar chart (HF, CCSD, Neural QMC)
- [ ] Correlation energy plot
- [ ] Error bar comparison
- [ ] Wall time comparison

**Phase 6 (Stretch):**
- [ ] H₂ → H₄ → H₆ scaling analysis
- [ ] Optimizer comparison (Adam vs KFAC vs SR)
- [ ] Parameter count vs accuracy trade-off
