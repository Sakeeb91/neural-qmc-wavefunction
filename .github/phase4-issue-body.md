## Summary

Scale the Neural QMC implementation from H₂ to H₄ linear hydrogen chain, handling 4 electrons (2 up, 2 down) and generating potential energy surface data.

**Parent Issue:** #1
**Depends On:** #4 (Phase 3)

## Objectives

- Extend molecular system to handle hydrogen chains
- Scale wavefunction architecture for 4 electrons
- Implement proper spin block handling (2×2 determinants)
- Generate H₄ potential energy surface at various bond lengths
- Compare against PySCF HF and CCSD baselines

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/systems/hydrogen_chain.py` | ~60 | Create | H₄/H₆ system generators |
| `src/nqmc/wavefunction/antisymmetric.py` | ~30 | Modify | Handle larger determinants |
| `scripts/train_h4.py` | ~100 | Create | H₄ training script |
| `scripts/scan_bond_length.py` | ~80 | Create | PES scanning script |
| `notebooks/h4_analysis.ipynb` | - | Create | Results analysis |

## Implementation Tasks

### 1. Hydrogen Chain System (`src/nqmc/systems/hydrogen_chain.py`)

- [ ] Create `HydrogenChain` class extending `Molecule`
- [ ] Support variable number of atoms (4, 6, 8...)
- [ ] Support variable bond lengths for PES scans
- [ ] Add methods for equilibrium geometry estimation

📄 **Starter Code:**

```python
class HydrogenChain:
    """Linear hydrogen chain generator."""

    def __init__(self, n_atoms: int, bond_length: float = 1.8):
        if n_atoms % 2 != 0:
            raise ValueError("n_atoms must be even for closed-shell")

        self.n_atoms = n_atoms
        self.bond_length = bond_length
        self.n_electrons = n_atoms  # One electron per H
        self.n_up = n_atoms // 2
        self.n_down = n_atoms // 2

    def get_molecule(self) -> Molecule:
        """Generate Molecule object for current geometry."""
        positions = np.array([
            [i * self.bond_length, 0.0, 0.0]
            for i in range(self.n_atoms)
        ])
        elements = ["H"] * self.n_atoms
        return Molecule.from_atoms(elements, positions)

    def scan_bond_lengths(self, lengths: np.ndarray) -> List[Molecule]:
        """Generate molecules at various bond lengths."""
        molecules = []
        for R in lengths:
            self.bond_length = R
            molecules.append(self.get_molecule())
        return molecules

    @property
    def equilibrium_bond_length(self) -> float:
        """Approximate equilibrium bond length in Bohr."""
        return 1.8  # ~0.95 Angstrom
```

### 2. Update Wavefunction for H₄

- [ ] Handle 2×2 Slater determinants for spin blocks
- [ ] Increase orbital hidden dimensions for expressiveness
- [ ] Scale Jastrow factor for more electron pairs

📄 **Key Changes:**

```python
# For H₄: 2 up, 2 down
# Orbital matrix for spin-up: (2, 2) - 2 electrons, 2 orbitals
# Determinant is now a proper 2×2 determinant

class AntisymmetricWavefunction(nn.Module):
    n_up: int = 2
    n_down: int = 2
    orbital_hidden_dims: Tuple[int, ...] = (64, 64)  # Larger for H₄

    def compute_slater(self, r, molecule):
        r_up = r[:self.n_up]
        r_down = r[self.n_up:]

        # Build orbital matrices
        phi_up = jnp.stack([
            self.orbitals_up(r_up[i], molecule.positions, molecule.charges)
            for i in range(self.n_up)
        ])  # (n_up, n_orbitals)

        phi_down = jnp.stack([
            self.orbitals_down(r_down[i], molecule.positions, molecule.charges)
            for i in range(self.n_down)
        ])  # (n_down, n_orbitals)

        # Compute determinants
        sign_up, log_det_up = jnp.linalg.slogdet(phi_up)
        sign_down, log_det_down = jnp.linalg.slogdet(phi_down)

        return sign_up * sign_down, log_det_up + log_det_down
```

### 3. Training Script (`scripts/train_h4.py`)

- [ ] Set up H₄ system at equilibrium geometry
- [ ] Configure larger network and longer training
- [ ] Implement learning rate schedule (warmup + decay)
- [ ] Save checkpoints and generate visualizations

📄 **Configuration:**

```python
@dataclass
class H4Config:
    # System
    n_atoms: int = 4
    bond_length: float = 1.8

    # Wavefunction
    orbital_hidden_dims: Tuple[int, ...] = (64, 64)
    jastrow_hidden_dims: Tuple[int, ...] = (32, 32)
    use_cusp: bool = True
    use_backflow: bool = True

    # MCMC
    n_walkers: int = 1024
    n_steps_per_param_update: int = 10
    step_size: float = 0.5

    # Optimization
    n_iterations: int = 20000
    learning_rate: float = 1e-3
    lr_decay_rate: float = 0.99
    gradient_clip: float = 1.0

    # Logging
    log_every: int = 100
    checkpoint_every: int = 1000
```

### 4. PES Scanning Script (`scripts/scan_bond_length.py`)

- [ ] Scan bond lengths from 1.0 to 4.0 Bohr
- [ ] Train or load checkpoints for each geometry
- [ ] Collect energies and uncertainties
- [ ] Save results for plotting

📄 **Scan Script:**

```python
def scan_pes(
    bond_lengths: np.ndarray,
    n_iterations_per_point: int = 5000,
    output_dir: str = "results/h4_pes"
):
    """Scan H₄ potential energy surface."""
    results = []

    for R in bond_lengths:
        print(f"Bond length: {R:.2f} Bohr")

        # Create system
        chain = HydrogenChain(n_atoms=4, bond_length=R)
        molecule = chain.get_molecule()

        # Train or load
        checkpoint_path = f"{output_dir}/checkpoint_R{R:.2f}.pkl"
        if os.path.exists(checkpoint_path):
            params, energy = load_checkpoint(checkpoint_path)
        else:
            params, energy, std = train_vmc(molecule, n_iterations_per_point)
            save_checkpoint(checkpoint_path, params, energy)

        results.append({
            'bond_length': R,
            'energy': energy,
            'std': std
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    bond_lengths = np.linspace(1.0, 4.0, 16)
    df = scan_pes(bond_lengths)
    df.to_csv("results/h4_pes_data.csv", index=False)
```

### 5. Analysis Notebook (`notebooks/h4_analysis.ipynb`)

- [ ] Load Neural QMC results
- [ ] Compute PySCF baselines (HF, CCSD)
- [ ] Plot comparison PES curves
- [ ] Calculate correlation energy
- [ ] Generate publication figures

## Visualizations for Phase 4

- [ ] `visualizations/energy_curves/h4_pes.png` - Full PES comparison
- [ ] `visualizations/energy_curves/h4_convergence.png` - Training curves at equilibrium
- [ ] `visualizations/comparison_charts/h4_method_comparison.png` - Bar chart at equilibrium
- [ ] `visualizations/wavefunction_plots/h4_density.png` - Electron density along chain

## Definition of Done

- [ ] H₄ training completes without numerical failures
- [ ] Energy at equilibrium within 50 mHa of CCSD baseline
- [ ] PES curve generated for 10+ bond lengths
- [ ] Training time < 2 hours on laptop CPU (per geometry)
- [ ] Comparison table: Neural QMC vs HF vs CCSD

## Benchmark Targets

| Method | H₄ Equilibrium Energy (Ha) | Correlation Captured |
|--------|---------------------------|---------------------|
| HF | ~-1.95 | 0% |
| MP2 | ~-2.00 | ~60% |
| CCSD | ~-2.05 | ~95% |
| Neural QMC | < -2.00 | Target > 50% |
| Exact (FCI) | ~-2.08 | 100% |

## Technical Notes

**Scaling Considerations:**
- Configuration space: 12D (4 electrons × 3 coordinates) vs 6D for H₂
- MCMC decorrelation time increases
- Determinant computation scales as O(n³)

**GPU Acceleration:**
- If CPU is too slow, use JAX GPU backend
- Google Colab provides free T4 GPU
- Vectorize walker updates with vmap

**Initialization:**
- Initialize orbitals from HF solution for faster convergence
- Use atomic orbital-like features in neural network
