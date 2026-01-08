## Summary

Implement the foundational VMC loop for H₂ molecule to validate the core algorithm before scaling to larger systems.

**Parent Issue:** #1

## Objectives

- Create molecular system dataclass with distance computations
- Implement basic feedforward neural network wavefunction
- Build molecular Hamiltonian (kinetic + Coulomb operators)
- Create Metropolis-Hastings MCMC sampler
- Implement VMC energy and gradient estimation
- Validate against exact H₂ ground state (-1.174 Ha)

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/systems/molecule.py` | ~100 | Create | Molecular system dataclass |
| `src/nqmc/wavefunction/base.py` | ~50 | Create | Abstract wavefunction interface |
| `src/nqmc/wavefunction/simple.py` | ~80 | Create | Simple feedforward NN |
| `src/nqmc/hamiltonian/molecular.py` | ~150 | Create | Kinetic + Coulomb operators |
| `src/nqmc/sampler/metropolis.py` | ~120 | Create | Metropolis-Hastings sampler |
| `src/nqmc/optimizer/vmc.py` | ~100 | Create | VMC optimizer |
| `scripts/train_h2.py` | ~80 | Create | H₂ training script |

## Implementation Tasks

### 1. Molecular System (`src/nqmc/systems/molecule.py`)

- [ ] Create `Molecule` dataclass with atoms, charges, n_electrons, spin
- [ ] Implement `from_atoms()` factory method
- [ ] Implement `electron_nuclear_distances()`
- [ ] Implement `electron_electron_distances()`
- [ ] Add `hydrogen_molecule()` and `hydrogen_chain()` helpers

📄 **Starter Code:**

```python
@dataclass
class Molecule:
    atoms: List[Tuple[str, np.ndarray]]
    charges: np.ndarray
    n_electrons: int
    spin: Tuple[int, int]

    def electron_nuclear_distances(self, r: np.ndarray) -> np.ndarray:
        """Shape (n_electrons, 3) -> (n_electrons, n_atoms)"""
        diff = r[:, None, :] - self.positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)
```

### 2. Wavefunction Interface (`src/nqmc/wavefunction/base.py`)

- [ ] Create abstract `Wavefunction` class
- [ ] Define `__call__(params, r)` interface
- [ ] Define `log_prob(params, r)` for MCMC
- [ ] Define `init(key, r_sample)` for parameter initialization

### 3. Simple Neural Network (`src/nqmc/wavefunction/simple.py`)

- [ ] Implement `SimpleWavefunction(Wavefunction)`
- [ ] Use Flax `nn.Module` for MLP
- [ ] Input: flattened electron positions (n_electrons * 3)
- [ ] Output: scalar log|ψ|

📄 **Starter Code:**

```python
class SimpleWavefunction(nn.Module):
    hidden_dims: Tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, r):
        # r: (n_electrons, 3)
        x = r.flatten()
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        log_psi = nn.Dense(1)(x).squeeze()
        return log_psi
```

### 4. Hamiltonian (`src/nqmc/hamiltonian/molecular.py`)

- [ ] Implement kinetic energy via Laplacian (∇²ψ/ψ)
- [ ] Implement electron-nuclear Coulomb potential
- [ ] Implement electron-electron Coulomb potential
- [ ] Implement `local_energy(wf, params, r)` function

📄 **Starter Code:**

```python
def local_energy(wf, params, r, molecule):
    """Compute E_L = Hψ/ψ at configuration r."""
    # Kinetic: -0.5 * ∇²ψ/ψ = -0.5 * (∇²log|ψ| + |∇log|ψ||²)
    log_psi = lambda x: wf.apply(params, x)
    grad_log = jax.grad(log_psi)(r)
    laplacian_log = jnp.trace(jax.hessian(log_psi)(r))

    kinetic = -0.5 * (laplacian_log + jnp.sum(grad_log**2))

    # Potential
    r_en = molecule.electron_nuclear_distances(r)
    r_ee = molecule.electron_electron_distances(r)

    v_en = -jnp.sum(molecule.charges / r_en)
    v_ee = jnp.sum(1.0 / (r_ee + jnp.eye(len(r_ee)) * 1e10).flatten()[::len(r_ee)+1])

    return kinetic + v_en + v_ee
```

### 5. MCMC Sampler (`src/nqmc/sampler/metropolis.py`)

- [ ] Implement `MetropolisSampler` class
- [ ] Implement `step(key, r, log_prob)` for single MH step
- [ ] Implement `sample(key, n_samples, n_chains)` for batch sampling
- [ ] Track acceptance rate
- [ ] Implement burn-in and thinning

### 6. VMC Optimizer (`src/nqmc/optimizer/vmc.py`)

- [ ] Implement energy estimation with MC samples
- [ ] Implement gradient via log-derivative trick
- [ ] Use Optax for parameter updates
- [ ] Implement training loop with checkpointing

📄 **Gradient Formula:**

```python
# ∂E/∂θ = 2 * ⟨(E_L - ⟨E_L⟩) * ∂log|ψ|/∂θ⟩
def compute_gradient(wf, params, samples, local_energies):
    mean_energy = jnp.mean(local_energies)
    centered_energies = local_energies - mean_energy

    grad_log_psi = jax.vmap(lambda r: jax.grad(lambda p: wf.apply(p, r))(params))(samples)
    gradient = jax.tree_map(lambda g: 2 * jnp.mean(centered_energies[:, None] * g, axis=0), grad_log_psi)

    return gradient
```

### 7. Training Script (`scripts/train_h2.py`)

- [ ] Command-line interface with Hydra
- [ ] Training loop with logging
- [ ] Save checkpoints and visualizations
- [ ] Print energy vs exact comparison

## Visualizations for Phase 1

- [ ] `visualizations/energy_curves/h2_convergence.png` - Energy vs step
- [ ] `visualizations/sampling_diagnostics/mcmc_trace.png` - Electron trajectories
- [ ] `visualizations/sampling_diagnostics/acceptance_rate.png` - MH acceptance
- [ ] `visualizations/sampling_diagnostics/local_energy_dist.png` - E_L histogram

## Definition of Done

- [ ] All unit tests pass
- [ ] H₂ energy converges to < -1.10 Ha (within 70 mHa of exact -1.174 Ha)
- [ ] Training completes in < 30 minutes on laptop CPU
- [ ] Energy convergence plot generated
- [ ] Code reviewed and documented

## Technical Notes

**Numerical Stability:**
- Use `log_psi` instead of `psi` to avoid overflow
- Regularize 1/r terms with small epsilon near coalescence
- Use log-sum-exp for probability ratios in MH

**Testing:**
- Test H atom first (exact solution known: E = -0.5 Ha)
- Verify gradients with finite differences
