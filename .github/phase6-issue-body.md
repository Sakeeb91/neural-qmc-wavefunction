## Summary

**STRETCH GOAL:** Scale the implementation to H₆ (6 electrons) and implement advanced optimization techniques (KFAC, Stochastic Reconfiguration) for improved convergence.

**Parent Issue:** #1
**Depends On:** #6 (Phase 5)

## Objectives

- Extend system to H₆ (3 up, 3 down electrons)
- Implement KFAC (Kronecker-Factored Approximate Curvature) optimizer
- Implement Stochastic Reconfiguration (SR) optimizer
- Analyze scaling behavior (H₂ → H₄ → H₆)
- Document performance and accuracy trade-offs

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/optimizer/kfac.py` | ~200 | Create | KFAC optimizer |
| `src/nqmc/optimizer/sr.py` | ~150 | Create | Stochastic Reconfiguration |
| `scripts/train_h6.py` | ~100 | Create | H₆ training script |
| `scripts/scaling_analysis.py` | ~80 | Create | Scaling study |
| `notebooks/scaling_analysis.ipynb` | - | Create | Scaling visualization |

## Implementation Tasks

### 1. H₆ System Extension

- [ ] Configure H₆ chain (3 up, 3 down)
- [ ] Scale wavefunction architecture
- [ ] Handle 3×3 determinants for spin blocks
- [ ] Optimize MCMC for 18D configuration space

📄 **H₆ Configuration:**

```python
@dataclass
class H6Config:
    # System
    n_atoms: int = 6
    bond_length: float = 1.8
    n_up: int = 3
    n_down: int = 3

    # Wavefunction (larger for H₆)
    orbital_hidden_dims: Tuple[int, ...] = (128, 128)
    n_determinants: int = 1  # Could use multi-determinant expansion

    # MCMC
    n_walkers: int = 2048  # More walkers for higher D
    n_steps_per_update: int = 20
    step_size: float = 0.3  # Smaller step in higher D

    # Optimization
    n_iterations: int = 50000
    optimizer: str = "kfac"  # or "sr" or "adam"
```

### 2. KFAC Optimizer (`src/nqmc/optimizer/kfac.py`)

- [ ] Implement Kronecker-factored Fisher approximation
- [ ] Compute layer-wise curvature estimates
- [ ] Update with natural gradient direction
- [ ] Damping for numerical stability

📄 **KFAC Overview:**

```python
class KFACOptimizer:
    """Kronecker-Factored Approximate Curvature optimizer.

    Approximates Fisher information matrix as Kronecker product of
    activation covariance and gradient covariance per layer.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        damping: float = 1e-3,
        cov_ema_decay: float = 0.95,
    ):
        self.lr = learning_rate
        self.damping = damping
        self.ema_decay = cov_ema_decay
        self.cov_a = {}  # Activation covariances
        self.cov_g = {}  # Gradient covariances

    def update_curvature(self, params, activations, grads):
        """Update running estimates of curvature factors."""
        for layer_name in params.keys():
            a = activations[layer_name]  # (batch, in_features)
            g = grads[layer_name]        # (batch, out_features)

            # Update covariances with EMA
            cov_a_new = jnp.einsum('bi,bj->ij', a, a) / a.shape[0]
            cov_g_new = jnp.einsum('bi,bj->ij', g, g) / g.shape[0]

            if layer_name in self.cov_a:
                self.cov_a[layer_name] = (
                    self.ema_decay * self.cov_a[layer_name] +
                    (1 - self.ema_decay) * cov_a_new
                )
                self.cov_g[layer_name] = (
                    self.ema_decay * self.cov_g[layer_name] +
                    (1 - self.ema_decay) * cov_g_new
                )
            else:
                self.cov_a[layer_name] = cov_a_new
                self.cov_g[layer_name] = cov_g_new

    def compute_update(self, params, grads):
        """Compute natural gradient update."""
        updates = {}
        for layer_name, grad in grads.items():
            # Inverse of Kronecker factors with damping
            cov_a_inv = jnp.linalg.inv(
                self.cov_a[layer_name] + self.damping * jnp.eye(self.cov_a[layer_name].shape[0])
            )
            cov_g_inv = jnp.linalg.inv(
                self.cov_g[layer_name] + self.damping * jnp.eye(self.cov_g[layer_name].shape[0])
            )

            # Natural gradient: F^{-1} * grad ≈ (A^{-1} ⊗ G^{-1}) * grad
            updates[layer_name] = -self.lr * cov_a_inv @ grad @ cov_g_inv

        return updates
```

### 3. Stochastic Reconfiguration (`src/nqmc/optimizer/sr.py`)

- [ ] Implement SR matrix S = ⟨O†O⟩ - ⟨O†⟩⟨O⟩
- [ ] Compute gradient in SR basis
- [ ] Add regularization for ill-conditioned S
- [ ] Compare convergence with Adam/KFAC

📄 **SR Overview:**

```python
class StochasticReconfiguration:
    """Stochastic Reconfiguration optimizer for VMC.

    Updates parameters as: δθ = S^{-1} * ∂E/∂θ
    where S is the covariance matrix of log-derivatives.
    """

    def __init__(
        self,
        learning_rate: float = 1e-2,
        regularization: float = 1e-4,
    ):
        self.lr = learning_rate
        self.reg = regularization

    def compute_update(self, log_psi_grads, local_energies):
        """Compute SR update.

        Args:
            log_psi_grads: Gradients of log|ψ| w.r.t. params, shape (n_samples, n_params)
            local_energies: Local energies, shape (n_samples,)

        Returns:
            Parameter update direction
        """
        n_samples, n_params = log_psi_grads.shape

        # Centered quantities
        O = log_psi_grads - jnp.mean(log_psi_grads, axis=0)
        E = local_energies - jnp.mean(local_energies)

        # S matrix: covariance of log-derivatives
        S = jnp.einsum('ni,nj->ij', O, O) / n_samples
        S += self.reg * jnp.eye(n_params)  # Regularization

        # Energy gradient in original basis
        f = 2 * jnp.einsum('n,ni->i', E, O) / n_samples

        # Solve S * δθ = f
        delta_theta = jnp.linalg.solve(S, f)

        return -self.lr * delta_theta
```

### 4. Scaling Analysis Script (`scripts/scaling_analysis.py`)

- [ ] Train models on H₂, H₄, H₆
- [ ] Measure wall-clock time per iteration
- [ ] Measure energy accuracy vs iteration
- [ ] Measure memory usage
- [ ] Compare optimizers

📄 **Scaling Study:**

```python
def run_scaling_study(output_dir: str = "results/scaling"):
    """Run comprehensive scaling analysis."""
    systems = [
        ("H2", 2, 1.4),
        ("H4", 4, 1.8),
        ("H6", 6, 1.8),
    ]

    optimizers = ["adam", "kfac", "sr"]

    results = []

    for name, n_atoms, R in systems:
        for opt in optimizers:
            print(f"Running {name} with {opt}...")

            # Configure
            molecule = HydrogenChain(n_atoms, R).get_molecule()
            config = get_config(n_atoms, opt)

            # Train and measure
            start_time = time.time()
            energies, params = train_vmc(molecule, config)
            wall_time = time.time() - start_time

            # Record
            results.append({
                'system': name,
                'n_electrons': n_atoms,
                'optimizer': opt,
                'final_energy': energies[-1],
                'energy_std': np.std(energies[-100:]),
                'wall_time': wall_time,
                'n_iterations': len(energies),
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/scaling_results.csv", index=False)
    return df
```

## Visualizations for Phase 6

- [ ] `visualizations/comparison_charts/scaling_electrons.png` - Time vs n_electrons
- [ ] `visualizations/comparison_charts/optimizer_comparison.png` - Adam vs KFAC vs SR
- [ ] `visualizations/energy_curves/h6_convergence.png` - H₆ training curves
- [ ] `visualizations/comparison_charts/accuracy_vs_time.png` - Pareto frontier
- [ ] `visualizations/comparison_charts/memory_scaling.png` - Memory vs system size

## Definition of Done

- [ ] H₆ training completes successfully
- [ ] KFAC and SR optimizers implemented and tested
- [ ] Scaling analysis with 3+ system sizes
- [ ] Optimizer comparison documented
- [ ] All visualizations generated

## Expected Scaling Behavior

| System | n_electrons | Config Dim | Det Size | Est. Time/iter |
|--------|-------------|------------|----------|----------------|
| H₂ | 2 | 6D | 1×1 | 0.01s |
| H₄ | 4 | 12D | 2×2 | 0.05s |
| H₆ | 6 | 18D | 3×3 | 0.2s |
| H₈ | 8 | 24D | 4×4 | 0.8s |

## Technical Notes

**KFAC Considerations:**
- Requires storing activation/gradient statistics per layer
- Memory scales with layer sizes, not sample count
- May need gradient accumulation for stability

**SR Considerations:**
- S matrix is n_params × n_params (can be large)
- Use diagonal approximation for very large networks
- Natural for VMC but requires more samples per update

**GPU Recommendations:**
- H₆ likely needs GPU for reasonable training times
- Use jax.pmap for multi-GPU if available
- Monitor memory usage and reduce batch size if needed
