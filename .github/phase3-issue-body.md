## Summary

Implement electron-nucleus and electron-electron cusp conditions to enforce correct wavefunction behavior at particle coalescence, significantly reducing local energy variance.

**Parent Issue:** #1
**Depends On:** #3 (Phase 2)

## Objectives

- Implement Kato cusp conditions for electron-nucleus coalescence
- Implement cusp correction for electron-electron coalescence
- Add backflow transformation for improved correlation
- Reduce local energy variance by > 20%
- Achieve H₂ energy within 5 mHa of exact (-1.174 Ha)

## Theoretical Background

**Kato Cusp Conditions:**
At electron-nucleus coalescence (r → 0):
```
(∂ψ/∂r) / ψ |_{r=0} = -Z
```
where Z is the nuclear charge.

At electron-electron coalescence:
```
(∂ψ/∂r_{12}) / ψ |_{r_{12}=0} = 1/2 (same spin) or 1/4 (opposite spin)
```

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/wavefunction/cusp.py` | ~150 | Create | Cusp correction layers |
| `src/nqmc/wavefunction/backflow.py` | ~120 | Create | Backflow transformation |
| `src/nqmc/wavefunction/antisymmetric.py` | ~50 | Modify | Integrate cusp/backflow |
| `tests/test_cusp.py` | ~80 | Create | Cusp verification tests |

## Implementation Tasks

### 1. Electron-Nuclear Cusp (`src/nqmc/wavefunction/cusp.py`)

- [ ] Implement cusp envelope: `ψ_cusp = ψ_NN * Π_A exp(-Z_A * r_iA) * (1 + f(r_iA))`
- [ ] Neural network `f(r)` with `f(0) = 0` constraint
- [ ] Ensure smooth transition from cusp to NN at larger r

📄 **Starter Code:**

```python
class ElectronNuclearCusp(nn.Module):
    """Enforce electron-nucleus cusp condition.

    ψ → ψ * Π_{i,A} exp(-Z_A * r_iA + g(r_iA))
    where g(0) = 0 to preserve cusp.
    """

    @nn.compact
    def __call__(self, r_en, charges):
        """
        Args:
            r_en: Electron-nuclear distances (n_el, n_atoms)
            charges: Nuclear charges (n_atoms,)

        Returns:
            log of cusp correction factor
        """
        # Exact cusp: -Z * r
        cusp_term = -jnp.sum(charges * r_en, axis=-1)  # (n_el,)

        # Smooth correction that vanishes at r=0
        # Use r * tanh(neural_net(features)) so it's 0 at r=0
        correction = 0.0
        for i, Z in enumerate(charges):
            r = r_en[:, i]
            # Features that are smooth at r=0
            features = jnp.stack([r, r**2, jnp.exp(-r)], axis=-1)
            g = nn.Dense(16)(features)
            g = nn.tanh(g)
            g = nn.Dense(1)(g).squeeze(-1)
            # Multiply by r to ensure g(0) = 0
            correction += r * g

        return jnp.sum(cusp_term) + jnp.sum(correction)
```

### 2. Electron-Electron Cusp

- [ ] Implement e-e cusp correction based on spin
- [ ] Same spin: cusp coefficient = 1/2
- [ ] Opposite spin: cusp coefficient = 1/4

📄 **Starter Code:**

```python
class ElectronElectronCusp(nn.Module):
    """Enforce electron-electron cusp condition."""

    @nn.compact
    def __call__(self, r_ee, spin_mask):
        """
        Args:
            r_ee: Electron-electron distances (n_el, n_el)
            spin_mask: Same-spin mask (n_el, n_el), True if same spin

        Returns:
            log of cusp correction factor
        """
        # Cusp coefficients
        cusp_same = 0.5
        cusp_diff = 0.25

        cusp_coeff = jnp.where(spin_mask, cusp_same, cusp_diff)

        # Only upper triangle (i < j)
        triu_mask = jnp.triu(jnp.ones_like(r_ee), k=1)

        # Cusp term: a * r_{ij} / (1 + b * r_{ij})
        # Derivative at r=0 gives cusp coefficient
        b = self.param('b_ee', nn.initializers.ones, ())
        cusp_term = cusp_coeff * r_ee / (1 + jnp.abs(b) * r_ee)

        return jnp.sum(cusp_term * triu_mask)
```

### 3. Backflow Transformation (`src/nqmc/wavefunction/backflow.py`)

- [ ] Transform electron positions: `x_i = r_i + Σ_j η(r_ij)(r_i - r_j)`
- [ ] Use these transformed coordinates in orbitals
- [ ] Implement backflow function η as neural network

📄 **Starter Code:**

```python
class BackflowTransform(nn.Module):
    """Transform electron coordinates to capture correlation.

    x_i = r_i + Σ_{j≠i} η(r_ij) * (r_i - r_j)
    """
    hidden_dims: Tuple[int, ...] = (16, 16)

    @nn.compact
    def __call__(self, r, r_ee):
        """
        Args:
            r: Electron positions (n_el, 3)
            r_ee: Electron-electron distances (n_el, n_el)

        Returns:
            x: Transformed positions (n_el, 3)
        """
        n_el = r.shape[0]

        # Compute η(r_ij) for each pair
        def eta(r_ij):
            """Backflow function, should decay at large r."""
            x = jnp.array([r_ij, jnp.exp(-r_ij)])
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.tanh(x)
            return nn.Dense(1)(x).squeeze() * jnp.exp(-0.5 * r_ij)

        eta_matrix = jax.vmap(jax.vmap(eta))(r_ee)  # (n_el, n_el)

        # Compute displacement for each electron
        r_diff = r[:, None, :] - r[None, :, :]  # (n_el, n_el, 3)
        displacement = jnp.sum(eta_matrix[:, :, None] * r_diff, axis=1)  # (n_el, 3)

        # Exclude self-interaction (diagonal is 0 anyway due to r_diff)
        return r + displacement
```

### 4. Integrate into Wavefunction

- [ ] Modify `AntisymmetricWavefunction` to use cusp and backflow
- [ ] Order: backflow → orbitals → determinant → cusp correction

📄 **Updated Wavefunction:**

```python
class AntisymmetricWavefunction(nn.Module):
    n_up: int
    n_down: int
    use_cusp: bool = True
    use_backflow: bool = True

    def setup(self):
        self.orbitals_up = OrbitalSet(self.n_up)
        self.orbitals_down = OrbitalSet(self.n_down)
        self.jastrow = JastrowFactor()
        if self.use_cusp:
            self.en_cusp = ElectronNuclearCusp()
            self.ee_cusp = ElectronElectronCusp()
        if self.use_backflow:
            self.backflow = BackflowTransform()

    def __call__(self, r, molecule):
        r_ee = molecule.electron_electron_distances(r)
        r_en = molecule.electron_nuclear_distances(r)

        # Apply backflow
        if self.use_backflow:
            r_bf = self.backflow(r, r_ee)
        else:
            r_bf = r

        # Slater determinant with backflow coordinates
        sign, log_det = self.compute_slater(r_bf, molecule)

        # Jastrow
        log_j = self.jastrow(r_ee, r_en)

        # Cusp corrections
        log_cusp = 0.0
        if self.use_cusp:
            log_cusp += self.en_cusp(r_en, molecule.charges)
            spin_mask = self.get_spin_mask()
            log_cusp += self.ee_cusp(r_ee, spin_mask)

        return sign, log_det + log_j + log_cusp
```

### 5. Cusp Verification Tests (`tests/test_cusp.py`)

- [ ] Test numerical derivative at nucleus matches -Z
- [ ] Test e-e derivative at coalescence matches 1/2 or 1/4
- [ ] Test variance reduction compared to no-cusp baseline

## Visualizations for Phase 3

- [ ] `visualizations/wavefunction_plots/radial_cusp.png` - ψ(r) near nucleus
- [ ] `visualizations/wavefunction_plots/cusp_derivative.png` - dψ/dr verification
- [ ] `visualizations/sampling_diagnostics/variance_comparison.png` - Before/after cusp
- [ ] `visualizations/wavefunction_plots/backflow_coords.png` - Backflow displacement

## Definition of Done

- [ ] Local energy variance reduced by > 20% compared to Phase 2
- [ ] Cusp conditions verified numerically in tests
- [ ] H₂ energy reaches < -1.17 Ha (within 5 mHa of exact -1.174 Ha)
- [ ] Backflow coordinates visualized
- [ ] No numerical instabilities during training

## Technical Notes

**Cusp Stability:**
- The cusp correction should not dominate the wavefunction
- Use smooth transitions (tanh, sigmoid) to blend cusp with NN
- Monitor local energy histogram for outliers near coalescence

**Backflow Jacobian:**
- Backflow changes the Jacobian of the coordinate transformation
- For energy computation, we use physical coordinates r, not backflow x
- Backflow only affects orbital evaluation
