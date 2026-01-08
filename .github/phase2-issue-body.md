## Summary

Implement fermionic antisymmetry via Slater determinant-based architecture to properly handle electron exchange symmetry.

**Parent Issue:** #1
**Depends On:** #2 (Phase 1)

## Objectives

- Implement Slater determinant computation with stable gradients
- Create neural network orbital functions
- Build full antisymmetric wavefunction (Slater-Jastrow form)
- Verify antisymmetry property under electron exchange
- Improve H₂ energy from Phase 1

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/wavefunction/slater.py` | ~100 | Create | Slater determinant layer |
| `src/nqmc/wavefunction/orbitals.py` | ~120 | Create | Neural network orbitals |
| `src/nqmc/wavefunction/antisymmetric.py` | ~150 | Create | Full antisymmetric ψ |
| `src/nqmc/wavefunction/jastrow.py` | ~80 | Create | Jastrow correlation factor |
| `tests/test_antisymmetry.py` | ~60 | Create | Antisymmetry verification |

## Implementation Tasks

### 1. Slater Determinant (`src/nqmc/wavefunction/slater.py`)

- [ ] Implement `slater_determinant(orbitals_matrix)`
- [ ] Use log-det decomposition for stability: `sign, log_abs_det = slogdet(A)`
- [ ] Implement gradient via adjugate matrix method
- [ ] Handle spin blocks (spin-up and spin-down determinants)

📄 **Starter Code:**

```python
def log_slater_determinant(orbital_matrix):
    """Compute log|det(A)| and sign.

    Args:
        orbital_matrix: Shape (n_electrons, n_orbitals)

    Returns:
        sign: +1 or -1
        log_abs_det: log|det(A)|
    """
    sign, log_abs_det = jnp.linalg.slogdet(orbital_matrix)
    return sign, log_abs_det


def slater_with_spin(orbitals_up, orbitals_down, r_up, r_down):
    """Compute ψ = det(φ_up) * det(φ_down).

    For H₂: 1 up, 1 down -> det is just the orbital value.
    For H₄: 2 up, 2 down -> 2x2 determinants.
    """
    # Evaluate orbitals at electron positions
    phi_up = jax.vmap(orbitals_up)(r_up)    # (n_up, n_orbitals)
    phi_down = jax.vmap(orbitals_down)(r_down)  # (n_down, n_orbitals)

    sign_up, log_det_up = log_slater_determinant(phi_up)
    sign_down, log_det_down = log_slater_determinant(phi_down)

    return sign_up * sign_down, log_det_up + log_det_down
```

### 2. Neural Orbitals (`src/nqmc/wavefunction/orbitals.py`)

- [ ] Create `NeuralOrbital(nn.Module)` base class
- [ ] Input: single electron position (3,) + nuclear positions
- [ ] Output: orbital value at that position
- [ ] Include electron-nuclear features (distances, angles)

📄 **Starter Code:**

```python
class NeuralOrbital(nn.Module):
    """Neural network that represents a single orbital."""
    hidden_dims: Tuple[int, ...] = (32, 32)

    @nn.compact
    def __call__(self, r_electron, r_nuclei, charges):
        # Compute features
        r_en = jnp.linalg.norm(r_electron - r_nuclei, axis=-1)  # (n_atoms,)

        # Concatenate features
        features = jnp.concatenate([
            r_electron,              # (3,)
            r_en,                    # (n_atoms,)
            jnp.exp(-r_en),          # (n_atoms,) - exponential envelope
        ])

        x = features
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)

        return nn.Dense(1)(x).squeeze()


class OrbitalSet(nn.Module):
    """Set of n_orbitals neural orbitals."""
    n_orbitals: int
    hidden_dims: Tuple[int, ...] = (32, 32)

    @nn.compact
    def __call__(self, r_electron, r_nuclei, charges):
        orbitals = []
        for i in range(self.n_orbitals):
            phi = NeuralOrbital(self.hidden_dims, name=f'orbital_{i}')
            orbitals.append(phi(r_electron, r_nuclei, charges))
        return jnp.array(orbitals)
```

### 3. Jastrow Factor (`src/nqmc/wavefunction/jastrow.py`)

- [ ] Implement electron-electron Jastrow: `J_ee = Σ_{i<j} u(r_ij)`
- [ ] Implement electron-nuclear Jastrow: `J_en = Σ_{iA} χ(r_iA)`
- [ ] Use neural network or Padé form for u and χ

📄 **Starter Code:**

```python
class JastrowFactor(nn.Module):
    """Jastrow correlation factor: ψ → ψ * exp(J)."""

    @nn.compact
    def __call__(self, r_ee, r_en):
        """
        Args:
            r_ee: Electron-electron distances (n_el, n_el)
            r_en: Electron-nuclear distances (n_el, n_atoms)
        """
        # Electron-electron correlation
        # Padé form: u(r) = a*r / (1 + b*r)
        a_ee = self.param('a_ee', nn.initializers.ones, ())
        b_ee = self.param('b_ee', nn.initializers.ones, ())

        # Only upper triangle (i < j)
        triu_mask = jnp.triu(jnp.ones_like(r_ee), k=1)
        u_ee = a_ee * r_ee / (1 + b_ee * r_ee)
        j_ee = jnp.sum(u_ee * triu_mask)

        # Electron-nuclear correlation
        a_en = self.param('a_en', nn.initializers.ones, ())
        b_en = self.param('b_en', nn.initializers.ones, ())

        u_en = a_en * r_en / (1 + b_en * r_en)
        j_en = jnp.sum(u_en)

        return j_ee + j_en
```

### 4. Full Antisymmetric Wavefunction (`src/nqmc/wavefunction/antisymmetric.py`)

- [ ] Combine Slater determinant with Jastrow factor
- [ ] ψ = det(φ_up) * det(φ_down) * exp(J)
- [ ] Implement `log_psi` for numerical stability

📄 **Starter Code:**

```python
class AntisymmetricWavefunction(nn.Module):
    """Full neural QMC wavefunction: ψ = det * exp(J)."""
    n_up: int
    n_down: int
    hidden_dims: Tuple[int, ...] = (32, 32)

    def setup(self):
        self.orbitals_up = OrbitalSet(self.n_up, self.hidden_dims)
        self.orbitals_down = OrbitalSet(self.n_down, self.hidden_dims)
        self.jastrow = JastrowFactor()

    @nn.compact
    def __call__(self, r, molecule):
        """
        Args:
            r: Electron positions (n_electrons, 3)
            molecule: Molecule object

        Returns:
            log|ψ|, sign
        """
        r_up = r[:self.n_up]
        r_down = r[self.n_up:]

        # Slater determinants
        sign, log_det = slater_with_spin(
            lambda x: self.orbitals_up(x, molecule.positions, molecule.charges),
            lambda x: self.orbitals_down(x, molecule.positions, molecule.charges),
            r_up, r_down
        )

        # Jastrow factor
        r_ee = molecule.electron_electron_distances(r)
        r_en = molecule.electron_nuclear_distances(r)
        log_jastrow = self.jastrow(r_ee, r_en)

        return sign, log_det + log_jastrow
```

### 5. Antisymmetry Test (`tests/test_antisymmetry.py`)

- [ ] Test ψ(r_swap) = -ψ(r) for electron exchange
- [ ] Test with random configurations
- [ ] Test for both same-spin and opposite-spin exchanges

📄 **Test Code:**

```python
def test_antisymmetry():
    """Verify ψ changes sign under same-spin electron exchange."""
    wf = AntisymmetricWavefunction(n_up=2, n_down=2)
    molecule = hydrogen_chain(4)

    key = jax.random.PRNGKey(0)
    r = jax.random.normal(key, (4, 3))
    params = wf.init(key, r, molecule)

    # Swap first two electrons (both spin-up)
    r_swapped = r.at[[0, 1]].set(r[[1, 0]])

    sign1, log_psi1 = wf.apply(params, r, molecule)
    sign2, log_psi2 = wf.apply(params, r_swapped, molecule)

    # log|ψ| should be same, sign should flip
    assert jnp.allclose(log_psi1, log_psi2, rtol=1e-5)
    assert sign1 == -sign2
```

## Visualizations for Phase 2

- [ ] `visualizations/wavefunction_plots/orbitals_2d.png` - 2D orbital contours
- [ ] `visualizations/wavefunction_plots/electron_density.png` - |ψ|² density
- [ ] `visualizations/sampling_diagnostics/antisymmetry_test.png` - ψ vs -ψ plot

## Definition of Done

- [ ] Antisymmetry property verified in tests (sign flips on exchange)
- [ ] H₂ energy improves by > 20 mHa over Phase 1 baseline
- [ ] No NaN/Inf during 10k training steps
- [ ] Determinant gradients match finite differences (< 1e-4 error)
- [ ] Orbital visualization plots generated

## Technical Notes

**Determinant Stability:**
- Use `jnp.linalg.slogdet` instead of `jnp.linalg.det`
- For gradients, use adjugate: `∂det(A)/∂A = det(A) * A^{-T}`
- Watch for near-singular matrices (regularize or restart)

**Spin Handling:**
- First `n_up` electrons are spin-up, rest are spin-down
- Antisymmetry only applies to same-spin exchanges
- For H₂: 1 up + 1 down → determinant is just orbital value (trivial)
