## Summary

Implement comprehensive benchmarking framework with PySCF integration for comparing Neural QMC against classical quantum chemistry methods (HF, MP2, CCSD, CCSD(T)).

**Parent Issue:** #1
**Depends On:** #5 (Phase 4)

## Objectives

- Create PySCF wrapper for automated baseline calculations
- Implement statistical error estimation for VMC energies
- Generate publication-quality comparison figures
- Create analysis notebook with full benchmark suite
- Document basis set and method comparisons

## Files to Create/Modify

| File | Lines | Action | Description |
|------|-------|--------|-------------|
| `src/nqmc/baselines/pyscf_runner.py` | ~150 | Create | PySCF wrapper |
| `src/nqmc/analysis/comparison.py` | ~100 | Create | Comparison utilities |
| `src/nqmc/analysis/statistics.py` | ~80 | Create | Error estimation |
| `scripts/run_benchmark.py` | ~120 | Create | Full benchmark suite |
| `notebooks/benchmark_analysis.ipynb` | - | Create | Publication notebook |

## Implementation Tasks

### 1. PySCF Runner (`src/nqmc/baselines/pyscf_runner.py`)

- [ ] Create wrapper for HF, MP2, CCSD, CCSD(T) calculations
- [ ] Handle geometry specification consistent with Neural QMC
- [ ] Support multiple basis sets (STO-3G, cc-pVDZ, cc-pVTZ)
- [ ] Return energies with basis set extrapolation

📄 **Starter Code:**

```python
from pyscf import gto, scf, mp, cc

class PySCFRunner:
    """Wrapper for PySCF calculations."""

    def __init__(self, molecule: Molecule, basis: str = "cc-pVDZ"):
        self.molecule = molecule
        self.basis = basis
        self.mol = self._build_mol()

    def _build_mol(self) -> gto.Mole:
        """Convert our Molecule to PySCF format."""
        atom_str = ""
        for el, pos in self.molecule.atoms:
            # PySCF expects Angstrom, we have Bohr
            pos_ang = pos * 0.529177  # Bohr to Angstrom
            atom_str += f"{el} {pos_ang[0]:.6f} {pos_ang[1]:.6f} {pos_ang[2]:.6f}; "

        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = self.basis
        mol.spin = self.molecule.spin[0] - self.molecule.spin[1]
        mol.build()
        return mol

    def run_hf(self) -> float:
        """Run Hartree-Fock calculation."""
        mf = scf.RHF(self.mol)
        return mf.kernel()

    def run_mp2(self) -> Tuple[float, float]:
        """Run MP2 calculation."""
        mf = scf.RHF(self.mol).run()
        mp2 = mp.MP2(mf).run()
        return mf.e_tot, mp2.e_tot

    def run_ccsd(self) -> Tuple[float, float]:
        """Run CCSD calculation."""
        mf = scf.RHF(self.mol).run()
        mycc = cc.CCSD(mf).run()
        return mf.e_tot, mycc.e_tot

    def run_ccsd_t(self) -> Tuple[float, float, float]:
        """Run CCSD(T) calculation."""
        mf = scf.RHF(self.mol).run()
        mycc = cc.CCSD(mf).run()
        e_t = mycc.ccsd_t()
        return mf.e_tot, mycc.e_tot, mycc.e_tot + e_t

    def run_all(self) -> Dict[str, float]:
        """Run all available methods."""
        mf = scf.RHF(self.mol).run()
        mp2_calc = mp.MP2(mf).run()
        cc_calc = cc.CCSD(mf).run()
        e_t = cc_calc.ccsd_t()

        return {
            'HF': mf.e_tot,
            'MP2': mp2_calc.e_tot,
            'CCSD': cc_calc.e_tot,
            'CCSD(T)': cc_calc.e_tot + e_t
        }
```

### 2. Statistical Analysis (`src/nqmc/analysis/statistics.py`)

- [ ] Implement blocking analysis for autocorrelation
- [ ] Compute statistical error in VMC energy
- [ ] Implement bootstrap resampling for error bars
- [ ] Calculate effective sample size

📄 **Starter Code:**

```python
def blocking_analysis(data: np.ndarray, n_blocks: int = 20) -> Tuple[float, float]:
    """Compute mean and error using blocking method.

    This accounts for autocorrelation in MCMC samples.
    """
    n = len(data)
    block_size = n // n_blocks
    block_means = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(np.mean(data[start:end]))

    block_means = np.array(block_means)
    mean = np.mean(block_means)
    std_error = np.std(block_means) / np.sqrt(n_blocks)

    return mean, std_error


def compute_autocorrelation(data: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.mean((data[:n-lag] - mean) * (data[lag:] - mean)) / var

    return acf


def effective_sample_size(data: np.ndarray) -> float:
    """Estimate effective sample size accounting for autocorrelation."""
    acf = compute_autocorrelation(data)
    # Integrated autocorrelation time
    tau = 1 + 2 * np.sum(acf[1:])
    return len(data) / tau
```

### 3. Comparison Utilities (`src/nqmc/analysis/comparison.py`)

- [ ] Create comparison tables
- [ ] Calculate correlation energy percentages
- [ ] Generate method comparison plots
- [ ] Export results to CSV/JSON

📄 **Starter Code:**

```python
@dataclass
class MethodResult:
    name: str
    energy: float
    error: Optional[float] = None
    basis: Optional[str] = None


def compute_correlation_energy(
    results: Dict[str, float],
    reference: str = 'HF',
    exact: Optional[float] = None
) -> Dict[str, float]:
    """Compute correlation energy recovered by each method."""
    e_ref = results[reference]
    e_exact = exact or results.get('FCI') or results.get('CCSD(T)')

    corr_pct = {}
    for method, energy in results.items():
        if method == reference:
            corr_pct[method] = 0.0
        else:
            e_corr = e_ref - energy
            e_corr_exact = e_ref - e_exact
            corr_pct[method] = 100 * e_corr / e_corr_exact

    return corr_pct


def create_comparison_table(
    results: Dict[str, MethodResult],
    molecule_name: str
) -> pd.DataFrame:
    """Create publication-ready comparison table."""
    rows = []
    for name, result in results.items():
        row = {
            'Method': name,
            'Energy (Ha)': f"{result.energy:.6f}",
            'Error (mHa)': f"±{result.error*1000:.2f}" if result.error else "-",
            'Basis': result.basis or "N/A"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs['molecule'] = molecule_name
    return df
```

### 4. Benchmark Script (`scripts/run_benchmark.py`)

- [ ] Run Neural QMC at multiple geometries
- [ ] Run PySCF at corresponding geometries
- [ ] Collect all results with error bars
- [ ] Generate comparison figures
- [ ] Save results to files

📄 **Benchmark Script:**

```python
def run_full_benchmark(
    molecules: List[Tuple[str, Molecule]],
    qmc_checkpoints: Dict[str, str],
    output_dir: str = "results/benchmark"
):
    """Run full benchmark suite."""
    all_results = []

    for name, molecule in molecules:
        print(f"\nBenchmarking {name}...")

        # PySCF calculations
        pyscf_runner = PySCFRunner(molecule, basis="cc-pVTZ")
        classical_results = pyscf_runner.run_all()

        # Neural QMC results (load from checkpoint)
        qmc_energy, qmc_error = load_qmc_results(qmc_checkpoints[name])

        results = {
            'molecule': name,
            **classical_results,
            'Neural QMC': qmc_energy,
            'Neural QMC Error': qmc_error
        }
        all_results.append(results)

        # Generate comparison plot
        plot_method_comparison(results, name, f"{output_dir}/{name}_comparison.png")

    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)

    return df
```

### 5. Analysis Notebook (`notebooks/benchmark_analysis.ipynb`)

- [ ] Load all benchmark results
- [ ] Create summary tables
- [ ] Generate publication figures
- [ ] Write analysis text

## Visualizations for Phase 5

- [ ] `visualizations/comparison_charts/method_comparison_h2.png`
- [ ] `visualizations/comparison_charts/method_comparison_h4.png`
- [ ] `visualizations/comparison_charts/correlation_energy.png`
- [ ] `visualizations/comparison_charts/error_comparison.png`
- [ ] `visualizations/comparison_charts/basis_convergence.png`
- [ ] `visualizations/energy_curves/pes_all_methods.png`

## Definition of Done

- [ ] PySCF benchmarks complete for H₂ and H₄
- [ ] All energies have statistical error estimates
- [ ] Comparison table with 5+ methods generated
- [ ] Publication-ready figures in `visualizations/`
- [ ] README updated with benchmark results
- [ ] Analysis notebook renders correctly

## Expected Results Table

| System | HF | MP2 | CCSD | CCSD(T) | Neural QMC | Exact |
|--------|-----|-----|------|---------|------------|-------|
| H₂ (R=1.4) | -1.110 | -1.144 | -1.166 | -1.172 | -1.17±0.01 | -1.174 |
| H₄ (R=1.8) | -1.95 | -2.00 | -2.05 | -2.07 | -2.0±0.02 | -2.08 |

## Technical Notes

**Basis Set Effects:**
- PySCF uses finite basis sets; Neural QMC is basis-free
- Compare against cc-pVTZ or extrapolated CBS limit
- Note: Neural QMC should approach FCI, not CCSD

**Statistical Significance:**
- Report 1σ error bars (68% confidence)
- Use at least 10 blocks for blocking analysis
- Target effective sample size > 1000
