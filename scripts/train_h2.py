#!/usr/bin/env python3
"""Train neural network wavefunction on H₂ molecule.

This script demonstrates the basic VMC workflow:
1. Create H₂ molecular system
2. Initialize SimpleWavefunction
3. Run VMC optimization
4. Plot energy convergence
5. Compare to exact result

Usage:
    python scripts/train_h2.py [--n_steps N] [--lr LR] [--seed S]

Example:
    python scripts/train_h2.py --n_steps 500 --lr 0.001
"""
import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nqmc.optimizer import VMCOptimizer
from nqmc.systems.molecule import hydrogen_molecule
from nqmc.wavefunction import SimpleWavefunction

# Exact H₂ ground state energy (approximately)
H2_EXACT_ENERGY = -1.174  # Hartree


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VMC on H₂ molecule")
    parser.add_argument(
        "--n_steps", type=int, default=200,
        help="Number of optimization steps (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--hidden_dims", type=str, default="64,64",
        help="Hidden layer dimensions (default: '64,64')"
    )
    parser.add_argument(
        "--n_samples", type=int, default=300,
        help="MCMC samples per step (default: 300)"
    )
    parser.add_argument(
        "--n_chains", type=int, default=32,
        help="Number of MCMC chains (default: 32)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--bond_length", type=float, default=1.4,
        help="H-H bond length in Bohr (default: 1.4)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations/energy_curves",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--log_every", type=int, default=10,
        help="Logging frequency (default: 10)"
    )
    return parser.parse_args()


def plot_convergence(history, output_path, exact_energy=H2_EXACT_ENERGY):
    """Plot energy convergence curve.

    Args:
        history: List of metrics dicts with 'energy' and 'energy_std'
        output_path: Path to save the plot
        exact_energy: Exact ground state energy for reference
    """
    energies = [m["energy"] for m in history]
    stds = [m["energy_std"] for m in history]
    steps = range(1, len(energies) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Energy convergence
    ax1.plot(steps, energies, "b-", linewidth=1, label="VMC Energy")
    ax1.fill_between(
        steps,
        np.array(energies) - np.array(stds),
        np.array(energies) + np.array(stds),
        alpha=0.3,
        color="blue",
    )
    ax1.axhline(exact_energy, color="r", linestyle="--", label=f"Exact ({exact_energy:.3f} Ha)")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_title("H₂ VMC Energy Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set y-axis limits to zoom in on relevant range
    final_energy = np.mean(energies[-50:]) if len(energies) >= 50 else np.mean(energies)
    ax1.set_ylim(min(exact_energy - 0.5, final_energy - 0.2), max(energies[0] + 0.1, 0))

    # Error from exact
    errors = [e - exact_energy for e in energies]
    ax2.semilogy(steps, np.abs(errors), "g-", linewidth=1, label="|E - E_exact|")
    ax2.axhline(0.07, color="orange", linestyle=":", label="Target (70 mHa)")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("|E - E_exact| (Ha)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved convergence plot to {output_path}")


def main():
    """Run H₂ VMC training."""
    args = parse_args()

    print("=" * 60)
    print("Neural QMC Wavefunction - H₂ Training")
    print("=" * 60)

    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    print(f"\nConfiguration:")
    print(f"  Bond length: {args.bond_length} Bohr")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Learning rate: {args.lr}")
    print(f"  N steps: {args.n_steps}")
    print(f"  N samples: {args.n_samples}")
    print(f"  N chains: {args.n_chains}")
    print(f"  Seed: {args.seed}")

    # Create molecular system
    print("\nCreating H₂ molecule...")
    molecule = hydrogen_molecule(bond_length=args.bond_length)
    print(f"  N electrons: {molecule.n_electrons}")
    print(f"  Spin: {molecule.spin}")
    print(f"  Nuclear repulsion: {molecule.nuclear_repulsion_energy():.6f} Ha")

    # Initialize wavefunction
    print("\nInitializing wavefunction...")
    key = random.PRNGKey(args.seed)
    key, init_key = random.split(key)

    wavefunction = SimpleWavefunction(
        hidden_dims=hidden_dims,
        envelope_decay=1.0,
        nuclear_positions=molecule.positions,
    )

    # Initialize parameters
    r_sample = jnp.zeros((molecule.n_electrons, 3))
    params = wavefunction.init(init_key, r_sample)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Network parameters: {n_params}")

    # Create optimizer
    print("\nSetting up VMC optimizer...")
    optimizer = VMCOptimizer(
        learning_rate=args.lr,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_burn=300,
        step_size=0.2,
        clip_grad=1.0,
    )

    # Train
    print("\nStarting training...")
    print("-" * 60)

    key, train_key = random.split(key)
    final_params, history = optimizer.train(
        train_key,
        wavefunction,
        params,
        molecule,
        n_steps=args.n_steps,
        log_every=args.log_every,
    )

    print("-" * 60)

    # Results
    final_energies = [m["energy"] for m in history[-50:]]
    final_energy = np.mean(final_energies)
    final_std = np.std(final_energies) / np.sqrt(len(final_energies))

    print("\nResults:")
    print(f"  Final energy: {final_energy:.6f} ± {final_std:.6f} Ha")
    print(f"  Exact energy: {H2_EXACT_ENERGY:.6f} Ha")
    print(f"  Error: {(final_energy - H2_EXACT_ENERGY)*1000:.1f} mHa")

    success = final_energy < -1.10
    print(f"\n  Target (< -1.10 Ha): {'PASSED ✓' if success else 'FAILED ✗'}")

    # Plot convergence
    output_path = os.path.join(args.output_dir, "h2_convergence.png")
    plot_convergence(history, output_path)

    print("\nDone!")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
