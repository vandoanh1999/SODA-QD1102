"""
SODA-Q 3D Flow Visualization
=============================

This module simulates a compact SODA-Q evolution and produces a 3D "Flow" visualization.
The code runs a short SODA-Q style simulation and creates a 3D matplotlib figure showing
nucleus trajectories in phase-space (cos(phi), sin(phi)) vs Quantum Order (R_q).

Visualization Axes:
    - X = cos(φ)
    - Y = sin(φ)
    - Z = R_q (global quantum order per time step)
    - Point size ~ entropy_j (or fallback 1/fitness)
    - Color ~ entropy_j (cold -> hot using plasma colormap)
    - Connections drawn when J_jk > J_th at each time step

Output:
    - Saves visualization to output/soda_q_evolution.png
    - Displays figure inline

Author: vandoanh1999
Date: 2025-12-08
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import time
from pathlib import Path


# -------------------- Helper: minimal Nucleus classes --------------------
class MiniNucleus:
    """Minimal nucleus representation for simulation."""
    
    def __init__(self, id):
        """
        Initialize a nucleus with random initial conditions.
        
        Args:
            id: Unique identifier for the nucleus
        """
        self.id = id
        self.phase = random.uniform(0, 2*math.pi)
        self.psi = np.exp(1j * self.phase)  # complex quantum state
        self.fitness = random.uniform(1e3, 1e4)
        self.entropy = random.uniform(1.0, 4.0)
        self.spike_count = random.randint(1, 5)
        self.genome = ["x0", "+", "x1"]  # minimal genome placeholder

    def update_spikes(self):
        """Simulate spike activity using Poisson distribution."""
        lam = max(0.5, self.spike_count * 0.8)
        self.spike_count = np.random.poisson(lam=lam) + 1

    def __repr__(self):
        return f"Nucleus({self.id}, phase={self.phase:.3f}, entropy={self.entropy:.3f})"


# -------------------- Quantum Kuramoto Map (compact) --------------------
def quantum_kuramoto_update(nuclei, quantum_k=1.0):
    """
    Quantum Kuramoto Map (QKM) update rule.
    
    Quantum state evolution:
        ψ_j(t+1) = Z⁻¹ [ ψ_j(t) 
                         + Σ_k J_jk * exp(i * (θ_k - θ_j)) 
                         + Δω_j * ψ_j(t) ]
    
    Where:
        - J_jk = exp( -|fitness_j - fitness_k| / T ) * exp( -|entropy_j - entropy_k| )
        - Δω_j = firing_rate_j / normalization
        - Z normalizes to unit probability (Σ_j |ψ_j|² = 1)
    
    Args:
        nuclei: List of MiniNucleus objects
        quantum_k: Quantum coupling strength
        
    Returns:
        Tuple of (R_q, J_matrix)
            - R_q: Quantum order parameter (float)
            - J_matrix: Coupling matrix (N x N array)
    """
    N = len(nuclei)
    if N == 0:
        return 0.0, np.zeros((0, 0))

    # Collect state arrays
    phases = np.array([n.phase for n in nuclei])
    fitness = np.array([n.fitness for n in nuclei])
    entropy = np.array([n.entropy for n in nuclei])

    # Ensure ψ exists for all nuclei
    for n in nuclei:
        if not hasattr(n, "psi"):
            n.psi = np.exp(1j * random.uniform(0, 2*np.pi))

    psi_old = np.array([n.psi for n in nuclei], dtype=complex)

    # Build coupling matrix J (symmetric)
    J = np.zeros((N, N), dtype=float)
    T = 1.0 + np.std(fitness)  # Temperature-like scaling
    
    for j in range(N):
        for k in range(N):
            df = abs(fitness[j] - fitness[k])
            de = abs(entropy[j] - entropy[k])
            J[j, k] = math.exp(-df / (T + 1e-9)) * math.exp(-de)

    # Firing rates from spike counts → delta omega
    firing_rate = np.array([max(1e-6, n.spike_count) for n in nuclei], dtype=float)
    delta_omega = firing_rate / (1000.0 + firing_rate.max())

    # Update quantum state ψ
    psi_new = np.zeros(N, dtype=complex)
    for j in range(N):
        coupling_sum = 0.0 + 0.0j
        for k in range(N):
            coupling_sum += J[j, k] * np.exp(1j * (phases[k] - phases[j]))
        
        psi_new[j] = (psi_old[j] + 
                     quantum_k * coupling_sum + 
                     delta_omega[j] * psi_old[j])

    # Normalize to unit probability mass
    norm = np.sqrt(np.sum(np.abs(psi_new)**2)) + 1e-12
    psi_new = psi_new / norm

    # Write back states and update phases
    for j in range(N):
        nuclei[j].psi = psi_new[j]
        nuclei[j].phase = np.angle(psi_new[j])

    # Quantum order parameter (coherence measure)
    R_q = abs(np.sum(psi_new))

    return R_q, J


# -------------------- Main simulation --------------------
def run_simulation(n_nuclei=18, n_steps=220, quantum_k=1.0, seed=42):
    """
    Run SODA-Q evolution simulation.
    
    Args:
        n_nuclei: Number of nuclei to simulate
        n_steps: Number of time steps
        quantum_k: Quantum coupling strength
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing simulation results
    """
    np.random.seed(seed)
    random.seed(seed)

    # Initialize nuclei
    nuclei = [MiniNucleus(f"n{idx}") for idx in range(n_nuclei)]

    # Records
    history_phases = np.zeros((n_steps, n_nuclei))
    history_entropy = np.zeros((n_steps, n_nuclei))
    history_fitness = np.zeros((n_steps, n_nuclei))
    history_Rq = np.zeros(n_steps)
    history_J_matrices = []

    start = time.time()
    
    for t in range(n_steps):
        # Small internal updates to fitness/entropy (emulate learning)
        for n in nuclei:
            n.fitness *= (1.0 - 0.0005 * random.random())
            n.entropy *= (1.0 + 0.0008 * (random.random() - 0.5))
            n.update_spikes()

        # Apply quantum Kuramoto update
        Rq, J = quantum_kuramoto_update(nuclei, quantum_k=quantum_k)

        # Record results
        history_Rq[t] = Rq
        history_J_matrices.append(J.copy())
        for i, n in enumerate(nuclei):
            history_phases[t, i] = n.phase
            history_entropy[t, i] = n.entropy
            history_fitness[t, i] = n.fitness

    elapsed = time.time() - start

    return {
        'nuclei': nuclei,
        'phases': history_phases,
        'entropy': history_entropy,
        'fitness': history_fitness,
        'Rq': history_Rq,
        'J_matrices': history_J_matrices,
        'elapsed_time': elapsed,
        'n_nuclei': n_nuclei,
        'n_steps': n_steps
    }


# -------------------- 3D Visualization --------------------
def visualize_flow(sim_data, output_path="soda_q_evolution.png", show=True):
    """
    Create and save 3D flow visualization.
    
    Args:
        sim_data: Dictionary returned from run_simulation()
        output_path: Path to save PNG figure
        show: Whether to display the figure
    """
    history_phases = sim_data['phases']
    history_entropy = sim_data['entropy']
    history_Rq = sim_data['Rq']
    history_J_matrices = sim_data['J_matrices']
    T_steps, N = history_phases.shape

    # Build 3D trajectory data
    X_trajs = np.cos(history_phases)
    Y_trajs = np.sin(history_phases)
    Z_trajs = np.tile(history_Rq.reshape(-1, 1), (1, N))

    # Size mapping: size ~ entropy_j (scaled)
    ent_min, ent_max = history_entropy.min(), history_entropy.max()
    size_scale = 80.0
    sizes = size_scale * ((history_entropy - ent_min) / (ent_max - ent_min + 1e-9) + 0.2)

    # Color mapping: entropy → cold to hot
    cmap = cm.get_cmap("plasma")
    norm = Normalize(vmin=ent_min, vmax=ent_max)

    # Connection threshold (high percentile to avoid clutter)
    def connection_threshold(J, perc=95):
        flat = J.flatten()
        return np.percentile(flat, perc)

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-60)

    # Plot nucleus trajectories
    for j in range(N):
        x = X_trajs[:, j]
        y = Y_trajs[:, j]
        z = Z_trajs[:, j]

        mean_ent = history_entropy[:, j].mean()
        color = cmap(norm(mean_ent))
        
        # Main trajectory line
        ax.plot(x, y, z, linewidth=1.5, alpha=0.9, color=color)

        # Scatter points as evolution markers
        step_mark = max(1, T_steps // 30)
        ax.scatter(x[::step_mark], y[::step_mark], z[::step_mark],
                  s=sizes[::step_mark, j],
                  color=[color],
                  alpha=0.95,
                  edgecolors='k',
                  linewidths=0.2)

    # Draw connection lines at several time slices
    time_slices = list(range(0, T_steps, max(1, T_steps // 6)))
    for t_idx in time_slices:
        J = history_J_matrices[t_idx]
        thr = connection_threshold(J, perc=97)
        Rq_t = history_Rq[t_idx]
        
        for a in range(N):
            for b in range(a + 1, N):
                if J[a, b] > thr:
                    xa, ya = X_trajs[t_idx, a], Y_trajs[t_idx, a]
                    xb, yb = X_trajs[t_idx, b], Y_trajs[t_idx, b]
                    za, zb = Rq_t, Rq_t
                    ax.plot([xa, xb], [ya, yb], [za, zb], 
                           linewidth=0.8, alpha=0.45, color='gray')

    # Labels and aesthetics
    ax.set_xlabel("cos(φ)", fontsize=11)
    ax.set_ylabel("sin(φ)", fontsize=11)
    ax.set_zlabel("Quantum Order (R_q)", fontsize=11)
    ax.set_title("SODA-Q Evolution Flow: Trajectories in Phase-Space vs Quantum Order", 
                fontsize=13, fontweight='bold')

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(history_entropy)
    cbar = fig.colorbar(mappable, shrink=0.6, pad=0.1)
    cbar.set_label("Entropy (per nucleus)", fontsize=10)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    
    if show:
        plt.show()


# -------------------- Main execution --------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SODA-Q Evolution Flow Visualization")
    print("=" * 60)
    
    # Create output directory if needed
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "soda_q_evolution.png"

    # Run simulation
    print(f"\nRunning simulation with 18 nuclei for 220 steps...")
    sim_data = run_simulation(n_nuclei=18, n_steps=220, quantum_k=1.0, seed=42)

    print(f"✓ Simulation completed in {sim_data['elapsed_time']:.2f}s")
    print(f"  - Number of nuclei: {sim_data['n_nuclei']}")
    print(f"  - Number of steps: {sim_data['n_steps']}")
    print(f"  - Final R_q: {sim_data['Rq'][-1]:.4f}")
    print(f"  - Mean R_q: {sim_data['Rq'].mean():.4f}")

    # Create visualization
    print(f"\nGenerating 3D flow visualization...")
    visualize_flow(sim_data, output_path=str(output_path), show=True)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
