import torch
import math
import random
import time
from sodaq_ext import compute_and_update     # CUDA fused kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================================================================
# 1. Nucleus types (Chaos / Bayes) — same logic, except GPU-backed psi/phase evolution
# =======================================================================================

class ChaosNucleus:
    def __init__(self, id, dim=20):
        self.id = id
        self.dim = dim
        self.w = torch.randn(dim, device=device).float()
        self.genome = ["x0","+","x1"]
        self.fitness = 1e9
        self.entropy = 1.0
        self.phase = random.uniform(0, 2*math.pi)
        self.spike = random.randint(1,5)

    def update_local(self, X, y_true):
        y_hat = (torch.tanh(torch.dot(self.w, X/1000.0))*1000).item()
        err = abs(y_true - y_hat)
        self.fitness = err
        self.w += 0.002 * (y_true - y_hat) * X/1000.0
        self.entropy = float(torch.mean(torch.abs(self.w)).item())
        self.spike = max(1, self.spike + random.randint(-1,1))
        return err


class BayesNucleus:
    def __init__(self, id, dim=20):
        self.id = id
        self.dim = dim
        self.mu = torch.zeros(dim, device=device)
        self.sigma = torch.ones(dim, device=device)
        self.genome = ["x0","+","x1"]
        self.fitness = 1e9
        self.entropy = 1.0
        self.phase = random.uniform(0, 2*math.pi)
        self.spike = random.randint(1,5)

    def update_local(self, X, y_true):
        y_hat = (torch.dot(self.mu, X/1000.0)*1000).item()
        err = abs(y_true - y_hat)
        self.fitness = err
        self.mu += 0.01 * (y_true - y_hat) * X/1000.0
        self.sigma = 0.99*self.sigma + 0.01*abs(y_true-y_hat)/1000.0
        self.entropy = float(torch.sum(torch.log(self.sigma+1e-9)).item())
        self.spike = max(1, self.spike + random.randint(-1,1))
        return err

# =======================================================================================
# 2. GPU Quantum Engine Integration
# =======================================================================================

class SODAEngineGPU:
    def __init__(self, dim=20, n_init=12):
        self.dim = dim
        self.nuclei = []
        for i in range(n_init):
            if random.random() < 0.5:
                self.nuclei.append(ChaosNucleus(f"c_{i}", dim))
            else:
                self.nuclei.append(BayesNucleus(f"b_{i}", dim))

        # initialize ψ for quantum dynamics
        N = len(self.nuclei)
        self.psi = torch.exp(1j * (2 * math.pi * torch.rand(N, device=device))).to(torch.cfloat)
        self.phases = torch.angle(self.psi).real

        self.fitness = torch.tensor([n.fitness for n in self.nuclei], device=device).float()
        self.entropy = torch.tensor([n.entropy for n in self.nuclei], device=device).float()
        self.spikes = torch.tensor([n.spike for n in self.nuclei], device=device).float()

        self.errors = []
        self.entropies = []
        self.sync = []
        self.pop_sizes = []

    # =========================================================
    # GPU Quantum Kuramoto + Nucleus update
    # =========================================================
    def evolve_step(self, X, y_true):
        # Local nucleus learning (CPU+GPU mix)
        preds = []
        for j,n in enumerate(self.nuclei):
            err = n.update_local(X, y_true)
            self.fitness[j] = n.fitness
            self.entropy[j] = n.entropy
            self.spikes[j] = n.spike
            preds.append(n.fitness)

        error = min(preds)

        # ---------- GPU quantum update ----------
        psi_r, psi_i, _, _, Rq = compute_and_update(
            self.fitness, self.entropy, self.phases, self.psi.real,
            self.psi.imag, self.spikes, 1.0
        )

        self.psi = torch.complex(psi_r, psi_i)
        self.phases = torch.angle(self.psi).real

        # update nuclei phases
        for j, n in enumerate(self.nuclei):
            n.phase = math.atan2(float(psi_i[j]), float(psi_r[j]))

        # record logs
        self.errors.append(error)
        self.entropies.append(float(torch.mean(self.entropy).item()))
        self.sync.append(Rq)
        self.pop_sizes.append(len(self.nuclei))

        return error, Rq

    # =========================================================
    def run(self, generations=300):
        for gen in range(generations):
            X = torch.rand(self.dim, device=device)*1000
            y_true = torch.sum(X**1.5).item()

            start = time.time()
            error, Rq = self.evolve_step(X, y_true)
            dt = (time.time() - start)*1000

            if gen % 20 == 0:
                print(f"Gen {gen:03d} | Error: {error:8.1f} | R_q: {Rq:.3f} | Pop: {len(self.nuclei)} | Ent: {self.entropies[-1]:.3f} | GPU: {dt:.2f} ms")

        return self