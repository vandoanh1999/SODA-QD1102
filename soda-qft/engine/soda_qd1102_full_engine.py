import torch
import torch.nn.functional as F
import math

from fused_qkm import qkm_fused   # CUDA kernel (real) – in concept we assume available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
#  SODA-QD1102 FULL ENGINE
# ================================================================
class SODA_QD1102_Engine:
    def __init__(self, W_max=8192, dim=32, init_N=32):
        """
        W_max : maximum number of nuclei to store without realloc
        dim   : feature dimension for learning
        init_N: initial active nuclei
        """

        self.W_max = W_max
        self.dim = dim

        # Full parameter buffer (fixed)
        self.W = torch.randn(W_max, dim, device=device) * 0.05

        # Only first init_N nuclei are alive
        self.alive_mask = torch.zeros(W_max, dtype=torch.bool, device=device)
        self.alive_mask[:init_N] = True

        # Track nucleus types (0=Chaos, 1=Bayes)
        self.nucleus_type = torch.randint(0, 2, (W_max,), device=device)

        # Cached training state
        self.current_N = init_N

        # Quantum phases ψ (only for alive)
        self.psi = torch.exp(1j * (2 * torch.pi * torch.rand(W_max, device=device)))

        # metrics history
        self.history_Rq = []
        self.history_population = []

    # ================================================================
    # 1) Vectorized Fitness + Entropy
    # ================================================================
    def compute_fitness_entropy(self, X_batch, y_batch):
        """
        Compute vectorized fitness and entropy ONLY for alive nuclei.
        """

        alive_idx = self.alive_mask.nonzero(as_tuple=False).view(-1)
        W_alive = self.W[alive_idx]        # (N, D)

        # Pred = X @ W^T
        preds = X_batch @ W_alive.t()      # (B, N)
        B = X_batch.shape[0]

        if y_batch.dim() == 1:
            y_batch = y_batch.view(B, 1)

        # MSE over batch
        fitness = ((preds - y_batch)**2).mean(dim=0)   # (N,)

        # Shannon entropy of softmax(W)
        p = F.softmax(W_alive, dim=1)
        entropy = -(p * (p + 1e-12).log()).sum(dim=1)  # (N,)

        return alive_idx, fitness, entropy

    # ================================================================
    # 2) Quantum Kernel (Fused QKM)
    # ================================================================
    def quantum_dynamics(self, phases, fitness, entropy):
        """
        Call fused CUDA kernel to compute couplings S_j for alive nuclei.
        """
        T = 1.0 + torch.std(fitness)
        S = qkm_fused(phases, fitness, entropy, float(T))   # returns complex(N,)
        R_q = torch.abs(S.sum())

        return S, R_q

    # ================================================================
    # 3) Decomposition (Top 3 worst nuclei)
    # ================================================================
    def decomposition(self, alive_idx, fitness):
        """
        Select worst-performing nuclei (top-3 fitness)
        """
        k = min(3, fitness.numel())
        _, worst_local_idx = torch.topk(fitness, k, largest=True)
        worst_global_idx = alive_idx[worst_local_idx]
        return worst_global_idx

    # ================================================================
    # 4) Fusion (Top entropy nuclei)
    # ================================================================
    def fusion(self, alive_idx, entropy, top_m=4):
        """
        Weighted fusion of top-m high entropy nuclei.
        """

        m = min(top_m, entropy.numel())
        _, top_local_idx = torch.topk(entropy, k=m, largest=True)
        top_global_idx = alive_idx[top_local_idx]

        W_sel = self.W[top_global_idx]               # (m, D)
        ent_sel = entropy[top_local_idx].view(m,1)   # (m,1)

        W_fused = (ent_sel * W_sel).sum(dim=0) / (ent_sel.sum() + 1e-9)

        return top_global_idx, W_fused

    # ================================================================
    # 5) Structural Update (SODA Law)
    # ================================================================
    def structural_update(self, worst_idx, fused_idx, W_fused, R_q):

        # Quantum Tunneling Probability
        R_th = 0.78
        gap = max(0.0, R_th - float(R_q))
        P_dec = math.exp(-2.0 * gap)    # simplified for concept

        # (a) Decomposition: mark worst as dead, spawn children
        for j in worst_idx:
            self.alive_mask[j] = False

        # Add new nuclei (descendants)
        children_count = 3 * worst_idx.numel()
        added = 0
        for slot in range(self.W_max):
            if added >= children_count:
                break
            if not self.alive_mask[slot]:
                # new random nucleus (mutation)
                self.W[slot] = W_fused + 0.01 * torch.randn_like(W_fused)
                self.psi[slot] = torch.exp(1j * (2 * torch.pi * torch.rand(1, device=device)))
                self.alive_mask[slot] = True
                added += 1

        # (b) Fusion: remove fused originals, add fused nucleus
        for j in fused_idx:
            self.alive_mask[j] = False

        # assign fused nucleus into first free slot
        for slot in range(self.W_max):
            if not self.alive_mask[slot]:
                self.W[slot] = W_fused.clone()
                self.psi[slot] = torch.exp(1j * (2 * torch.pi * torch.rand(1, device=device)))
                self.alive_mask[slot] = True
                break

        # update count
        self.current_N = int(self.alive_mask.sum())

        return P_dec

    # ================================================================
    # 6) FULL EVOLUTION CYCLE
    # ================================================================
    def update_evolution_cycle(self, X_batch, y_batch):

        # (1) LOCAL LEARNING
        alive_idx, fitness, entropy = self.compute_fitness_entropy(X_batch, y_batch)

        # (2) QUANTUM DYNAMICS
        phases_alive = torch.angle(self.psi[alive_idx])
        S, R_q = self.quantum_dynamics(phases_alive, fitness, entropy)

        # update phases
        self.psi[alive_idx] = torch.exp(1j * (torch.angle(self.psi[alive_idx]) + 0.01 * torch.real(S)))

        # (3) STRUCTURAL EVOLUTION
        worst_idx = self.decomposition(alive_idx, fitness)
        fused_idx, W_fused = self.fusion(alive_idx, entropy)
        P_dec = self.structural_update(worst_idx, fused_idx, W_fused, R_q)

        # RECORD
        self.history_Rq.append(float(R_q))
        self.history_population.append(self.current_N)

        return {
            "R_q": float(R_q),
            "N": self.current_N,
            "P_dec": P_dec,
            "worst_idx": worst_idx,
            "fused_idx": fused_idx
        }