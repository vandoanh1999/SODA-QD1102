import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================
# SODA-QD1102: Vectorized Quantum Kuramoto Engine (GPU)
# ==============================================
class SODA_QD1102:
    def __init__(self, N, quantum_k=1.0):
        self.N = N
        self.K = quantum_k

        # initialize quantum wavefunction
        phase = 2 * math.pi * torch.rand(N, device=device)
        self.psi = torch.exp(1j * phase)                       # complex (N,)
        self.phase = phase.clone()                             # real (N,)

        # initialize sigma-free learning states
        self.fitness = 1e3 + 9e3 * torch.rand(N, device=device)
        self.entropy = 1.0 + 3.0 * torch.rand(N, device=device)
        self.spike = 1 + torch.randint(1, 6, (N,), device=device)

    # =============================================================
    # Δω_j computed from spike activity
    # =============================================================
    def compute_delta_omega(self):
        max_spike = self.spike.max()
        return self.spike / (1000.0 + max_spike)

    # =============================================================
    # Full GPU vectorized QKM step
    # =============================================================
    def qkm_step(self):
        phi = self.phase
        psi = self.psi

        # ------------------------
        # 1. Compute J-matrix (vectorized)
        # ------------------------
        f = self.fitness
        e = self.entropy

        # Broadcasting to NxN
        df = torch.abs(f[:, None] - f[None, :])
        de = torch.abs(e[:, None] - e[None, :])

        T = 1.0 + torch.std(f)
        J = torch.exp(-df / (T + 1e-12)) * torch.exp(-de)

        # ------------------------
        # 2. Build phase difference matrix
        # ------------------------
        dphi = phi[None, :] - phi[:, None]               # (N,N)
        M = torch.exp(1j * dphi)                         # (N,N) complex

        # ------------------------
        # 3. Compute global coupling (sum over k)
        # ------------------------
        coupling = torch.sum(J * M, dim=1)               # (N,) complex

        # ------------------------
        # 4. Compute delta omega from spike activity
        # ------------------------
        delta_omega = self.compute_delta_omega()         # (N,)

        # ------------------------
        # 5. Update ψ
        # ψ_new = ψ + K * coupling + Δω * ψ
        # ------------------------
        psi_new = psi + self.K * coupling + (delta_omega * psi)

        # ------------------------
        # 6. Normalize ψ_new
        # ------------------------
        norm = torch.sqrt(torch.sum(torch.abs(psi_new)**2))
        psi_new = psi_new / (norm + 1e-12)

        # update state
        self.psi = psi_new
        self.phase = torch.angle(psi_new)

        # ------------------------
        # 7. Quantum Order Parameter
        # ------------------------
        R_q = torch.abs(torch.sum(psi_new)).item()
        return R_q

    # =============================================================
    # run simulation
    # =============================================================
    def run(self, steps=50):
        R_history = []
        for t in range(steps):
            Rq = self.qkm_step()
            R_history.append(Rq)
        return R_history