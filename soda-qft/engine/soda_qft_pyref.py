import torch
import math
import time
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SODA_QFT_Py:
    """
    PyTorch reference Multi-field Quantum Kuramoto (SODA-QFT).
    - N: number of nuclei
    - F: number of fields (e.g., 3: Chaos, Bayes, SNN)
    - D: parameter dim (for local learning)
    """

    def __init__(self, N:int, F:int=3, D:int=20, quantum_k:float=1.0):
        self.N = N
        self.F = F
        self.D = D
        self.K = quantum_k

        # Parameter buffers per nucleus (for local learning) shape (N, D)
        self.W = torch.randn(N, D, device=device) * 0.05

        # Per-field wavefunction psi (complex): shape (N, F) complex64
        phases = 2*math.pi*torch.rand(N, F, device=device)
        self.psi = torch.exp(1j * phases)            # complex tensor (N,F)
        self.phase = torch.angle(self.psi)           # (N,F) real tensor

        # Per-field fitness & entropy (N,F)
        self.fitness = torch.abs(torch.randn(N, F, device=device)) * 1000.0
        # Entropy per-field computed from per-nucleus per-field parameterization
        self.entropy = torch.abs(torch.randn(N, F, device=device)) * 1.0

        # Inter-field coupling matrix G (F,F)
        self.G = torch.ones(F, F, device=device)     # can be tuned / learned

        # spiking/firing per nucleus per field
        self.spikes = torch.randint(1, 6, (N, F), device=device).float()

    def compute_delta_omega(self):
        # delta omega per nucleus per field: normalized firing across fields
        max_spike = self.spikes.max(dim=0)[0].clamp(min=1.0)  # (F,)
        # broadcast to (N,F)
        return self.spikes / (1000.0 + max_spike.view(1, -1))

    def local_learning_update(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Vectorized local learning producing per-field fitness & entropy.
        Simple strategy: for each field a, compute preds = X @ w_a^T, where w_a are transformations of W.
        For clarity: use small per-field linear heads as W_field = W * r_a (random mask) -- this is illustrative.
        """
        B = X_batch.shape[0]
        # Create per-field parameter heads by linear projection from W (cheap)
        # Heads: shape (N, D) -> replicate or parametrize per-field
        # Here: create F small linear projection matrices R_a: (D,D) small overhead
        R = torch.stack([torch.eye(self.D, device=device) for _ in range(self.F)], dim=0)  # (F,D,D)
        # Compute predictions per field: preds_b_j_a = X (B,D) @ (W_j (D) * R_a^T)  -> (B,N,F)
        # Efficient approach: compute W_heads_a = W @ R_a^T  -> (N,D) per a
        W_heads = torch.einsum("nd,fdk->fnk", self.W, R)   # results (F,N,D) [note: small F]
        # reshape to (B,N,F)
        preds = torch.zeros(B, self.N, self.F, device=device, dtype=X_batch.dtype)
        for a in range(self.F):
            preds[:,:,a] = X_batch @ W_heads[a].t()   # (B,N)
        if y_batch.dim()==1:
            y = y_batch.view(B,1,1)
        else:
            y = y_batch.view(B,1,1)
        mse = ((preds - y)**2).mean(dim=0)    # (N,F)
        self.fitness = mse

        # entropy: compute softmax over D of per-nucleus pseudo-logits for each field.
        # Build pseudo logits by slicing W and applying small transform per field
        ent = torch.zeros(self.N, self.F, device=device)
        for a in range(self.F):
            logits = self.W  # (N,D) simple placeholder; production should use field-specific param
            p = torch.softmax(logits, dim=1)
            ent[:,a] = -(p * (p+1e-12).log()).sum(dim=1)
        self.entropy = ent

    def qkm_multifield_step(self):
        """
        Compute S_j^{(a)} for all j,a in vectorized manner.
        Implementation loops over field pairs (a,b) in Python (F small) and computes N×N matrices per pair.
        """
        N, F = self.N, self.F
        T = 1.0 + torch.std(self.fitness)   # scalar

        # prepare outputs
        S = torch.zeros(N, F, dtype=torch.complex64, device=device)

        # phases: (N,F)
        phi = self.phase        # (N,F)
        psi = self.psi          # (N,F) complex

        # loop over target field a and source field b
        # This creates per-pair intermediates of size (N,N) — acceptable if F small
        for a in range(F):
            # we'll accumulate S[:,a]
            S_a = torch.zeros(N, dtype=torch.complex64, device=device)
            phi_ja = phi[:, a].unsqueeze(1)           # (N,1)
            f_ja = self.fitness[:, a].unsqueeze(1)    # (N,1)
            e_ja = self.entropy[:, a].unsqueeze(1)    # (N,1)

            for b in range(F):
                # source field b
                phi_kb = phi[:, b].unsqueeze(0)      # (1,N)
                f_kb = self.fitness[:, b].unsqueeze(0) # (1,N)
                e_kb = self.entropy[:, b].unsqueeze(0) # (1,N)

                # df, de -> (N,N)
                df = (f_ja - f_kb).abs()            # (N,N)
                de = (e_ja - e_kb).abs()            # (N,N)

                J = torch.exp(-df / (T + 1e-12)) * torch.exp(-de)   # (N,N)

                # phase diff (phi_kb - phi_ja) -> (N,N)
                dphi = (phi_kb - phi_ja)           # (N,N)
                M = torch.exp(1j * dphi)           # (N,N) complex64

                # psi_k^b: (N,) complex, need shape (1,N) to broadcast
                psi_kb = psi[:, b].unsqueeze(0)    # (1,N) complex

                # interaction amplitude includes inter-field scalar G_ab
                G_ab = float(self.G[a,b])

                # contribution: for each j: sum_k J[j,k] * M[j,k] * psi_kb[0,k] * G_ab
                # compute elementwise then sum over k
                contrib = (J.to(torch.complex64) * M * (psi_kb * G_ab))  # (N,N) complex
                S_a += contrib.sum(dim=1)    # sum over k -> (N,)
            S[:, a] = S_a

        # Update psi per field
        delta_omega = self.compute_delta_omega()   # (N,F)
        psi_new = self.psi + self.K * S + (delta_omega.to(torch.complex64) * self.psi)
        # normalize per nucleus across fields (optional). Here normalize per nucleus (sum |psi|^2 =1)
        norms = torch.sqrt(torch.sum(torch.abs(psi_new)**2, dim=1, keepdim=True))
        psi_new = psi_new / (norms + 1e-12)
        self.psi = psi_new
        self.phase = torch.angle(self.psi)
        Rq = torch.abs(self.psi.sum(dim=(0,1))).item()  # global coherence scalar (sum over j,a)
        return S, Rq

    def step(self, X_batch, y_batch):
        self.local_learning_update(X_batch, y_batch)
        S, Rq = self.qkm_multifield_step()
        return {"R_q": Rq, "S": S}