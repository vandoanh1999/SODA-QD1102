# soda_qft_engine_gpu.py
import torch, math
from fused_qkm_multifield import qkm_fused_multi
from compute_semantic_interaction_matrix import compute_semantic_interaction_matrix, semantic_to_G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SODA_QFT_GPU_Engine:
    def __init__(self, N=4096, F=3, D=20, W_max=None):
        self.N = N
        self.F = F
        self.D = D
        self.W_max = W_max if W_max is not None else N
        # parameter buffer (N, D)
        self.W = torch.randn(self.W_max, D, device=device) * 0.05
        self.alive_mask = torch.zeros(self.W_max, dtype=torch.bool, device=device)
        self.alive_mask[:N] = True
        # initialize multi-field psi (Nmax, F) complex
        phases = 2*math.pi*torch.rand(self.W_max, self.F, device=device)
        self.psi = torch.exp(1j * phases).to(torch.complex64)
        self.phase = torch.angle(self.psi)
        # per-field fitness & entropy (Nmax, F)
        self.fitness = torch.zeros(self.W_max, self.F, device=device)
        self.entropy = torch.zeros(self.W_max, self.F, device=device)
        # default G (F,F) identity
        self.G = torch.eye(F, device=device)
        # initialize psi_real/imag slices for kernel
        # records
        self.history_Rq = []

    def bootstrap_G_from_semantics(self, field_embeddings, method="cosine", base_scale=1.0):
        Lambda = compute_semantic_interaction_matrix(field_embeddings, method=method)
        self.G = semantic_to_G(Lambda.to(device), base_scale)

    def compute_local_learning(self, X_batch, y_batch):
        # example: compute per-field fitness and entropy (vectorized)
        alive_idx = self.alive_mask.nonzero(as_tuple=False).view(-1)
        N_alive = alive_idx.numel()
        if N_alive == 0:
            return alive_idx

        W_alive = self.W[alive_idx]  # (N_alive, D)
        B = X_batch.shape[0]
        # simple pred per nucleus: X @ w^T -> (B, N_alive)
        preds = X_batch @ W_alive.t()
        if y_batch.dim() == 1:
            y = y_batch.view(B,1)
        else:
            y = y_batch
        mse = ((preds - y)**2).mean(dim=0)   # (N_alive,)
        # replicate for fields as simple baseline
        self.fitness[alive_idx] = mse.unsqueeze(1).repeat(1, self.F)
        # entropy from W slice softmax
        p = torch.softmax(W_alive, dim=1)
        ent = -(p * (p + 1e-12).log()).sum(dim=1)
        self.entropy[alive_idx] = ent.unsqueeze(1).repeat(1, self.F)
        return alive_idx

    def step(self, X_batch, y_batch):
        alive_idx = self.compute_local_learning(X_batch, y_batch)
        if alive_idx.numel() == 0:
            return None

        # pack per-field tensors (N_alive, F)
        phases_alive = self.phase[alive_idx]         # (N_alive,F)
        fitness_alive = self.fitness[alive_idx].float()
        entropy_alive = self.entropy[alive_idx].float()
        psi_alive = self.psi[alive_idx].to(torch.complex64)  # (N_alive,F)

        # split psi into real/imag float tensors for kernel
        psi_real = psi_alive.real.float()
        psi_imag = psi_alive.imag.float()

        T = 1.0 + torch.std(fitness_alive)

        # call fused kernel
        out = qkm_fused_multi(phases_alive.float(), fitness_alive, entropy_alive,
                              psi_alive, self.G.float(), float(T))
        # out: complex (N_alive,F)
        S = out

        # update psi for alive indices
        delta_omega = (torch.rand_like(fitness_alive) / 1000.0).to(torch.complex64)  # placeholder
        psi_new = psi_alive + 1.0 * S + delta_omega * psi_alive
        # normalize per nucleus across fields
        norms = torch.sqrt(torch.sum(torch.abs(psi_new)**2, dim=1, keepdim=True))
        psi_new = psi_new / (norms + 1e-12)
        self.psi[alive_idx] = psi_new

        # compute global order
        Rq = torch.abs(psi_new.sum(dim=(0,1))).item()
        self.history_Rq.append(Rq)

        # structural evolution: vectorized selection + update
        # decomposition: select worst 3 by aggregated fitness
        agg_fitness = self.fitness[alive_idx].mean(dim=1)
        _, worst_local = torch.topk(agg_fitness, k=min(3, agg_fitness.numel()), largest=True)
        worst_global = alive_idx[worst_local]

        # fusion: top entropy
        agg_entropy = self.entropy[alive_idx].mean(dim=1)
        _, top_local = torch.topk(agg_entropy, k=min(4, agg_entropy.numel()), largest=True)
        top_global = alive_idx[top_local]

        # compute W_fused as entropy-weighted avg of parameters (vectorized)
        W_sel = self.W[top_global]      # (m, D)
        ent_sel = agg_entropy[top_local].view(-1,1)
        W_fused = (ent_sel * W_sel).sum(dim=0) / (ent_sel.sum() + 1e-12)

        # apply structural updates in-place using alive_mask; find free slots
        # mark worst dead
        self.alive_mask[worst_global] = False
        # mark top fused dead
        self.alive_mask[top_global] = False

        # insert fused nucleus in first free slot
        free_slots = (self.alive_mask == False).nonzero(as_tuple=False).view(-1)
        if free_slots.numel() > 0:
            slot = free_slots[0].item()
            self.W[slot] = W_fused.clone()
            # init psi for fused
            self.psi[slot] = torch.exp(1j * 2*math.pi * torch.rand(self.F, device=device)).to(torch.complex64)
            self.alive_mask[slot] = True

        # spawn children for each worst: create mutated copies in available free slots
        free_slots = (self.alive_mask == False).nonzero(as_tuple=False).view(-1)
        child_needed = worst_global.numel() * 2
        for i in range(min(child_needed, free_slots.numel())):
            slot = free_slots[i].item()
            parent = worst_global[i % worst_global.numel()]
            self.W[slot] = self.W[parent] + 0.01 * torch.randn_like(self.W[parent])
            self.psi[slot] = torch.exp(1j * 2*math.pi * torch.rand(self.F, device=device)).to(torch.complex64)
            self.alive_mask[slot] = True

        return {"R_q": Rq, "N_alive": int(self.alive_mask.sum().item())}