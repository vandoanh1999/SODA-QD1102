import torch
import torch.nn.functional as F
import math

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# 1) Vectorized Fitness & Entropy
# ----------------------
def compute_fitness_entropy(X_batch: torch.Tensor,
                            y_batch: torch.Tensor,
                            W: torch.Tensor):
    """
    Vectorized computation of fitness and entropy for N nuclei.

    Inputs:
        X_batch: (B, D) input batch (torch.float, device)
        y_batch: (B,) or (B,1) target values
        W: (N, D) weight/parameter matrix for N nuclei

    Outputs:
        fitness: (N,) mean squared error per nucleus over batch (higher = worse)
        entropy: (N,) Shannon entropy of softmax(W_i) (measure of parameter diversity)
    Notes:
        - All ops are vectorized and run on the same device as inputs.
        - Complexity: memory O(B*N) for preds; choose B small if N large.
    """
    assert X_batch.dim() == 2 and W.dim() == 2
    B, D = X_batch.shape
    N, D2 = W.shape
    assert D == D2, "Dimension mismatch"

    # Move to same device
    X = X_batch.to(device)
    y = y_batch.to(device)
    W = W.to(device)

    # Predictions for all nuclei: preds shape (B, N)  via matrix multiply X @ W^T
    # This is the heavy op but vectorized on GPU
    preds = X @ W.t()                              # (B, N)

    # If y_batch is (B,) make it (B,1) to broadcast
    if y.dim() == 1:
        y = y.view(B, 1)

    # Compute MSE per nucleus across batch: mean over B
    # mse_j = mean_b ( (pred_bj - y_b)^2 )
    errors = (preds - y).pow(2).mean(dim=0)        # (N,)

    # Fitness: we want higher = worse (so errors as fitness)
    fitness = errors

    # Entropy: Shannon entropy of softmax weights per nucleus (treat W_i as logits)
    # Compute softmax across feature dim D for each nucleus
    # W: (N,D) -> p: (N,D)
    p = F.softmax(W, dim=1)                        # (N, D)
    # entropy_j = - sum_d p_jd * log(p_jd)
    entropy = -(p * (p + 1e-12).log()).sum(dim=1)  # (N,)

    return fitness, entropy


# ----------------------
# 2) Vectorized Decomposition Selection (choose worst k)
# ----------------------
def select_worst_k(fitness: torch.Tensor, k: int = 3):
    """
    Select indices of k worst-performing nuclei (largest fitness/error).
    Inputs:
        fitness: (N,) tensor
        k: int
    Output:
        worst_idx: (k,) long tensor of indices (on same device)
    """
    N = fitness.shape[0]
    k = min(k, N)
    # topk with largest=True gives worst
    vals, idx = torch.topk(fitness, k=k, largest=True, sorted=True)
    return idx  # (k,)


# ----------------------
# 3) Vectorized Fusion (create fused W from top-entropy nuclei)
# ----------------------
def fuse_nuclei(W: torch.Tensor, entropy: torch.Tensor, top_m: int = 4):
    """
    Fuse top_m nuclei by entropy into a single super-nucleus parameter vector.

    Inputs:
        W: (N, D)
        entropy: (N,)
        top_m: number of nuclei with highest entropy to fuse

    Outputs:
        W_fused: (D,) fused parameter vector (entropy-weighted average)
        fused_idx: indices used for fusion (top_m,)
        W_updated: (N - top_m + 1, D) optional new population param matrix
                   (this simply removes fused elements and appends fused vector)
    Notes:
        - All ops vectorized on GPU.
    """
    N, D = W.shape
    m = min(top_m, N)

    # get indices of top-m entropy nuclei
    top_vals, top_idx = torch.topk(entropy, k=m, largest=True, sorted=False)  # (m,)

    # gather weights
    W_sel = W[top_idx]                    # (m, D)
    ent_sel = top_vals.view(m, 1)         # (m,1)

    # weighted average by entropy (so higher entropy contributes more)
    weight_sum = ent_sel.sum(dim=0) + 1e-12
    W_fused = (ent_sel * W_sel).sum(dim=0) / weight_sum  # (D,)

    # Build updated W: remove fused indices, append fused vector at end
    mask = torch.ones(N, dtype=torch.bool, device=W.device)
    mask[top_idx] = False
    W_remaining = W[mask]                 # (N-m, D)
    W_updated = torch.cat([W_remaining, W_fused.view(1, D)], dim=0)  # (N-m+1, D)

    return W_fused, top_idx, W_updated


# ----------------------
# 4) Quantum Tunneling Probability & dN/dt vectorized variant
# ----------------------
def compute_P_dec(R: float, R_th: float = 0.78, C: float = 1.0, hbar: float = 1.0, DeltaR: float = 1.0):
    """
    Compute scalar quantum tunneling probability P_dec(t) using formula provided:
    P_dec = C * exp( -2/hbar * DeltaR * max(0, R_th - R) )

    R: current global quantum order parameter (scalar)
    Returns: scalar float
    """
    gap = max(0.0, R_th - float(R))
    exponent = - (2.0 / hbar) * DeltaR * gap
    return C * math.exp(exponent)


def vectorized_dN_dt(N_current: int, P_dec: float, W3_count: int):
    """
    Compute dN/dt as per SODA law in vectorized form (scalar update).
    dN/dt = 3 * P_dec * |W3| - 3 * P_dec
    """
    return 3.0 * P_dec * float(W3_count) - 3.0 * P_dec


# ----------------------
# Example end-to-end demo
# ----------------------
if __name__ == "__main__":
    # toy sizes (adjust to your GPU memory)
    N = 4096   # population size (test smaller if OOM)
    D = 20
    B = 32     # batch size

    # initialize random tensors on GPU
    torch.manual_seed(0)
    X = torch.randn(B, D, device=device) * 10.0
    # make artificial target y as some nonlinear function of X (for realism)
    y = (X.pow(1.5).sum(dim=1) + 500.0 * torch.sin(X.sum(dim=1) / 1000.0)).to(device)

    W = torch.randn(N, D, device=device) * 0.1  # parameter matrix for N nuclei

    # 1) compute fitness & entropy (vectorized)
    fitness_vec, entropy_vec = compute_fitness_entropy(X, y, W)
    print("fitness_vec shape:", fitness_vec.shape, "entropy_vec shape:", entropy_vec.shape)
    print("sample fitness (min/mean/max):", fitness_vec.min().item(), fitness_vec.mean().item(), fitness_vec.max().item())

    # 2) select worst k
    worst_idx = select_worst_k(fitness_vec, k=3)
    print("worst indices:", worst_idx)

    # 3) fuse highest-entropy nuclei
    W_fused, fused_idx, W_updated = fuse_nuclei(W, entropy_vec, top_m=4)
    print("fused_idx:", fused_idx)
    print("W_fused shape:", W_fused.shape, "W_updated shape:", W_updated.shape)

    # 4) compute global order R (example compute using psi if available)
    # For demo, compute a mock R in [0,1]
    R_example = float(torch.rand(1).item())
    P_dec = compute_P_dec(R_example, R_th=0.78, C=1.0, hbar=1.0, DeltaR=1.0)
    deltaN = vectorized_dN_dt(N_current=N, P_dec=P_dec, W3_count=3)
    print(f"R={R_example:.4f}, P_dec={P_dec:.6e}, dN/dt={deltaN:.6f}")

    # All results computed on device; transfer to host for I/O only if needed
    # (e.g., fused_idx.cpu().numpy())