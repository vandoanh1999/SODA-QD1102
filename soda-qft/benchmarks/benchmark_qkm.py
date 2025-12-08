import torch
import time
from fused_qkm import qkm_fused    # <-- CUDA kernel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PyTorch BROADCASTING QKM (Baseline)
# ==========================================
def qkm_broadcast(phases, fitness, entropy):
    """
    Baseline implementation using PyTorch broadcasting.
    Complexity: O(N^2) memory + kernel launches
    """
    N = phases.shape[0]
    T = 1.0 + torch.std(fitness)

    # Δf, Δe
    df = (fitness[:, None] - fitness[None, :]).abs()      # (N,N)
    de = (entropy[:, None] - entropy[None, :]).abs()      # (N,N)

    # J
    J = torch.exp(-df / T) * torch.exp(-de)                # (N,N)

    # phase difference
    dphi = phases[None, :] - phases[:, None]               # (N,N)
    M = torch.exp(1j * dphi)                               # (N,N)

    # Coupling
    S = (J * M).sum(dim=1)                                 # (N,)
    return S


# ==========================================
# Benchmark runner
# ==========================================
def benchmark(N=2048, repeats=10):
    torch.cuda.empty_cache()

    print("====================================")
    print(f"Benchmarking QKM: N={N}, repeats={repeats}")
    print("Device:", device)
    print("====================================")

    # Initialize test data
    phases  = torch.rand(N, device=device) * 2 * torch.pi
    fitness = torch.rand(N, device=device) * 10000
    entropy = torch.rand(N, device=device) * 4.0

    # Warm-up GPU
    for _ in range(3):
        qkm_fused(phases, fitness, entropy, float(1.0 + torch.std(fitness)))

    # ---------------------------
    # Baseline PyTorch Broadcasting
    # ---------------------------
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(repeats):
        S_base = qkm_broadcast(phases, fitness, entropy)

    torch.cuda.synchronize()
    t1 = time.time()
    baseline_time = (t1 - t0) / repeats * 1000

    # ---------------------------
    # Fused CUDA Kernel
    # ---------------------------
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(repeats):
        S_fused = qkm_fused(phases, fitness, entropy, float(1.0 + torch.std(fitness)))

    torch.cuda.synchronize()
    t1 = time.time()
    fused_time = (t1 - t0) / repeats * 1000

    # ---------------------------
    # Accuracy Check (L2 error)
    # ---------------------------
    l2_error = torch.norm(S_base - S_fused).item()

    speedup = baseline_time / fused_time

    print("RESULTS")
    print("------------------------------------")
    print(f"Baseline (Broadcast):  {baseline_time:.3f} ms/run")
    print(f"Fused CUDA Kernel:     {fused_time:.3f} ms/run")
    print(f"Speedup:               {speedup:.2f}× faster")
    print(f"L2 Error:              {l2_error:.6f}   (should be < 1e-4)")
    print("====================================")
    return {
        "baseline_ms": baseline_time,
        "fused_ms": fused_time,
        "speedup": speedup,
        "l2_error": l2_error
    }


if __name__ == "__main__":
    benchmark(N=1024, repeats=10)
    benchmark(N=2048, repeats=10)