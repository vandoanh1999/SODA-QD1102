import torch
import time
import sys
import pandas as pd
device = torch.device("cpu")

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
    print(f"Benchmarking QKM on CPU: N={N}, repeats={repeats}...")

    # Initialize test data
    phases  = torch.rand(N, device=device) * 2 * torch.pi
    fitness = torch.rand(N, device=device) * 10000
    entropy = torch.rand(N, device=device) * 4.0

    # Warm-up
    for _ in range(3):
        qkm_broadcast(phases, fitness, entropy)

    # ---------------------------
    # Baseline PyTorch Broadcasting
    # ---------------------------
    t0 = time.time()

    for _ in range(repeats):
        S_base = qkm_broadcast(phases, fitness, entropy)

    t1 = time.time()
    baseline_time = (t1 - t0) / repeats * 1000

    return {
        "N": N,
        "repeats": repeats,
        "baseline_ms": baseline_time,
    }


def main():
    args = sys.argv[1:]
    run_all = "all" in args
    report = "report" in args

    if run_all:
        benchmark_Ns = [256, 512, 1024, 2048, 4096]
        results = []
        for n in benchmark_Ns:
            result = benchmark(N=n, repeats=10)
            results.append(result)

        if report:
            df = pd.DataFrame(results)
            df['baseline_ms'] = df['baseline_ms'].round(3)
            print("\n--- Scientific Standard Results ---")
            print(df.to_markdown(index=False))

    else:
        benchmark(N=1024, repeats=10)
        benchmark(N=2048, repeats=10)


if __name__ == "__main__":
    main()