#!/usr/bin/env python3
# soda_hybrid_v3_gpu_benchmark.py
# Run SODA-Q GPU benchmark at scale N=4096. Requires compiled CUDA extension `sodaq_ext` with compute_and_update.

import time, os, math, random
import numpy as np
import torch
import matplotlib.pyplot as plt

# --------- CONFIG ----------
N = 4096                   # number of nuclei to benchmark
GENERATIONS = 200          # measured generations (after warm-up)
WARMUP = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTUM_K = 1.0
USE_FP16 = True            # try mixed precision to reduce memory & speed up
LOG_INTERVAL = 10
OUT_DIR = "./soda_benchmark_out"
os.makedirs(OUT_DIR, exist_ok=True)

# --------- import compiled extension (must be available) ----------
try:
    import saoq_ext as sodaq_ext  # if you named module differently, change here
    # older name used in prior messages: from sodaq_ext import compute_and_update
    compute_and_update = sodaq_ext.compute_and_update
except Exception:
    # fallback name: compute_and_update directly available
    try:
        from saodq_cuda_extension import step_sodaq_cuda as compute_and_update
    except Exception as e:
        raise RuntimeError("CUDA extension not found. Compile and ensure module is importable.") from e

# --------- initialize tensors/state ----------
torch.backends.cudnn.benchmark = True

# random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# initialize psi (complex), phases, fitness, entropy, spikes
if USE_FP16:
    # keep real/imag float16 on GPU, but extension expects float32 arrays for real/imag;
    # we'll store in fp16 but cast to fp32 for kernel or adapt extension to fp16.
    dtype_real = torch.float32
else:
    dtype_real = torch.float32

# psi represented as complex via two float tensors
psi_phase = 2 * math.pi * torch.rand(N, device=DEVICE)
psi_real = torch.cos(psi_phase).to(dtype_real, device=DEVICE)
psi_imag = torch.sin(psi_phase).to(dtype_real, device=DEVICE)

phases = torch.atan2(psi_imag, psi_real).to(dtype_real, device=DEVICE)
fitness = (1e3 + 9e3 * torch.rand(N, device=DEVICE)).to(dtype_real)
entropy = (1.0 + 3.0 * torch.rand(N, device=DEVICE)).to(dtype_real)
spikes = (1 + torch.randint(1,6,(N,), device=DEVICE)).to(dtype_real)

# optional move heavy arrays to pinned memory for host<->device transfers if needed
# --------- benchmark records ----------
times = []
R_qs = []
errors = []
ent_avgs = []

# small helper: simple local update to simulate local learning (keeps things realistic)
def local_update_batch(fitness, entropy, spikes, X_vec, y_true):
    # X_vec is CPU/torch float vector used only for simulated local update
    # We'll emulate updates: small random walk in fitness/entropy and spike fluctuation
    # Return simulated min error
    # This is cheap and runs on CPU
    # But to be realistic, we update tensors on GPU
    # For performance, do vector ops on GPU
    noise = (torch.rand_like(fitness) - 0.5) * 0.01
    fitness = fitness * (1.0 + 0.001*noise)
    entropy = entropy * (1.0 + 0.001*(torch.rand_like(entropy)-0.5))
    spikes = (spikes + (torch.randint(-1,2,(N,), device=DEVICE).float())).clamp(min=1)
    return fitness, entropy, spikes

# warm-up
print(f"DEVICE = {DEVICE}; N = {N}; Generations = {GENERATIONS}; Warmup = {WARMUP}; FP16={USE_FP16}")
torch.cuda.synchronize()

for t in range(WARMUP):
    # small synthetic X,y
    X = torch.rand(20, device=DEVICE) * 1000.0
    y_true = float((X**1.5).sum().item())
    fitness, entropy, spikes = local_update_batch(fitness, entropy, spikes, X, y_true)

    # call CUDA fused kernel; note kernel signature may vary depending on compiled module
    # adjust call to your extension: here expecting (fitness, entropy, phases, psi_real, psi_imag, spikes, quantum_k)
    t0 = time.time()
    psi_new, couple_r, couple_i, Rq = compute_and_update(
        fitness.contiguous(), entropy.contiguous(), phases.contiguous(),
        psi_real.contiguous(), psi_imag.contiguous(), spikes.contiguous(), float(QUANTUM_K)
    )
    torch.cuda.synchronize()
    t1 = time.time()

    # update psi buffers
    psi_real = psi_new.real.to(dtype_real)
    psi_imag = psi_new.imag.to(dtype_real)
    phases = torch.atan2(psi_imag, psi_real)
    if (t+1) % 5 == 0:
        print(f"Warmup {t+1}/{WARMUP} | Rq {Rq:.4f} | step {(t1-t0)*1000:.2f} ms")

# measured run
print("Starting measured run...")
torch.cuda.synchronize()
for gen in range(GENERATIONS):
    X = torch.rand(20, device=DEVICE) * 1000.0
    y_true = float((X**1.5).sum().item())

    # local update (simulate learning)
    fitness, entropy, spikes = local_update_batch(fitness, entropy, spikes, X, y_true)

    t0 = time.time()
    psi_new, couple_r, couple_i, Rq = compute_and_update(
        fitness.contiguous(), entropy.contiguous(), phases.contiguous(),
        psi_real.contiguous(), psi_imag.contiguous(), spikes.contiguous(), float(QUANTUM_K)
    )
    torch.cuda.synchronize()
    t1 = time.time()
    dt_ms = (t1 - t0) * 1000.0

    # record
    psi_real = psi_new.real.to(dtype_real)
    psi_imag = psi_new.imag.to(dtype_real)
    phases = torch.atan2(psi_imag, psi_real)

    times.append(dt_ms)
    R_qs.append(Rq)
    ent_avgs.append(float(entropy.mean().item()))
    errors.append(float(fitness.min().item()))

    if gen % LOG_INTERVAL == 0 or gen == GENERATIONS-1:
        print(f"Gen {gen:04d} | Error: {errors[-1]:8.2f} | R_q: {R_qs[-1]:.4f} | Ent: {ent_avgs[-1]:.3f} | GPU step: {dt_ms:.2f} ms")

# save results
np.savez(os.path.join(OUT_DIR, "benchmark_results.npz"),
         times=np.array(times), R_qs=np.array(R_qs),
         ent_avgs=np.array(ent_avgs), errors=np.array(errors))

# plot
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(times, label="ms/step")
plt.ylabel("ms")
plt.legend()
plt.subplot(2,1,2)
plt.plot(R_qs, label="R_q")
plt.plot(ent_avgs, label="entropy_avg")
plt.plot(errors, label="min_error")
plt.legend()
plt.xlabel("Generation")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "benchmark_plots.png"), dpi=200)
print("Benchmark complete. Outputs in", OUT_DIR)