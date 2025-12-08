# saodq_cuda_extension.py
import torch
from torch.utils.cpp_extension import load
import os

# compile (this will create a .so in the build dir)
module = load(
    name="sodaq_ext",
    sources=["sodaq_cuda_bind.cpp", "sodaq_cuda_kernel.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)

# Python wrapper
def step_sodaq_cuda(psi_complex, phases, fitness, entropy, spikes, quantum_k=1.0):
    """
    psi_complex: complex tensor as two tensors or torch.cfloat (N,)
    returns: psi_new (complex as real+imag), coupling real/imag, R_q
    """
    device = psi_complex.device
    # split into real/imag if complex dtype
    if psi_complex.is_complex():
        psi_r = psi_complex.real.contiguous()
        psi_i = psi_complex.imag.contiguous()
    else:
        raise RuntimeError("psi must be complex tensor (torch.cfloat)")

    fitness = fitness.contiguous().float()
    entropy = entropy.contiguous().float()
    phases = phases.contiguous().float()
    spikes = spikes.contiguous().float()

    out = module.compute_and_update(fitness, entropy, phases, psi_r, psi_i, spikes, float(quantum_k))
    # out = [psi_new_real, psi_new_imag, couple_real, couple_imag, R_q]
    psi_new_r, psi_new_i, couple_r, couple_i, Rq = out
    psi_new = torch.complex(psi_new_r.to(device), psi_new_i.to(device))
    return psi_new, couple_r, couple_i, Rq.item()


# Quick test (small N)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 512
    psi = torch.exp(1j * 2 * 3.1415 * torch.rand(N, device=device)).to(torch.cfloat)
    phases = torch.angle(psi).real
    fitness = torch.rand(N, device=device) * 1e4
    entropy = 1.0 + torch.rand(N, device=device) * 3.0
    spikes = (torch.randint(1, 6, (N,), device=device)).float()
    psi_new, cr, ci, Rq = step_sodaq_cuda(psi, phases, fitness, entropy, spikes, quantum_k=1.0)
    print("Rq", Rq)