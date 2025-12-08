# fused_qkm_multifield.py
import torch
from torch.autograd import Function
import importlib

# compile/load extension with torch.utils.cpp_extension.load if not prebuilt
# Example (run once):
# from torch.utils.cpp_extension import load
# fused_mod = load(name="fused_qkm_multifield", sources=["fused_qkm_multifield.cu"], verbose=True)

# Assume compiled as fused_qkm_multifield
import fused_qkm_multifield as fused_mod

class FusedQKMMulti(Function):
    @staticmethod
    def forward(ctx, phases, fitness, entropy, psi_complex, G, T):
        # phases, fitness, entropy: (N,F) float32
        # psi_complex: complex tensor (N,F) cfloat -> split
        ctx.save_for_backward(phases, fitness, entropy, psi_complex, G)
        psi_real = psi_complex.real.contiguous().float()
        psi_imag = psi_complex.imag.contiguous().float()

        out_real, out_imag = fused_mod.qkm_fused_multi_forward(
            phases.contiguous(), fitness.contiguous(), entropy.contiguous(),
            psi_real, psi_imag, G.contiguous(), float(T)
        )
        out = torch.complex(out_real, out_imag)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Not implemented: engine uses forward-only (no gradients)
        return None, None, None, None, None, None

def qkm_fused_multi(phases, fitness, entropy, psi_complex, G, T):
    return FusedQKMMulti.apply(phases, fitness, entropy, psi_complex, G, T)