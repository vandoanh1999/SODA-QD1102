import torch
from torch.autograd import Function
import fused_qkm_cuda  # compiled CUDA extension

class FusedQKMCoupling(Function):

    @staticmethod
    def forward(ctx, phases, fitness, entropy, T):
        # Save for backward if needed
        ctx.save_for_backward(phases, fitness, entropy)
        ctx.T = T

        # Call CUDA kernel
        out_real, out_imag = fused_qkm_cuda.qkm_fused_forward(
            phases.contiguous(),
            fitness.contiguous(),
            entropy.contiguous(),
            float(T)
        )
        return torch.complex(out_real, out_imag)

    @staticmethod
    def backward(ctx, grad_output):
        # No backward for now (we don’t need gradients in SODA)
        return None, None, None, None


def qkm_fused(phases, fitness, entropy, T):
    return FusedQKMCoupling.apply(phases, fitness, entropy, T)