import torch
from fused_qkm import qkm_fused

device = 'cuda'

N = 4096

phases  = torch.rand(N, device=device) * 2 * torch.pi
fitness = torch.rand(N, device=device) * 10000
entropy = torch.rand(N, device=device) * 4.0

T = 1.0 + torch.std(fitness)

coupling = qkm_fused(phases, fitness, entropy, T)
print(coupling.shape)   # (N,)
print(coupling[:5])