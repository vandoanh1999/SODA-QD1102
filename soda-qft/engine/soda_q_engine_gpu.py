import torch
import math
import random
from sodaq_ext import compute_and_update   # CUDA extension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NucleusGPU:
    def __init__(self, id):
        self.id = id
        self.fitness = random.uniform(1000, 9000)
        self.entropy = random.uniform(1.0, 4.0)
        self.phase = random.uniform(0, 2 * math.pi)
        self.spike_count = random.randint(1, 5)

    def update_spikes(self):
        lam = max(0.5, self.spike_count * 0.8)
        self.spike_count = torch.poisson(torch.tensor([lam])).item() + 1


class SODAEngineGPU:
    def __init__(self, N=128, quantum_k=1.0):
        self.N = N
        self.quantum_k = quantum_k

        # Initialize N nuclei
        self.nuclei = [NucleusGPU(f"n{j}") for j in range(N)]

        # allocate tensors on GPU
        self.psi = torch.exp(1j * (2 * math.pi * torch.rand(N, device=device))).to(torch.cfloat)
        self.phases = torch.angle(self.psi).real

        self.fitness = torch.tensor([n.fitness for n in self.nuclei], device=device).float()
        self.entropy = torch.tensor([n.entropy for n in self.nuclei], device=device).float()
        self.spikes = torch.tensor([n.spike_count for n in self.nuclei], device=device).float()

    def step(self):
        # update spikes & refresh tensor
        for idx, n in enumerate(self.nuclei):
            n.update_spikes()
            self.spikes[idx] = n.spike_count

        # compute psi_new using CUDA fused kernel
        psi_new_real, psi_new_imag, couple_r, couple_i, Rq = compute_and_update(
            self.fitness, self.entropy, self.phases, self.psi.real, self.psi.imag, self.spikes, float(self.quantum_k)
        )

        psi_new = torch.complex(psi_new_real, psi_new_imag)

        # update local state
        self.psi = psi_new
        self.phases = torch.angle(self.psi).real

        # update Nucleus objects
        psi_np_real = psi_new_real.detach().cpu().numpy()
        psi_np_imag = psi_new_imag.detach().cpu().numpy()

        for j, n in enumerate(self.nuclei):
            n.phase = math.atan2(psi_np_imag[j], psi_np_real[j])

        # small fitness/entropy updates to resemble learning
        self.fitness *= (1.0 - 0.0005 * torch.rand(1, device=device))
        self.entropy *= (1.0 + 0.0005 * (2 * torch.rand(1, device=device) - 1))

        return Rq

    def run(self, steps=200):
        R_history = []
        for t in range(steps):
            Rq = self.step()
            R_history.append(Rq)
        return R_history