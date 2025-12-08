#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define MAX_F 8
#ifndef TILE_K
#define TILE_K 128
#endif

// -------------------------------------------------------------
// KERNEL
// -------------------------------------------------------------
__global__ void qkm_fused_multi_kernel(
    const float* __restrict__ phases,
    const float* __restrict__ fitness,
    const float* __restrict__ entropy,
    const float* __restrict__ psi_real,
    const float* __restrict__ psi_imag,
    const float* __restrict__ G,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    int N,
    int F,
    float T
) {
    extern __shared__ float shbuf[];

    int j = blockIdx.x;
    if (j >= N) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float acc_re[MAX_F];
    float acc_im[MAX_F];
    for (int a = 0; a < F; a++) {
        acc_re[a] = 0.f;
        acc_im[a] = 0.f;
    }

    float phi_j[MAX_F];
    float f_j[MAX_F];
    float e_j[MAX_F];
    float psi_jr[MAX_F];
    float psi_ji[MAX_F];

    for (int a = 0; a < F; a++) {
        int idx = j * F + a;
        phi_j[a] = phases[idx];
        f_j[a] = fitness[idx];
        e_j[a] = entropy[idx];
        psi_jr[a] = psi_real[idx];
        psi_ji[a] = psi_imag[idx];
    }

    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        int tile_sz = min(TILE_K, N - k0);

        int per_field = tile_sz * F;
        float* sh_phases   = shbuf;
        float* sh_fitness  = sh_phases   + per_field;
        float* sh_entropy  = sh_fitness  + per_field;
        float* sh_psir     = sh_entropy  + per_field;
        float* sh_psii     = sh_psir     + per_field;

        for (int t = tid; t < per_field; t += nthreads) {
            int kk = k0 + (t / F);
            int b  = t % F;
            int idx = kk * F + b;

            sh_phases[t]  = (kk < N) ? phases[idx]    : 0.f;
            sh_fitness[t] = (kk < N) ? fitness[idx]   : 0.f;
            sh_entropy[t] = (kk < N) ? entropy[idx]   : 0.f;
            sh_psir[t]    = (kk < N) ? psi_real[idx]  : 0.f;
            sh_psii[t]    = (kk < N) ? psi_imag[idx]  : 0.f;
        }
        __syncthreads();

        for (int local_k = tid; local_k < tile_sz; local_k += nthreads) {
            float phi_k[MAX_F], f_k[MAX_F], e_k[MAX_F], pr[MAX_F], pi[MAX_F];
            for (int b = 0; b < F; b++) {
                int idx = local_k * F + b;
                phi_k[b] = sh_phases[idx];
                f_k[b]   = sh_fitness[idx];
                e_k[b]   = sh_entropy[idx];
                pr[b]    = sh_psir[idx];
                pi[b]    = sh_psii[idx];
            }

            for (int a = 0; a < F; a++) {
                for (int b = 0; b < F; b++) {
                    float df = fabsf(f_j[a] - f_k[b]);
                    float de = fabsf(e_j[a] - e_k[b]);
                    float J = __expf(-df / (T + 1e-12f)) * __expf(-de);

                    float G_ab = G[a * F + b];

                    float dphi = phi_k[b] - phi_j[a];
                    float c = cosf(dphi);
                    float s = sinf(dphi);

                    float mul_re = c * pr[b] - s * pi[b];
                    float mul_im = c * pi[b] + s * pr[b];

                    acc_re[a] += J * G_ab * mul_re;
                    acc_im[a] += J * G_ab * mul_im;
                }
            }
        }
        __syncthreads();
    }

    int out_base = j * F;
    for (int a = 0; a < F; a++) {
        out_real[out_base + a] = acc_re[a];
        out_imag[out_base + a] = acc_im[a];
    }
}


// -------------------------------------------------------------
// WRAPPER
// -------------------------------------------------------------
std::vector<torch::Tensor> qkm_fused_multi_forward(
    torch::Tensor phases,
    torch::Tensor fitness,
    torch::Tensor entropy,
    torch::Tensor psi_real,
    torch::Tensor psi_imag,
    torch::Tensor G,
    float T
) {
    int N = phases.size(0);
    int F = phases.size(1);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(phases.device());
    auto out_r = torch::zeros({N, F}, opts);
    auto out_i = torch::zeros({N, F}, opts);

    dim3 blocks(N);
    int threads = 256;

    int per_field = TILE_K * F;
    int sh_tile_bytes = sizeof(float) * (5 * per_field);
    int shared_bytes = sh_tile_bytes;

    qkm_fused_multi_kernel<<<blocks, threads, shared_bytes>>>(
        phases.data_ptr<float>(),
        fitness.data_ptr<float>(),
        entropy.data_ptr<float>(),
        psi_real.data_ptr<float>(),
        psi_imag.data_ptr<float>(),
        G.data_ptr<float>(),
        out_r.data_ptr<float>(),
        out_i.data_ptr<float>(),
        N, F, T
    );

    return {out_r, out_i};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkm_fused_multi_forward", &qkm_fused_multi_forward,
        "Fused Multi-field QKM Forward (CUDA, sm_75)");
}