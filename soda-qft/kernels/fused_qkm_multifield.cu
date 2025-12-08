// fused_qkm_multifield.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define MAX_F 8   // support up to F=8 fields; increase if needed

// Shared memory tile size for K loop
#ifndef TILE_K
#define TILE_K 128
#endif

extern "C" {

// Kernel: each block handles one j (row), threads loop over k tiles.
// Uses shared memory tiling for phases/fitness/entropy/psi for k block to reduce global loads.
// Accumulates S_j^a for all a in registers and writes out per-j per-field results.
__global__ void qkm_fused_multi_kernel(
    const float* __restrict__ phases,   // (N * F) flattened: index = j*F + a
    const float* __restrict__ fitness,  // (N * F)
    const float* __restrict__ entropy,  // (N * F)
    const float* __restrict__ psi_real, // (N * F)
    const float* __restrict__ psi_imag, // (N * F)
    const float* __restrict__ G,        // (F * F)
    float* __restrict__ out_real,       // (N * F) output
    float* __restrict__ out_imag,       // (N * F)
    int N,
    int F,
    float T
) {
    extern __shared__ float shbuf[]; // dynamic shared mem
    // Layout of shbuf per tile: phases_k[F * TILE_K] | fitness_k[F * TILE_K] | entropy_k[F*TILE_K] | psi_real_k[F*TILE_K] | psi_imag_k[F*TILE_K]
    // To simplify indexing, we compute offsets dynamically.

    int j = blockIdx.x;
    if (j >= N) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // per-thread accumulators for all fields a
    // F <= MAX_F
    float acc_re[MAX_F];
    float acc_im[MAX_F];
    for (int a = 0; a < F; ++a) { acc_re[a] = 0.0f; acc_im[a] = 0.0f; }

    // load j's per-field values into registers
    float phi_j[MAX_F];
    float f_j[MAX_F];
    float e_j[MAX_F];
    float psi_jr[MAX_F];
    float psi_ji[MAX_F];
    for (int a = 0; a < F; ++a) {
        int idx = j * F + a;
        phi_j[a] = phases[idx];
        f_j[a] = fitness[idx];
        e_j[a] = entropy[idx];
        psi_jr[a] = psi_real[idx];
        psi_ji[a] = psi_imag[idx];
    }

    // tile over k dimension
    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        int tile_sz = min(TILE_K, N - k0);

        // compute shared memory offsets
        int per_field = tile_sz * F;
        float* sh_phases = shbuf;
        float* sh_fitness = sh_phases + per_field;
        float* sh_entropy = sh_fitness + per_field;
        float* sh_psi_real = sh_entropy + per_field;
        float* sh_psi_imag = sh_psi_real + per_field;
        // each array length = per_field

        // load tile into shared memory: each thread loads multiple elements
        for (int t = tid; t < per_field; t += nthreads) {
            int kk = k0 + (t / F);
            int b = t % F;
            int idx = kk * F + b;
            sh_phases[t] = (kk < N) ? phases[idx] : 0.0f;
            sh_fitness[t] = (kk < N) ? fitness[idx] : 0.0f;
            sh_entropy[t] = (kk < N) ? entropy[idx] : 0.0f;
            sh_psi_real[t] = (kk < N) ? psi_real[idx] : 0.0f;
            sh_psi_imag[t] = (kk < N) ? psi_imag[idx] : 0.0f;
        }
        __syncthreads();

        // each thread processes subset of k in this tile
        for (int local_k = tid; local_k < tile_sz; local_k += nthreads) {
            int k = k0 + local_k;
            // for each source field b, load values from shared memory
            float phi_k[MAX_F];
            float f_k[MAX_F];
            float e_k[MAX_F];
            float psi_kr[MAX_F];
            float psi_ki[MAX_F];
            for (int b = 0; b < F; ++b) {
                int idx_sh = local_k * F + b;
                phi_k[b] = sh_phases[idx_sh];
                f_k[b] = sh_fitness[idx_sh];
                e_k[b] = sh_entropy[idx_sh];
                psi_kr[b] = sh_psi_real[idx_sh];
                psi_ki[b] = sh_psi_imag[idx_sh];
            }

            // for each pair (a,b) accumulate contributions
            for (int a = 0; a < F; ++a) {
                for (int b = 0; b < F; ++b) {
                    // compute J_{jk}^{(a,b)}
                    float df = fabsf(f_j[a] - f_k[b]);
                    float de = fabsf(e_j[a] - e_k[b]);
                    float J = expf(- df / (T + 1e-12f)) * expf(- de);
                    float G_ab = G[a * F + b];

                    // phase difference
                    float dphi = phi_k[b] - phi_j[a];
                    float c = cosf(dphi);
                    float s = sinf(dphi);

                    // psi_k^b complex
                    float pr = psi_kr[b];
                    float pi = psi_ki[b];

                    // contribution = J * G_ab * exp(i dphi) * psi_kb
                    // compute (exp(i dphi) * psi_kb) first:
                    // (c + i s) * (pr + i pi) = (c*pr - s*pi) + i(c*pi + s*pr)
                    float mul_re = c * pr - s * pi;
                    float mul_im = c * pi + s * pr;

                    // scale by J and G_ab
                    float contrib_re = J * G_ab * mul_re;
                    float contrib_im = J * G_ab * mul_im;

                    acc_re[a] += contrib_re;
                    acc_im[a] += contrib_im;
                }
            }
        }
        __syncthreads();
    }

    // Reduce across threads in block: use shared mem buffers then thread0 writes result
    // Use shared mem region at start of shbuf
    float* red_re = shbuf; // reuse
    float* red_im = red_re + F * nthreads;
    // Ensure enough shared memory allocated by caller: F * nthreads * 2 floats
    for (int a = 0; a < F; ++a) {
        // each thread writes its acc into shared
        red_re[a * nthreads + tid] = acc_re[a];
        red_im[a * nthreads + tid] = acc_im[a];
    }
    __syncthreads();

    if (tid == 0) {
        // sum over threads
        for (int a = 0; a < F; ++a) {
            double s_re = 0.0;
            double s_im = 0.0;
            for (int t = 0; t < nthreads; ++t) {
                s_re += (double)red_re[a * nthreads + t];
                s_im += (double)red_im[a * nthreads + t];
            }
            int out_idx = j * F + a;
            out_real[out_idx] = (float)s_re;
            out_imag[out_idx] = (float)s_im;
        }
    }
}

// launcher
std::vector<torch::Tensor> qkm_fused_multi_forward(
    torch::Tensor phases,        // (N,F) float
    torch::Tensor fitness,       // (N,F)
    torch::Tensor entropy,       // (N,F)
    torch::Tensor psi_real,      // (N,F)
    torch::Tensor psi_imag,      // (N,F)
    torch::Tensor G,             // (F,F)
    float T
) {
    const int N = phases.size(0);
    const int F = phases.size(1);
    TORCH_CHECK(F <= MAX_F, "F exceeds MAX_F");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(phases.device());
    auto out_real = torch::zeros({N, F}, opts);
    auto out_imag = torch::zeros({N, F}, opts);

    int threads = 256;
    dim3 blocks(N);
    // compute required shared memory size:
    // per tile arrays: per_field = TILE_K * F
    int per_field = TILE_K * F;
    // shared for phases/fitness/entropy/psi_real/psi_imag => 5 * per_field floats
    int sh_tile_bytes = sizeof(float) * (5 * per_field);
    // reduction shared: F * threads * 2 floats
    int sh_red_bytes = sizeof(float) * (2 * F * threads);
    size_t shared_bytes = sh_tile_bytes + sh_red_bytes;

    qkm_fused_multi_kernel<<<blocks, threads, shared_bytes>>>(
        phases.data_ptr<float>(),
        fitness.data_ptr<float>(),
        entropy.data_ptr<float>(),
        psi_real.data_ptr<float>(),
        psi_imag.data_ptr<float>(),
        G.data_ptr<float>(),
        out_real.data_ptr<float>(),
        out_imag.data_ptr<float>(),
        N,
        F,
        T
    );
    cudaDeviceSynchronize();

    return {out_real, out_imag};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkm_fused_multi_forward", &qkm_fused_multi_forward, "Fused QKM Multi-field Forward");
}