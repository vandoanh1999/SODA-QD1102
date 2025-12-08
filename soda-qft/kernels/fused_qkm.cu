#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define IDX(j, k, N) ((j) * (N) + (k))

// ==========================================
// FUSED QKM COUPLING KERNEL
// Each block computes S_j, one j per block.
// ==========================================
__global__ void qkm_fused_kernel(
    const float* __restrict__ phases,     // (N)
    const float* __restrict__ fitness,    // (N)
    const float* __restrict__ entropy,    // (N)
    float* __restrict__ out_real,         // (N)
    float* __restrict__ out_imag,         // (N)
    const int N,
    const float T
){
    int j = blockIdx.x;                     // each block handles a row j
    if (j >= N) return;

    float phi_j = phases[j];
    float f_j = fitness[j];
    float e_j = entropy[j];

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    // each thread processes multiple ks
    for (int k = threadIdx.x; k < N; k += blockDim.x) {

        float df = fabsf(f_j - fitness[k]);   // |f_j - f_k|
        float de = fabsf(e_j - entropy[k]);   // |H_j - H_k|

        // compute coupling coefficient J_jk
        float J = expf(-df / T) * expf(-de);

        // compute phase difference exp(i*(phi_k - phi_j))
        float dphi = phases[k] - phi_j;
        float c = cosf(dphi);
        float s = sinf(dphi);

        // accumulate
        sum_real += J * c;
        sum_imag += J * s;
    }

    // reduce within block
    __shared__ float buf_real[256];
    __shared__ float buf_imag[256];

    int tid = threadIdx.x;
    buf_real[tid] = sum_real;
    buf_imag[tid] = sum_imag;
    __syncthreads();

    // parallel reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf_real[tid] += buf_real[tid + stride];
            buf_imag[tid] += buf_imag[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_real[j] = buf_real[0];
        out_imag[j] = buf_imag[0];
    }
}


// WRAPPER
std::vector<torch::Tensor> qkm_fused_forward(
    torch::Tensor phases,
    torch::Tensor fitness,
    torch::Tensor entropy,
    float T
) {
    const int N = phases.size(0);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(phases.device());
    torch::Tensor out_real = torch::zeros({N}, opts);
    torch::Tensor out_imag = torch::zeros({N}, opts);

    const dim3 blocks(N);
    const dim3 threads(256);

    qkm_fused_kernel<<<blocks, threads>>>(
        phases.data_ptr<float>(),
        fitness.data_ptr<float>(),
        entropy.data_ptr<float>(),
        out_real.data_ptr<float>(),
        out_imag.data_ptr<float>(),
        N,
        T
    );

    return {out_real, out_imag};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkm_fused_forward", &qkm_fused_forward, "Fused Quantum Kuramoto Coupling");
}