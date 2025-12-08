#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

__global__ void qkm_fused_multi_kernel(
    const float* phases,     // (N*F) flattened [j*F + a]
    const float* fitness,    // (N*F)
    const float* entropy,    // (N*F)
    const float* G,          // (F*F) inter-field matrix
    float* out_real,         // (N*F) output
    float* out_imag,
    int N,
    int F,
    float T
) {
    int j = blockIdx.x;
    if (j >= N) return;

    // allocate per-thread accumulators in registers for all fields (a)
    extern __shared__ float s_buf[]; // optional shared mem for reduction if needed

    // accumulators per a: use double precision accumulators split into real/imag arrays
    // For simplicity assume F small <= 8; so allocate fixed small arrays
    float acc_re[8]; // support up to F=8
    float acc_im[8];
    for (int a=0;a<F;++a){ acc_re[a]=0.0f; acc_im[a]=0.0f; }

    // load j-field values
    // phases_j_a = phases[j*F + a]
    // f_j_a = fitness[j*F + a]
    // e_j_a = entropy[j*F + a]
    float phi_j_a[8];
    float f_j_a[8];
    float e_j_a[8];
    for (int a=0; a<F; ++a) {
        int idx = j*F + a;
        phi_j_a[a] = phases[idx];
        f_j_a[a] = fitness[idx];
        e_j_a[a] = entropy[idx];
    }

    // loop over k
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        // load k-field values once
        float phi_k_b[8];
        float f_k_b[8];
        float e_k_b[8];
        for (int b=0;b<F;++b){
            int idxk = k*F + b;
            phi_k_b[b] = phases[idxk];
            f_k_b[b] = fitness[idxk];
            e_k_b[b] = entropy[idxk];
        }

        // for each pair (a,b) accumulate
        for (int a=0; a<F; ++a){
            for (int b=0; b<F; ++b){
                float df = fabsf(f_j_a[a] - f_k_b[b]);
                float de = fabsf(e_j_a[a] - e_k_b[b]);
                float J = expf(- df / T) * expf(- de);
                float dphi = phi_k_b[b] - phi_j_a[a];
                float c = cosf(dphi);
                float s = sinf(dphi);
                float G_ab = G[a*F + b];

                // assume psi_kb is encoded via phase (magnitude 1) or provided separately if needed
                // For generality, if psi provided, multiply here. For simplification, assume psi magnitude 1:
                acc_re[a] += (J * G_ab) * c; // * |psi_kb|
                acc_im[a] += (J * G_ab) * s;
            }
        }
    }

    // reduce across threads in block (shared mem)
    // store local acc into shared memory arrays and perform reduction; then thread 0 writes outputs.
    // For brevity: do simple atomic add to global output (works but slower); better: shared mem + reduction.
    for (int a=0; a<F; ++a){
        atomicAdd(&out_real[j*F + a], acc_re[a]);
        atomicAdd(&out_imag[j*F + a], acc_im[a]);
    }
}

} // extern "C"