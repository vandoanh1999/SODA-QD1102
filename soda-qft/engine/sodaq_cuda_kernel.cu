// saodq_cuda_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

// Compute for each j: coupling_real[j], coupling_imag[j]
// Inputs are device pointers to floats.
// N: number of nuclei
// quantum_k: scalar
// T: temperature scalar
// Note: this kernel uses one thread per j and loops k in global memory.
// For better perf, can tile into shared memory (optional optimization).
__global__ void compute_coupling_kernel(
    const float *fitness,    // [N]
    const float *entropy,    // [N]
    const float *phi,        // [N] phases
    const float *psi_real,   // [N] real part of psi_old
    const float *psi_imag,   // [N] imag part
    const float *spike,      // [N]
    int N,
    float quantum_k,
    float T,
    float *couple_real,      // [N] output
    float *couple_imag       // [N] output
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    float fj = fitness[j];
    float ej = entropy[j];
    float phij = phi[j];

    double acc_re = 0.0;
    double acc_im = 0.0;

    // loop over k
    for (int k = 0; k < N; ++k) {
        float dk_f = fabsf(fj - fitness[k]);
        float dk_e = fabsf(ej - entropy[k]);

        float J = expf(- dk_f / (T + 1e-12f)) * expf(- dk_e);

        // phase diff = phi_k - phi_j
        float pd = phi[k] - phij;
        // sincosf for speed
        float s = __sinf(pd);
        float c = __cosf(pd);

        // contribution: J * exp(i*pd)
        // (real, imag) = J * (c, s)
        acc_re += (double)(J * c);
        acc_im += (double)(J * s);
    }

    // multiply by quantum_k at host-level update step; we return coupling only
    couple_real[j] = (float)acc_re;
    couple_imag[j] = (float)acc_im;
}

// After compute coupling, update psi_new in-place (vectorized kernel).
// psi_old_real/imag -> psi_new_real/imag
// delta_omega is float per-j
__global__ void update_psi_kernel(
    const float *psi_old_real,
    const float *psi_old_imag,
    const float *couple_real,
    const float *couple_imag,
    const float *delta_omega, // [N]
    float quantum_k,
    int N,
    float *psi_new_real,
    float *psi_new_imag
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    float orr = psi_old_real[j];
    float oi = psi_old_imag[j];

    // coupling (complex)
    float cr = couple_real[j];
    float ci = couple_imag[j];

    // psi_new = psi_old + quantum_k * coupling + delta_omega * psi_old
    // compute quantum_k * coupling
    float qr = quantum_k * cr;
    float qi = quantum_k * ci;

    float domega = delta_omega[j];

    // psi_new = (orr + qr + domega * orr) + i(oi + qi + domega * oi)
    float nr = orr + qr + domega * orr;
    float ni = oi + qi + domega * oi;

    psi_new_real[j] = nr;
    psi_new_imag[j] = ni;
}

} // extern "C"