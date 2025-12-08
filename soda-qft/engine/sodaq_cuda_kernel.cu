// saodq_cuda_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

extern "C" {

__global__ void compute_coupling_kernel_device(
    const float *fitness, const float *entropy, const float *phi,
    const float *psi_real, const float *psi_imag, const float *spike,
    int N, float quantum_k, float T, float *couple_real, float *couple_imag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float fj = fitness[j];
    float ej = entropy[j];
    float phij = phi[j];
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int k = 0; k < N; ++k) {
        float df = fabsf(fj - fitness[k]);
        float de = fabsf(ej - entropy[k]);
        float J = expf(- df / (T + 1e-12f)) * expf(- de);
        float pd = phi[k] - phij;
        float s = __sinf(pd);
        float c = __cosf(pd);
        acc_re += (double)(J * c);
        acc_im += (double)(J * s);
    }
    couple_real[j] = (float)acc_re;
    couple_imag[j] = (float)acc_im;
}

__global__ void update_psi_kernel_device(
    const float *psi_old_real, const float *psi_old_imag,
    const float *couple_real, const float *couple_imag,
    const float *delta_omega, float quantum_k, int N,
    float *psi_new_real, float *psi_new_imag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float orr = psi_old_real[j];
    float oi = psi_old_imag[j];
    float cr = couple_real[j];
    float ci = couple_imag[j];
    float qr = quantum_k * cr;
    float qi = quantum_k * ci;
    float domega = delta_omega[j];
    float nr = orr + qr + domega * orr;
    float ni = oi + qi + domega * oi;
    psi_new_real[j] = nr;
    psi_new_imag[j] = ni;
}

// Extern C launchers
void compute_coupling_kernel(
    const float *fitness, const float *entropy, const float *phi,
    const float *psi_real, const float *psi_imag, const float *spike,
    int N, float quantum_k, float T, float *couple_real, float *couple_imag) {

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_coupling_kernel_device<<<blocks, threads>>>(
        fitness, entropy, phi, psi_real, psi_imag, spike, N, quantum_k, T, couple_real, couple_imag);
    cudaDeviceSynchronize();
}

void update_psi_kernel(
    const float *psi_old_real, const float *psi_old_imag,
    const float *couple_real, const float *couple_imag,
    const float *delta_omega, float quantum_k, int N,
    float *psi_new_real, float *psi_new_imag) {

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_psi_kernel_device<<<blocks, threads>>>(
        psi_old_real, psi_old_imag, couple_real, couple_imag, delta_omega, quantum_k, N, psi_new_real, psi_new_imag);
    cudaDeviceSynchronize();
}

} // extern "C"