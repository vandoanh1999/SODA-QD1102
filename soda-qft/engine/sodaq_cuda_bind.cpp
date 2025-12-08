// saodq_cuda_bind.cpp
#include <torch/extension.h>
#include <vector>

// declare kernels
void compute_coupling_kernel(const float*, const float*, const float*, const float*, const float*, const float*, int, float, float, float*, float*);
void update_psi_kernel(const float*, const float*, const float*, const float*, const float*, float, int, float*, float*);

// wrappers
std::vector<torch::Tensor> sodaq_compute_and_update(
    torch::Tensor fitness,
    torch::Tensor entropy,
    torch::Tensor phi,
    torch::Tensor psi_real,
    torch::Tensor psi_imag,
    torch::Tensor spike,
    float quantum_k
) {
    // Ensure tensors are contiguous and on CUDA
    auto fitness_c = fitness.contiguous();
    auto entropy_c = entropy.contiguous();
    auto phi_c = phi.contiguous();
    auto psi_r_c = psi_real.contiguous();
    auto psi_i_c = psi_imag.contiguous();
    auto spike_c = spike.contiguous();

    int N = fitness.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(fitness.device());
    auto couple_real = torch::zeros({N}, options);
    auto couple_imag = torch::zeros({N}, options);
    auto psi_new_real = torch::zeros({N}, options);
    auto psi_new_imag = torch::zeros({N}, options);

    // Launch kernel parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Temperature T
    float T = 1.0f + fitness.std().item<float>();

    // Get raw pointers
    float* f_ptr = (float*)fitness_c.data_ptr<float>();
    float* e_ptr = (float*)entropy_c.data_ptr<float>();
    float* phi_ptr = (float*)phi_c.data_ptr<float>();
    float* psi_r_ptr = (float*)psi_r_c.data_ptr<float>();
    float* psi_i_ptr = (float*)psi_i_c.data_ptr<float>();
    float* spike_ptr = (float*)spike_c.data_ptr<float>();
    float* couple_r_ptr = (float*)couple_real.data_ptr<float>();
    float* couple_i_ptr = (float*)couple_imag.data_ptr<float>();
    float* psi_new_r_ptr = (float*)psi_new_real.data_ptr<float>();
    float* psi_new_i_ptr = (float*)psi_new_imag.data_ptr<float>();

    // Call kernels (via launcher)
    // Use CUDA launch from C++ -> we need to use ATen's CUDA API or rely on extern "C" functions compiled with default kernel launches.
    // For simplicity, call wrapper functions that internally launch kernels (implemented in .cu with grid/thread launches).
    // We'll declare them as extern "C" above and link.
    compute_coupling_kernel(f_ptr, e_ptr, phi_ptr, psi_r_ptr, psi_i_ptr, spike_ptr, N, quantum_k, T, couple_r_ptr, couple_i_ptr);
    update_psi_kernel(psi_r_ptr, psi_i_ptr, couple_r_ptr, couple_i_ptr, spike_ptr, quantum_k, N, psi_new_r_ptr, psi_new_i_ptr);

    // Now normalize psi_new (sum |psi|^2)
    auto abs2 = psi_new_real.mul(psi_new_real) + psi_new_imag.mul(psi_new_imag);
    float norm = std::sqrt(abs2.sum().item<float>() + 1e-12f);
    psi_new_real = psi_new_real.div(norm);
    psi_new_imag = psi_new_imag.div(norm);

    // Compute R_q = |sum psi_new|
    auto sum_re = psi_new_real.sum();
    auto sum_im = psi_new_imag.sum();
    auto R_q = torch::sqrt(sum_re.mul(sum_re) + sum_im.mul(sum_im));

    return {psi_new_real, psi_new_imag, couple_real, couple_imag, R_q};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_and_update", &sodaq_compute_and_update, "SODA-Q compute coupling and update (CUDA)");
}