from torch.utils.cpp_extension import load
fused_mod = load(
    name="fused_qkm_multifield",
    sources=["fused_qkm_multifield.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
    verbose=True
)