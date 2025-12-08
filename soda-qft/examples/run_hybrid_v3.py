from soda_hybrid_v3_gpu import SODAEngineGPU
engine = SODAEngineGPU(dim=20, n_init=20)
engine.run(generations=300)