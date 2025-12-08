from soda_q_engine_gpu import SODAEngineGPU
import time

engine = SODAEngineGPU(N=512, quantum_k=1.0)

start = time.time()
R_hist = engine.run(steps=200)
elapsed = time.time() - start

print("Completed 200 steps in", elapsed, "seconds")
print("Average R_q =", sum(R_hist)/len(R_hist))