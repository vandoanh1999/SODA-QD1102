# compile once as above, then
from soda_qft_engine_gpu import SODA_QFT_GPU_Engine
import torch
engine = SODA_QFT_GPU_Engine(N=2048, F=3, D=20)
# bootstrap G from synthetic field embeddings (F,E)
field_emb = torch.randn(3, 128)  # e.g., aggregated semantic footprints
engine.bootstrap_G_from_semantics(field_emb, method="cosine", base_scale=1.0)

X = torch.randn(16, 20, device=engine.W.device)
y = (X**1.5).sum(dim=1).to(engine.W.device)
out = engine.step(X, y)
print(out)