N=1024; F=3; D=20
engine = SODA_QFT_Py(N=N, F=F, D=D, quantum_k=1.0)
X = torch.randn(16, D, device=device)
y = (X**1.5).sum(dim=1).to(device)
out = engine.step(X, y)
print("R_q", out["R_q"], "S shape", out["S"].shape)