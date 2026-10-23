"""Lay qwen_base2.npz out for the GPU: slabs [in][out] int16 with their powers of two, the per-head gains, the tables."""
import json, os, numpy as np
M = np.load("qwen_base2.npz"); OUT = "qwen_gpu/"; os.makedirs(OUT, exist_ok=True); banks = []
for k in M.files:
    if k.endswith(".w"):
        n = k[:-2]; w = M[k]; np.ascontiguousarray(w).astype(np.int16).tofile(OUT + n + ".wc"); M[n + ".e"].astype(np.int32).tofile(OUT + n + ".e")
        banks.append(dict(name=n, nin=int(w.shape[0]), nout=int(w.shape[1])))
gains = []
for l in range(28):
    for g in ("qg", "kg"): M[f"{g}.{l}"].astype(np.int64).tofile(OUT + f"{g}.{l}.bin"); gains.append(int(M[f"{g[0]}ge.{l}"]))
M["embed"].astype(np.int32).tofile(OUT + "embed.bin")
for k in ("cos", "sin", "sig", "arrival", "log2", "exp2"): M[k].astype(np.int64).tofile(OUT + k + ".bin")
json.dump(dict(banks=banks, gain_exps=gains, eps40=int(M["eps40"]), arrival_len=int(len(M["arrival"]))), open(OUT + "manifest.json", "w"))
print(f"wrote {OUT}: {len(banks)} slabs")
