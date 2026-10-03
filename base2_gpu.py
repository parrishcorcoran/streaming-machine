"""Lay base2_gpt2.npz out for the GPU, nothing else changed: every weight bank input-major ([input][output], 16 bits),
its powers of two and its biases, the tables, and a manifest.
    python3 base2_gpu.py
"""
import json, os, numpy as np
M = np.load("base2_gpt2.npz")
OUT = "base2_gpu/"; os.makedirs(OUT, exist_ok=True)
man = []
for name in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    w = M[name + ".w"]; nin, nout = w.shape
    np.ascontiguousarray(w).astype(np.int16).tofile(OUT + f"{name}.wc")
    M[name + ".e"].astype(np.int32).tofile(OUT + f"{name}.e")
    M[name + ".b"].astype(np.int64).tofile(OUT + f"{name}.b")
    man.append(dict(name=name, nin=nin, nout=nout))
for k in ("wte", "wpe", "phi", "arrival", "log2", "exp2"):
    M[k].astype(np.int64).tofile(OUT + f"{k}.bin")
json.dump(dict(banks=man, log2_768=int(M["log2_768"]), eps40=int(M["eps40"]), arrival_len=int(len(M["arrival"]))), open(OUT + "manifest.json", "w"))
print(f"wrote {OUT}: {len(man)} banks")
