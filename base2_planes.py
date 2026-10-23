"""Lay base2_gpt2.npz out for the GPU: every weight bank as bit planes, nothing else changed.
A plane is one binary digit (0..15, 15 = the two's complement top digit) of every weight in a column, 32 inputs per
32-bit word:  planes[column][digit][word].  Everything else is written as raw whole numbers.
    python3 base2_planes.py
"""
import json, numpy as np
M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
OUT = "/Users/abundancemachine/streaming-machine/base2_gpu/"
import os; os.makedirs(OUT, exist_ok=True)

def planes(w16):                                   # [nin][nout] int16 -> [nout][16][nin/32] uint32
    nin, nout = w16.shape
    u = w16.astype(np.int64) & 0xFFFF
    bits = ((u[None, :, :] >> np.arange(16)[:, None, None]) & 1).astype(np.uint64)      # [16][nin][nout]
    bits = bits.reshape(16, nin // 32, 32, nout)
    words = (bits << np.arange(32, dtype=np.uint64)[None, None, :, None]).sum(2).astype(np.uint32)   # [16][nin/32][nout]
    return np.ascontiguousarray(words.transpose(2, 0, 1))

man = []
for name in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    w = M[name + ".w"]; nin, nout = w.shape
    planes(w).tofile(OUT + f"{name}.planes")
    M[name + ".e"].astype(np.int32).tofile(OUT + f"{name}.e")
    M[name + ".b"].astype(np.int64).tofile(OUT + f"{name}.b")
    man.append(dict(name=name, nin=nin, nout=nout))
for k in ("wte", "wpe", "phi", "arrival", "log2", "exp2"):
    M[k].astype(np.int64).tofile(OUT + f"{k}.bin")
json.dump(dict(banks=man, log2_768=int(M["log2_768"]), eps40=int(M["eps40"]), arrival_len=int(len(M["arrival"]))), open(OUT + "manifest.json", "w"))
print(f"wrote {OUT}: {len(man)} banks as bit planes")
