"""The last water level's gain (GPT-2 ln_f.weight), read ONCE from the checkpoint and written as whole numbers at
2^24 -- the same one-time float read the converter (base2_convert.py) does. Everything after this is whole numbers.
It is side information of the top-table codec (768 numbers, counted in its size).    python3 fax_gain.py
"""
import json, struct, numpy as np
p = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
a, b = h["ln_f.weight"]["data_offsets"]; f.seek(8 + n + a)
g = np.frombuffer(f.read(b - a), "<f4").astype(np.float64)
G = np.round(g * 2 ** 24).astype(np.int64)
np.save("/Users/abundancemachine/streaming-machine/shrink/fax_gain24.npy", G)
print(f"768 gains, whole numbers at 2^24: {G.min()} .. {G.max()} ({int(G.max()).bit_length()} bits each)")
