"""DEVELOP family: the lens between the entry table E and the top table T, written ONCE as whole numbers -- the last
water level's gain (GPT-2 ln_f.weight) read from the checkpoint and rounded at 2^24, the same one-time float read the
converter (base2_convert.py) does for every other number. After this, printing T from E is whole numbers only:
    s = G x E ;  c = 768 s - (sum of s over the word's 768 lines) ;  y = (c + 2^23) >> 24 ;  T = (y x 1431655766 + 2^43) >> 44
(the centring with no divide; 768 x 16 = 3 x 4096, and 1431655766 = 2^32 / 3 rounded up).   python3 develop_gain.py
"""
import json, struct, numpy as np
p = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
a, b = h["ln_f.weight"]["data_offsets"]; f.seek(8 + n + a)
g = np.frombuffer(f.read(b - a), "<f4").astype(np.float64)
G = np.round(g * 2 ** 24).astype(np.int64)
np.save("/Users/abundancemachine/streaming-machine/shrink/develop_gain24.npy", G)
print(f"768 gains, whole numbers at 2^24: {G.min()} .. {G.max()} ({int(G.max()).bit_length()} bits each); "
      f"places: {np.bincount(np.array([int(x).bit_length() - 1 for x in G]))}")
