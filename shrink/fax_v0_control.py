"""FAX family, version 0 (the control): store EVERY pixel of every place's page, no position coding at all.
  a) the pages bit-packed as they are (sign page + one page per place): the fixed-width baseline
  b) the same packed pages through lzma (its own LZ + adaptive range coder finds any repeats or skew)
  c) the whole numbers as 4-byte little-endian words through lzma (no pages at all)
Decoded back and checked equal.      python3 fax_v0_control.py E|T|fc.0
"""
import sys, json, time, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); N = A.size; nb = nplanes(A); a = np.abs(A)
res = []
t = time.time()
pages = [raw_bits((A < 0).ravel())] + [raw_bits(((a >> k) & 1).ravel()) for k in range(nb)]
raw = b"".join(pages); del pages
res.append(report("v0a every pixel, pages packed", name, len(raw), 0, N, True))
z = lzma.compress(raw, preset=9); del raw
pb = (N + 7) // 8; u = lzma.decompress(z)
B = np.zeros(A.shape, np.int64)
for k in range(nb): B |= unraw_bits(u[pb * (k + 1): pb * (k + 2)], N).reshape(A.shape).astype(np.int64) << k
B[unraw_bits(u[:pb], N).reshape(A.shape)] *= -1; del u
res.append(report("v0b every pixel, pages packed, lzma", name, len(z), 32, N, bool(np.array_equal(B, A))))
del B
z = lzma.compress(A.astype("<i4").tobytes(), preset=9)
B = np.frombuffer(lzma.decompress(z), "<i4").reshape(A.shape)
res.append(report("v0c whole numbers as 4-byte words, lzma", name, len(z), 0, N, bool(np.array_equal(B, A))))
print(f"  ({time.time() - t:.0f} s)")
json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v0_{name}.json", "w"))
