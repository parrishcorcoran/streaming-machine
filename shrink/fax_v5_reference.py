"""FAX family, version 5: plain base-2 pages coded from the top place down, the page above as the reference line
(the way a fax's 2-D mode codes a line against the line above it). At each place the pixels split in two:
  - values already lit higher up: this pixel is their next digit -> stored as it is, one bit each (a lit-pixel list
    would cost more: at half lit, a position costs 2 bits per lit pixel = 1 bit per pixel anyway)
  - values still dark: a lit pixel here is where the value's light STARTS -> sparse -> only these lit pixels are stored,
    as Rice-coded gaps over the still-dark pixels
plus one sign bit per lit value. Decoded and checked equal.      python3 fax_v5_reference.py E|T|fc.0
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

def code(A):
    S = A.ravel(); a = np.abs(S); N = S.size; nb = nplanes(S)
    sig = np.zeros(N, bool); chunks = []; side = 2 * 32 + 5; nbytes = 0
    for k in range(nb - 1, -1, -1):
        bit = ((a >> k) & 1).astype(bool)
        ref = raw_bits(bit[sig]); buf, n, kk, s = gaps_encode(bit[~sig])
        chunks.append((ref, buf, n, kk)); side += s; nbytes += len(ref) + len(buf); sig |= bit
    sg = raw_bits(S[S != 0] < 0); nbytes += len(sg)
    return (chunks, sg, nb, N), nbytes, side

def decode(pkg, shape):
    chunks, sg, nb, N = pkg; a = np.zeros(N, np.int32); sig = np.zeros(N, bool)
    for i, (ref, buf, n, kk) in enumerate(chunks):
        k = nb - 1 - i; bit = np.zeros(N, bool)
        bit[sig] = unraw_bits(ref, int(sig.sum())); bit[~sig] = gaps_decode(buf, n, kk, N - int(sig.sum()))
        a |= bit.astype(np.int32) << k; sig |= bit
    nz = np.flatnonzero(a); a[nz[unraw_bits(sg, len(nz))]] *= -1
    return a.reshape(shape)

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); t = time.time()
pkg, nbytes, side = code(A)
ok = bool(np.array_equal(decode(pkg, A.shape), A))
res = [report("v5 page above as reference (start pixels only)", name, nbytes, side, A.size, ok)]
print(f"     ({time.time() - t:.0f} s)")
json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v5_{name}.json", "w"))
