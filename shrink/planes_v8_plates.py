"""PLANES family, version 8: the same bit-plane coder on all 48 plates (qkv/proj/fc/mproj x 12, int16, stored as
(lines in, neurons out)). Each plate:
  a) version 1 contexts (higher places only)                          -- the plain chain rule
  b) version 7b: flat field per neuron (dark level = mean of the first 2^m lines in, a shift) + higher places + the
     neuron (column) number + the line-in meter (how many neurons of this line are lit above this place) + line-aware
     places below the top + sign context (neuron x top place)
Here "line" in the coder is the neuron out (each neuron's column has its own power of two), and the "word" meter
runs along the line in (each line in carries its layer-norm gain folded in).
Prediction: like E and T, every plate's places below ~3 under the top are coin flips, so the plates land at their
per-neuron floor, about 14.5 bits a value (stored at 16); b) a few tenths better than a), mostly from the line-in
meter, because the folded layer-norm gain makes some lines in much brighter than others.
    python3 planes_v8_plates.py
"""
import sys, json; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
import planes_lib as PL

def H(c):
    c = c[c > 0].astype(np.float64); p = c / c.sum(); return float(-(c * np.log2(p)).sum())

names = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
tot = dict(a=0, b=0, f0=0.0, n=0); rows = []
for nm in names:
    A = load(nm); R, C = A.shape
    _, cnt = np.unique(A, return_counts=True); f0 = H(cnt)
    ra, _ = run("v8a plate, higher places", nm, A, M_HIGH)
    n2 = 1 << (R.bit_length() - 1); sh = n2.bit_length() - 1
    off = ((A[:n2].astype(np.int64).sum(0) + (n2 >> 1)) >> sh).astype(np.int32)
    side = C * (int(np.abs(off).max()).bit_length() + 1)
    rb, data = run("v8b plate, flat field + v6b contexts", nm, A - off[None, :], M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE, extra_side=side)
    nb = nplanes(A - off[None, :]); B = np.zeros_like(A); m2 = np.zeros(nb + 1, np.int64)
    lib().planes(1, B.ctypes.data, R, C, nb, M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE, None, data.ctypes.data, len(data), m2.ctypes.data)
    ok = bool(np.array_equal(B + off[None, :], A))
    print(f"   {nm}: order-0 floor {f0 / A.size:.4f}; from bytes + dark levels back equal: {ok}", flush=True)
    tot["a"] += ra["bits_per_value"] * A.size; tot["b"] += rb["bits_per_value"] * A.size; tot["f0"] += f0; tot["n"] += A.size
    rows.append(dict(plate=nm, order0=round(f0 / A.size, 4), v8a=ra["bits_per_value"], v8b=rb["bits_per_value"], ok_a=ra["roundtrip_ok"], ok_b=ok and rb["roundtrip_ok"]))
n = tot["n"]
summary = dict(values=n, stored_16bit_MB=round(n * 16 / 8e6, 3), order0_bits=round(tot["f0"] / n, 4),
               v8a_bits=round(tot["a"] / n, 4), v8a_MB=round(tot["a"] / 8e6, 3), v8b_bits=round(tot["b"] / n, 4), v8b_MB=round(tot["b"] / 8e6, 3),
               all_roundtrip_ok=all(r["ok_a"] and r["ok_b"] for r in rows), plates=rows)
json.dump(summary, open("/Users/abundancemachine/streaming-machine/shrink/planes_out_v8_plates.json", "w"), indent=1)
print(f"ALL 48 PLATES: {n:,} values, stored at 16 bits = {n * 16 / 8e6:.2f} MB; order-0 floor {tot['f0'] / n:.4f} bits; "
      f"v8a {tot['a'] / n:.4f} bits = {tot['a'] / 8e6:.2f} MB; v8b {tot['b'] / n:.4f} bits = {tot['b'] / 8e6:.2f} MB; all roundtrip {summary['all_roundtrip_ok']}")
