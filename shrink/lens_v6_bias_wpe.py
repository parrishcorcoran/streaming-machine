"""LENS v6: the rest of what the converter wrote through the same lens or along the same light, all lossless.

(a) head.b = stream(bfin @ wte.T): each word's offset at the top is the entry film read through the last water level's
    offset bfin (768 numbers). Store B = bfin at 2^-24 (whole), recompute b^ = (sum_i B_i E_i) >> 24, store b - b^.
    Predicted: E's rounding (+-1/2 unit per line) times bfin adds up to an error of rms sqrt(sum bf^2 / 12) units
    -> a few units, ~3-4 bits a word instead of ~20.
(b) wpe (1024 x 768 at 2^16): no lens makes it from anything else. The fixed rule a fax uses instead: predict each
    position's line from the position before it (the natural order, not a sort), store the step. Predicted: the
    position table changes slowly from position to position, so the step is narrower than the value.
(c) the films' own floors, for the record: order-0 of E and T with one histogram, and with one histogram per line.
"""
import sys, time, json, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

K = 24
t0 = time.time()
E, T, TE, HB, WPE = load()
N, L = E.shape
res = {}

# (a) the bias read at the top
B = whole(lens_float("ln_f.bias"), K)
bh = np.empty(N, np.int64)
for a in range(0, N, 4096):
    bh[a:a + 4096] = rshift_round((E[a:a + 4096] * B[None, :]).sum(1), K)
rb = HB - bh
bHB, hHB = pack(HB); bRB, hRB = pack(rb); bB, hB = pack(B)
res["headb_alone_bytes"] = nbytes(bHB, hHB); res["headb_residual_bytes"] = nbytes(bRB, hRB); res["headb_lens_bytes"] = nbytes(bB, hB)
res["headb_residual_range"] = [int(rb.min()), int(rb.max())]; res["headb_residual_order0_bits"] = order0_bits(rb)
B2 = unpack(bB, hB); bh2 = np.empty(N, np.int64)
for a in range(0, N, 4096):
    bh2[a:a + 4096] = rshift_round((E[a:a + 4096] * B2[None, :]).sum(1), K)
res["headb_roundtrip_ok"] = bool(np.array_equal(bh2 + unpack(bRB, hRB), HB))
Bs = B[np.random.default_rng(0).permutation(L)]
bhs = np.empty(N, np.int64)
for a in range(0, N, 4096):
    bhs[a:a + 4096] = rshift_round((E[a:a + 4096] * Bs[None, :]).sum(1), K)
bRs, hRs = pack(HB - bhs); res["headb_control_shuffled_bytes"] = nbytes(bRs, hRs)
print(f"head.b alone {res['headb_alone_bytes']:,} B; through the bias lens: residual {res['headb_residual_bytes']:,} B + lens {res['headb_lens_bytes']:,} B; "
      f"residual range {res['headb_residual_range']}; control {res['headb_control_shuffled_bytes']:,} B; roundtrip {res['headb_roundtrip_ok']}", flush=True)

# (b) the position table, step from the previous position
bW, hW = pack(WPE.T); res["wpe_alone_bytes"] = nbytes(bW, hW)
step = WPE.copy(); step[1:] = WPE[1:] - WPE[:-1]
bS, hS = pack(step.T); res["wpe_step_bytes"] = nbytes(bS, hS)
back = np.cumsum(np.ascontiguousarray(unpack(bS, hS).T), axis=0)
res["wpe_roundtrip_ok"] = bool(np.array_equal(back, WPE))
res["wpe_order0_bits"] = order0_bits(WPE); res["wpe_step_order0_bits"] = order0_bits(step[1:])
res["wpe_photons"] = float(ones(WPE).mean()); res["wpe_step_photons"] = float(ones(step[1:]).mean())
sh = WPE[np.random.default_rng(0).permutation(1024)]
st2 = sh.copy(); st2[1:] = sh[1:] - sh[:-1]
bC, hC = pack(st2.T); res["wpe_control_shuffled_positions_step_bytes"] = nbytes(bC, hC)
print(f"wpe alone {res['wpe_alone_bytes']:,} B (order-0 {res['wpe_order0_bits']:.2f} bits, photons {res['wpe_photons']:.2f}); "
      f"step from previous position {res['wpe_step_bytes']:,} B (order-0 {res['wpe_step_order0_bits']:.2f} bits, photons {res['wpe_step_photons']:.2f}); "
      f"roundtrip {res['wpe_roundtrip_ok']}; control (positions shuffled, then step) {res['wpe_control_shuffled_positions_step_bytes']:,} B", flush=True)

# (c) floors of the films
for name, A in (("E", E), ("T", T)):
    g = order0_bits(A)
    per = float(np.mean([order0_bits(A[:, i].astype(np.int64) >> 3) + 3 for i in range(L)]))   # top part per line + 3 raw low places (keeps the histograms from being too sparse)
    res[f"{name}_order0_bits_one_histogram"] = g; res[f"{name}_order0_bits_per_line_histogram"] = per
    print(f"{name}: order-0 {g:.3f} bits/value (one histogram) -> {g * A.size / 8e6:.2f} MB; per line {per:.3f} -> {per * A.size / 8e6:.2f} MB", flush=True)
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
