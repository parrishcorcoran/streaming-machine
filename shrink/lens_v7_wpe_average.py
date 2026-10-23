"""LENS v7: the position table again (v6 found the step from the previous position saves only 3%).
Why v6 barely helped: each line of wpe is a slow pattern plus a grain of about the same size; the step from one
position to the next removes the slow part but adds two grains (spread x sqrt 2, +0.5 bit), so the net is small.
Fixed rule tried here instead: predict from the AVERAGE of the previous 4 positions (sum >> 2, rounded): the slow
part is still removed, but the grain is averaged over 4 (spread x sqrt(1 + 1/4), +0.16 bit).
Predicted: better than the plain step, by a few percent of wpe's ~1.05 MB. Lossless, roundtrip checked.
Control: the same rule on randomly permuted positions.
"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from lens_lib import *

_, _, _, _, W = load()
P, L = W.shape


def residual(W):
    R = W.copy()
    for p in range(1, P):
        n = min(p, 4); prev = W[p - n:p].sum(0)
        pred = rshift_round(prev, 2) if n == 4 else div_round(prev, n)   # first 3 positions: average of what exists
        R[p] = W[p] - pred
    return R


def rebuild(R):
    W2 = np.zeros_like(R); W2[0] = R[0]
    for p in range(1, P):
        n = min(p, 4); prev = W2[p - n:p].sum(0)
        W2[p] = R[p] + (rshift_round(prev, 2) if n == 4 else div_round(prev, n))
    return W2


R = residual(W)
b, h = pack(R.T)
ok = bool(np.array_equal(rebuild(np.ascontiguousarray(unpack(b, h).T)), W))
Ws = W[np.random.default_rng(0).permutation(P)]
bs, hs = pack(residual(Ws).T)
res = {"wpe_alone_bytes": 1078304, "wpe_step_bytes_v6": 1046912, "wpe_avg4_bytes": nbytes(b, h), "roundtrip_ok": ok,
       "avg4_order0_bits": order0_bits(R[1:]), "avg4_photons": float(ones(R[1:]).mean()), "control_shuffled_positions_bytes": nbytes(bs, hs)}
print(json.dumps(res, indent=1))
