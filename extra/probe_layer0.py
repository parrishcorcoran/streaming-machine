"""Probe: what layer 0's water level does to the light. For every position of the 30 refs texts: the shift s the
water level picks at the entry (qkv.0), at the arrival's output (proj.0) and at neurons-in (fc.0); the carry that
appears when word and position are rounded on their own; photons in each light.
    python3 extra/probe_layer0.py
"""
import sys, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(_src)
refs = json.load(open(f"{HERE}/refs_many.json"))
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
WTE, WPE = M["wte"], M["wpe"]
def shift_of(X):
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)]); return np.maximum(top - 15, 0)
S0, S1, SO, CARRY, PH = [], [], [], [], {"x16": [], "carry": [], "o16": [], "x16fc": [], "fc_extra": []}
for r in refs:
    ids = r["ids"] + r["ref"]; P = len(ids); a, b = WTE[ids], WPE[:P]; R = a + b
    s = shift_of(R); S0 += s.tolist()
    x16, ex = to16(R, -ES)
    c = x16 - rshift(a, s[:, None]) - rshift(b, s[:, None]); CARRY.append(c)
    PH["x16"].append(ones(x16).sum(1)); PH["carry"].append(ones(c).sum(1))
    qkv = gauged_bank(R, "qkv.0"); o = attention(qkv); o16, eo = to16(o, -ES); SO += (eo + ES).tolist(); PH["o16"].append(ones(o16).sum(1))
    R1 = R + plain_bank(o, "proj.0"); s1 = shift_of(R1); S1 += s1.tolist(); x1, _ = to16(R1, -ES); PH["x16fc"].append(ones(x1).sum(1))
    bp = W["proj.0"][2]
    m = x1 - rshift(a, s1[:, None]) - rshift(b + bp[None, :], s1[:, None]); PH["fc_extra"].append(ones(m).sum(1))
C = np.concatenate(CARRY)
print("shift at entry (qkv.0):", np.unique(S0, return_counts=True))
print("shift at arrival out (proj.0):", np.unique(SO, return_counts=True))
print("shift at neurons-in (fc.0):", np.unique(S1, return_counts=True))
print("carry values:", np.unique(C, return_counts=True), " nonzero share", (C != 0).mean())
for k, v in PH.items(): v = np.concatenate(v); print(f"photons per position, {k:9s}: mean {v.mean():9.1f}  (per line {v.mean()/768:.2f})")
