"""Shared harness for shrinking the stored plate and the word table, counting only 1s and holes. Every system must
(a) decode back EXACTLY (lossless, checked) or (b) if it changes values, be judged by right next word out of 30.
Usage:  from table_bits import *
  E            the entry table: 50,257 words x 768 lines, whole numbers at 2^16 (int64, about 18 bits) -- the token
               embeddings, what a token's light IS when it enters ("the 50k numbers")
  T            the top table: 50,257 words x 768 lines, int16 (each word's own power of two in TE) -- the read at the top
               (the same embedding with the last water level's gain and centring folded in)
  plate(name)  a bank's plate, int16, (lines in, neurons out): names qkv.L proj.L fc.L mproj.L, L = 0..11
  entropy_bits(sym)   Shannon floor (bits) of a sequence of symbols coded one at a time with its own histogram
                      (what an ideal entropy coder needs, model not counted)
  words(table=None, plates=None)   right next word / 30 with a changed table (int16, same shape) and/or plates {name: int16}
  ones(A)      photons (1s) per value, sign apart;  naf(A) photons in the single-wave (up/down) writing
"""
import sys, json, numpy as np
_src = open("/Users/abundancemachine/streaming-machine/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv
refs = json.load(open("/Users/abundancemachine/streaming-machine/refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
T = M["head.w"].T.copy()                              # (50257, 768) int16: the top table
E = M["wte"].copy()                                   # (50257, 768) int64 at 2^16: the entry table
TE = M["head.e"]                                      # each word's power of two
_WTE0 = M["wte"]                                      # the entry light at 2^16 (what the machine uses going in)
def plate(name): return M[name + ".w"]
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
def naf(A): A = np.abs(np.asarray(A).astype(np.int64)); return ones(A ^ (3 * A))
def entropy_bits(sym):
    sym = np.asarray(sym).ravel(); _, c = np.unique(sym, return_counts=True); p = c / c.sum(); return float(-(c * np.log2(p)).sum())
def _run(ids, wte):
    P = len(ids); R = wte[ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
_TRUTH = None
def words(entry=None, top=None, plates=None):
    """right next word / 30. entry: (50257,768) whole numbers replacing E; top: (50257,768) int16 replacing T;
    plates: {name: int16 (lines in, neurons out)} replacing banks."""
    global _TRUTH
    if _TRUTH is None: _TRUTH = [_run(i, _WTE0) for i in inputs]
    saved = {}; wte = _WTE0
    try:
        if entry is not None: wte = np.asarray(entry).astype(np.int64)
        if top is not None:
            saved["head"] = W["head"]; W["head"] = (np.asarray(top).astype(np.int64).T.astype(np.float64), W["head"][1], W["head"][2])
        for n, w in (plates or {}).items():
            saved[n] = W[n]; W[n] = (np.asarray(w).astype(np.float64), W[n][1], W[n][2])
        return sum(_run(i, wte) == t for i, t in zip(inputs, _TRUTH))
    finally:
        for n, v in saved.items(): W[n] = v
