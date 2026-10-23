"""Probe 2: where the photons of layer 0's lights are. At neurons-in (fc.0) the light is word + position + the proj
bias + the arrival's part; at the proj bank the light is the arrival's output. Photons per line of each part at the
water level's own shift, and with a shuffled word (control: a random other word's row, same distribution).
    python3 extra/probe_parts.py
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
WTE, WPE = M["wte"], M["wpe"]; rng = np.random.default_rng(1)
bp, bv = W["proj.0"][2], W["qkv.0"][2][1536:]
acc = {}
def add(k, v): acc.setdefault(k, []).append(np.asarray(v, float))
for r in refs:
    ids = r["ids"] + r["ref"]; P = len(ids); a, b = WTE[ids], WPE[:P]; R = a + b
    x16, ex = to16(R, -ES); s = (ex + ES)[:, None]
    ash = WTE[rng.integers(0, 50257, P)]
    add("entry: light", ones(x16).mean(1)); add("entry: carry (right word)", ones(x16 - rshift(a, s) - rshift(b, s)).mean(1))
    add("entry: carry (shuffled word)", ones(x16 - rshift(ash, s) - rshift(b, s)).mean(1))
    QKV = gauged_bank(R, "qkv.0"); o = attention(QKV); o16, eo = to16(o, -ES); so = (eo + ES)[:, None]
    att = plain_bank(o, "proj.0"); R1 = R + att; x1, e1 = to16(R1, -ES); s1 = (e1 + ES)[:, None]
    add("proj: light o16", ones(o16).mean(1)); add("proj: o16 - v bias", ones(o16 - rshift(bv, so)).mean(1))
    add("proj: o16 - own value", ones(o16 - rshift(QKV[:, 1536:], so)).mean(1))
    add("fc: light", ones(x1).mean(1)); add("fc: word part", ones(rshift(a, s1)).mean(1)); add("fc: position part", ones(rshift(b, s1)).mean(1))
    add("fc: proj bias part", ones(rshift(np.broadcast_to(bp, a.shape), s1)).mean(1)); add("fc: arrival part (proj out - bias)", ones(rshift(att - bp, s1)).mean(1))
    add("fc: extra = light - word - (position + bias)", ones(x1 - rshift(a, s1) - rshift(b + bp, s1)).mean(1))
    add("fc: extra, shuffled word", ones(x1 - rshift(ash, s1) - rshift(b + bp, s1)).mean(1))
    top = np.abs(R1).max(0).argsort()[-3:]
    add("fc: largest lines' share of |light|", np.abs(R1[:, top]).sum(1) / np.abs(R1).sum(1))
print("photons per line (mean over every position of the 30 refs texts):")
for k, v in acc.items(): print(f"  {k:46s} {np.concatenate(v).mean():7.3f}")
x = np.concatenate([WTE[r['ids'] + r['ref']] for r in refs])
print("lines where proj bias is largest:", np.argsort(-np.abs(bp))[:5], (np.abs(bp)[np.argsort(-np.abs(bp))[:5]] / 65536).round(2))
