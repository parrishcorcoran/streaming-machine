"""A real finished model with its published blank: Pythia-70m, step0 (blank, fp32) and main (finished, fp16).
Try both; remove the blank exactly, two ways: X = W - S (blank as published) and X16 = W - fp16(S) (blank on the
model's own fp16 grid, a fixed rule from the blank, so also free). Every weight rebuilt and checked bit for bit.
Exact bits a weight; how much of the blank is still inside each piece (c = <W,S>/<S,S>)."""
import math, json, numpy as np, torch
from fractions import Fraction
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import GPTNeoXForCausalLM, AutoTokenizer
torch.set_num_threads(8)
R = "EleutherAI/pythia-70m"
tok = AutoTokenizer.from_pretrained(R)
ids = torch.tensor(tok(open("../tiny/shakespeare.txt").read()).input_ids)
ho = ids[int(0.9 * len(ids)):]; T = 128
g = torch.Generator().manual_seed(1234)
HO = torch.stack([ho[i:i+T+1] for i in torch.randint(0, len(ho) - T - 1, (64,), generator=g).tolist()])

@torch.no_grad()
def try_it(rev):
    m = GPTNeoXForCausalLM.from_pretrained(R, revision=rev, torch_dtype=torch.float32).eval(); ls, hs = [], []
    for c in range(0, len(HO), 8):
        lg = m(HO[c:c+8, :T]).logits
        ls.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), HO[c:c+8, 1:].reshape(-1)).item())
        hs.append((lg.argmax(-1) == HO[c:c+8, 1:]).float().mean().item()); del lg
    p = "ROMEO:\nO"; pl = m(torch.tensor([tok(p).input_ids])).logits[0, -1].softmax(-1); top = pl.topk(5)
    words = ", ".join(f"{tok.decode([int(i)])!r} {v:.3f}" for v, i in zip(top.values, top.indices))
    print(f"[{rev}] held-out loss {np.mean(ls):.3f} (pure guess {math.log(m.config.vocab_size):.3f}); next word right "
          f"{100*np.mean(hs):.1f}%; after {p!r}: {words}", flush=True)
    del m; return float(np.mean(ls)), float(np.mean(hs))
r0 = try_it("step0"); r1 = try_it("main")

def H(keys):
    _, c = np.unique(keys, return_counts=True); c = c.astype(np.float64); p = c / c.sum()
    return float(-(c * np.log2(p)).sum())
def parts_float(a):
    mant, e = np.frexp(np.abs(a)); M = (mant * 2.0 ** 53).astype(np.int64)
    assert np.all(M.astype(np.float64) == mant * 2.0 ** 53)
    tz = np.log2((M & -M).astype(np.float64)).astype(np.int64)
    return np.sign(a), e - 1, 52 - tz
def parts_frac(f):
    a = abs(f); num, den = a.numerator, a.denominator; d = den.bit_length() - 1
    return (1 if f > 0 else -1), num.bit_length() - 1 - d, (num.bit_length() - 1) - ((num & -num).bit_length() - 1)
def exact_size(fl, fracs=()):
    keys = np.zeros(fl.size + len(fracs), np.int64); nz = np.flatnonzero(fl != 0)
    sg, top, ln = parts_float(fl[nz]); keys[nz] = 1 + (sg > 0) + 2 * ((top + 400) * 64 + ln); raw = float(ln.sum())
    for i, f in enumerate(fracs):
        if f == 0: continue
        a, b, c = parts_frac(f); keys[fl.size + i] = 1 + (a > 0) + 2 * ((b + 400) * 64 + c); raw += c
    return H(keys) + raw

def kind(n):
    for k in ("embed_in", "embed_out", "query_key_value", "attention.dense", "dense_h_to_4h", "dense_4h_to_h"):
        if k in n: return k
    return "norms+biases"
f0 = safe_open(hf_hub_download(R, "model.safetensors", revision="step0"), framework="pt")
f1 = safe_open(hf_hub_download(R, "model.safetensors", revision="main"), framework="pt")
names = [n for n in f0.keys() if n in set(f1.keys())]
skipped = [n for n in f1.keys() if n not in set(f0.keys())]
print(f"pieces in both: {len(names)}; only in the finished file (not weights): {len(skipped)} e.g. {skipped[:2]}")
agg = {}; tot = dict(n=0, W=0.0, X=0.0, X16=0.0, z=0, z16=0, bad=0, far=0)
for n in names:
    s32 = f0.get_tensor(n).numpy().ravel(); w16 = f1.get_tensor(n).numpy().ravel()
    assert s32.dtype == np.float32 and w16.dtype == np.float16
    s = s32.astype(np.float64); w = w16.astype(np.float64); s16 = s32.astype(np.float16).astype(np.float64)
    # X = W - S, exact
    x = w - s; ew, es = np.frexp(w)[1], np.frexp(s)[1]
    far = np.flatnonzero((w != 0) & (s != 0) & (np.abs(ew - es) > 28)); fr = [Fraction(w[i]) - Fraction(s[i]) for i in far]
    back = s + x; back[far] = [float(Fraction(s[i]) + f) for i, f in zip(far, fr)]
    keep = np.ones(w.size, bool); keep[far] = False
    # X16 = W - fp16(S): both on fp16 grids, float64 holds the difference exactly
    x16 = w - s16; back16 = s16 + x16
    bad = int(np.sum(back != w)) + int(np.sum(back16 != w)) + int(np.sum(back.astype(np.float16) != w16))
    bW, bX, bX16 = exact_size(w), exact_size(x[keep], fr), exact_size(x16)
    ss = float(np.dot(s, s)); c = float(np.dot(w, s) / ss) if ss else 0.0
    Lw = float(np.linalg.norm(w)); share = abs(c) * math.sqrt(ss) / Lw if Lw else 0.0
    k = kind(n); a = agg.setdefault(k, dict(n=0, W=0.0, X=0.0, X16=0.0, z=0, z16=0, c=[], share=[]))
    a["n"] += w.size; a["W"] += bW; a["X"] += bX; a["X16"] += bX16
    a["z"] += int(np.sum(x[keep] == 0)) + sum(1 for f in fr if f == 0); a["z16"] += int(np.sum(x16 == 0))
    if ss: a["c"].append((c, w.size)); a["share"].append((share, w.size))
    for kk, v in (("n", w.size), ("W", bW), ("X", bX), ("X16", bX16), ("bad", bad), ("far", len(far)),
                  ("z", int(np.sum(x[keep] == 0)) + sum(1 for f in fr if f == 0)), ("z16", int(np.sum(x16 == 0)))):
        tot[kk] += v
print(f"\n{'piece':16s} {'weights':>11s} {'W bits':>7s} {'W-S bits':>9s} {'W-fp16(S)':>10s} {'unchanged':>10s} {'c (blank left)':>15s} {'blank share of length':>22s}")
for k, a in agg.items():
    wa = lambda L: sum(v * n for v, n in L) / max(1, sum(n for _, n in L)) if L else float("nan")
    print(f"{k:16s} {a['n']:>11,} {a['W']/a['n']:7.2f} {a['X']/a['n']:9.2f} {a['X16']/a['n']:10.2f} {100*a['z16']/a['n']:9.2f}% "
          f"{wa(a['c']):15.3f} {100*wa(a['share']):21.1f}%")
N = tot["n"]
print(f"\nALL {N:,} weights: rebuilt wrong {tot['bad']} | exact bits a weight: W {tot['W']/N:.2f}, W - S {tot['X']/N:.2f}, "
      f"W - fp16(S) {tot['X16']/N:.2f} | unchanged from fp16 blank {100*tot['z16']/N:.2f}% | via fractions {tot['far']}")
json.dump(dict(blank=r0, finished=r1, total={k: v for k, v in tot.items()},
               pieces={k: {kk: (vv if kk not in ("c", "share") else None) for kk, vv in a.items()} for k, a in agg.items()}),
          open("real_pythia70m.json", "w"), indent=1)
