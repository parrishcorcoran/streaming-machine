"""Remove the blank from the taught tiny model exactly. S is remade from seed 0 and checked against start.pt;
X = W - S exact for every weight (float64 where it holds it, exact fractions for the rest); S + X rebuilt and
checked against W bit for bit; exact bits a weight for W and X."""
import numpy as np, torch
from fractions import Fraction
from transformers import GPT2Config, GPT2LMHeadModel
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=128, vocab_size=50257)
torch.manual_seed(0); fresh = {n: p.detach() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
S = torch.load("start.pt"); W = torch.load("taught.pt")
print("start remade from seed 0 equals start.pt bit for bit:", all(torch.equal(fresh[n], S[n]) for n in S))

def H(keys):
    _, c = np.unique(keys, return_counts=True); c = c.astype(np.float64); p = c / c.sum()
    return float(-(c * np.log2(p)).sum())

def key_of(sg, top, ln):
    return 1 + (1 if sg > 0 else 0) + 2 * ((top + 400) * 64 + ln)

def parts_float(a):   # nonzero float64 values -> sign, top place, places below the top
    mant, e = np.frexp(np.abs(a)); M = (mant * 2.0 ** 53).astype(np.int64)
    assert np.all(M.astype(np.float64) == mant * 2.0 ** 53)
    tz = np.log2((M & -M).astype(np.float64)).astype(np.int64)
    return np.sign(a), e - 1, 52 - tz

def parts_frac(f):    # exact fraction with a power-of-two bottom
    a = abs(f); num, den = a.numerator, a.denominator; d = den.bit_length() - 1
    top = num.bit_length() - 1 - d; low = (num & -num).bit_length() - 1 - d
    return (1 if f > 0 else -1), top, top - low

def exact_size(fl, fracs=()):
    """bits to write every value exactly: (sign, top place, length) packed with one table per tensor,
    places below the top written plainly; zero is its own symbol."""
    keys = np.zeros(fl.size + len(fracs), np.int64); nz = np.flatnonzero(fl != 0)
    sg, top, ln = parts_float(fl[nz]); keys[nz] = 1 + (sg > 0) + 2 * ((top + 400) * 64 + ln); raw = float(ln.sum())
    for i, f in enumerate(fracs):
        if f == 0: continue
        a, b, c = parts_frac(f); keys[fl.size + i] = key_of(a, b, c); raw += c
    return H(keys) + raw, H(keys), raw

tot = dict(n=0, zero=0, far=0, bad=0, W=0.0, X=0.0, Xo=0.0)
for n in S:
    w = W[n].numpy().ravel().astype(np.float64); s = S[n].numpy().ravel().astype(np.float64)
    x = w - s
    ew, es = np.frexp(w)[1], np.frexp(s)[1]
    far = (w != 0) & (s != 0) & (np.abs(ew - es) > 28)          # float64 may not hold w - s exactly
    fi = np.flatnonzero(far); fr = [Fraction(w[i]) - Fraction(s[i]) for i in fi]
    back = s + x; back[fi] = [float(Fraction(s[i]) + f) for i, f in zip(fi, fr)]
    bad = int(np.sum(back != w)) + int(np.sum(back.astype(np.float32) != W[n].numpy().ravel()))
    keep = np.ones(w.size, bool); keep[fi] = False
    bx, px, rx = exact_size(x[keep], fr); bw, pw, rw = exact_size(w)
    zero = int(np.sum(x[keep] == 0)) + sum(1 for f in fr if f == 0)
    Ls, Lx = float(np.linalg.norm(s)), float(np.linalg.norm(x))
    for k, v in (("n", w.size), ("zero", zero), ("far", len(fi)), ("bad", bad), ("W", bw), ("X", bx), ("Xo", px)): tot[k] += v
    print(f"{n:32s} {w.size:>9,}  X exactly 0 {100*zero/w.size:5.1f}%   exact bits a weight: W {bw/w.size:5.2f}  "
          f"X {bx/w.size:5.2f} (outer {px/w.size:4.2f} + inner {rx/w.size:5.2f})   length X/S "
          f"{(f'{Lx/Ls:6.3f}' if Ls else 'start 0'):>7s}   fractions {len(fi)}  rebuilt wrong {bad}")
N = tot["n"]
print(f"\nALL {N:,} weights: S + X rebuilt wrong {tot['bad']} | X exactly 0 {100*tot['zero']/N:.3f}% | exact bits a weight: "
      f"W {tot['W']/N:.2f}, X {tot['X']/N:.2f} (outer {tot['Xo']/N:.2f}), S 0 (the seed) | via fractions {tot['far']}")
