"""PHASE A, part 4: WHY THE TURN GATHERS NOTHING. A reflection is a mirror between two patterns: it sends d onto the
centered line t = 768 e_k - 1, and sends t onto d. So after the turn the light's share on line k is its old share along
d, and its share along d is its old share on line k. The turn gathers light onto the line only when
    share along d  >  share already on the centered line k.
Exact Fractions; float only to print.
  1. per plate, S and F: mean share of the centered light on the centered line k, against the mean share along d.
     Exact check of the swap at qkv.6 and fc.11 (S): share of H x on t == share of x along d, to the leftover.
  2. the top read: neurons whose largest |value| sits on line 447 (why the S turn rescales 95% of them one place finer).
  3. work totals with and without the top read.
PREDICTION (written before running): from layer 3 on, the light's share on line 447 is about its share along S
  (S is 92-96% on line 447), so the swap trades like for like; for F the share on line 266/481 is larger than the
  share along F, so the turn throws light out (more photons). In the top plate, most neurons peak on line 447.
    python3 recording/why.py
"""
import os, json, math, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
L = np.load("recording/light.npz"); PLATES = list(L["plates"]); POS = L["pos"]; later = np.flatnonzero(POS >= 1)
SH = json.load(open("recording/share_stats.json")); M = np.load("base2_gpt2.npz")
def obj(a): return np.array([int(v) for v in np.asarray(a).ravel()], dtype=object).reshape(np.shape(a))
def centered(f): f = obj(f); return 768 * f - sum(f)
def dot(a, b): return sum(int(x) * int(y) for x, y in zip(a, b))
def shares(Xo, d):
    dd = dot(d, d); xd = Xo.dot(d); S1 = Xo.sum(1); S2 = (Xo * Xo).sum(1)
    return [Fraction(768 * a * a, (768 * s2 - s1 * s1) * dd) for a, s1, s2 in zip(xd, S1, S2)]
def mean(fr): return float(sum(fr) / len(fr))
print(__doc__.split("    python3")[0])
print("1. share of the later light (positions >= 1) ALONG d  vs  ON the centered line k (mean %); the turn swaps them")
print("   plate     S: k   along   on-line  |  F: k   along   on-line  |  light on line 447: mean |value| vs median line's mean |value|")
for k in PLATES:
    X = L[f"X|{k}"][later]; Xo = obj(X); row = []
    for n, key in (("S", "S The"), ("F", "F")):
        d = centered(L[f"{key}|{k}"])
        if dot(d, d) == 0: row.append(f"{n}:  -     none    none  "); continue
        kk = max(range(768), key=lambda i: abs(int(d[i]))); t = np.full(768, -1, dtype=object); t[kk] = 767
        row.append(f"{n}: {kk:3d} {mean(shares(Xo, d)) * 100:6.2f}% {mean(shares(Xo, t)) * 100:6.2f}%")
    Xc = 768 * X - X.sum(1, keepdims=True); m = np.abs(Xc).mean(0) / 768 / 65536
    print(f"   {k:8s} " + "  |  ".join(row) + f"  |  {m[447]:8.3f} vs {np.median(m):6.3f}")

print("\n   exact check of the swap (S turn, m = 48, as in turn.py): share of H x on t vs share of x along d, max difference")
MB = 48
for k in ("qkv.6", "fc.11"):
    X = L[f"X|{k}"][later]; Xo = obj(X); d = centered(L[f"S The|{k}"])
    kk = max(range(768), key=lambda i: abs(int(d[i]))); sg = 1 if d[kk] > 0 else -1
    t = np.full(768, -1, dtype=object); t[kk] = 767
    C = math.isqrt(dot(d, d) * 4 ** MB // (768 * 767)); v = d * (1 << MB) - sg * C * t; D = dot(v, v)
    Xp = D * Xo - 2 * np.multiply.outer(Xo.dot(v), v)
    a, b = shares(Xp, t), shares(Xo, d)
    print(f"   {k:8s} max |difference| {max(abs(float(x - y)) for x, y in zip(a, b)):.1e}  (mean {mean(a) * 100:.4f}% vs {mean(b) * 100:.4f}%)")

print("\n2. THE TOP PLATE: neurons whose largest |value| sits on each line (the five most common lines)")
w = M["head.w"]; arg = np.abs(w.astype(np.int32)).argmax(0); cnt = np.bincount(arg, minlength=768)
for line in np.argsort(-cnt, kind="stable")[:5]: print(f"   line {line:3d}: {cnt[line]:6d} of {w.shape[1]} neurons")
print(f"   the stored top plate's line 447: mean |value| {np.abs(w[447].astype(np.int64)).mean():.0f}, other lines {np.abs(w.astype(np.int64)).mean():.0f}")

print("\n3. WORK TOTALS, % of the unturned machine (sum over the plates that have the direction)")
DIRS = ["F", "S", "R", "F~", "S~", "F|", "S|"]
for lab, sel in (("qkv and fc only", [k for k in PLATES if k != "head"]), ("with the top read", PLATES)):
    out = []
    for n in DIRS:
        a = b = 0
        for k in sel:
            I = json.load(open(f"recording/turn/{k}__{n}.json")); B = json.load(open(f"recording/turn/{k}__base.json"))
            if I is None: continue
            a += I["work"]; b += B["work"]
        out.append(f"{n} {100 * a / b:7.2f}%")
    print(f"   {lab:18s} " + "  ".join(out))
