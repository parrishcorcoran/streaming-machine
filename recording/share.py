"""PHASE A, part 2: THE TWO FOUNDATION DIRECTIONS, AND HOW MUCH OF THE LIGHT LIES ALONG THEM. Exact whole numbers and
Fractions throughout; float only to print percentages.

The plates read the centered stream (the water level's centering is folded into every plate that reads the stream:
plate = C diag(gain) W), scaled by the water level's gauge (one number per position, no effect on direction). So the
direction used at a plate is the centered foundation stream at that plate:
    d = 768 F - (sum of F) 1          (whole numbers; 768 x the centered F; its lines sum to exactly 0)
The share of a position's centered light x_c along d, exactly:
    (x_c . d)^2 / (|x_c|^2 |d|^2) = 768 (x . d)^2 / ((768 S2 - S1^2) |d|^2)
    (x . d = x_c . d because d's lines sum to 0; 768 S2 - S1^2 is the machine's own water-level sum, 768 |x_c|^2)
Directions at each plate: F (the weights alone), S (the stone of ' The'), OWN (each text's own position-0 stream).
CONTROLS: R, a random direction (whole numbers, fixed seed per plate, centered so it is orthogonal to the all-ones
direction); F~ and S~, d with its lines shuffled (same values, fixed seed).
ORDER-FREE CHECK: shuffling the lines of BOTH the light and the direction the same way must give every share exactly
(it is a relation between the two patterns, not a property of the line order).
PREDICTION (written before running, from the plan): the share along the stone grows with depth and is modest; the
share along the weights-alone foundation is smaller.
Claude's numbers: S mean share 1-5% at layers 0-1, 10-40% from the middle layers on; F below S at every plate;
random and shuffled at about 1/767 = 0.13%.
    python3 recording/share.py     (reads recording/light.npz)
"""
import os, json, time, math, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
print("PREDICTION (written before running): S share grows with depth and is modest; F share smaller.\n"
      "  Claude's numbers: S 1-5% at layers 0-1, 10-40% from the middle on; F below S at every plate; controls ~0.13%.")
L = np.load("recording/light.npz"); PLATES = list(L["plates"]); POS, TXT = L["pos"], L["txt"]
WORDS = list(L["words"]); NT = int(TXT.max()) + 1
later = np.flatnonzero(POS >= 1); first = np.flatnonzero(POS == 0)

def obj(a): return np.array([int(v) for v in np.asarray(a).ravel()], dtype=object).reshape(np.shape(a))
def centered(f): f = obj(f); return 768 * f - sum(f)          # 768 x (f - mean), whole numbers, lines sum to 0
def dot(a, b): return sum(int(x) * int(y) for x, y in zip(a, b))
def pct(fr): return 100.0 * float(fr)

def shares(X, d):
    """exact share of each row's centered light along d (Fractions); d has lines summing to 0"""
    dd = dot(d, d)
    if dd == 0: return None
    Xo = obj(X); xd = Xo.dot(d); S1 = Xo.sum(1); S2 = (Xo * Xo).sum(1)
    return [Fraction(768 * a * a, (768 * s2 - s1 * s1) * dd) for a, s1, s2 in zip(xd, S1, S2)]

def stats(fr):
    if fr is None: return None
    s = sorted(fr); n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return sum(s) / n, med, s[math.ceil(0.9 * n) - 1]

def fmt(st): return "      (no light)       " if st is None else f"{pct(st[0]):6.2f} {pct(st[1]):6.2f} {pct(st[2]):6.2f}"

t0 = time.time()
# ------------------------------------------------------------------ 1. the two foundation directions
print("\n1. THE TWO FOUNDATION DIRECTIONS (sizes in stream units, the stream's whole numbers / 2^16; float only to print)")
print("   plate     |F|      max|F| line  |centered F| | |S The|   max|S| line | cos(F,S) centered | F at qkv.l vs F at fc.l")
D = {}
for i, k in enumerate(PLATES):
    F, S = L[f"F|{k}"], L[f"S The|{k}"]
    dF, dS = centered(F), centered(S)
    D[k] = dict(F=dF, S=dS)
    nF2, nS2 = dot(F, F), dot(S, S); cF2, cS2 = dot(dF, dF), dot(dS, dS)
    cosFS = "   -  " if cF2 == 0 else f"{math.copysign(math.sqrt(float(Fraction(dot(dF, dS) ** 2, cF2 * cS2))), dot(dF, dS)):+.4f}"
    kF, kS = int(np.argmax(np.abs(F))), int(np.argmax(np.abs(S)))
    extra = ""
    if k.startswith("fc."):
        q = D[k.replace("fc", "qkv")]["F"]; qq = dot(q, q)
        extra = "  -" if qq == 0 else f"  cos {math.copysign(math.sqrt(float(Fraction(dot(q, dF) ** 2, qq * cF2))), dot(q, dF)):+.4f}"
    print(f"   {k:8s} {math.sqrt(nF2) / 65536:8.2f} {abs(int(F[kF])) / 65536:8.2f} {kF:4d}  {math.sqrt(cF2) / 768 / 65536:8.2f}    |"
          f" {math.sqrt(nS2) / 65536:8.2f} {abs(int(S[kS])) / 65536:8.2f} {kS:4d} | {cosFS:>8s}          |{extra}")

print("\n   THE STONE AGAINST THE WORD: centered S for another first word against ' The': lines with the same sign;"
      " |S_w - S_The| / |S_The| (centered)")
print("   plate    " + "  ".join(f"{w!r:>18s}" for w in WORDS if w != " The"))
for k in PLATES:
    dS = D[k]["S"]; nS = dot(dS, dS); row = []
    for w in WORDS:
        if w == " The": continue
        dw = centered(L[f"S{w}|{k}"]); same = sum((a > 0) == (b > 0) for a, b in zip(dw, dS))
        diff = dw - dS; rel = math.sqrt(float(Fraction(dot(diff, diff), nS)))
        row.append(f"{100 * same / 768:5.1f}% {100 * rel:7.2f}%   ")
    print(f"   {k:8s} " + "  ".join(row))

# ------------------------------------------------------------------ 2. the share of the light along each direction
print(f"\n2. HOW MUCH OF THE LIGHT LIES ALONG IT: share of the centered light's energy, exactly, at every plate that reads the"
      f" stream, positions >= 1 ({len(later)} positions of {NT} texts). mean / median / 90th percentile, %")
print("   plate     F (weights alone)      S (stone ' The')       OWN (text's own stone) | R random             F~ shuffled          S~ shuffled")
RES = {}; ORDER_OK = [0, 0]; JOINT = np.random.default_rng(99).permutation(768)
for i, k in enumerate(PLATES):
    X = L[f"X|{k}"]; Xl = X[later]
    r = np.random.default_rng(20261004 + i).integers(-(1 << 20), 1 << 20, 768)
    pF, pS = np.random.default_rng(7700 + i).permutation(768), np.random.default_rng(8800 + i).permutation(768)
    dirs = {"F": D[k]["F"], "S": D[k]["S"], "R": centered(r), "F~": D[k]["F"][pF], "S~": D[k]["S"][pS]}
    res = {n: shares(Xl, d) for n, d in dirs.items()}
    own = []; ownd = {t: centered(X[first[t]]) for t in range(NT)}
    for p in later: own += shares(X[p:p + 1], ownd[TXT[p]])   # each position against its own text's position-0 stream
    res["OWN"] = own
    for n in ("F", "S"):                                     # the order-free check: lines of both shuffled together
        if res[n] is None: continue
        ORDER_OK[0] += shares(Xl[:, JOINT], dirs[n][JOINT]) == res[n]; ORDER_OK[1] += 1
    RES[k] = {n: stats(v) for n, v in res.items()}
    print(f"   {k:8s} " + "  ".join(fmt(RES[k][n]) for n in ("F", "S", "OWN")) + " | " + "  ".join(fmt(RES[k][n]) for n in ("R", "F~", "S~")),
          flush=True)
print(f"\n   order-free check (lines of the light and the direction shuffled together give every share exactly): {ORDER_OK[0]}/{ORDER_OK[1]} plates")
print("   1/767 (a direction with no relation to the light, on average): 0.13%")
json.dump({k: {n: [float(x) for x in st] if st else None for n, st in v.items()} for k, v in RES.items()},
          open("recording/share_stats.json", "w"), indent=1)
print(f"({time.time() - t0:.0f}s)")
