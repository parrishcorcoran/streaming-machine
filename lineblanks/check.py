"""A SECOND CODE PATH for lineblanks/way2.py, pairs.py and totals.py (same session: NOT an independent rebuild).

Recounts samples of every part by different arithmetic and compares with the numbers the three scripts wrote:
  1. way 2 (plates proj.0, qkv.3, mproj.<last>; blank and finished, both lights; fresh_b at proj.0): plain from
     Python's own bit counts; the walk's gap digits from a digit-by-digit signed-digit writing (x odd: digit 2 - x mod 4,
     subtract, halve), on each line's values sorted with Python's sorted(); the pours from every product x w formed
     directly (no walk) and its 1s counted; the writing. Each must equal way2_<size>.json's count.
  2. pairing by size (proj.0, fc.0, mproj.<last>, blank and finished, rows and columns, all lines): pure Python lists:
     ups and downs sorted, zip_longest with 0, sum of |differences|; whole; |net|. Must equal pairs_<size>.json's.
  3A. the lane totals of the word table, the attention-out rows and the MLP-out rows (blank, finished), summed within
     each power of two in int64 and the powers joined as Python whole numbers (foundation_share.py's way, not
     common.real's shifts). Must equal totals_<size>.json's "T". And r(E, M) recomputed with fractions.Fraction.
  3B. Q of the machine's own writes recomputed with math.fsum over float64 squares (each square exact below 2^26 x
     2^26; asserted): must equal totals_<size>.json's Q within 1e-12.
PREDICTION: every count equal, digit for digit (3B within 1e-12).
    python3 check.py size=70m
"""
import os, sys, json, math, time, itertools, numpy as np
from fractions import Fraction
from common import fresh

A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m")
OUT = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(OUT), "foundation0")
LOGF = open(os.path.join(OUT, f"check_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
t0 = time.time(); SID = {"70m": 70, "160m": 160}[SIZE]
Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L_ = int(Z["L"]); d = int(Z["d"])
W2J, PJ, TJ = (json.load(open(os.path.join(OUT, f"{s}_{SIZE}.json"))) for s in ("way2", "pairs", "totals"))
fails = 0
def same(label, a, b):
    global fails
    ok = a == b; fails += not ok; log(f"  {'OK  ' if ok else 'FAIL'} {label}: {a} {'==' if ok else '!='} {b}")

def sd_digits(v):                                       # lit digits of the signed-digit writing, digit by digit
    v = np.abs(np.asarray(v, np.int64)).copy(); c = np.zeros(v.shape, np.int64)
    while v.any():
        odd = (v & 1) == 1; dig = np.where(odd, 2 - (v & 3), 0); c += odd; v = (v - dig) >> 1
    return c
def pyones(v): return np.vectorize(lambda t: bin(abs(int(t))).count("1"), otypes=[np.int64])(v)

log(f"SECOND CODE PATH, pythia-{SIZE}")
log("1. way 2 counts")
LIGHT = {"Lf": np.load(os.path.join(OUT, f"light_{SIZE}_finished.npz")), "Lb": np.load(os.path.join(OUT, f"light_{SIZE}_start.npz"))}
PLATES = [f"{k}.{l}" for l in range(L_) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
for name in ("proj.0", "qkv.3", f"mproj.{L_ - 1}"):
    pi = PLATES.index(name)
    vers = [("blank", np.asarray(Z[name + ".s"])), ("finished", np.asarray(Z[name + ".w"]))]
    if name == "proj.0":
        vers.append(("fresh_b", fresh(np.asarray(Z[name + ".s"]), np.asarray(Z[name + ".e"]), 1, np.random.default_rng([20261006, SID, pi, 2]))))
    for v, w in vers:
        w = w.astype(np.int64); srt = [sorted(row) for row in w.tolist()]
        GN = np.array([sd_digits(np.array([r[0]] + [b - a for a, b in zip(r, r[1:])])).sum() for r in srt])
        W1 = pyones(w).sum(1)
        for lt in ("Lf", "Lb"):
            x = np.asarray(LIGHT[lt]["light." + name]).astype(np.int64)
            plain = int((pyones(x) * W1[None, :]).sum()); walk = int((sd_digits(x) * GN[None, :]).sum())
            pours = 0
            for i in range(w.shape[0]):
                xi = x[:, i]; nz = xi != 0
                if nz.any(): pours += int(np.bitwise_count(np.abs(np.outer(xi[nz], w[i]))).sum())
            c = W2J[name][v][lt]
            same(f"{name} {v} {lt} plain", plain, c["plain"]); same(f"{name} {v} {lt} walk", walk, c["walk"])
            same(f"{name} {v} {lt} pours", pours, c["pours"]); same(f"{name} {v} {lt} writing", int(pyones(x).sum()), c["writing"])

log("2. pairing by size")
def pyleft(lines):
    left = whole = absnet = 0
    for row in lines:
        U = sorted((t for t in row if t > 0), reverse=True); D = sorted((-t for t in row if t < 0), reverse=True)
        left += sum(abs(a - b) for a, b in itertools.zip_longest(U, D, fillvalue=0)); whole += sum(abs(t) for t in row); absnet += abs(sum(row))
    return left, whole, absnet
for name in ("proj.0", "fc.0", f"mproj.{L_ - 1}"):
    e = [int(t) for t in np.asarray(Z[name + ".e"])]; emin = min(e)
    for v, key in (("blank", ".s"), ("finished", ".w")):
        q = np.asarray(Z[name + key]).tolist()
        V = [[t << (e[j] - emin) for j, t in enumerate(row)] for row in q]          # per column power, Python whole numbers
        for along, lines in (("rows", V), ("cols", [list(c) for c in zip(*V)])):
            l, w, n = pyleft(lines); c = PJ[name][v][along]
            same(f"{name} {v} {along} (left, whole, |net|)", (l, w, n), (c["left"], c["whole"], c["absnet"]))

log("3A. lane totals by power of two, joined as Python whole numbers")
EMIN = min(int(np.asarray(Z[n + ".e"]).min()) for n in ["embed"] + [f"{k}.{l}" for l in range(L_) for k in ("proj", "mproj")])
for v, key in (("blank", ".s"), ("finished", ".w")):
    T = {}
    for g, names, axis in (("E", ["embed"], 0), ("A", [f"proj.{l}" for l in range(L_)], 1), ("M", [f"mproj.{l}" for l in range(L_)], 1)):
        tot = [0] * d
        for nm in names:
            q = np.asarray(Z[nm + key]).astype(np.int64); e = np.asarray(Z[nm + ".e"])
            for ev in np.unique(e):
                part = q[e == ev].sum(0) if axis == 0 else np.where((e == ev)[None, :], q, 0).sum(0)
                tot = [a + (int(b) << (int(ev) - EMIN)) for a, b in zip(tot, part)]
        T[g] = tot
        same(f"{v} lane totals {g} (all {d} lanes)", tot, TJ["A"][v]["T"][g])
    ab = sum(a * b for a, b in zip(T["E"], T["M"])); aa = sum(a * a for a in T["E"]); bb = sum(b * b for b in T["M"])
    rr = ab / math.sqrt(aa * bb)
    same(f"{v} r(E, M) to 12 places", round(rr, 12), round(TJ["A"][v]["agree"]["EM"]["r"], 12))
    cr = Fraction(aa + 2 * ab + bb, aa + bb)
    same(f"{v} cancel(E, M) to 12 places", round(float(cr), 12), round(TJ["A"][v]["agree"]["EM"]["cancel"], 12))

log("3B. Q of the machine's own writes by math.fsum")
for run in ("start", "finished"):
    W = np.load(os.path.join(OUT, f"light_{SIZE}_{run}.npz"))["writes"]; assert np.abs(W).max() < 2 ** 26 and np.abs(W.sum(1)).max() < 2 ** 26
    q = math.fsum((W.sum(1).astype(np.float64) ** 2).ravel()) / math.fsum((W.astype(np.float64) ** 2).ravel())
    ok = abs(q - TJ["B"][run]["Q"]) < 1e-12; fails += not ok
    log(f"  {'OK  ' if ok else 'FAIL'} {run} Q {q:.15f} vs {TJ['B'][run]['Q']:.15f}")
log(f"\n{fails} failures ({time.time() - t0:.0f}s)")
