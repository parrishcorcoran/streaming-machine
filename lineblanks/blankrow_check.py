"""SECOND CODE PATH for blankrow.py and where_blank2.py (same session, written after their runs). 2026-10-06.

Recomputes their key numbers by another route and compares with blankrow_<size>.json / where_blank2_<size>.json:
  - each position's power of two found by the SIZE RATIO of the machine's write to the whole-number product
    (log2 of the largest |write - bias| over the largest |product x 2^e|), then verified digit for digit (no search);
  - the blank's writes from int64 sums (no float in the product), the plate's input lanes in a random order;
  - the stream's lanes in a random order and the positions reversed before any measure; every sum by math.fsum;
  - the word rows from the stored stream rows as the machine added them (writes[:, 0]), not from the table: the built
    part as writes[:, 0] - c x (the blank row on the stream scale), so the row route differs too.
Checked: part 2 (whole, built_c: cos, k, g, cos_word), part 3 (mean B against the table's lane totals: cos), the floor
(the blank machine: cos, k), where_blank2 at layer 0 and 1 (lean, along, perp).
EXPECTED (written before the run): every number the same to 1e-6 or better (cos and k), apart from the word rows taken
from the rounded stream rows: there the cosines move by under 1e-4 (rounding to 2^-16 of a row of size ~0.6 x 2^16).

    python3 blankrow_check.py size=70m
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m"); ES = 16
LOGF = open(os.path.join(HERE, f"blankrow_check_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L, d = int(Z["L"]), int(Z["d"])
SH = json.load(open(os.path.join(F0, "foundation_share.json")))["sizes"][SIZE]["plates"]
C = json.load(open(os.path.join(F0, "foundation_check.json")))[f"{SIZE}@143000"]["embed"]["c_med"]
REF = json.load(open(os.path.join(HERE, f"blankrow_{SIZE}.json"))); REF2 = json.load(open(os.path.join(HERE, f"where_blank2_{SIZE}.json")))
rng = np.random.default_rng([int(SIZE[:-1]), 999])
LANES = rng.permutation(d)

def rs(X, s):                                         # X x 2^-s rounded to nearest, s per element (Python whole numbers)
    f = np.frompyfunc(lambda x, k: (x << -k) if k <= 0 else ((x + (1 << (k - 1))) >> k), 2, 1)
    return f(X.astype(object), np.broadcast_to(s, X.shape).astype(object)).astype(np.int64)

def blank_writes(run):
    z = np.load(os.path.join(HERE, f"light_{SIZE}_{run}.npz")); WR = z["writes"]; P_ = WR.shape[0]
    B = np.zeros((L, P_, d)); FULL = np.zeros((L, P_, d))
    for j, (p, l) in enumerate([(f"{k}.{l}", l) for l in range(L) for k in ("proj", "mproj")]):
        x = z[f"light.{p}"].astype(np.int64); e = Z[p + ".e"].astype(np.int64)
        Wq = np.asarray(Z[p + (".w" if run == "finished" else ".s")]).astype(np.int64)
        b = Z[p + (".b" if run == "finished" else ".bs")].astype(np.int64)
        pin = rng.permutation(x.shape[1])
        Pw = x[:, pin] @ Wq[pin]                                          # int64 sums, input lanes in a random order
        tgt = WR[:, 1 + j] - b[None, :]
        num = np.abs(tgt).max(1).astype(np.float64); den = np.abs(np.ldexp(Pw.astype(np.float64), e[None, :])).max(1)
        sh = np.round(np.log2(np.maximum(num, 1) / np.maximum(den, 1e-300))).astype(np.int64)
        good = np.zeros(P_, bool)
        for dd in (0, -1, 1):                                             # the ratio lands within one place; verify exactly
            t = sh + dd; ok = ~good & (rs(Pw, -(e[None, :] + t[:, None])) == tgt).all(1); sh[ok] = t[ok]; good |= ok
        assert good.all(), f"{run} {p}: the size ratio did not give the write back at {(~good).sum()} positions"
        P0 = x[:, pin] @ np.asarray(Z[p + ".s"]).astype(np.int64)[pin]
        assert np.abs(P0).max() < 2 ** 53 and np.abs(Pw).max() < 2 ** 53
        sfac = SH[p]["s"] if run == "finished" else 1.0
        B[l] += sfac * np.ldexp(P0.astype(np.float64), e[None, :] + sh[:, None])
        FULL[l] += np.ldexp(Pw.astype(np.float64), e[None, :] + sh[:, None])
    o = slice(None, None, -1)
    return WR[o][:, :, LANES], B[:, o][:, :, LANES], FULL[:, o][:, :, LANES]

def fs(x): return math.fsum(np.asarray(x, dtype=np.float64).ravel().tolist())
def pair(a, b):
    ab, aa, bb = fs(a * b), fs(a * a), fs(b * b); ac, bc = a - a.mean(0), b - b.mean(0)
    return dict(cos=ab / math.sqrt(aa * bb), k=-2 * ab / (aa + bb), g=-ab / aa, cos_word=fs(ac * bc) / math.sqrt(fs(ac * ac) * fs(bc * bc)))
rows = []
def cmp(label, mine, ref, tol):
    dv = abs(mine - ref); rows.append(dv <= tol)
    log(f"  {label:58s} check {mine:+.6f}  run {ref:+.6f}  |diff| {dv:.1e}  {'SAME' if dv <= tol else 'DIFFERENT'} (tol {tol:.0e})")

refs = json.load(open(os.path.join(F0, f"refs_pythia-{SIZE}.json")))[:30]
IDS = np.concatenate([np.array(r["ids"] + r["ref"][:15], dtype=np.int64) for r in refs])[::-1]
EE = Z["embed.e"].astype(np.int64)
WR, B, FULL = blank_writes("finished"); Bs = B.sum(0)
row = WR[:, 0].astype(np.float64)                                         # the stream row as the machine added it
blank_row = np.ldexp(np.asarray(Z["embed.s"])[IDS].astype(np.float64), (EE[IDS] + ES)[:, None])[:, LANES]
log(f"SECOND PATH, pythia-{SIZE}")
for rk, b in (("whole", row), ("built_c", row - C * blank_row)):
    m = pair(Bs, b)
    for k in ("cos", "k", "g", "cos_word"): cmp(f"part 2 row {rk} {k}", m[k], REF["same_spot"][rk]["real"][k], 2e-4)
EW = Z["embed.w"]; T = np.zeros(d)
for i0 in range(0, EW.shape[0], 2048):
    T += np.ldexp(np.asarray(EW[i0:i0 + 2048]).astype(np.float64), (EE[i0:i0 + 2048] + ES)[:, None]).sum(0)
mB = Bs.mean(0)[None]; T = T[LANES][None]
cmp("part 3 mean B vs table lane totals (whole) cos", fs(mB * T) / math.sqrt(fs(mB * mB) * fs(T * T)), REF["spread_thin"]["table lane totals (whole)"]["real"]["cos"], 1e-6)
cen = lambda X: X - X.mean(0)
for l in (0, 1):
    b, w = cen(B[l]), cen(FULL[l]); lat = cen(WR[:, 3 + 2 * l:].sum(1).astype(np.float64))
    c = (b * w).sum(1) / (w * w).sum(1); bp = b - c[:, None] * w; bb = fs(b * b)
    cmp(f"where_blank2 L{l} lean", fs(b * w) / bb, REF2["layers"][l]["real"]["lean"], 1e-6)
    cmp(f"where_blank2 L{l} along", -fs(c[:, None] * w * lat) / bb, REF2["layers"][l]["real"]["along"], 1e-6)
    cmp(f"where_blank2 L{l} perp", -fs(bp * lat) / bb, REF2["layers"][l]["real"]["perp"], 1e-6)
del WR, B, FULL, Bs
WR, B, FULL = blank_writes("start"); Bs = B.sum(0)
m = pair(Bs, blank_row)
for k in ("cos", "k"): cmp(f"floor (blank machine) {k}", m[k], REF["floor"]["real"][k], 1e-6)
log(f"{sum(rows)} of {len(rows)} numbers the same within tolerance")
