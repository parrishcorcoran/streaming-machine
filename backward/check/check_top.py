"""CHECK 1 (independent): at 20 positions, does what STEP 1 says the top read needs of the final light (every line's
places >= c5, the rest unknown, the gauge bounded from the same light) really fix the top 5 (set and order)?

The checker's own prediction, written before the run: proved at c5 at 20/20 (the session's prover and the checker's are
the same exact bounds written twice, so they should agree); not proved at c5+1 at most of the 20; no adversarial light
inside the c5 box changes the top 5 (if one did, the bound would be wrong and the claim refuted); at c5+1 some
adversarial lights do change it at the close positions.

    python3 backward/check/check_top.py -> backward/check/check_top.log
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward/check")
from common import *
LOG = open(f"{CK}/check_top.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

REFS = json.load(open(f"{HERE}/refs_many.json")); RTEXT = [r["ids"] + r["ref"] for r in REFS]
WINS = json.load(open(f"{HERE}/judge5_texts.json"))
Zr = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); Zw = np.load(f"{HERE}/fringes_machine/windows_top_light.npz")
T5r = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")["top5"]; T5w = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5)
Cr = np.load(f"{HERE}/backward/out/top_refs.npz")["c5"]; Cw = np.load(f"{HERE}/backward/out/top_windows.npz")["c5"]
roff = np.concatenate([[0], np.cumsum([len(t) for t in RTEXT])]); assert roff[-1] == 648

rng = np.random.default_rng(81008)
def pick(C, plan):
    out = []
    for c, n in plan:
        cand = np.flatnonzero(C == c); out += sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())
    return out
SEL = [("refs", g) for g in pick(Cr, [(5, 4), (4, 2), (3, 2), (2, 2), (1, 2)])] + \
      [("windows", g) for g in pick(Cw, [(5, 3), (4, 1), (3, 1), (2, 1), (1, 2)])]
log(f"CHECK 1: {len(SEL)} positions (refs {sum(s == 'refs' for s, _ in SEL)}, windows {sum(s == 'windows' for s, _ in SEL)}); "
    f"the session's c5 at them: {[int((Cr if s == 'refs' else Cw)[g]) for s, g in SEL]}")

# the final light from the checker's own forward run (base2.py's functions), compared with the saved light
X = np.zeros((len(SEL), 768), np.int64); EX = np.zeros(len(SEL), np.int64); LSv = np.zeros(len(SEL), np.int64)
T5 = np.zeros((len(SEL), 5), np.int64); C5 = np.zeros(len(SEL), np.int64); same_saved = 0
cache = {}
for i, (s, g) in enumerate(SEL):
    if s == "refs":
        t = int(np.searchsorted(roff, g, side="right") - 1); p = g - roff[t]; key = ("r", t); ids = RTEXT[t]
        Zx, Zex, ZLS, t5, c5 = Zr["x16"][g], Zr["ex"][g], Zr["LS"][g], T5r[g], Cr[g]
    else:
        t, p = divmod(g, 20); p += 3; key = ("w", t); ids = WINS[t]["ids"]
        Zx, Zex, ZLS, t5, c5 = Zw["x16"][g], Zw["ex"][g], Zw["LS"][g], T5w[g], Cw[g]
    if key not in cache: cache[key] = forward_keep(ids, keep=-1)
    rec = cache[key]
    X[i], EX[i], LSv[i], T5[i], C5[i] = rec["x16"][p], rec["ex"][p], rec["LS"][p], rec["top5"][p], c5
    assert gauge_exact(X[i:i + 1], EX[i:i + 1])[0] == LSv[i]
    same_saved += int(np.array_equal(X[i], Zx) and EX[i] == Zex and LSv[i] == ZLS and np.array_equal(T5[i], t5))
del cache
log(f"  own forward run: light, level, gauge and top 5 equal to the saved ones at {same_saved}/{len(SEL)} ({time.time() - t0:.0f}s)")

def place_box(x, c):
    """the light known by its places >= c only. A line whose read part K is not 0 has its sign known and |x| in
    [K, K + 2^c - 1]; a line whose read part is 0 is anywhere in [-(2^c - 1), 2^c - 1]."""
    c = np.asarray(c, np.int64)[:, None]; u = (np.int64(1) << c) - 1; K = (np.abs(x) >> c) << c
    lo = np.where(K == 0, -u, np.where(x > 0, K, -K - u)); hi = np.where(K == 0, u, np.where(x > 0, K + u, -K))
    assert ((lo <= x) & (x <= hi)).all()
    return lo, hi

def branch(lo, hi, ex):
    LSlo, LShi = gauge_range(lo, hi, ex)
    return [dict(xlo=lo, xhi=hi, ex=ex, LSlo=LSlo, LShi=LShi, valid=np.ones(len(lo), bool))]

lo, hi = place_box(X, C5); assert ((gauge_range(lo, hi, EX)[0] <= LSv) & (LSv <= gauge_range(lo, hi, EX)[1])).all()
p_at, np_at = prove_order(branch(lo, hi, EX), T5)
log(f"  proved at the session's c5 (checker's own prover): {int(p_at.sum())}/{len(SEL)}; pair bounds used per position {np_at.tolist()}")
cz = C5 + 1; lo1, hi1 = place_box(X, cz)
p_up, _ = prove_order(branch(lo1, hi1, EX), T5)
log(f"  proved one place coarser (c5+1): {int(p_up.sum())}/{len(SEL)}  (0 expected if c5 is the coarsest)")

# adversarial lights inside the box: for each requirement pair (a, b), every unread grain falls against a
def machine_top5(x, ex):
    LS = gauge_exact(x, ex); return top5_of(head_scores(x, ex, LS))
def attack(lo, hi, x, t5, ex, n_ch=30, n_rand=10):
    """returns the number of lights tried and how many changed the top 5"""
    S = head_scores(x[None], np.array([ex]), gauge_exact(x[None], np.array([ex])))[0]
    rest = np.argsort(-S); rest = [int(j) for j in rest if j not in set(t5.tolist())][:n_ch]
    pairs = [(t5[k], t5[k + 1]) for k in range(4)] + [(t5[4], j) for j in rest]
    L = []
    for a, b in pairs:
        d = HW[:, a].astype(np.int64) - HW[:, b].astype(np.int64)
        L.append(np.where(d >= 0, lo, hi))                         # the light that makes pile_a - pile_b smallest
    for _ in range(n_rand): L.append(lo + (rng.random(768) * (hi - lo + 1)).astype(np.int64).clip(0, hi - lo))
    L = np.stack(L); bad = 0
    for q in range(0, len(L), 16):
        T = machine_top5(L[q:q + 16], np.full(len(L[q:q + 16]), ex)); bad += int((T != t5[None]).any(1).sum())
    return len(L), bad
tried = bad = 0; flips_up = 0; tried_up = 0
for i in range(len(SEL)):
    n, b = attack(lo[i], hi[i], X[i], T5[i], EX[i]); tried += n; bad += b
    n, b = attack(lo1[i], hi1[i], X[i], T5[i], EX[i]); tried_up += n; flips_up += int(b > 0)
log(f"  adversarial and random lights inside the c5 box: {tried} tried, top 5 changed by {bad}  (0 required)")
log(f"  the same one place coarser (c5+1): positions where some light changes the top 5: {flips_up}/{len(SEL)}")
for i, (s, g) in enumerate(SEL):
    log(f"    {s:7s} {g:5d}: c5 {C5[i]}  proved at c5 {bool(p_at[i])}, at c5+1 {bool(p_up[i])}")

# the top plate's extra at the 12 refs positions, against the session's saved per-position numbers
Zt = np.load(f"{HERE}/backward/out/top_refs.npz"); ri = [i for i, (s, _) in enumerate(SEL) if s == "refs"]; rg = [SEL[i][1] for i in ri]
xs = X[ri]; P_ = len(ri); sg = np.sign(xs); ax = np.abs(xs)
Xs = np.concatenate([(sg * ((ax >> a) & 1)) for a in range(16)]).astype(np.float32)        # (16 P, 768), rows a-major
Xu = np.abs(Xs)
ext = np.zeros(P_, np.int64); meet = np.zeros(P_, np.int64); ext_read = np.zeros(P_, np.int64)
aw = np.abs(HW.astype(np.int32)); sw = np.sign(HW.astype(np.int32))
for b in range(16):
    bit = ((aw >> b) & 1)
    Ts = (Xs @ (sw * bit).astype(np.float32)).reshape(16, P_, NW)                            # exact: |.| <= 768
    Cu = (Xu @ bit.astype(np.float32)).reshape(16, P_, NW)
    A_ = np.abs(Ts).sum(2, dtype=np.float64).astype(np.int64)                                # (16 a, P)
    ext += A_.sum(0); meet += Cu.sum((0, 2), dtype=np.float64).astype(np.int64)
    ext_read += np.where(np.arange(16)[:, None] >= C5[ri][None, :], A_, 0).sum(0)
    del Ts, Cu, bit
log(f"  top plate at the {P_} refs positions: meetings equal to the session's plain_meet {int((meet == Zt['plain_meet'][rg]).sum())}/{P_}; "
    f"extra equal to the session's ext_all {int((ext == Zt['ext_all'][rg]).sum())}/{P_}; read-part extra equal {int((ext_read == Zt['ext_read'][rg]).sum())}/{P_}")
log(f"  extra at these positions {ext.sum() / meet.sum():.2%} of the top plate's meetings")
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
