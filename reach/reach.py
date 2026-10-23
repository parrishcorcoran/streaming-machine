"""reach/reach.py: (1) BIT-EXACT REACH. Which weight digits can never reach a kept value, at any position.

HIS PICTURE, RESTATED (the session's words, not his): a lit digit at place b on line i of neuron j adds exactly
+-2^b x (the light on line i) to that neuron's pile. The pile is never kept whole: the bank's own water level (qkv, fc,
the top read: the gauge's power, a table read and a shift) or its own shift (proj, mproj) keeps only the places of the
pile that survive the rounding. A pile can move anywhere inside its rounding bucket and the kept value stays the same.
So a set of digits can go if, at every position, the sum of their largest possible moves (each +-2^b x |light|,
summed so they cannot add up past it) stays inside the bucket on both sides.

THE RELATION, applied as it stands (no search, no winner): for neuron j, at position t, the bucket's near edge is
m[t, j] = min(pile - lowest pile with the same kept value, highest pile with the same kept value - pile), found exactly
from the bank's own rounding (checked: the kept value at both ends is the same, one step past either end differs).
The digits are taken a whole place at a time from place 0 up, the order in which they land on the pile's low places:
place p goes if sum over places q <= p of 2^q x (sum over the lines lit at place q of |light|) <= m[t, j] at every one
of the positions. A line whose light is 0 at every position moves the pile by exactly 0, so its digits go at any place.
Both stated per neuron, every neuron of every plate.
  - the banks: the light at all 7,848 positions that run through the layers (648 refs + 300 windows x 24);
  - the top table: its kept values are every word's score at the 6,480 read positions (480 refs + 6,000 judge).
  - "an exact bound over all of them": also the stricter uniform form, sum over the set of 2^q x (largest |light| on
    the line over all positions) <= the smallest m over all positions, which holds for any light no larger than
    any seen.
THEN: the machine with exactly those digits removed, run on all 7,848 positions: every kept value of every bank, every
light, every stream, the light at the top and every word's score at the 6,480 read positions compared with the
unchanged machine (sha256 of each array). Piles move only by the removed digits' light.
COUNTED: digits removed (share of all lit signed digits), photon meetings removed (on each line, the light's lit places
in signed digits x the removed digits; share of the plates' lit-pixel meetings, as recording/C_machine.py counts them),
and the footprint: every changed plate packed again by the planes coder (shrink/planes_rc.dylib, single-wave entry,
v9 contexts, the plate's own line order, as recording/C_store.py's control (i)), before and after.
CONTROLS: SHUFFLE, each neuron's edge distances m[t, j] shuffled across positions (same values, pairing with the light
broken), the same rule; ORDER-FREE, one bank redone with the positions and lines in a random order.

PREDICTIONS (written before the run)
  The task's: 2-5% of digits removed (the review: under 5%).
  Claude's: under 0.1% of digits, and about 0% of photon meetings. With 7,848 positions, some pile of every neuron sits
    within a few units of its bucket's edge (a pile lands within m of an edge about 2m / bucket of the time, so the
    closest of 7,848 is about bucket / 16,000 away, and often exactly at the edge), while one place-0 digit alone moves
    the pile by the light on its line, tens to thousands of units. So no neuron loses even its place-0 sheet. What goes
    is only digits on lines whose light is 0 at every position (bend outputs that stay shut, feeding mproj), and those
    meet no photons, so they save storage only. The uniform form removes the same or less. The shuffle control gives
    the same count (the result is set by the closest edge and the light's size, not by which position meets which).
    The machine with them removed: every kept value, light, stream and score identical.
    python3 reach/reach.py      -> reach/reach.log, reach/reach.json, reach/reach_removed.npz
"""
import sys, os, json, time, hashlib, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastm as F
HERE = F.HERE; OUT = f"{HERE}/reach"; LOG = open(f"{OUT}/reach.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
PLACES = 17; RNG = np.random.default_rng(20261005)

# ---------- the bank's own rounding: the bucket of piles that give the same kept value ----------
def bucket_gauged(P, F_):
    """kept = power(P, F) = rshift(P x EXP2[f], 16 - I). Returns (lo, hi) of P with the same kept value."""
    I, f = np.right_shift(F_, 16), np.bitwise_and(F_, 0xFFFF); g = F.EXP2[f]; u = 16 - I
    lo = P.copy(); hi = P.copy(); s = u > 0
    us = np.where(s, u, 1); h = np.left_shift(np.int64(1), us - 1)
    k = np.right_shift(P * g + h, us); A = np.left_shift(k, us) - h; Bv = A + np.left_shift(np.int64(1), us) - 1
    lo = np.where(s, -((-A) // g), P); hi = np.where(s, Bv // g, P)
    return lo, hi
def bucket_plain(P, sh):
    """kept = rshift(P, sh). Returns (lo, hi)."""
    s = sh > 0; ss = np.where(s, sh, 1); h = np.left_shift(np.int64(1), ss - 1)
    k = np.right_shift(P + h, ss); lo = np.left_shift(k, ss) - h; hi = lo + np.left_shift(np.int64(1), ss) - 1
    return np.where(s, lo, P), np.where(s, hi, P)
def margins(P, kept_fn, lo, hi):
    """m = min(P - lo, hi - P), and the bucket checked exactly: same kept value at lo and hi, a different one past them
    (where the bucket is wider than one pile)."""
    K = kept_fn(P); assert np.array_equal(kept_fn(lo), K) and np.array_equal(kept_fn(hi), K)
    assert (kept_fn(lo - 1) != K).all() and (kept_fn(hi + 1) != K).all()
    return np.minimum(P - lo, hi - P)

# ---------- the relation, per neuron ----------
def places_out(absx, lit, m, mshuf, cols=None):
    """b[j]: how many places from place 0 up go (the relation at every position); also for the shuffled edges.
    absx: (T, L) |light| float64; lit: (L, N) lit-place masks; m, mshuf: (T, N). Returns b, bshuf, first-place stats."""
    T, N = m.shape; b = np.zeros(N, np.int64); bs = np.zeros(N, np.int64)
    alive = np.arange(N); alive_s = np.arange(N); cum = np.zeros((T, N), np.int64); stat = {}
    for p in range(PLACES):
        need = np.union1d(alive, alive_s)
        if len(need) == 0: break
        Lp = ((lit[:, need] >> p) & 1).astype(np.float64)
        Cp = absx @ Lp; assert Cp.max(initial=0) < 2 ** 52
        cum[:, need] += np.left_shift(Cp.astype(np.int64), p)
        ok = (cum[:, alive] <= m[:, alive]).all(0); oks = (cum[:, alive_s] <= mshuf[:, alive_s]).all(0)
        if p == 0:
            over = cum > m                                     # positions where place 0 alone crosses the edge
            stat = dict(place0_neurons_failing=int((over.any(0)).sum()), place0_positions_crossing_median=int(np.median(over.sum(0))))
        b[alive[ok]] = p + 1; bs[alive_s[oks]] = p + 1; alive = alive[ok]; alive_s = alive_s[oks]
    return b, bs, stat
def places_out_uniform(A, lit, mmin):
    N = lit.shape[1]; b = np.zeros(N, np.int64); alive = np.arange(N); cum = np.zeros(N, np.int64)
    for p in range(PLACES):
        if len(alive) == 0: break
        cum[alive] += np.left_shift((A @ ((lit[:, alive] >> p) & 1).astype(np.float64)).astype(np.int64), p)
        ok = cum[alive] <= mmin[alive]; b[alive[ok]] = p + 1; alive = alive[ok]
    return b

# ---------- the planes coder, both ways (recording/C_store.py's code(), the same library and contexts) ----------
LIB = ctypes.CDLL(f"{HERE}/shrink/planes_rc.dylib")
_P, _I, _i = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
LIB.naf_planes.argtypes = [_i, _P, _I, _I, _i, _i, _P, _I, _P]; LIB.naf_planes.restype = _I
MODE = 1 | 32 | 64
def code(A):
    A = np.ascontiguousarray(A, np.int32); R, C = A.shape; nb = int(np.abs(A).max()).bit_length() + 1
    cap = A.size * 3 + (1 << 20); buf = np.zeros(cap, np.uint8); mark = np.zeros(nb + 1, np.int64)
    n = LIB.naf_planes(0, A.ctypes.data, R, C, nb, MODE, buf.ctypes.data, cap, mark.ctypes.data); assert n <= cap
    data = buf[:n].copy(); del buf
    Bk = np.zeros_like(A); m2 = np.zeros(nb + 1, np.int64)
    LIB.naf_planes(1, Bk.ctypes.data, R, C, nb, MODE, data.ctypes.data, n, m2.ctypes.data)
    return int(n), bool(np.array_equal(Bk, A))

# ---------- one plate: the relation, the count, the plate with its digits removed (a piece of neurons at a time) ----------
RES = {}; NEW = {}; CH = 512
def plate_pass(name, x16, piece, order_free=False):
    """piece(cs) -> m (positions x neurons cs): each pile's distance to its bucket's near edge."""
    w16 = F.plate(name); L, N = w16.shape; T = len(x16)
    absx = np.abs(x16).astype(np.float64); A = absx.max(0); Z = A == 0; nx = F.naf(x16).sum(0)
    r = dict(lines=L, neurons=N, positions=T, digits=0, removed=0, removed_uniform=0, removed_shuffled=0, dark_lines=int(Z.sum()),
             neurons_with_places_out=0, places_out_hist=[0] * (PLACES + 1), meetings=0, meetings_removed=0, edge_zero=0, cells=0,
             place0_neurons_failing=0, min_edge=[], light_median=int(np.median(np.abs(x16))))
    keep = np.empty((L, N), np.int16); removed_line = np.zeros(L, np.int64); nrow_all = np.zeros(L, np.int64)
    for c0 in range(0, N, CH):
        cs = slice(c0, min(c0 + CH, N)); w = w16[:, cs].astype(np.int64)
        up, dn = F.signed_digits(w); lit = up | dn
        m = piece(cs); mshuf = RNG.permuted(m, axis=0)            # SHUFFLE: each neuron's edges across positions
        b, bs, st = places_out(absx, lit, m, mshuf); bU = places_out_uniform(A, lit, m.min(0))
        if order_free:                                            # ORDER-FREE: positions and lines in a random order
            pt, pl = RNG.permutation(T), RNG.permutation(L)
            b2, _, _ = places_out(absx[pt][:, pl], lit[pl], m[pt], m[pt])
            r["order_free_identical"] = r.get("order_free_identical", True) and bool(np.array_equal(b, b2))
        nrow_all += np.bitwise_count(lit).sum(1).astype(np.int64)
        kc = np.zeros_like(w)
        for p in range(PLACES):
            lp = ((lit >> p) & 1).astype(bool)
            go = lp & ((p < b)[None, :] | Z[:, None])
            removed_line += go.sum(1)
            r["removed_uniform"] += int((lp & ((p < bU)[None, :] | Z[:, None])).sum())
            r["removed_shuffled"] += int((lp & ((p < bs)[None, :] | Z[:, None])).sum())
            kc += ((((up >> p) & 1) - ((dn >> p) & 1)) * (~go)) << p
        keep[:, cs] = kc
        r["neurons_with_places_out"] += int((b > 0).sum())
        for v in b: r["places_out_hist"][int(v)] += 1
        r["edge_zero"] += int((m == 0).sum()); r["cells"] += int(m.size); r["place0_neurons_failing"] += st["place0_neurons_failing"]
        r["min_edge"].append(m.min(0))
        del m, mshuf, w, up, dn, lit, kc
    r["digits"] = int(nrow_all.sum()); r["removed"] = int(removed_line.sum())
    r["meetings"] = int(nx @ nrow_all); r["meetings_removed"] = int(nx @ removed_line)
    me = np.concatenate(r.pop("min_edge")); r["min_edge_median"] = int(np.median(me)); r["min_edge_zero_neurons"] = int((me == 0).sum())
    if r["removed"]: NEW[name] = keep
    RES[name] = r
    log(f"  {name:8s} {L:4d}x{N:5d}: digits {r['digits']:>10,} removed {r['removed']:>8,} ({100 * r['removed'] / r['digits']:.4f}%) "
        f"[uniform {r['removed_uniform']:,}; shuffled {r['removed_shuffled']:,}]; dark lines {r['dark_lines']}; neurons with >=1 place out "
        f"{r['neurons_with_places_out']}; meetings removed {r['meetings_removed']:,} of {r['meetings']:,}; "
        f"edge 0 at {r['edge_zero']:,}/{r['cells']:,} piles; neurons whose closest pile is at its edge {r['min_edge_zero_neurons']}/{N}; "
        f"median |light| {r['light_median']}; place 0 alone crosses at {r['place0_neurons_failing']}/{N} neurons"
        + (f"; order-free identical {r['order_free_identical']}" if order_free else "") + f" ({time.time() - t0:.0f}s, {F.rss()} MB)")

# ---------- run 1: the unchanged machine; every bank's relation; hashes of everything kept ----------
H1 = {}
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def hook1(name, l, d):
    if name == "stream in": return
    if name in ("stream a", "stream"): H1[f"{name} {l}"] = sha(d); return
    x16, P, ex, LS, we = d["x16"], d["pile"], d["ex"], d["LS"], d["we"]
    H1[f"{name} kept"] = sha(d["kept"]); H1[f"{name} light"] = sha(x16); H1[f"{name} pile"] = sha(P)
    def piece(cs):
        Pc = P[:, cs]
        if LS is not None:
            F_ = (we[None, cs] + ex[:, None] + F.ES) * 65536 - LS[:, None]
            lo, hi = bucket_gauged(Pc, F_); return margins(Pc, lambda X: F.power(X, F_), lo, hi)
        sh = -(we[None, cs] + ex[:, None] + F.ES)
        lo, hi = bucket_plain(Pc, sh); return margins(Pc, lambda X: F.rshift(X, sh), lo, hi)
    plate_pass(name, x16, piece, order_free=(name == "proj.0"))
log("RUN 1: the unchanged machine, all 7,848 positions; the relation at every bank")
x16, ex, LS = F.run(hook=hook1, log=log)
RR = F.READ_ROWS; xt, et, lt = x16[RR], ex[RR], LS[RR]; H1["top light"] = sha(np.concatenate([x16, ex[:, None], LS[:, None]], 1))

# ---------- the top table: its kept values are every word's score at the 6,480 read positions ----------
log("THE TOP TABLE at the 6,480 read positions")
WH = F.plate("head"); HE = F.E_["head"]
def piece_head(cs):
    P = F.product(xt, WH[:, cs].astype(np.float64)); F_ = (HE[None, cs] + et[:, None] + F.ES) * 65536 - lt[:, None]
    lo, hi = bucket_gauged(P, F_); return margins(P, lambda X: F.power(X, F_), lo, hi)
CH = 2048; plate_pass("head", xt, piece_head)
H1["scores"] = [sha(S) for _, S in F.top_scores(xt, et, lt)]

# ---------- run 2: the machine with exactly those digits removed ----------
log(f"RUN 2: the machine with the digits removed ({len(NEW)} plates changed: {sorted(NEW)})")
H2 = {}
def hook2(name, l, d):
    if name == "stream in": return
    if name in ("stream a", "stream"): H2[f"{name} {l}"] = sha(d); return
    H2[f"{name} kept"] = sha(d["kept"]); H2[f"{name} light"] = sha(d["x16"]); H2[f"{name} pile"] = sha(d["pile"])
x16b, exb, LSb = F.run(plates={k: v for k, v in NEW.items() if k != "head"}, hook=hook2, log=log)
H2["top light"] = sha(np.concatenate([x16b, exb[:, None], LSb[:, None]], 1))
H2["scores"] = [sha(S) for _, S in F.top_scores(x16b[RR], exb[RR], LSb[RR], head=NEW.get("head"))]
diff_kept = [k for k in H1 if not k.endswith("pile") and H1[k] != H2[k]]
diff_pile = [k for k in H1 if k.endswith("pile") and H1[k] != H2[k]]
log(f"  every kept value, light, stream, the top light and every word's score at 6,480 positions: "
    f"{'IDENTICAL' if not diff_kept else 'DIFFERENT at ' + str(diff_kept)}; piles that moved (inside their buckets): {len(diff_pile)} {diff_pile[:10]}")

# ---------- footprint: every changed plate packed again ----------
FOOT = {}
for name, wn in NEW.items():
    n0, ok0 = code(F.plate(name)); n1, ok1 = code(wn); assert ok0 and ok1
    FOOT[name] = dict(before=n0, after=n1)
    log(f"  packed {name}: {n0:,} -> {n1:,} bytes (decoded back equal)")

tot = {k: sum(RES[p][k] for p in RES) for k in ("digits", "removed", "removed_uniform", "removed_shuffled", "meetings", "meetings_removed")}
pl = {k: sum(RES[p][k] for p in RES if p != "head") for k in ("digits", "removed", "meetings", "meetings_removed")}
log(f"\nALL 49 PLATES: digits removed {tot['removed']:,} of {tot['digits']:,} ({100 * tot['removed'] / tot['digits']:.5f}%); "
    f"uniform form {tot['removed_uniform']:,}; shuffle control {tot['removed_shuffled']:,}")
log(f"  48 plates alone: {pl['removed']:,} of {pl['digits']:,} ({100 * pl['removed'] / pl['digits']:.5f}%); photon meetings removed "
    f"{pl['meetings_removed']:,} of {pl['meetings']:,} ({100 * pl['meetings_removed'] / max(pl['meetings'], 1):.5f}%) over 7,848 positions")
log(f"  with the top table: meetings removed {tot['meetings_removed']:,} of {tot['meetings']:,}")
dB = sum(v["before"] - v["after"] for v in FOOT.values())
log(f"  footprint: changed plates packed {sum(v['before'] for v in FOOT.values()):,} -> {sum(v['after'] for v in FOOT.values()):,} bytes "
    f"(saves {dB:,} bytes); against 153.01 MB (48 plates, own line order, same coder) + the top table")
json.dump(dict(plates=RES, footprint=FOOT, totals=tot, plates48=pl, identical=not diff_kept, piles_moved=diff_pile), open(f"{OUT}/reach.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/reach_removed.npz", **{k: v for k, v in NEW.items()})
log(f"saved reach/reach.json ({time.time() - t0:.0f}s, max memory {F.rss()} MB)")
