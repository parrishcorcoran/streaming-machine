"""reach/answer.py: (2) ANSWER-EXACT MARGIN. Can a plate's low digits go without any top 5 changing, PROVEN by exact ranges?

HIS PICTURE, RESTATED (the session's words, not his): at each judge position the top 5 words stand in an order, each
one's score above the next by a gap, the 5th above the 6th. Removing a plate's digits below place b moves every pile of
that plate by at most (largest |light|) x |removed part| summed over the lines. That move is carried forward through
every later step of the machine as an exact range of whole numbers (reach/ival.py: the water levels, the knife edges
by the share table's own steps, the arrival, the top read). At the top every word's score has a range. The top 5
(set and order) cannot change where the 1st's low end is above the 2nd's high end, ..., the 5th's low end above the
high end of every word outside the top 5. If that holds at all 6,000 judge positions, the digits go; if not, they stay.

THE RELATION, applied per plate, no search: b = 1 first (the place-0 digits, the smallest removal there is). It holds
or it does not. Only if it holds at all 6,000 positions is b = 2 carried, and so on.
CONTROL (a measurement, not the method): the machine actually run with the same digits removed: each position's
largest real move of any word's score, and whether its top 5 (set and order) changed. Every range is checked to contain
the value the machine actually takes, at every bank, every stream and every score (a range that missed would be a
fault in the ranges).
SELF-CHECK (selftest=1): the range form carried with nothing removed must close to the machine's own value everywhere.
Rows: judge5's 300 windows (7,200 positions run, 6,000 read); the refs are checked only on a final removal.

PREDICTIONS (written before the run)
  The task's: the exact bounds through 12 layers of knife edges are loose, so the proven removal is small, while the
  measured effect shows much more could go.
  Claude's:
  - Proven removal: 0 digits on every plate, the top table included. The closest judge position has a gap of 2 units
    (2^-16 of a score unit) between two of its top words (reach/base_check.log: 21 positions below 16 units). At the
    top table alone, one place-0 sheet moves a score by up to sum over lines of |light| x gain, hundreds to thousands
    of units at every position, so even the top table fails at those positions, and every plate below it fails by
    more.
  - The bound at the 6,000 positions (largest score move it allows) for the top table: 10^2-10^4 units; for mproj.11:
    10^4-10^6; each layer further down multiplies it by 10-100 through the four banks; for plates in layers 0-5 the
    stream's range passes the light's own scale (the water level's size can reach 0) and the bound becomes unbounded.
  - The real effect of removing the place-0 digits of one plate: the largest score move 10-1,000 units, the top 5
    changing at 0-20 of 6,000 positions (only where gaps are a few units). Bound / real: about 3-30 for the top table,
    growing past 10^6 and then unbounded for the lower plates.
    python3 reach/answer.py [plates=head,mproj.11,...] [b=1] [selftest=1]   -> reach/answer/<plate>_b<b>.json, reach/answer.log
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastm as F
import ival as V
F.ROWS = 256
A = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
HERE = F.HERE; OUT = f"{HERE}/reach/answer"; os.makedirs(OUT, exist_ok=True)
SELF = A.get("selftest", "0") == "1"; B0 = int(A.get("b", 1))
PLATES = A["plates"].split(",") if "plates" in A else F.PLATES[::-1]
LOG = open(f"{HERE}/reach/answer{'_selftest' if SELF else ''}.log", "a"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
PLACES = 17

WIN = F.WIN_TEXTS; OFF = np.concatenate([[0], np.cumsum([len(t) for t in WIN])])
RROWS = np.concatenate([OFF[t] + np.arange(3, 23) for t in range(len(WIN))])            # the 6,000 judge rows
BT = np.load(f"{HERE}/reach/base_top.npz"); T5 = BT["top5"][480:]; S6 = BT["s6"][480:]
J5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); assert np.array_equal(T5, J5)
GAP = (S6[:, :5] - S6[:, 1:]).min(1)                                                       # each position's smallest gap

def removed_part(w16, b):                                      # (kept weights, removed part), int32, a piece of neurons at a time
    W16 = np.asarray(w16); keep = np.empty(W16.shape, np.int32); rem = np.empty(W16.shape, np.int32)
    for c0 in range(0, W16.shape[1], 4096):
        w = W16[:, c0:c0 + 4096].astype(np.int64); up, dn = F.signed_digits(w); k = np.zeros_like(w)
        for p in range(b, PLACES): k += (((up >> p) & 1) - ((dn >> p) & 1)) << p
        keep[:, c0:c0 + 4096] = k; rem[:, c0:c0 + 4096] = w - k
    return keep, rem

CACHE = f"{HERE}/reach/cache"
def make_cache():
    """the unchanged machine's stream at the start of every half layer on the 300 windows, written once (int32, checked)"""
    os.makedirs(CACHE, exist_ok=True)
    def hook(n, l, d):
        if n in ("stream in", "stream a"):
            assert np.abs(d).max() < 2 ** 31; np.save(f"{CACHE}/{n.replace(' ', '_')}_{l}.npy", d.astype(np.int32))
    F.run(texts=WIN, hook=hook)
def entry(name):
    """the unchanged machine's stream where the plate's half layer begins, on the 300 windows"""
    if name == "head":                                        # the finished stream: layer 11's neurons' half, unchanged
        R = np.load(f"{CACHE}/stream_a_11.npy").astype(np.int64)
        x16, ex, LS = F.gauge_log(R); Hp = F.gauged_kept(F.product_lean(x16, F.plate("fc.11").astype(np.float64)), F.E_["fc.11"], ex, LS, F.B_["fc.11"])
        x16, ex = F.to16_rows(F.edge_rows(Hp), -F.ES)
        return R + F.plain_kept(F.product_lean(x16, F.plate("mproj.11").astype(np.float64)), F.E_["mproj.11"], ex, F.B_["mproj.11"])
    l0 = int(name.split(".")[1]); key = "stream_in" if name.split(".")[0] in ("qkv", "proj") else "stream_a"
    f = f"{CACHE}/{key}_{l0}.npy"
    if not os.path.exists(f): make_cache()
    return np.load(f).astype(np.int64)

def pstats(X, real):
    """one piece of rows: each row's range half-width (-1 where blown), each row's real move, ranges that missed"""
    ok = ~X.bl; rm = np.where(ok, np.maximum(X.c - X.lo, X.hi - X.c).max(1), -1); rr = np.abs(real - X.c).max(1)
    missed = int((ok[:, None] & ((real < X.lo) | (real > X.hi))).sum())
    if SELF: assert X.bl.sum() == 0 and np.array_equal(X.lo, X.c) and np.array_equal(X.hi, X.c), "self-check: a range did not close"
    return rm, rr, missed, X.bl.copy()
def fin(where, parts):
    rm = np.concatenate([p[0] for p in parts]); rr = np.concatenate([p[1] for p in parts]); missed = sum(p[2] for p in parts)
    bl = np.concatenate([p[3] for p in parts])
    out = dict(where=where, blown_rows=int(bl.sum()), real_max=int(rr.max()), real_row_median=int(np.median(rr)), missed=missed)
    if (rm >= 0).any(): out.update(range_max=int(rm.max()), range_row_median=int(np.median(rm[rm >= 0])))
    assert missed == 0, f"a range missed the machine's actual value at {where}"
    return out
def stats(X, real, where):
    return fin(where, [pstats(X.rows(slice(r0, r0 + 256)), real[r0:r0 + 256]) for r0 in range(0, len(X.c), 256)])

def carry(name, b):
    l0 = 12 if name == "head" else int(name.split(".")[1]); half = 1 if name.split(".")[0] in ("fc", "mproj") else 0
    keep, r = removed_part(F.plate(name), b)
    if SELF: keep, r = np.asarray(F.plate(name), np.int32), np.zeros_like(r)
    nrem = sum(int(F.naf(r[:, c0:c0 + 4096]).sum()) for c0 in range(0, r.shape[1], 4096))   # digits removed (each a lit place)
    R0 = entry(name); R = V.exact(R0); Rm = R0.copy(); REC = []; settled = None
    def P(n): return V.Plate(n, F.plate(n), removed=r if n == name else None)
    def Wm(n): return (keep if n == name else np.asarray(F.plate(n))).astype(np.float64)
    def gk(Rm_, n):                                            # the machine's own gauged bank, the modified plate
        x16, ex, LS = F.gauge_log(Rm_); return F.gauged_kept(F.product_lean(x16, Wm(n)), F.E_[n], ex, LS, F.B_[n])
    def pk(X, n):
        x16, ex = F.to16_rows(X, -F.ES); return F.plain_kept(F.product_lean(x16, Wm(n)), F.E_[n], ex, F.B_[n])
    for l in range(l0, 12):
        if not (l == l0 and half == 1):
            n = f"qkv.{l}"; QKV = V.fill_rows(V.gauged, R, P(n)); QKVm = gk(Rm, n); REC.append(stats(QKV, QKVm, n))
            o = V.attention(QKV, OFF); om = np.concatenate([F.attention(QKVm[OFF[t]:OFF[t + 1]]) for t in range(len(WIN))])
            REC.append(stats(o, om, f"arrival.{l}")); del QKV, QKVm
            n = f"proj.{l}"; att = V.fill_rows(V.plain, o, P(n)); attm = pk(om, n); REC.append(stats(att, attm, n)); del o, om
            R = V.add(R, att); Rm = Rm + attm; REC.append(stats(R, Rm, f"stream a {l}")); del att, attm
            if R.bl.all(): settled = (l, 1); break
        n1, n2 = f"fc.{l}", f"mproj.{l}"; P1, P2 = P(n1), P(n2); A1, A2, A3, MO, MOM = [], [], [], [], []
        for r0 in range(0, len(R.c), 512):                    # the neurons' half, row by row: fc, the bend, mproj
            sl = slice(r0, r0 + 512); Rp = R.rows(sl); Rmp = Rm[sl]
            Hv = V.gauged(Rp, P1); Hm = gk(Rmp, n1); A1.append(pstats(Hv, Hm))
            Hb = V.bend(Hv); Hbm = F.edge_rows(Hm); A2.append(pstats(Hb, Hbm)); del Hv, Hm
            mo = V.plain(Hb, P2); mom = pk(Hbm, n2); A3.append(pstats(mo, mom)); del Hb, Hbm
            MO.append(mo); MOM.append(mom)
        REC += [fin(n1, A1), fin(f"bend.{l}", A2), fin(n2, A3)]
        mo = V.concat(MO); mom = np.concatenate(MOM); del MO, MOM, P1, P2
        R = V.add(R, mo); Rm = Rm + mom; REC.append(stats(R, Rm, f"stream {l}")); del mo, mom
        s = REC[-1]
        log(f"    {name} b={b}: layer {l:2d} stream range max {s.get('range_max', '-')}, row median {s.get('range_row_median', '-')}; "
            f"blown rows {s['blown_rows']}/{len(R.c)}; real move max {s['real_max']} ({time.time() - t0:.0f}s, {F.rss()} MB)")
        if R.bl.all() and l < 11: settled = (l + 1, 0); break
    if settled is not None:
        # every row's range is unbounded and stays so (a blown row never closes again): the bound has settled. Only the
        # control goes on: the machine itself, with the plate's digits removed, to the top.
        log(f"    {name} b={b}: every row unbounded after {REC[-1]['where']}; the real machine carried on to the top")
        got = {}
        def hk(n_, l_, d_):
            if (n_, l_) == ("stream", 11): got["R"] = d_.copy()
        F.run(texts=WIN, plates={name: keep}, hook=hk, start=(settled[0], settled[1], Rm)); Rm = got["R"]
        F.run(texts=WIN, hook=hk, start=(settled[0], settled[1], R.c)); Rc = got["R"]       # and the unchanged machine
        R = V.IV(Rc, Rc.copy(), Rc.copy(), np.ones(len(Rc), bool))     # every row blown: no range, only the center
    # ---------- the top read at the 6,000 judge rows ----------
    Rt = R.rows(RROWS); Rmt = Rm[RROWS]; del R, Rm
    WH = F.plate("head"); he, hb = F.E_["head"], F.B_["head"]; cut = r if name == "head" else None
    WHm = keep if name == "head" else WH
    NR = len(RROWS); lo5 = np.zeros((NR, 5), np.int64); hi5 = np.zeros((NR, 5), np.int64); other = np.full(NR, np.iinfo(np.int64).min)
    bound = np.zeros(NR, np.int64); real = np.zeros(NR, np.int64); t5m = np.zeros((NR, 5), np.int64); blown = Rt.bl.copy(); missed = 0
    RP, WC = 250, 8192
    for r0 in range(0, NR, RP):
        rs = slice(r0, min(r0 + RP, NR)); X = Rt.rows(rs); n_ = rs.stop - rs.start
        x16c, exc, LSc = F.gauge_log(X.c); x16m, exm, LSm = F.gauge_log(Rmt[rs])
        opts = []
        if (X.lo != X.hi).any() or cut is not None or V.FORCE:
            for sh, X16, ex in V.to16_opts(X):
                LSlo, LShi, b1 = V.size_range(X16, ex); opts.append((X16, ex, LSlo, LShi)); blown[rs] |= b1 | X16.bl
        Sm_full = np.zeros((n_, WH.shape[1]), np.int64)
        for c0 in range(0, WH.shape[1], WC):
            cs = slice(c0, min(c0 + WC, WH.shape[1])); w = WH[:, cs].astype(np.float64)
            Sc = F.power(F.product_lean(x16c, w), (he[None, cs] + exc[:, None] + F.ES) * 65536 - LSc[:, None]) + hb[cs]
            Sm = F.power(F.product_lean(x16m, WHm[:, cs].astype(np.float64)), (he[None, cs] + exm[:, None] + F.ES) * 65536 - LSm[:, None]) + hb[cs]
            Sm_full[:, cs] = Sm
            if opts:
                wp, wn = np.maximum(w, 0), np.minimum(w, 0); lo = hi = None
                for X16, ex, LSlo, LShi in opts:
                    _, Plo, Phi = V.pile_range(X16, w, wp, wn, None if cut is None else np.abs(cut[:, cs]).astype(np.float64))
                    base = (he[None, cs] + ex[:, None] + F.ES) * 65536
                    olo, ohi, b2 = V.power_range(Plo, Phi, base - LShi[:, None], base - LSlo[:, None]); blown[rs] |= b2
                    lo = olo if lo is None else np.minimum(lo, olo); hi = ohi if hi is None else np.maximum(hi, ohi)
                lo = lo + hb[cs]; hi = hi + hb[cs]
            else: lo, hi = Sc, Sc
            ok = ~blown[rs]
            missed += int((ok[:, None] & ((Sm < lo) | (Sm > hi))).sum())
            assert ((lo <= Sc) & (Sc <= hi))[ok].all(), "a score range lost its center"
            bound[rs] = np.maximum(bound[rs], np.maximum(Sc - lo, hi - Sc).max(1)); real[rs] = np.maximum(real[rs], np.abs(Sm - Sc).max(1))
            t5 = T5[rs]; inn = (t5 >= cs.start) & (t5 < cs.stop); loc = np.clip(t5 - cs.start, 0, cs.stop - cs.start - 1)
            lo5[rs] = np.where(inn, np.take_along_axis(lo, loc, 1), lo5[rs]); hi5[rs] = np.where(inn, np.take_along_axis(hi, loc, 1), hi5[rs])
            hm = hi.copy(); rr = np.repeat(np.arange(n_), 5)[inn.reshape(-1)]; hm[rr, loc[inn]] = np.iinfo(np.int64).min
            other[rs] = np.maximum(other[rs], hm.max(1))
            if SELF: assert np.array_equal(lo, Sc) and np.array_equal(hi, Sc) and np.array_equal(Sm, Sc)
        t5m[rs] = F.top5(Sm_full); del Sm_full
    assert missed == 0, "a score range missed the machine's actual score"
    proven = ~blown & (lo5[:, :4] > hi5[:, 1:]).all(1) & (lo5[:, 4] > other)
    changed = ~(t5m == T5).all(1)
    okr = ~blown
    res = dict(plate=name, b=b, digits_removed=nrem, rows=NR, proven=int(proven.sum()), blown_at_top=int(blown.sum()),
               bound_median=int(np.median(bound[okr])) if okr.any() else None, bound_max=int(bound[okr].max()) if okr.any() else None,
               bound_min=int(bound[okr].min()) if okr.any() else None,
               real_median=int(np.median(real)), real_max=int(real.max()), top5_changed=int(changed.sum()),
               top5_changed_rows=np.flatnonzero(changed).tolist()[:50],
               bound_over_real_median=float(np.median(bound[okr & (real > 0)] / real[okr & (real > 0)])) if (okr & (real > 0)).any() else None,
               worst_rows=[dict(row=int(i), gap=int(GAP[i]), bound=int(bound[i]) if okr[i] else None, real=int(real[i]), proven=bool(proven[i]))
                           for i in np.argsort(GAP)[:10]],
               layers=REC)
    json.dump(res, open(f"{OUT}/{name}_b{b}{'_selftest' if SELF else ''}.json", "w"), indent=1)
    log(f"  {name} b={b}: digits removed {nrem:,}; PROVEN at {res['proven']}/{NR} judge positions (blown at the top {res['blown_at_top']}); "
        f"bound on a score's move: median {res['bound_median']}, max {res['bound_max']}; REAL move: median {res['real_median']}, max "
        f"{res['real_max']}; top 5 changed (real) at {res['top5_changed']}/{NR}; bound/real median {res['bound_over_real_median']} "
        f"({time.time() - t0:.0f}s, {F.rss()} MB)")
    return res

if A.get("cache") == "1":
    make_cache(); log(f"cache written: {sorted(os.listdir(CACHE))} ({time.time() - t0:.0f}s, max memory {F.rss()} MB)"); sys.exit(0)
if SELF:
    V.FORCE = True
    log(f"SELF-CHECK: the range form with nothing removed, every step carried in range form ({PLATES})")
for name in PLATES:
    f = f"{OUT}/{name}_b{B0}{'_selftest' if SELF else ''}.json"
    if os.path.exists(f) and not SELF: log(f"  {name} b={B0}: already done ({f})"); continue
    b = B0
    while True:
        res = carry(name, b)
        if SELF or res["proven"] < res["rows"] or b >= PLACES: break
        b += 1                                                 # the relation held at every position: carry the next place
log(f"done ({time.time() - t0:.0f}s, max memory {F.rss()} MB)")
