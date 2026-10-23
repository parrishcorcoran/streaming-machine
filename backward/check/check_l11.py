"""CHECK 2 and 3 (independent): layer 11's fc and mproj plates read top diagonal first.

For every pile of fc.11 and mproj.11 at every judged position: the checker's own diagonal tables (place pairs (a, b) of
the light and the plate, a + b = d), the read part K_D (diagonals >= D), the plain bound on the unread part
B_D = sum over a + b < D of 2^(a+b) x min(lines lit at light place a, lines lit at plate place b).
  - the bound must cover the unread remainder: |pile - K_D| <= B_D at every D, at every pile;
  - (a) stop: the largest D at which what the pile hands on is the machine's own digit for digit, for every value
    the interval allows (fc: the bend's output after the bend's water level, and that level; mproj.11: its line of
    the final light after the last water level, and that level);
  - (b) stop: (a), or the output interval no wider than 2^k, k from the session's req files (the box +-2^k proved there);
  - the work: meetings of the diagonals read; the extra: |net count| per place pair, summed.
Then each plate, cut alone at its stops, is carried to the top by the checker's own interval steps:
  (a) the final light, its level and its gauge must be the machine's own;  (b) the top 5 (set and order) must be proved.
And the session's req boxes at Mw and H (+-2^k) are carried up and proved (and +-2^(k+1) tried).

The checker's prediction, written before the run: the counts reproduce the session's (refs: mproj.11 (a) 89.67%,
(b) 84.67%, extra 3.34%; fc.11 (a) 89.20%, (b) 88.79%, extra 7.02%) to the last meeting if both read the same rule; the
bound holds at every pile and stage (it is a count of lit meetings); (a) gives the machine's own final light 648/648 and
300/300; (b) proves 647/648 (refs position 494 the exact tie) and 300/300 on the windows.

    python3 backward/check/check_l11.py set=refs|windows   -> backward/check/check_l11_<set>.log / .json
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward/check")
from common import *
ARG = dict(a.split("=") for a in sys.argv[1:]); SET = ARG.get("set", "refs")
LOG = open(f"{CK}/check_l11_{SET}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if SET == "refs":
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    ROWS = [np.arange(len(t)) for t in TEXTS]; T5ALL = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")["top5"]
    Q = np.load(f"{HERE}/backward/out/req_refs_L11.npz"); GIDX = np.arange(648)
else:
    WINS = json.load(open(f"{HERE}/judge5_texts.json")); WSEL = list(range(0, 300, 20))
    TEXTS = [WINS[w]["ids"] for w in WSEL]; ROWS = [np.arange(3, 23) for _ in WSEL]
    T5ALL = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5)
    Q = np.load(f"{HERE}/backward/out/req_windows_L11_QKVq-A-H-Mw.npz")
    GIDX = np.concatenate([np.arange(20 * w, 20 * w + 20) for w in WSEL])
KH, KM = Q["11__H"][GIDX], Q["11__Mw"][GIDX]
T5 = T5ALL[GIDX]; NPOS = len(GIDX)
log(f"CHECK 2-3: {SET}, {len(TEXTS)} texts, {NPOS} positions; layer 11's fc and mproj plates")

def planes_plate(name):
    w = M[name + ".w"].astype(np.int32); a = np.abs(w); s = np.sign(w)
    Wu = [((a >> b) & 1).astype(np.float32) for b in range(16)]
    Ws = [(s * ((a >> b) & 1)).astype(np.float32) for b in range(16)]
    m_b = np.stack([((a >> b) & 1).sum(0) for b in range(16)]).astype(np.int64)        # (16, M) lit lines per place
    return Ws, Wu, m_b

def tables(x, Ws, Wu, m_b):
    """x (P, N): Td (31,P,M) signed net count x 2^d; Cd meetings; Ed |net count| summed over the pairs; Bd bound terms"""
    P, N = x.shape; M_ = m_b.shape[1]; ax = np.abs(x); sg = np.sign(x); assert ax.max() <= 1 << 15
    xu = np.concatenate([((ax >> a) & 1) for a in range(16)]).astype(np.float32)          # (16P, N), a-major
    xs = np.concatenate([sg * ((ax >> a) & 1) for a in range(16)]).astype(np.float32)
    n_a = np.stack([((ax >> a) & 1).sum(1) for a in range(16)]).astype(np.int64)          # (16, P)
    Td = np.zeros((31, P, M_), np.int64); Cd = np.zeros_like(Td); Ed = np.zeros_like(Td); Bd = np.zeros_like(Td)
    for b in range(16):
        T = (xs @ Ws[b]).reshape(16, P, M_); C = (xu @ Wu[b]).reshape(16, P, M_)
        assert np.abs(T).max() <= N and C.max() <= N
        T = T.astype(np.int64); C = C.astype(np.int64)
        for a in range(16):
            d = a + b; Td[d] += T[a] << d; Cd[d] += C[a]; Ed[d] += np.abs(T[a])
            Bd[d] += np.minimum(n_a[a][:, None], m_b[b][None, :]) << d
    return Td, Cd, Ed, Bd

def fixed(lo, hi, sh, amax, cap_ok=None):
    """(.., P, M) intervals of what one water level reads; sh (P,) that level; amax (P,) the line holding the largest
    value. Fixed: the line's digits after the level are one number, and the level cannot move (no value of the line
    reaches 2^(sh+15); the largest line cannot fall below 2^(sh+14))."""
    s = sh[:, None]; one = rshift(lo, s) == rshift(hi, s)
    under = np.maximum(np.abs(lo), np.abs(hi)) < (np.int64(1) << (s + 15))
    mlo = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(np.abs(lo), np.abs(hi)))
    floor = np.where(sh > 0, np.int64(1) << np.maximum(sh + 14, 0), 0)[:, None]
    isam = np.arange(lo.shape[-1])[None, :] == amax[:, None]
    return one & under & (~isam | (mlo >= floor))

def stop_of(cond):
    """cond (32, P, M): the largest D with cond true at D and at every D below it; also the largest D with cond true"""
    good = np.cumprod(cond, axis=0).astype(bool)
    return np.where(good, np.arange(32)[:, None, None], -1).max(0), np.where(cond, np.arange(32)[:, None, None], -1).max(0)

BIGX = np.int64(1) << 44
REC = {nm: dict(plain=0, extra=0, a=0, b=0, hist_a=np.zeros(32, np.int64), hist_b=np.zeros(32, np.int64), piles=0,
                bound_bad=0, nest_bad=0) for nm in ("fc", "mproj")}
VER = dict(a_fc=0, a_mproj=0, b_fc=0, b_mproj=0, b_fc_tie_exact=0, b_mproj_tie_exact=0, req_Mw=0, req_Mw_n=0, req_Mw_up=0,
           req_H=0, req_H_n=0, req_H_up=0, own_top5=0, own_light=0)
wf, wef, bfc = M["fc.11.w"], M["fc.11.e"].astype(np.int64), M["fc.11.b"].astype(np.int64)
wm, wem, bm = M["mproj.11.w"], M["mproj.11.e"].astype(np.int64), M["mproj.11.b"].astype(np.int64)
PL = {"fc": planes_plate("fc.11"), "mproj": planes_plate("mproj.11")}

def mproj_box(Elo, Ehi):
    """mproj.11 on a box of the bend's outputs: every water level the box allows, the pile's exact range, the shift,
    the bias; the union over the levels (the output is in the stream's own units at every level)"""
    s0, s1 = lvl_range(Elo, Ehi); olo = ohi = None
    for o in range(int((s1 - s0).max()) + 1):
        sh = s0 + o; ok = sh <= s1
        plo, phi = dot_box(rshift(Elo, sh[:, None]), rshift(Ehi, sh[:, None]), wm)
        s = -(wem[None, :] + (sh - ES)[:, None] + ES)
        a_, b_ = rshift(plo, s) + bm, rshift(phi, s) + bm
        if olo is None: olo, ohi = np.where(ok[:, None], a_, np.iinfo(np.int64).max), np.where(ok[:, None], b_, np.iinfo(np.int64).min)
        else: olo = np.where(ok[:, None], np.minimum(olo, a_), olo); ohi = np.where(ok[:, None], np.maximum(ohi, b_), ohi)
    return olo, ohi

def prove_stream(Rlo, Rhi, t5):
    br, nlev = final_branches(Rlo, Rhi); p, _ = prove_order(br, t5); return p

g0 = 0
for ti, ids in enumerate(TEXTS):
    rows = ROWS[ti]; g = np.arange(g0, g0 + len(rows)); g0 += len(rows)
    rec = forward_keep(ids, keep=11)
    VER["own_top5"] += int((rec["top5"][rows] == T5[g]).all(1).sum())
    if SET == "refs":
        Zl = np.load(f"{HERE}/fringes_machine/refs_top_light.npz")
        VER["own_light"] += int((rec["x16"][rows] == Zl["x16"][g]).all(1).sum())
    Rmid, Rout = rec["Rmid"][rows], rec["Rout"][rows]
    # fc.11: its light, its gauge, its piles (base2's own steps)
    x16f, exf, LSf = gauge_log(rec["Rmid"]); x16f, exf, LSf = x16f[rows], exf[rows], LSf[rows]
    Ff = (wef[None, :] + exf[:, None] + ES) * 65536 - LSf[:, None]
    pile_f = product(x16f, wf.astype(np.float64)); H = power(pile_f, Ff) + bfc; assert np.array_equal(H, rec["H"][rows])
    E = edge(H); x16e, exe = to16(E, -ES); she = exe + ES; ame = np.abs(E).argmax(1)
    pile_m = product(x16e, wm.astype(np.float64)); sm = -(wem[None, :] + exe[:, None] + ES)
    Mw = rshift(pile_m, sm) + bm; assert np.array_equal(Mw, rec["Mw"][rows]) and np.array_equal(Rmid + Mw, Rout)
    shf = rec["ex"][rows] + ES; amf = np.abs(Rout).argmax(1)
    assert np.array_equal(np.maximum(nbits(np.abs(Rout).max(1)) - 15, 0), shf)
    stops = {}
    for nm, x, pile in (("fc", x16f, pile_f), ("mproj", x16e, pile_m)):
        Ws, Wu, m_b = PL[nm]; r = REC[nm]
        Td, Cd, Ed, Bd = tables(x, Ws, Wu, m_b)
        assert ((Cd.sum(0)) == np.bitwise_count(np.abs(x)).astype(np.float64) @ np.bitwise_count(np.abs(M[nm + ".11.w"].astype(np.int64))).astype(np.float64)).all()
        K = np.zeros((32,) + pile.shape, np.int64); K[:31] = np.cumsum(Td[::-1], 0)[::-1]
        B = np.zeros_like(K); B[1:] = np.cumsum(Bd, 0)
        assert np.array_equal(K[0], pile) and (B[0] == 0).all()
        r["bound_bad"] += int((np.abs(pile[None] - K) > B).sum())                       # the bound covers the remainder
        lo, hi = K - B, K + B
        wide = (np.maximum(np.abs(lo), np.abs(hi)) >= BIGX)                             # too wide to be fixed (treated as not fixed)
        lo_ = np.where(wide, 0, lo); hi_ = np.where(wide, 0, hi)
        if nm == "fc":
            olo, ohi = power(lo_, Ff[None]) + bfc, power(hi_, Ff[None]) + bfc         # power rises with the pile at a fixed gauge
            elo, ehi = edge_box(olo, ohi)
            ca = np.stack([fixed(elo[D], ehi[D], she, ame) for D in range(32)]) & ~wide
            kk = KH[g]
        else:
            olo, ohi = rshift(lo_, sm[None]) + bm, rshift(hi_, sm[None]) + bm
            ca = np.stack([fixed(Rmid + olo[D], Rmid + ohi[D], shf, amf) for D in range(32)]) & ~wide
            kk = KM[g]
        assert ca[0].all()
        wok = (kk >= 0)[None, :, None] & ((ohi - olo) <= (np.int64(1) << np.maximum(kk, 0))[None, :, None]) & ~wide
        Da, Da_any = stop_of(ca); Db, Db_any = stop_of(ca | wok)
        r["nest_bad"] += int((Da != Da_any).sum() + (Db != Db_any).sum())
        Ccum = np.zeros_like(K); Ccum[:31] = np.cumsum(Cd[::-1], 0)[::-1]
        tk = lambda A, D: np.take_along_axis(A, D[None], 0)[0]
        r["plain"] += int(Ccum[0].sum()); r["extra"] += int(Ed.sum()); r["piles"] += Da.size
        r["a"] += int(tk(Ccum, Da).sum()); r["b"] += int(tk(Ccum, Db).sum())
        r["hist_a"] += np.bincount(Da.ravel(), minlength=32); r["hist_b"] += np.bincount(Db.ravel(), minlength=32)
        stops[nm] = dict(alo=tk(lo, Da), ahi=tk(hi, Da), blo=tk(lo, Db), bhi=tk(hi, Db))
        del Td, Cd, Ed, Bd, K, B, lo, hi, Ccum, ca, wok, olo, ohi
    # ---- carried to the top, each plate cut alone
    # mproj.11 (a): the final light, its level and its gauge exact
    s = stops["mproj"]
    Rlo, Rhi = Rmid + rshift(s["alo"], sm) + bm, Rmid + rshift(s["ahi"], sm) + bm
    br, nlev = final_branches(Rlo, Rhi); b0 = br[0]
    ok = (nlev == 0) & (b0["xlo"] == b0["xhi"]).all(1) & (b0["xlo"] == rec["x16"][rows]).all(1) & (b0["LSlo"] == b0["LShi"]) & (b0["LSlo"] == rec["LS"][rows])
    VER["a_mproj"] += int(ok.sum())
    # mproj.11 (b)
    Rlo, Rhi = Rmid + rshift(s["blo"], sm) + bm, Rmid + rshift(s["bhi"], sm) + bm
    pb = prove_stream(Rlo, Rhi, T5[g]); VER["b_mproj"] += int(pb.sum())
    exact_b = (Rlo == Rhi).all(1); VER["b_mproj_tie_exact"] += int((~pb & exact_b).sum())
    # fc.11 (a): the bend's output after its water level exact, and that level
    s = stops["fc"]
    Hlo, Hhi = power_py(s["alo"], Ff) + bfc, power_py(s["ahi"], Ff) + bfc
    elo, ehi = edge_box(Hlo, Hhi); l0, l1 = lvl_range(elo, ehi)
    ok = (l0 == she) & (l1 == she) & (rshift(elo, she[:, None]) == x16e).all(1) & (rshift(ehi, she[:, None]) == x16e).all(1)
    VER["a_fc"] += int(ok.sum())
    # fc.11 (b): H box -> bend -> mproj.11 on the box -> stream -> top
    Hlo, Hhi = power_py(s["blo"], Ff) + bfc, power_py(s["bhi"], Ff) + bfc
    assert ((Hlo <= H) & (H <= Hhi)).all()
    elo, ehi = edge_box(Hlo, Hhi); assert ((elo <= E) & (E <= ehi)).all()
    mlo, mhi = mproj_box(elo, ehi); assert ((mlo <= Mw) & (Mw <= mhi)).all()
    pb = prove_stream(Rmid + mlo, Rmid + mhi, T5[g]); VER["b_fc"] += int(pb.sum())
    VER["b_fc_tie_exact"] += int((~pb & (mlo == mhi).all(1)).sum())
    # the session's req boxes at Mw and H (+-2^k proved; +-2^(k+1) tried)
    for key, kk in (("Mw", KM[g]), ("H", KH[g])):
        sel = np.flatnonzero(kk >= 0)
        if len(sel) == 0: continue
        for up in (0, 1):
            w_ = np.left_shift(np.int64(1), kk[sel] + up)[:, None]
            if key == "Mw":
                lo_, hi_ = Rmid[sel] + Mw[sel] - w_, Rmid[sel] + Mw[sel] + w_
            else:
                el, eh = edge_box(H[sel] - w_, H[sel] + w_); ml, mh = mproj_box(el, eh); lo_, hi_ = Rmid[sel] + ml, Rmid[sel] + mh
            p = prove_stream(lo_, hi_, T5[g][sel])
            if up == 0: VER[f"req_{key}"] += int(p.sum()); VER[f"req_{key}_n"] += len(sel)
            else: VER[f"req_{key}_up"] += int(p.sum())
    del rec
    log(f"  text {ti + 1}/{len(TEXTS)} done ({time.time() - t0:.0f}s, {rss()} MB)")

log(f"\nOWN FORWARD: top 5 equal to the saved reference {VER['own_top5']}/{NPOS}" + (f"; final light equal to the saved light {VER['own_light']}/{NPOS}" if SET == "refs" else ""))
log(f"\nNEEDED WORK ({SET}, {NPOS} positions)")
for nm in ("fc", "mproj"):
    r = REC[nm]
    log(f"  {nm}.11: plain/pos {r['plain'] / NPOS:,.0f}; extra {r['extra'] / r['plain']:.2%}; (a) {r['a'] / r['plain']:.2%} = {r['a'] / r['extra']:.1f}x extra; "
        f"(b) {r['b'] / r['plain']:.2%} = {r['b'] / r['extra']:.1f}x extra; piles where the bound failed to cover the remainder: {r['bound_bad']}; "
        f"non-nested stops: {r['nest_bad']}")
    for c in ("a", "b"):
        h = r[f"hist_{c}"]; log(f"     ({c}) D=0 {h[0] / r['piles']:.2%}; " + " ".join(f"D{D}:{h[D] / r['piles']:.2%}" for D in range(1, 32) if h[D]))
log(f"\nCARRIED TO THE TOP (each plate cut alone, the checker's interval steps and prover)")
log(f"  mproj.11 (a): final light, level and gauge the machine's own {VER['a_mproj']}/{NPOS}; (b) top 5 proved {VER['b_mproj']}/{NPOS} "
    f"(not proved but the final stream exact: {VER['b_mproj_tie_exact']})")
log(f"  fc.11    (a): the bend's light and level the machine's own {VER['a_fc']}/{NPOS}; (b) top 5 proved {VER['b_fc']}/{NPOS} "
    f"(not proved but the gate write exact: {VER['b_fc_tie_exact']})")
log(f"  the session's req boxes: Mw +-2^k proved {VER['req_Mw']}/{VER['req_Mw_n']}, +-2^(k+1) proved {VER['req_Mw_up']}; "
    f"H +-2^k proved {VER['req_H']}/{VER['req_H_n']}, +-2^(k+1) proved {VER['req_H_up']}")
json.dump(dict(rec={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in REC.items()}, ver=VER),
          open(f"{CK}/check_l11_{SET}.json", "w"), indent=0)
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
