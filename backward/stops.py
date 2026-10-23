"""STEPS 2-5: EVERY PILE READ TOP DIAGONAL FIRST AND STOPPED WHEN WHAT IS STILL UNREAD CANNOT CHANGE WHAT THE NEXT
STAGE NEEDS. Two criteria, counted separately.

HIS WORDS (2026-10-08): "Start backward I think that's your problem. Start on the next token and work backwards".
And: "THE FLAT IS THE CANCELLING." "the EXTRA photons are the base 2 photons rhar DONT cancel"

THE SESSION'S RESTATEMENT (the session's words). Every pile of every bank is read by diagonals, the place pairs
(light place a, plate place b) with the same a + b, top first (section 25's order, every diagonal a stage). After the
diagonals >= D are read, the pile is known as K_D +- B_D: K_D the read part, B_D the plain bound of section 25 on the
unread pairs, sum over a + b < D of 2^(a+b) x min(lines lit at light place a, lines lit at plate place b). The pile is
stopped at the first (coarsest) D where:
  (a) BIT-EXACT: what it hands on is the machine's own, digit for digit:
        qkv's q and k:  their line of the head's light after its water level (to16) is fixed, and that level is fixed
        qkv's v, proj (the arrival write), mproj below layer 11 (the gate write): the output itself is fixed
        fc:             the bend's output, after the bend's water level, is fixed (mproj's light, section 25's "output")
        mproj at layer 11: its line of the final light (after the last water level) is fixed, and that level
      Every pile then hands on exactly what the machine hands on, so every pile downstream is identical.
  (b) THE TOKEN (his answer-exact tier, labelled as such): the (a) condition, OR the output's interval is no wider
      than 2^k, where +-2^k is the box at that point that req.py proved still gives the same top 5 (set and order)
      at that position (QKV:q / QKV:k / QKV:v for qkv's three parts, A for proj, H for fc, Mw for mproj).
      Where req.py found the exact value needed (k = -1) the (a) condition is the stop.
The work needed is the meetings of the read diagonals (photon meetings, plain's unit: a light 1 meeting a plate 1).
THE EXTRA (the measure he named): per pile, per place pair, the net count of the meetings over the lines, |.| summed
(flat_design/1's sm_excess): the photons that do not cancel. Each plate's needed work is reported as a share of its
plain meetings and as a multiple of its extra.

PREDICTION (the session's, written before this ran): (a) fc stops at D24-D28 (section 25: 91.8% of lens 1), mproj below
layer 11 at the full pile on nearly every line (section 26: the value joining the stream is fixed before the full pile
on 0.0-0.5% of lines), proj the same, qkv's v the same, qkv's q and k a few % early (the water level of a 64-line head
drops 5-6 places); mproj at layer 11 like section 26's next light (fixed by D28 on most lines): about 90-95%. Whole
machine (a): 96-99% of the plain meetings at the banks. (b) adds only at layer 11: mproj.11 with a box of +-2^9 at the
median skips the bottom 5-8 diagonals, 80-90% needed; fc.11 (+-2^2 at the knife edge) about the same as (a); proj.11
and qkv.11 nothing. Every needed share is 10-30x the plate's extra.

    python3 backward/stops.py set=refs|windows layers=0,...,11 [win=a:b]   -> backward/out/stops_<set>....npz / .log
"""
import sys, glob, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward")
from core import *
ARG = dict(a.split("=") for a in sys.argv[1:]); SET = ARG.get("set", "refs")
LAYERS = [int(v) for v in ARG.get("layers", ",".join(map(str, range(12)))).split(",")]
WIN = ARG.get("win", None); PB = int(ARG.get("pbatch", 16)); LIGHT = ARG.get("light", "real")
ORDER = int(ARG["order"]) if "order" in ARG else None
TAG = ("smoke_" if "ntext" in ARG else "") + ("" if ORDER is None else f"order{ORDER}_") + f"stops_{SET}" + ("" if WIN is None else f"_win{WIN.replace(':', '-')}") + f"_L{'-'.join(map(str, LAYERS))}" + ("" if LIGHT == "real" else f"_{LIGHT}")
LOG = open(f"{OUT}/{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if SET == "refs":
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    JUDGE = [np.arange(len(t)) for t in TEXTS]
else:
    WINS = json.load(open(f"{HERE}/judge5_texts.json")); a_, b_ = (int(v) for v in (WIN or "0:300").split(":"))
    TEXTS = [w["ids"] for w in WINS[a_:b_]]; JUDGE = [np.arange(3, 23) for _ in TEXTS]
if "ntext" in ARG: TEXTS = TEXTS[:int(ARG["ntext"])]; JUDGE = JUDGE[:int(ARG["ntext"])]   # (a smoke test of the code)
NPOS_ALL = sum(len(j) for j in JUDGE) if "ntext" not in ARG else None
NPOS = sum(len(j) for j in JUDGE)

# the requirements found by req.py (k = -1 where a point was not run: the exact value, the (a) stop)
REQ = {}
for f in sorted(glob.glob(f"{OUT}/req_{SET}_L*.npz")) + sorted(glob.glob(f"{OUT}/req_{SET}_win*_L*.npz")):
    Z = np.load(f)
    for k in Z.files: REQ.setdefault(k, []).append((f, Z[k]))
def req_k(l, pt):
    import re
    key = f"{l}__{pt.replace(':', '__')}"
    if key not in REQ: return np.full(NPOS, -1, np.int64), "not run: exact"
    whole = [fv for fv in REQ[key] if not re.search(r"_win\d", fv[0])]
    if whole:
        f, v = whole[0]; v = v[:NPOS] if SET == "refs" else v
        if len(v) != NPOS: return np.full(NPOS, -1, np.int64), f"incomplete ({len(v)}): exact"
        return v, f.split("/")[-1]
    parts = sorted(REQ[key], key=lambda fv: int(re.search(r"_win(\d+)", fv[0]).group(1)))
    v = np.concatenate([p[1] for p in parts])
    if len(v) != NPOS: return np.full(NPOS, -1, np.int64), f"incomplete ({len(v)}): exact"
    return v, "windows parts"

def planes(v, signed):
    a = np.abs(v); s = np.sign(v) if signed else 1
    return [(s * ((a >> q) & 1)) for q in range(16)]

def plate_planes(name):
    """the plate's places as planes, one place at a time into float32 (memory): Ws signed, Wu lit"""
    w = np.asarray(plate16(name)).astype(np.int32); N, Mm = w.shape; a = np.abs(w); sg = np.sign(w).astype(np.float32)
    Ws = np.empty((N, 16, Mm), np.float32); Wu = np.empty((N, 16, Mm), np.float32); m_b = np.zeros((16, Mm), np.int64)
    for q in range(16):
        bit = ((a >> q) & 1); Wu[:, q, :] = bit; Ws[:, q, :] = sg * bit; m_b[q] = bit.sum(0)
    return Ws.reshape(N, 16 * Mm), Wu.reshape(N, 16 * Mm), m_b
import core as _core
def plate16(name): return np.asarray(_core._MZ[name + ".w"])

def diag_tables(x, Ws, Wu, m_b, Mm):
    """x (P, N) the light; the plate's planes. Returns per diagonal d (31): Td (signed net count x 2^d summed: the read
    part), Cd (meetings), Ed (|net count| per place pair, summed: the extra), Bd (the plain bound's terms)."""
    P, N = x.shape; assert np.abs(x).max() <= 1 << 15
    Xs = np.stack(planes(x, True)).reshape(16 * P, N).astype(np.float32)
    T = (Xs @ Ws).reshape(16, P, 16, Mm)                                                   # [a, p, b, j], exact (|.| <= N)
    C = (np.abs(Xs) @ Wu).reshape(16, P, 16, Mm)
    n_a = np.stack([((np.abs(x) >> q) & 1).sum(1) for q in range(16)])                     # (16, P)
    Td = np.zeros((31, P, Mm), np.int64); Cd = np.zeros((31, P, Mm), np.int64); Ed = np.zeros((31, P, Mm), np.int64)
    Bd = np.zeros((31, P, Mm), np.int64)
    for a in range(16):
        Ta = T[a].transpose(1, 0, 2).astype(np.int64); Ca = C[a].transpose(1, 0, 2).astype(np.int64)   # (b, P, M)
        Td[a:a + 16] += Ta; Cd[a:a + 16] += Ca; Ed[a:a + 16] += np.abs(Ta)
        Bd[a:a + 16] += np.minimum(n_a[a][None, :, None], m_b[:, None, :])
    pw = (np.int64(1) << np.arange(31, dtype=np.int64))[:, None, None]
    return Td * pw, Cd, Ed, Bd * pw

def stage_intervals(Td, Bd):
    """K_D (read part, diagonals >= D) and B_D (bound on the rest) for D = 0..31"""
    K = np.zeros((32,) + Td.shape[1:], np.int64); B = np.zeros_like(K)
    K[:31] = np.cumsum(Td[::-1], 0)[::-1]; B[1:] = np.cumsum(Bd, 0)
    return K, B

def first_stop(cond):
    """cond (32, P, M), True at D = 0 always: the largest D with cond true at D and every D below (nested, so the
    largest D with cond true)"""
    D = np.arange(32)[:, None, None]
    return np.where(cond, D, -1).max(0)

def fixed_at(olo, ohi, sh, amax):
    """olo, ohi (S, P, M); the reader's level sh (P,) and the line holding the largest value (P,): the line's digits after
    the water level are fixed and the level cannot move"""
    s = sh[None, :, None]; cap = np.left_shift(np.int64(1), sh + 15)[None, :, None]
    ok = (rshift(olo, s) == rshift(ohi, s)) & (np.maximum(np.abs(olo), np.abs(ohi)) < cap)
    mlo = np.where((olo <= 0) & (ohi >= 0), 0, np.minimum(np.abs(olo), np.abs(ohi)))
    floor = np.where(sh > 0, np.left_shift(np.int64(1), np.maximum(sh + 14, 0)), 0)[None, :, None]
    isar = (np.arange(olo.shape[2])[None, None, :] == amax[None, :, None])
    return ok & (~isar | (mlo >= floor))

def level_of(v):
    """to16's level and the line holding the largest |value|, per row"""
    top = bitlen(np.bitwise_or.reduce(np.abs(v), axis=1)); return np.maximum(top - 15, 0), np.abs(v).argmax(1)

CLIP = np.int64(1) << 44
REC = {f"{nm}.{l}": dict(plain=0, extra=0, a=0, b=0, a_extra=0, b_extra=0, piles=0, a_early=0, b_early=0,
                         hist_a=[0] * 32, hist_b=[0] * 32) for l in LAYERS for nm in ("qkv", "proj", "fc", "mproj")}
VER = {}                                                                       # the verification counts
LPERM = {}
KR = {}
# THE STREAMING RULE (fixed before the run, the session's): a value that a LATER token reads must be handed on exactly,
# since req.py proved each box only for this position's own top 5 with every earlier key and value exact. Every key and
# value (qkv's k and v parts, every layer) is read by later tokens, and every stream value below layer 11 feeds layer
# 11's keys and values. So (b) may use its proved box only for layer 11's q, its arrival write (A), its knife-edge
# values (H) and its gate write (Mw), which feed only this token's own top. Everything else stops at the (a) condition.
for l in LAYERS:
    for pt in ("QKV:q", "QKV:k", "QKV:v", "A", "H", "Mw"):
        k_, src = req_k(l, pt)
        if l < 11 or pt in ("QKV:k", "QKV:v"):
            KR[(l, pt)] = np.full(NPOS, -1, np.int64); src = f"read by later tokens: exact digits (req.py's own box: exact at {(k_ < 0).mean():.1%})"
        else: KR[(l, pt)] = k_
        log(f"  requirement {l} {pt}: {src}; exact needed at {(KR[(l, pt)] < 0).mean():.1%}")
VERIFY_LAYERS = [int(v) for v in ARG.get("verify", "11").split(",") if v != ""]
PILEKEY = {"qkv": "qkvpile", "proj": "projpile", "fc": "fcpile", "mproj": "mprojpile"}
CHT = 30 if SET == "refs" else 50
GOFF = np.concatenate([[0], np.cumsum([len(j) for j in JUDGE])])
take = lambda A, D: np.take_along_axis(A, D[None], 0)[0]

def plate_stops(rec, l, nm, t, Ws, Wu, m_b):
    """every pile of plate nm.l at the judged positions of text t: the (a) and (b) stops, the work, and the intervals"""
    d = rec["L"][l]; name = f"{nm}.{l}"; Mm = m_b.shape[1]; rows = JUDGE[t]; g = np.arange(GOFF[t], GOFF[t + 1])
    wf, we, b = plate(name); del wf
    x = {"qkv": d["x16q"], "proj": d["x16a"], "fc": d["x16f"], "mproj": d["x16e"]}[nm][rows]
    pile = d[PILEKEY[nm]][rows]
    if LIGHT == "shuffled":        # CONTROL: each position's light across its lines, every value kept; the plate's own outputs
        rng = np.random.default_rng([2026, l, "qpfm".index(nm[0]), t])          # on that light are the reference
        x = np.stack([v[rng.permutation(len(v))] for v in x]); pile = product(x, plate(name)[0])
    if ORDER is not None:          # ORDER-FREE: the light's lines and the plate's rows in one other order together
        x = x[:, LPERM[name]]
    if nm in ("qkv", "fc"):
        Fall = (we[None, :] + d["exq" if nm == "qkv" else "exf"][rows, None] + ES) * 65536 - d["LSq" if nm == "qkv" else "LSf"][rows, None]
        OUTT = power(pile, Fall) + b
    else:
        OUTT = rshift(pile, -(we[None, :] + d["exa" if nm == "proj" else "exe"][rows, None] + ES)) + b
    if LIGHT == "real": assert np.array_equal(OUTT, d[{"qkv": "QKV", "proj": "A", "fc": "H", "mproj": "Mw"}[nm]][rows])
    if nm == "fc":
        ET = edge(OUTT)
        if LIGHT == "real": assert np.array_equal(level_of(ET)[0], d["exe"][rows] + ES)
        ELEV = level_of(ET)[0] if LIGHT == "shuffled" else d["exe"][rows] + ES; EAM = level_of(ET)[1]
    if nm == "mproj" and l == 11:
        RT = d["Rmid"][rows] + OUTT; RLEV = level_of(RT)[0]; RAM = level_of(RT)[1]
        if LIGHT == "real": assert np.array_equal(RT, d["Rout"][rows]) and np.array_equal(RLEV, rec["top"]["ex"][rows] + ES)
    IA = [np.zeros((len(rows), Mm), np.int64) for _ in range(4)]               # (a) lo, hi; (b) lo, hi at the stops
    r = REC[name]; anyb = False
    for p0 in range(0, len(rows), PB):
        sl = slice(p0, p0 + PB); pr = rows[sl]; xs = x[sl]; pl = pile[sl]; gg = g[sl]
        Td, Cd, Ed, Bd = diag_tables(xs, Ws, Wu, m_b, Mm)
        K, B = stage_intervals(Td, Bd); del Td, Bd
        assert np.array_equal(K[0], pl), "the diagonals must give the machine's pile"
        assert (np.abs(pl[None] - K) <= B).all(), "the bound must hold at every stage"
        lo = np.clip(K - B, -CLIP, CLIP); hi = np.clip(K + B, -CLIP, CLIP); del K, B
        if nm in ("qkv", "fc"):
            F = (we[None, :] + d["exq" if nm == "qkv" else "exf"][pr, None] + ES) * 65536 - d["LSq" if nm == "qkv" else "LSf"][pr, None]
            olo = power(lo, F[None]) + b; ohi = power(hi, F[None]) + b
        else:
            s_ = -(we[None, :] + d["exa" if nm == "proj" else "exe"][pr, None] + ES)
            olo = rshift(lo, s_[None]) + b; ohi = rshift(hi, s_[None]) + b
        q_ = np.arange(p0, min(p0 + PB, len(rows)))
        if nm == "qkv":
            ca = olo == ohi; Q = OUTT[q_]
            for part in (0, 1):                                                  # q and k: their head's light
                for h in range(12):
                    c0 = part * 768 + h * 64; sh, am = level_of(Q[:, c0:c0 + 64])
                    ca[:, :, c0:c0 + 64] = fixed_at(olo[:, :, c0:c0 + 64], ohi[:, :, c0:c0 + 64], sh, am)
        elif nm == "fc":
            elo, ehi = i_edge(olo, ohi)
            ca = fixed_at(elo, ehi, ELEV[q_], EAM[q_]); del elo, ehi
        elif nm == "mproj" and l == 11:
            Rm = d["Rmid"][pr][None]
            ca = fixed_at(Rm + olo, Rm + ohi, RLEV[q_], RAM[q_])
        else:
            ca = olo == ohi
        Da = first_stop(ca)
        if nm == "qkv":
            kq = np.stack([KR[(l, "QKV:q")][gg], KR[(l, "QKV:k")][gg], KR[(l, "QKV:v")][gg]], 1)
            kc = np.repeat(kq, 768, axis=1)
        else:
            kc = np.repeat(KR[(l, {"proj": "A", "fc": "H", "mproj": "Mw"}[nm])][gg][:, None], Mm, axis=1)
        wok = (kc >= 0)[None] & ((ohi - olo) <= np.left_shift(np.int64(1), np.maximum(kc, 0))[None])
        Db = first_stop(ca | wok); del ca, wok, olo, ohi
        anyb |= bool((Db != Da).any())
        IA[0][sl], IA[1][sl] = take(lo, Da), take(hi, Da); IA[2][sl], IA[3][sl] = take(lo, Db), take(hi, Db)
        del lo, hi
        Ccum = np.zeros((32,) + Cd.shape[1:], np.int64); Ccum[:31] = np.cumsum(Cd[::-1], 0)[::-1]
        Ecum = np.zeros_like(Ccum); Ecum[:31] = np.cumsum(Ed[::-1], 0)[::-1]; del Cd, Ed
        r["plain"] += int(Ccum[0].sum()); r["extra"] += int(Ecum[0].sum()); r["piles"] += Da.size
        r["a"] += int(take(Ccum, Da).sum()); r["b"] += int(take(Ccum, Db).sum())
        r["a_extra"] += int(take(Ecum, Da).sum()); r["b_extra"] += int(take(Ecum, Db).sum())
        r["a_early"] += int((Da > 0).sum()); r["b_early"] += int((Db > 0).sum())
        ha = np.bincount(Da.ravel(), minlength=32); hb = np.bincount(Db.ravel(), minlength=32)
        r["hist_a"] = [int(u + v) for u, v in zip(r["hist_a"], ha)]; r["hist_b"] = [int(u + v) for u, v in zip(r["hist_b"], hb)]
        del Ccum, Ecum
    return IA, anyb

def verify(rec, l, nm, t, IA, anyb):
    """(a): the plate's (a) intervals carried up give the machine's own final light (every line, its level, its gauge);
    (b): the (b) intervals carried up prove the top 5 (set and order), and the proved words are the machine's"""
    rows = JUDGE[t]; name = f"{nm}.{l}"; v = VER.setdefault(name, dict(a_same=0, a_n=0, b_proved=0, b_n=0, b_wrong=0, b_blown=0))
    lo, hi, bl = push(rec, rows, l, PILEKEY[nm], IA[0], IA[1])
    br = top_branches(lo, hi); b0 = br[0]
    same = ~bl & (len(br) == 1) & (b0["xlo"] == b0["xhi"]).all(1) & (b0["xlo"] == rec["top"]["x16"][rows]).all(1) \
           & (b0["LSlo"] == b0["LShi"]) & (b0["LSlo"] == rec["top"]["LS"][rows])
    v["a_same"] += int(same.sum()); v["a_n"] += len(rows)
    if anyb or l == 11:
        lo, hi, bl = push(rec, rows, l, PILEKEY[nm], IA[2], IA[3])
        ok = np.flatnonzero(~bl); p5 = np.zeros(len(rows), bool); cand = np.zeros((len(rows), 5), np.int64)
        for c0 in range(0, len(ok), 64):
            ii = ok[c0:c0 + 64]; c = certify(top_branches(lo[ii], hi[ii])); p5[ii] = c["cert5"]; cand[ii] = c["cand"]
        v["b_proved"] += int(p5.sum()); v["b_n"] += len(rows); v["b_blown"] += int(bl.sum())
        v["b_wrong"] += int((p5 & ~(cand == rec["top"]["top5"][rows]).all(1)).sum())

for l in LAYERS:
    tl = time.time()
    cht = CHT if (l not in VERIFY_LAYERS or l == 11) else 6
    for c0 in range(0, len(TEXTS), cht):
        tt = list(range(c0, min(len(TEXTS), c0 + cht)))
        recs = {t: record(TEXTS[t], keep_from=l, keep_only=None if l in VERIFY_LAYERS else l,
                          keys=None if l == 11 or l not in VERIFY_LAYERS else None) for t in tt}
        for nm in ("qkv", "proj", "fc", "mproj"):
            Ws, Wu, m_b = plate_planes(f"{nm}.{l}")
            if ORDER is not None:
                Nn = Ws.shape[0]; LPERM[f"{nm}.{l}"] = np.random.default_rng([ORDER, l, Nn]).permutation(Nn)
                Ws = np.ascontiguousarray(Ws[LPERM[f"{nm}.{l}"]]); Wu = np.ascontiguousarray(Wu[LPERM[f"{nm}.{l}"]])
            for t in tt:
                IA, anyb = plate_stops(recs[t], l, nm, t, Ws, Wu, m_b)
                if l in VERIFY_LAYERS and LIGHT == "real": verify(recs[t], l, nm, t, IA, anyb)
            del Ws, Wu
        del recs
    log(f"  layer {l} done ({time.time() - tl:.0f}s; {time.time() - t0:.0f}s, {rss()} MB)")

log(f"\nNEEDED WORK BY PLATE ({SET}, {NPOS} positions): photon meetings of the diagonals read, against plain and the extra")
log(f"{'plate':9s} {'plain/pos':>12s} {'extra':>7s} | {'(a) needed':>10s} {'x extra':>8s} {'piles early':>11s} | {'(b) needed':>10s} {'x extra':>8s} {'piles early':>11s}")
tot = dict(plain=0, extra=0, a=0, b=0)
for l in LAYERS:
    for nm in ("qkv", "proj", "fc", "mproj"):
        r = REC[f"{nm}.{l}"]
        for k in tot: tot[k] += r[k]
        log(f"{nm + '.' + str(l):9s} {r['plain'] / NPOS:12,.0f} {r['extra'] / r['plain']:7.2%} | {r['a'] / r['plain']:10.2%} {r['a'] / r['extra']:8.1f} "
            f"{r['a_early'] / r['piles']:11.2%} | {r['b'] / r['plain']:10.2%} {r['b'] / r['extra']:8.1f} {r['b_early'] / r['piles']:11.2%}")
log(f"{'banks':9s} {tot['plain'] / NPOS:12,.0f} {tot['extra'] / tot['plain']:7.2%} | {tot['a'] / tot['plain']:10.2%} {tot['a'] / tot['extra']:8.1f} {'':11s} | "
    f"{tot['b'] / tot['plain']:10.2%} {tot['b'] / tot['extra']:8.1f}")
log(f"\nWHERE EACH PILE STOPS (D = the lowest diagonal read; D = 0: the whole pile; shares of the plate's piles, D >= 1 only)")
for l in LAYERS:
    for nm in ("qkv", "proj", "fc", "mproj"):
        r = REC[f"{nm}.{l}"]; n = sum(r["hist_a"])
        for crit in ("a", "b"):
            h = r[f"hist_{crit}"]
            if sum(h[1:]) or crit == "a":
                log(f"  {nm + '.' + str(l):9s} ({crit}) D=0 {h[0] / n:.2%}; " + " ".join(f"D{D}:{h[D] / n:.2%}" for D in range(1, 32) if h[D]))
log(f"\nVERIFIED (each plate cut alone at its stops, carried to the top by core.push):")
for name, v in VER.items():
    log(f"  {name:9s} (a) final light, its level and its gauge the machine's own at {v['a_same']}/{v['a_n']}"
        + (f"; (b) top 5 proved {v['b_proved']}/{v['b_n']} (blown {v['b_blown']}), proofs naming other words {v['b_wrong']}" if v["b_n"] else ""))
json.dump(dict(rec=REC, ver=VER), open(f"{OUT}/{TAG}.json", "w"), indent=0)
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
