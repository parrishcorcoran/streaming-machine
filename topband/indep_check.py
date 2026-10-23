"""INDEPENDENT CHECK OF THE BAND-BY-BAND TOP READ (a separate checker; it never imports or reads topband.py, band_walk.c,
check.py, report.py, judge.py, lineup2/ or fringes_machine/ code). Whole numbers in the read; base2.py's own steps
(exec'd from its source, as judge5.py does) for the machine, the light and the check path.

WHAT IS REBUILT, from the task's words only:
  - THE LIGHT, made again here: base2.py's own 12 layers run on every refs text (prompt + its 16 words, every position)
    and on judge5's windows (positions 3..22), then base2's gauge at the top. Compared with base2.py's saved run
    (recording/C_ref/plain_refs.npz) and with the light files the session read (fringes_machine/*_top_light.npz).
  - THE POUR, with no walk: each exposure formed directly as x_i * w_iw (my own C kernel, indep_trunc.c). The counted
    pile after band k is formed as the pile with every exposure cut toward 0 at place Q_k (Q = 26,22,18,14,10,6,2,0),
    not as a sum of band parts; Q_k = 0 is the whole pile and must equal base2's own product.
  - THE SET-ASIDE: V = standing amount + counted pile; shared bound R = (lit lines) x (2^Q_k - 1); v5 the 5th-best V
    among the words still in; set aside when V <= v5 - 2R - mu; mu = 2 gamma + delta made again from base2_fringes.py's
    stated bound (beta's rounding, one score step at pile scale), with Python fractions.
  - THE CHECKS: every word set aside scores strictly below the true 5th-best score (base2's plain read of every word);
    every counted pile within R of the whole pile for every word at every test; the words read to the end score as the
    plain read; top 5 (judge5.top5's definition, set and order) and race winner against the plain read, base2.py's saved
    refs run and judge5_ref.npz; judge5.judge's two counts made again on the windows checked.
  - THE PHOTONS: pours per band into the words still in (my own survivors), every band into every word, the walk
    (signed digits by my own step-by-step rule, gaps from my own sort), the light's writing, the standing amounts, the
    bound, the words read to the end, and the plain read's three parts; on every position checked, against the
    session's saved per-position arrays; and on 3 texts (0, 5, 22: the 25,920-survivor position and the 5th-place tie)
    the band photons again in plain numpy, independent of my C kernel.
  - CONTROL: the shuffled light (the session's stated draw: default_rng(2026), one permutation of 768 per position, in
    order over the 648 refs) through the same rebuild.
CHECKER'S PREDICTION (written before running): every top 5 and winner equal; every check 0; every count equal to the
session's arrays at every position; refs total 10.88% of the plain read, plain basis 904,017,313 a position.
    python3 topband/indep_check.py [windows=100]   -> topband/indep_check.log
"""
import os, sys, json, time, ctypes, subprocess, resource, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/topband"
ARG = dict(a.split("=") for a in sys.argv[1:]); NWIN = int(ARG.get("windows", 100)); STAGE = ARG.get("stage", "light")
LOG = open(f"{OUT}/indep_check_{STAGE}.log", "w"); T0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
Q = [26, 22, 18, 14, 10, 6, 2, 0]
def pc(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def pc1(v): return bin(abs(int(v))).count("1")
def ceil_frac(f): return -((-f.numerator) // f.denominator)

# ---------------- base2.py's own steps ----------------
src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
B = {}; _cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = _argv; os.chdir(_cwd)
M = B["M"]; W = B["W"]; power, race, product, gauge_log, EXP2, ES = B["power"], B["race"], B["product"], B["gauge_log"], B["EXP2"], B["ES"]

def top_light(ids, lo=None, hi=None):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + B["plain_bank"](B["attention"](B["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
        R = R + B["plain_bank"](B["edge"](B["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
    if lo is not None: R = R[lo:hi]
    return gauge_log(R)

def top5(S):                                               # judge5.top5's definition (the judge's two lines)
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

log(f"INDEPENDENT CHECK of the band-by-band top read; windows={NWIN}")
LIGHT = f"{OUT}/indep_light_w{NWIN}.npz"
PR = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
if STAGE == "light":
    # ---------------- the light, made again ----------------
    refs = json.load(open(f"{HERE}/refs_many.json")); wins = json.load(open(f"{HERE}/judge5_texts.json"))
    XR, ER, LR, NPOS = [], [], [], []
    for t in refs:
        x16, ex, LS = top_light(t["ids"] + t["ref"]); XR.append(x16); ER.append(ex); LR.append(LS); NPOS.append(len(ex))
    XR, ER, LR = np.concatenate(XR), np.concatenate(ER), np.concatenate(LR)
    log(f"  refs light made: {len(ER)} positions ({time.time() - T0:.0f}s, {rss()} MB)")
    XW, EW, LW = [], [], []
    for w in wins[:NWIN]:
        x16, ex, LS = top_light(w["ids"], 3, 23); XW.append(x16); EW.append(ex); LW.append(LS)
    XW, EW, LW = np.concatenate(XW), np.concatenate(EW), np.concatenate(LW)
    log(f"  windows light made: {len(EW)} positions ({time.time() - T0:.0f}s, {rss()} MB)")
    PR = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    FR = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); FW = np.load(f"{HERE}/fringes_machine/windows_top_light.npz")
    log(f"  refs light = base2.py's saved run: x {np.array_equal(XR, PR['x16'])}, ex {np.array_equal(ER, PR['ex'])}, "
        f"LS {np.array_equal(LR, PR['LS'])}; npos {np.array_equal(np.array(NPOS), PR['npos'])}")
    log(f"  refs light = the session's light file: {np.array_equal(XR, FR['x16'][np.argsort(FR['rows'])]) and np.array_equal(FR['rows'], np.arange(648))}")
    nW = len(EW)
    log(f"  windows light = the session's light file (first {nW}): x {np.array_equal(XW, FW['x16'][:nW])}, "
        f"ex {np.array_equal(EW, FW['ex'][:nW])}, LS {np.array_equal(LW, FW['LS'][:nW])}")
    np.savez(LIGHT, XR=XR, ER=ER, LR=LR, NPOS=np.array(NPOS), XW=XW, EW=EW, LW=LW)
    log(f"saved {LIGHT} ({time.time() - T0:.0f}s, max memory {rss()} MB)"); sys.exit(0)
Z = np.load(LIGHT); XR, ER, LR, NPOS, XW, EW, LW = (Z[k] for k in ("XR", "ER", "LR", "NPOS", "XW", "EW", "LW"))
log(f"light read from {LIGHT} (made by stage=light of this check)")

# ---------------- the top table; free the other banks ----------------
HW = W["head"][0]; HEAD_E = W["head"][1]; bias = np.asarray(W["head"][2], np.int64)
for k in list(W):
    if k != "head": del W[k]
T = np.asarray(M["head.w"]).astype(np.int64); L, NW = T.shape
assert np.array_equal(T.astype(np.float64), HW) and (HEAD_E == HEAD_E[0]).all()
assert np.abs(T).max() < 1 << 15
TW = np.ascontiguousarray(T.T.astype(np.int16))
BONES = pc(bias); BMAX = int(np.abs(bias).max()); PLAIN_LINE = pc(T).sum(1)

def naf_weight(A):                                         # signed digits, step by step: odd n -> digit 2 - (n mod 4)
    n = np.abs(np.asarray(A, np.int64)).copy(); w = np.zeros(n.shape, np.int64)
    while n.any():
        odd = (n & 1) == 1; d = np.where(odd, 2 - (n & 3), 0); w += odd; n = (n - d) >> 1
    return w
def naf_py(v):                                             # the same rule on one Python integer, digit list
    v = abs(int(v)); c = 0
    while v:
        if v & 1: d = 2 - (v & 3); v -= d; c += 1
        v >>= 1
    return c
_t = np.arange(0, 1 << 16); assert all(int(a) == naf_py(b) for a, b in zip(naf_weight(_t[::97]), _t[::97]))
WALK_LINE = np.zeros(L, np.int64); zero_steps = 0
for i0 in range(0, L, 64):
    s = np.sort(T[i0:i0 + 64], axis=1); g = np.concatenate([s[:, :1], s[:, 1:] - s[:, :-1]], axis=1)
    assert (g[:, 1:] >= 0).all() and np.array_equal(np.cumsum(g, axis=1), s)
    WALK_LINE[i0:i0 + 64] = naf_weight(g).sum(1); zero_steps += int((g[:, 1:] == 0).sum())
log(f"  top table {L} x {NW}; zero steps {zero_steps:,} of {L * (NW - 1):,} ({time.time() - T0:.0f}s, {rss()} MB)")

so = f"{OUT}/indep_trunc.dylib"; subprocess.check_call(["cc", "-O3", "-shared", "-o", so, f"{OUT}/indep_trunc.c"])
lib = ctypes.CDLL(so); P_, I_ = ctypes.c_void_p, ctypes.c_int64
lib.trunc_sums.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_]; lib.trunc_sums.restype = None
def kernel(X):
    X = np.ascontiguousarray(X, np.int64); P = len(X)
    TR = np.zeros((P, NW, 8), np.int64); PH = np.zeros((P, NW, 8), np.int32); BIG = np.zeros(P, np.int64)
    lib.trunc_sums(P, L, NW, X.ctypes.data, TW.ctypes.data, TR.ctypes.data, PH.ctypes.data, BIG.ctypes.data)
    assert not BIG.any(); return TR, PH

KEYS = ["n", "mu", "walk", "writing", "standing_written", "standing_poured", "end", "plain_meet", "plain_div", "plain_stand", "win", "pwin"]
def run(X, EX, LS, name, keep_ph=()):
    N_ = len(X); o = {k: np.zeros(N_, np.int64) for k in KEYS}
    o.update(pours=np.zeros((N_, 8), np.int64), hall=np.zeros((N_, 8), np.int64), surv=np.zeros((N_, 7), np.int64),
             bound=np.zeros((N_, 7), np.int64), cmp=np.zeros((N_, 7), np.int64), top5=np.zeros((N_, 5), np.int64),
             ptop5=np.zeros((N_, 5), np.int64))
    bad = dict(pile_vs_product=0, bound_broken=0, set_aside_not_below_5th=0, end_vs_plain=0, aside_margin_min=None)
    PHK = {}; t0 = time.time()
    for c0 in range(0, N_, 32):
        Xb = np.asarray(X[c0:c0 + 32], np.int64); P = len(Xb)
        TR, PH = kernel(Xb)
        F = (int(HEAD_E[0]) + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
        PILE = product(Xb, HW); S = power(PILE, F[:, None]) + bias[None, :]
        bad["pile_vs_product"] += int((TR[:, :, 7] != PILE).sum())
        for p in range(P):
            q = c0 + p; x = Xb[p]; n = int((x != 0).sum()); Fp = int(F[p])
            if q in keep_ph: PHK[q] = PH[p].copy(); PHK[("alive", q)] = []
            I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
            F2 = -Fp; I2 = F2 >> 16; e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - I2
            # mu: |beta_w - bias_w 2^s / e| <= 1/2 + |bias_w| |e2 / 2^s2 - 2^s / e|  (gamma), one score step = 2^s / e (delta)
            gam = Fraction(1, 2) + BMAX * abs(Fraction(e2) / Fraction(2) ** s2 - Fraction(2) ** s / e)
            gamma = ceil_frac(gam); delta = max(1, ceil_frac(Fraction(2) ** s / e)); mu = 2 * gamma + delta
            beta = power(bias, F2)
            o["n"][q] = n; o["mu"][q] = mu
            o["walk"][q] = int(naf_weight(x) @ WALK_LINE); o["writing"][q] = int(pc(x).sum())
            o["standing_written"][q] = int(BONES.sum()) * pc1(e2); o["standing_poured"][q] = int(pc(beta).sum())
            o["hall"][q] = PH[p].sum(0, dtype=np.int64)
            o["plain_meet"][q] = int(pc(x) @ PLAIN_LINE); o["plain_div"][q] = int(pc(PILE[p]).sum()) * pc1(e)
            o["plain_stand"][q] = int(BONES.sum())
            s5 = np.sort(S[p])[-5]                                    # the true 5th-best score
            alive = np.arange(NW)
            for k in range(8):
                o["pours"][q, k] = int(PH[p, alive, k].sum(dtype=np.int64))
                if q in keep_ph: PHK[("alive", q)].append(alive.copy())
                V = beta[alive] + (TR[p, alive, k] << Q[k])          # the counted pile after band k, cut at place Q_k
                if k == 7: break
                R = n * ((1 << Q[k]) - 1)
                bad["bound_broken"] += int((np.abs(PILE[p] - (TR[p, :, k] << Q[k])) > R).sum())
                na = len(alive); v5 = np.sort(V)[-5]; thr = v5 - 2 * R - mu
                keep = V > thr
                aside = alive[~keep]
                if len(aside):
                    bad["set_aside_not_below_5th"] += int((S[p, aside] >= s5).sum())
                    m = int(s5 - S[p, aside].max())
                    bad["aside_margin_min"] = m if bad["aside_margin_min"] is None else min(bad["aside_margin_min"], m)
                o["bound"][q, k] = pc1(n) + pc1(mu) + pc1(2 * R + mu); o["cmp"][q, k] = 2 * na
                alive = alive[keep]; o["surv"][q, k] = len(alive)
            pile = V - beta[alive]; sc = power(pile, Fp) + bias[alive]
            bad["end_vs_plain"] += int((sc != S[p, alive]).sum())
            o["end"][q] = int(pc(beta[alive]).sum() + pc(pile).sum() * pc1(e) + BONES[alive].sum())
            Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
            o["top5"][q] = top5(Sp[None])[0]; o["win"][q] = race(Sp)
            o["ptop5"][q] = top5(S[p:p + 1])[0]; o["pwin"][q] = race(S[p])
        del TR, PH, PILE, S
        if (c0 // 32) % 10 == 0: log(f"    {name}: {min(N_, c0 + 32)}/{N_} ({time.time() - t0:.0f}s, {rss()} MB)")
    return o, bad, PHK

def total(o): return o["walk"] + o["writing"] + o["standing_written"] + o["standing_poured"] + o["pours"].sum(1) + o["bound"].sum(1) + o["end"]
def plain(o): return o["plain_meet"] + o["plain_div"] + o["plain_stand"]
def pct(a, b): return f"{100 * a / b:.2f}%"
def compare(o, Z, rows, name):
    keys = KEYS[:-2] + ["pours", "hall", "surv", "bound", "cmp", "top5", "win", "ptop5", "pwin"]
    diff = {k: int((np.asarray(o[k]).reshape(len(rows), -1) != np.asarray(Z[k])[rows].reshape(len(rows), -1)).any(1).sum()) for k in keys}
    log(f"  {name}: positions differing from the session's saved arrays, per array: " + ", ".join(f"{k} {v}" for k, v in diff.items()))
    return sum(diff.values())

ALL_OK = True
ZR = np.load(f"{OUT}/top_refs.npz")
if STAGE == "refs":
    # PH kept for 3 texts' positions (photons again in numpy)
    cs = np.concatenate([[0], np.cumsum(NPOS)]); PHTXT = [0, 5, 22]
    keep_ph = set(int(q) for t in PHTXT for q in range(cs[t], cs[t + 1]))
    o, bad, PHK = run(XR, ER, LR, "refs", keep_ph=keep_ph)
    n = len(ER)
    log(f"\nREFS ({n} positions). checks: {bad}")
    t5 = int((o["top5"] == o["ptop5"]).all(1).sum()); t5s = int((np.sort(o["top5"], 1) == np.sort(o["ptop5"], 1)).all(1).sum())
    t5r = int((o["top5"] == PR["top5"]).all(1).sum()); wr = int((o["win"] == PR["win"]).sum()); wp = int((o["win"] == o["pwin"]).sum())
    log(f"  top 5 = plain read (set and order) {t5}/{n} (set {t5s}/{n}); = base2.py's saved run {t5r}/{n}; winner = plain {wp}/{n}, = saved {wr}/{n}")
    ALL_OK &= t5 == n and t5r == n and wr == n and wp == n
    ALL_OK &= bad["pile_vs_product"] == 0 and bad["bound_broken"] == 0 and bad["set_aside_not_below_5th"] == 0 and bad["end_vs_plain"] == 0
    TOT, PL = total(o), plain(o)
    log(f"  plain read {PL.sum() / n:,.1f} a position (meetings {o['plain_meet'].sum() / n:,.1f}); band read {TOT.sum() / n:,.1f} = {pct(TOT.sum(), PL.sum())}")
    for k in ("walk", "writing", "standing_written", "standing_poured", "end"):
        log(f"    {k:18s} {o[k].sum() / n:14,.1f}  {pct(o[k].sum(), PL.sum())}")
    log(f"    pours              {o['pours'].sum() / n:14,.1f}  {pct(o['pours'].sum(), PL.sum())}  (all bands into all words {o['hall'].sum() / n:,.1f} = {pct(o['hall'].sum(), o['plain_meet'].sum())} of the meetings)")
    log(f"    bound              {o['bound'].sum() / n:14,.1f}   compares {o['cmp'].sum() / n:,.1f}")
    for k in range(8):
        sv = o["surv"][:, k] if k < 7 else None
        svs = f"{int(np.median(sv)):6d} {sv.mean():9.1f} {sv.min():6d} {sv.max():6d}" if sv is not None else " " * 31
        log(f"  band {k} (from place {Q[k]:2d}): still in {svs}  pours {o['pours'][:, k].sum() / n:13,.1f} ({pct(o['pours'][:, k].sum(), PL.sum())})"
            f"  all words {o['hall'][:, k].sum() / n:13,.1f} ({pct(o['hall'][:, k].sum(), o['hall'].sum())})" + (f"  bound {o['bound'][:, k].mean():.1f}" if k < 7 else ""))
    whole = 1745178133 - 191031613 + int(round(TOT.sum() / n))
    log(f"  whole machine with this read in place of the collapsing read's 191,031,613: {whole:,} = {pct(whole, 3666596259)} of 3,666,596,259")
    ZR = np.load(f"{OUT}/top_refs.npz"); assert np.array_equal(ZR["sel"], np.arange(648))
    ALL_OK &= compare(o, ZR, np.arange(648), "refs") == 0

    # photons on 3 texts again in numpy (no C)
    log(f"\nPHOTONS AGAIN IN NUMPY, texts {PHTXT} ({len(keep_ph)} positions)")
    mism = 0; mism_pour = 0; mism_walk = 0
    for q in sorted(keep_ph):
        x = XR[q].astype(np.int64); ph = np.zeros((8, NW), np.int64)
        for i0 in range(0, L, 128):
            a = np.abs(x[i0:i0 + 128, None] * T[i0:i0 + 128])
            hi = 30
            for k in range(8):
                ph[k] += pc((a >> Q[k]) & ((1 << (hi - Q[k])) - 1)).sum(0); hi = Q[k]
        mism += int((ph.T != PHK[q]).sum())
        # the pours from these numpy photons into my own survivors before each band, against my run and the session's
        pn = [int(ph[k][PHK[("alive", q)][k]].sum()) for k in range(8)]
        mism_pour += int(pn != o["pours"][q].tolist()) + int(pn != ZR["pours"][q].tolist()) + int(ph.sum(1).tolist() != ZR["hall"][q].tolist())
        # walk once more in Python integers on this position: lit signed digits of x times each line's gap digits
        w_py = sum(naf_py(xi) * int(WALK_LINE[i]) for i, xi in enumerate(x.tolist()) if xi)
        mism_walk += int(w_py != ZR["walk"][q]) + int(w_py != o["walk"][q])
    log(f"  band photons per word per band, numpy vs my C: differing {mism}; pours per band (numpy photons, my survivors) vs mine and the session's, and every band over every word vs the session: differing {mism_pour}; walk (Python integers) vs session and mine: differing {mism_walk}")
    ALL_OK &= mism == 0 and mism_pour == 0 and mism_walk == 0
    for t in PHTXT:
        r = np.arange(cs[t], cs[t + 1])
        log(f"  text {t} ({len(r)} positions), per position mean: walk {o['walk'][r].mean():,.1f} (session {ZR['walk'][r].mean():,.1f}); "
            f"pours per band " + " ".join(f"{o['pours'][r, k].mean():,.1f}" for k in range(8))
            + " | session " + " ".join(f"{ZR['pours'][r, k].mean():,.1f}" for k in range(8)))

    np.savez(f"{OUT}/indep_refs_out.npz", **o)
if STAGE == "windows":
    # ---------------- windows ----------------
    o2, bad2, _ = run(XW, EW, LW, "windows")
    n2 = len(EW); J = np.load(f"{HERE}/judge5_ref.npz")["top5"][:NWIN].reshape(-1, 5)
    log(f"\nWINDOWS ({NWIN} windows, {n2} positions). checks: {bad2}")
    t5 = int((o2["top5"] == o2["ptop5"]).all(1).sum()); t5j = int((o2["top5"] == J).all(1).sum()); wp = int((o2["win"] == o2["pwin"]).sum())
    inn = int((J == o2["win"][:, None]).any(1).sum()); same = int((J[:, 0] == o2["win"]).sum())
    log(f"  top 5 = plain read {t5}/{n2}; = judge5_ref (set and order) {t5j}/{n2}; winner = plain {wp}/{n2}; judge5.judge's counts: in top 5 {inn}/{n2}, same top word {same}/{n2}")
    ALL_OK &= t5 == n2 and t5j == n2 and wp == n2 and inn == n2 and same == n2
    ALL_OK &= bad2["pile_vs_product"] == 0 and bad2["bound_broken"] == 0 and bad2["set_aside_not_below_5th"] == 0 and bad2["end_vs_plain"] == 0
    TOT2, PL2 = total(o2), plain(o2)
    log(f"  band read {pct(TOT2.sum(), PL2.sum())} of the plain read on these positions; after band 3 still in: median {int(np.median(o2['surv'][:, 3]))}, mean {o2['surv'][:, 3].mean():.1f}, least {o2['surv'][:, 3].min()}, most {o2['surv'][:, 3].max()}")
    ZW = np.load(f"{OUT}/top_windows_p1of2.npz"); assert np.array_equal(ZW["sel"][:n2], np.arange(n2))
    ALL_OK &= compare(o2, ZW, np.arange(n2), "windows") == 0

    # the session's 6,000 aggregates, from its saved arrays (both parts)
    Z1, Z2 = np.load(f"{OUT}/top_windows_p1of2.npz"), np.load(f"{OUT}/top_windows_p2of2.npz")
    cat = {k: np.concatenate([Z1[k], Z2[k]]) for k in ("walk", "writing", "standing_written", "standing_poured", "end", "pours", "bound", "plain_meet", "plain_div", "plain_stand", "surv")}
    tw, pw_ = total(cat), plain(cat)
    log(f"  the session's 6,000 from its arrays: band read {tw.sum() / 6000:,.1f} a position = {pct(tw.sum(), pw_.sum())}; after band 3 median {int(np.median(cat['surv'][:, 3]))} mean {cat['surv'][:, 3].mean():.1f} least {cat['surv'][:, 3].min()} most {cat['surv'][:, 3].max()}; "
        f"band 3 pours {pct(cat['pours'][:, 3].sum(), pw_.sum())}, band 2 {pct(cat['pours'][:, 2].sum(), pw_.sum())}")
    ZRd = {k: ZR[k] for k in ZR.files}
    allt = int(tw.sum()) + int(total(ZRd).sum()); allp = int(pw_.sum()) + int(plain(ZRd).sum())
    log(f"  all 6,648 (the session's arrays): {pct(allt, allp)}")

if STAGE == "shuffled":
    o = dict(np.load(f"{OUT}/indep_refs_out.npz")); n = len(ER)
    # ---------------- control: shuffled light ----------------
    rng = np.random.default_rng(2026); XS = np.stack([x[rng.permutation(768)] for x in XR])
    o3, bad3, _ = run(XS, ER, LR, "refs shuffled")
    t5 = int((o3["top5"] == o3["ptop5"]).all(1).sum()); wp = int((o3["win"] == o3["pwin"]).sum())
    TOT3, PL3 = total(o3), plain(o3)
    log(f"\nSHUFFLED LIGHT (refs): checks {bad3}; top 5 = plain {t5}/{n}; winner {wp}/{n}; band read {pct(TOT3.sum(), PL3.sum())} of the plain read; "
        f"places 14 and up {pct(o3['hall'][:, :4].sum(), o3['hall'].sum())} of the exposures' photons (real {pct(o['hall'][:, :4].sum(), o['hall'].sum())}); "
        f"after band 3 median {int(np.median(o3['surv'][:, 3]))}; positions where band 2's test set any word aside {int((o3['surv'][:, 2] < NW).sum())} (real {int((o['surv'][:, 2] < NW).sum())})")
    ZS = np.load(f"{OUT}/top_refs_shuffled.npz")
    ALL_OK &= t5 == n and wp == n and bad3["set_aside_not_below_5th"] == 0 and bad3["bound_broken"] == 0 and bad3["end_vs_plain"] == 0
    ALL_OK &= compare(o3, ZS, np.arange(648), "refs shuffled") == 0
log(f"\n{'ALL HOLD' if ALL_OK else 'SOMETHING DIFFERS'} ({time.time() - T0:.0f}s, max memory {rss()} MB)")
