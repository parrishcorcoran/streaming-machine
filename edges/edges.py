"""WHAT MOVES THE EDGES.  His words: "What makes that first wave matter.  What moves the edges.  Thats it."
Not words, not relations between texts: the knife edges themselves.

HIS PICTURE, restated before measuring. At every depth k the light (the stream) meets the fc plate after the water
level. Each of the 3,072 neurons piles up its meeting; the pile, with its bias, lands somewhere on that neuron's knife
edge (the bend), and where it lands picks the share that passes: dark side or bright side, and how far. The light at
depth k is not one thing: it is the word row, the position row, and every write the stream got below (each depth's
arrival write A_l and middle write M_l), plus the biases those writes carried. With the water level's scale held (one
number per position, the plate's power of two per neuron; the centering is folded in the plate), a meeting is a sum
of products, so each edge's pile splits EXACTLY into one push per source. The question: which pushes move the edge,
which decide its side, and what the newest arrival (the wave) does to edges already standing.

Units: the value each edge sees, H = power(pile, F) + bias, whole numbers at 2^-16 (the edge sits at H = 0, the share
table is read at (H + 8) >> 4).  pile = x16 met with the plate, x16 = the light after the water level.
    pile x 2^sh = sum over sources s of N_s,   N_s = S_s met with the plate (whole numbers),
where S_s is source s's part of the stream (whole numbers, 2^-16) and one more source, the water level's rounding,
S_r = x16 x 2^sh - R, makes it exact. In H units each push is N_s x EXP2[f] x 2^(I - 16 - sh) (an exact rational),
the fc bias adds directly, and the machine's own rounding of the result (at most 1/2 unit) is the last term.
Sources at depth k: (a) word row, (b) position row, (c) arrival writes A_0..A_k (A_k is this depth's own, the last
thing added before the plate), (d) middle writes M_0..M_{k-1}, (e) biases: the fc bias and the stream's biases
(every proj and mproj bias carried so far), (f) the two roundings. A second split: the whole light on line 447 (the
first-place stone's line) alone, and on the other 767 lines.

THE FIXED PART (the same machinery for every word): fc bias + stream biases + position row, and from depth 3 on the
whole light's push on line 447 in place of those three on that line. Line 447's value is the token's own; a variant
puts the median value over all positions in its place (the line is said to be the same for every word: that tests it).

Texts: the 30 refs texts (ids + ref), positions >= 1 for every measure (position 0 is the stone itself).
Controls: (i) each position's light shuffled across its lines (one order per position, the same for every source, so
the split stays exact; the water level is unchanged by a shuffle), everything measured again; (ii) the fixed part
with another position's row; (iii) the fixed part's sides against an UNRELATED token's true sides (same position, same
edge): the standing pattern alone; and two unrelated tokens' true sides against each other.
Order-free: the sources added in a random order give the same piles; the lines in a random order give the same pushes.

PREDICTIONS, written before running.
  The task's: the fixed part already puts 70-80% of the edges on their true side, about the share two unrelated texts
    agree on; the token's own light decides the other 20-30%, mostly through the writes of the last 2-3 depths below;
    the newest write alone flips 10-20% of edges at each depth.
  Claude's:
    - the bright side is the minority at every depth (5-25% of edges bright);
    - two unrelated tokens agree on 75-90% of sides, mostly the standing dark pattern;
    - the fixed part puts 75-90% on the true side, within 3 points of its score against an unrelated token's sides
      (it knows the standing pattern, not the token), and gets under 40% of the true-bright edges from depth 3 on;
    - with the light shuffled the fixed part scores about the same (the fc bias, untouched by a shuffle, holds most
      edges dark): the score is mostly the distribution;
    - the fixed part reads the true share within one table step for under 5% of edges;
    - sizes: the word row's push falls with depth (under 10% of all pushing from depth 6 on); the older writes together
      push most from depth 3 on; line 447's push is small (under 10%), because the fc plates are dim on that line --
      the stone acts through the water level, not through the plate; every pile is a balance, sum |push| 3-10x |pile|,
      half of it in 3-6 sources at the deep depths;
    - this depth's own arrival write alone flips 3-10% of edges, depth k-1's two writes 5-15%, the word row 10-25% at
      depths 0-2 and under 10% by depth 8; the shuffled light flips about as many (flips follow push size).
    python3 edges/edges.py          (needs edges/forward.npz from edges/forward.py)  -> edges/edges.npz, report printed
"""
import sys, os, json, time, numpy as np
from fractions import Fraction
np.seterr(all="ignore")                                   # numpy's float products on this Mac raise spurious flags
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE); ARGS = sys.argv[1:]
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)                                                 # the machine, untouched
STONE, STONE_FROM = 447, 3
refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]; NT = len(TEXTS)
FW = np.load("edges/forward.npz")
BPROJ = [W[f"proj.{l}"][2].astype(np.int64) for l in range(12)]
BMPROJ = [W[f"mproj.{l}"][2].astype(np.int64) for l in range(12)]
for k in list(W):                                         # only the fc plates are met here
    if not k.startswith("fc."): del W[k]
WPE = M["wpe"]
_wte = M["wte"]; WROWS = [_wte[s] for s in TEXTS]; del _wte    # each text's word rows, the table itself let go
AW = [FW[f"A{t}"].astype(np.int64) for t in range(NT)]   # (12, P, 768) arrival writes, bias included
MWr = [FW[f"M{t}"].astype(np.int64) for t in range(NT)]  # (12, P, 768) middle writes, bias included
HW = [FW[f"H{t}"].astype(np.int64) for t in range(NT)]   # (12, P, 3072) the machine's own edge values
for t in range(NT): assert list(FW[f"ids{t}"]) == TEXTS[t]

ROWS = [(t, p) for t in range(NT) for p in range(1, len(TEXTS[t]))]; NR = len(ROWS)
ROW = {tp: i for i, tp in enumerate(ROWS)}
rng = np.random.default_rng(12)                           # (iii) an unrelated token: another text, same position, other word
UNREL = np.full(NR, -1)
for i, (t, p) in enumerate(ROWS):
    cand = [u for u in range(NT) if u != t and len(TEXTS[u]) > p and TEXTS[u][p] != TEXTS[t][p]]
    if cand: UNREL[i] = ROW[(int(rng.choice(cand)), p)]
rng = np.random.default_rng(13)                           # (ii) another position's row
OTHERPOS = [np.array([0] + [int(rng.choice([q for q in range(1, 24) if q != p])) for p in range(1, len(s))]) for s in TEXTS]
PERM = [np.array([np.random.default_rng(1000 * t + p + 11).permutation(768) for p in range(len(s))]) for t, s in enumerate(TEXTS)]
RNG_ORDER, RNG_FRAC = np.random.default_rng(14), np.random.default_rng(15)
FALLBACK = [0, 0]                                         # decisions settled by Python whole numbers (float bound too close)

# ---------------------------------------------------------------- exact tools
def meet(S, w):
    """whole-number light (n, 768) met with the plate (768, J) holding int16, exactly: two 16-place halves of the light"""
    assert np.abs(S).max() < 2 ** 31
    lo, hi = (S & 0xFFFF).astype(np.float64), (S >> 16).astype(np.float64)
    a, b = hi @ w, lo @ w
    assert np.abs(a).max() < 2 ** 52 and np.abs(b).max() < 2 ** 52
    return a.astype(np.int64) * 65536 + b.astype(np.int64)

def side_of(N, E, e, badd):
    """exactly whether N x E x 2^e + badd > 0 (N, E, badd whole numbers): float with a safe bound, and Python whole
    numbers wherever the bound cannot decide"""
    a = N.astype(np.float64) * (E.astype(np.float64) * np.ldexp(1.0, e))
    bf = np.asarray(badd).astype(np.float64); h = a + bf
    out = h > 0
    amb = np.abs(h) <= (np.abs(a) + np.abs(bf)) * 2.0 ** -45
    if amb.any():
        Nb, Eb, eb, Bb = (np.broadcast_to(x, N.shape) for x in (N, E, e, badd))
        for i in zip(*np.nonzero(amb)):
            num, ee, bb = int(Nb[i]) * int(Eb[i]), int(eb[i]), int(Bb[i])
            out[i] = ((num << ee) + bb if ee >= 0 else num + (bb << -ee)) > 0
        FALLBACK[0] += int(amb.sum())
    return out

def index_of(N, E, e, badd):
    """the share-table entry the machine's own arithmetic reads for N x E x 2^e + badd: floor(x + 1/2) + badd, then
    (+8) >> 4, + 32768; -1 means below the table (share 0), 65536 above it (all passes)"""
    a = N.astype(np.float64) * (E.astype(np.float64) * np.ldexp(1.0, e))
    tt = a + 0.5; Hr = np.floor(tt).astype(np.int64) + np.broadcast_to(badd, N.shape)
    amb = np.abs(tt - np.round(tt)) <= np.abs(a) * 2.0 ** -45 + 2.0 ** -30
    if amb.any():
        Hr = Hr.copy(); Nb, Eb, eb, Bb = (np.broadcast_to(x, N.shape) for x in (N, E, e, badd))
        for i in zip(*np.nonzero(amb)):
            num, ee = int(Nb[i]) * int(Eb[i]), int(eb[i])
            Hr[i] = (num << ee if ee >= 0 else (num + (1 << (-ee - 1))) >> (-ee)) + int(Bb[i])
        FALLBACK[1] += int(amb.sum())
    return np.clip(((Hr + 8) >> 4) + 32768, -1, 65536)

def true_index(H): return np.clip(((H + 8) >> 4) + 32768, -1, 65536)
def share_at(c): return np.where(c < 0, 0, np.where(c > 65535, 65536, PHI[np.clip(c, 0, 65535)]))

CHK = dict(stream_split=0, stream_split_n=0, H_same=0, H_n=0, pile_split=0, pile_split_n=0, token_split=0,
           exact_obj=0, exact_obj_n=0, rho_half=0, rho_n=0, follow_H=0, follow_side=0, follow_share=0, follow_n=0,
           frac_ok=0, frac_n=0, int64_meet=0, int64_meet_n=0, order_src=0, order_src_n=0, order_lines=0, order_lines_n=0,
           shuffle_level_same=0, shuffle_n=0)

# ---------------------------------------------------------------- one view of one text at one depth
def view(k, S, Sr, x16, sh, F, H, st, w, wst, b, P, t=None, med=None, NposO=None, Spos_other=None):
    """S: (nS, P, 768) the sources (stream units), Sr: the water level's rounding, x16: the light after the water level,
    st: (P,) the stone's line at each position. Returns per-edge results for positions 1..P-1."""
    nS = S.shape[0]; iw, ip, ib = 0, 1, 2
    iA = lambda l: 3 + l; iM = lambda l: 3 + (k + 1) + l
    Sall = np.concatenate([S, Sr[None]])
    N = meet(Sall.reshape(-1, 768), w).reshape(nS + 1, P, 3072)
    pile = product(x16, w)
    target = pile << sh[:, None]; assert np.abs(pile).max() < 2 ** 46 and sh.max() < 16
    Nsum = N.sum(0); CHK["pile_split"] += int((Nsum == target).all()); CHK["pile_split_n"] += 1
    if t is not None:                                     # order-free: sources in a random order
        acc = np.zeros_like(Nsum)
        for s in RNG_ORDER.permutation(nS + 1): acc = acc + N[s]
        CHK["order_src"] += int((acc == target).all()); CHK["order_src_n"] += 1
    stv = Sall[:, np.arange(P), st]                       # (nS+1, P) each source's value on the stone line
    wrow = wst[st]                                        # (P, 3072) the plate's stone row at each position
    N447 = stv[:, :, None] * wrow[None]
    I = F >> 16; E = EXP2[F & 0xFFFF].astype(np.int64); e = I - 16 - sh[:, None]
    g = E.astype(np.float64) * np.ldexp(1.0, e)
    rho = H - (pile.astype(np.float64) * E * np.ldexp(1.0, I - 16) + b)          # the machine's last rounding (float, for sizes)
    # ---- part 2: sizes, H units
    PH = N * g[None]
    indiv = [PH[iw], PH[ip], PH[ib], np.broadcast_to(b.astype(np.float64), (P, 3072))] + \
            [PH[iA(l)] for l in range(k + 1)] + [PH[iM(l)] for l in range(k)] + [PH[nS], rho]
    z = np.zeros((P, 3072))
    groups = [PH[iw], PH[ip], PH[ib] + b, PH[iA(k)], (PH[iA(k - 1)] + PH[iM(k - 1)]) if k >= 1 else z,
              sum([PH[iA(l)] + PH[iM(l)] for l in range(k - 1)], z), PH[nS] + rho]
    L447 = (N447[:nS].sum(0)) * g; Lrest = (Nsum - N[nS] - N447[:nS].sum(0)) * g
    split = [L447, Lrest, np.broadcast_to(b.astype(np.float64), (P, 3072)), PH[nS] + rho]
    # ---- the fixed part and the token's part (whole numbers N)
    rest = (lambda s: N[s] - N447[s]) if k >= STONE_FROM else (lambda s: N[s])
    if k >= STONE_FROM: Nfix = rest(ip) + rest(ib) + N447[:nS].sum(0)
    else: Nfix = N[ip] + N[ib]
    comps = dict(word=rest(iw), newest=rest(iA(k)))
    if k >= 1: comps["M(k-1)"] = rest(iM(k - 1)); comps["A(k-1)"] = rest(iA(k - 1))
    if k >= 2: comps["older"] = sum(rest(iA(l)) + rest(iM(l)) for l in range(k - 1))
    comps["all writes"] = sum(rest(iA(l)) for l in range(k + 1)) + sum((rest(iM(l)) for l in range(k)), np.zeros_like(Nfix))
    comps["level rounding"] = N[nS]
    CHK["token_split"] += int((Nfix + comps["word"] + comps["all writes"] + N[nS] == target).all())
    side = H > 0
    sfix = side_of(Nfix, E, e, b)
    res = dict(side=side, fix=sfix)
    cfix, ctrue = index_of(Nfix, E, e, b), true_index(H)
    res["within"] = np.abs(cfix - ctrue) <= 1
    res["dshare"] = np.abs(share_at(cfix) - share_at(ctrue)).astype(np.float32)
    if med is not None:                                   # the stone line's value replaced by its median over all positions
        res["fix_med"] = side_of(Nfix - N447[:nS].sum(0) + med * wst[STONE][None, :], E, e, b) if k >= STONE_FROM else sfix
    if NposO is not None:                                 # (ii) another position's row
        Ro = NposO - (Spos_other[:, STONE][:, None] * wst[STONE][None, :] if k >= STONE_FROM else 0)
        res["fix_other"] = side_of(Nfix - rest(ip) + Ro, E, e, b)
    res["alone"] = {c: side_of(Nfix + v, E, e, b) for c, v in comps.items()}
    res["without"] = {c: side_of(-v, E, e, H) for c, v in comps.items()}
    wave = dict(newest=N[iA(k)], word=N[iw])
    if k >= 1:
        wave["M(k-1)"] = N[iM(k - 1)]; wave["A(k-1)"] = N[iA(k - 1)]; wave["pair k-1"] = N[iA(k - 1)] + N[iM(k - 1)]
        wave["since last row"] = N[iM(k - 1)] + N[iA(k)]
    res["wave_side"] = {c: side_of(-v, E, e, H) for c, v in wave.items()}
    res["wave_arr"] = {c: np.abs(v * g).astype(np.float32) for c, v in wave.items()}
    res["wave_stand"] = {c: np.abs(H - v * g).astype(np.float32) for c, v in wave.items()}
    res["indiv"] = np.abs(np.array(indiv)).astype(np.float32)
    res["groups"] = np.abs(np.array(groups)).astype(np.float32)
    res["split"] = np.abs(np.array(split)).astype(np.float32)
    res["absH"] = np.abs(H).astype(np.float32)
    srt = -np.sort(-res["indiv"], axis=0); cs = np.cumsum(srt, axis=0); tot = cs[-1]
    res["half"] = (cs < 0.5 * tot[None]).sum(0).astype(np.int16) + 1
    res["cancel"] = (tot / np.maximum(np.abs(H), 1)).astype(np.float32)
    # ---- exact checks (texts 0-2): the pushes add back as exact rationals; the machine's side and share follow
    if t is not None and t < 3:
        No = N.astype(object); tot_o = No.sum(0)
        p2 = np.array([1 << int(s_) for s_ in sh], dtype=object)[:, None]
        CHK["exact_obj"] += int((tot_o == pile.astype(object) * p2).all()); CHK["exact_obj_n"] += 1
        s16 = 16 - I; up = np.vectorize(lambda z_: 1 << int(z_), otypes=[object])
        lhs = (H - b).astype(object) * up(np.maximum(s16, 0)); rhs = pile.astype(object) * E.astype(object) * up(np.maximum(-s16, 0))
        num = lhs - rhs; den = up(np.maximum(s16, 0))
        CHK["rho_half"] += int(sum(2 * abs(int(a_)) <= int(d_) for a_, d_ in zip(num.ravel(), den.ravel()))); CHK["rho_n"] += num.size
        back = np.array([[int(x_) >> int(sh[p_]) for x_ in tot_o[p_]] for p_ in range(P)], dtype=np.int64)
        assert all(int(tot_o[p_, j_]) % (1 << int(sh[p_])) == 0 for p_ in range(P) for j_ in range(0, 3072, 97))
        Hb = power(back, F) + b
        CHK["follow_H"] += int((Hb == H).sum()); CHK["follow_side"] += int(((Hb > 0) == (H > 0)).sum())
        CHK["follow_share"] += int((share_at(true_index(Hb)) == share_at(true_index(H))).sum()); CHK["follow_n"] += H.size
        for _ in range(300):                              # the same in Fractions, edge by edge
            p_, j_ = int(RNG_FRAC.integers(P)), int(RNG_FRAC.integers(3072))
            gq = Fraction(int(E[p_, j_])) * Fraction(2) ** int(e[p_, j_])
            pushes = sum(Fraction(int(N[s_, p_, j_])) * gq for s_ in range(nS + 1))
            rq = Fraction(int(H[p_, j_])) - int(b[j_]) - Fraction(int(pile[p_, j_])) * int(E[p_, j_]) * Fraction(2) ** int(I[p_, j_] - 16)
            CHK["frac_ok"] += int(pushes + int(b[j_]) + rq == int(H[p_, j_]) and abs(rq) <= Fraction(1, 2)); CHK["frac_n"] += 1
        if t == 0:                                        # the meeting in plain int64 (no float at all), every source
            ok = True
            for s_ in range(nS + 1): ok &= np.array_equal(Sall[s_].astype(np.int64) @ wst, N[s_])
            CHK["int64_meet"] += int(ok); CHK["int64_meet_n"] += 1
            perm = RNG_ORDER.permutation(768)                 # the lines in a random order
            CHK["order_lines"] += int(np.array_equal(meet(Sall.reshape(-1, 768)[:, perm], w[perm]).reshape(N.shape), N)); CHK["order_lines_n"] += 1
    keep = slice(1, P)
    def cut(x): return {c: v[keep] for c, v in x.items()} if isinstance(x, dict) else (x[:, keep] if x.ndim == 3 else x[keep])
    return {kk: cut(v) for kk, v in res.items()}

# ---------------------------------------------------------------- one depth's arrays, filled text by text
def put(acc, key, x, r0, r1):
    if key not in acc: acc[key] = np.zeros(((x.shape[0], NR) + x.shape[2:]) if x.ndim == 3 else ((NR,) + x.shape[1:]), x.dtype)
    if x.ndim == 3: acc[key][:, r0:r1] = x
    else: acc[key][r0:r1] = x
def store(acc, res, r0, r1):
    for key, v in res.items():
        if isinstance(v, dict):
            for c, x in v.items(): put(acc, (key, c), x, r0, r1)
        else: put(acc, key, v, r0, r1)
def nest(acc):
    d = {}
    for key, v in acc.items():
        if isinstance(key, tuple): d.setdefault(key[0], {})[key[1]] = v
        else: d[key] = v
    return d

# ---------------------------------------------------------------- run
t0 = time.time(); OUT = {}; REPORT = {}
def pct(a, n): return 100.0 * a / n if n else float("nan")
DEP = range(12)
if ARGS: a_, b_ = ARGS[0].split("-"); DEP = range(int(a_), int(b_) + 1)       # optional: a range of depths, e.g. 0-5
for k in DEP:
    w, we, b = W[f"fc.{k}"]; wst = w.astype(np.int64)
    sb = sum(BPROJ[l] for l in range(k + 1)) + sum((BMPROJ[l] for l in range(k)), np.zeros(768, np.int64))
    def light(t):
        P = len(TEXTS[t]); ids = TEXTS[t]
        Srcs = [WROWS[t], WPE[:P], np.broadcast_to(sb, (P, 768))] + [AW[t][l] - BPROJ[l] for l in range(k + 1)] + \
               [MWr[t][l] - BMPROJ[l] for l in range(k)]
        return np.array(Srcs, dtype=np.int64)
    Rk = [AW[t][:k + 1].sum(0) + MWr[t][:k].sum(0) + WROWS[t] + WPE[:len(TEXTS[t])] for t in range(NT)]
    v447 = np.sort(np.concatenate([Rk[t][1:, STONE] for t in range(NT)])); med = int(v447[len(v447) // 2])
    acc = {"real": {}, "shuf": {}}; r0 = 0
    for t in range(NT):
        P = len(TEXTS[t]); S = light(t); R = Rk[t]
        CHK["stream_split"] += int(np.array_equal(S.sum(0), R)); CHK["stream_split_n"] += 1
        x16, ex, LS = gauge_log(R); sh = ex + ES
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
        H = power(product(x16, w), F) + b
        CHK["H_same"] += int(np.array_equal(H, HW[t][k])); CHK["H_n"] += 1
        Sr = (x16 << sh[:, None]) - R
        po = OTHERPOS[t]; Spo = WPE[po]
        store(acc["real"], view(k, S, Sr, x16, sh, F, H, np.full(P, STONE), w, wst, b, P, t=t, med=med,
                                NposO=meet(Spo, w), Spos_other=Spo), r0, r0 + P - 1)
        # (i) the light shuffled across its lines, one order per position, the same for every source
        pm = PERM[t]; ar = np.arange(P)[:, None]
        Sp, Rp, xp, Srp = S[:, ar, pm], R[ar, pm], x16[ar, pm], Sr[ar, pm]
        x2, ex2, LS2 = gauge_log(Rp)
        CHK["shuffle_level_same"] += int(np.array_equal(x2, xp) and np.array_equal(ex2, ex) and np.array_equal(LS2, LS)); CHK["shuffle_n"] += 1
        Hp = power(product(xp, w), F) + b
        stp = np.array([int(np.flatnonzero(pm[p_] == STONE)[0]) for p_ in range(P)])
        store(acc["shuf"], view(k, Sp, Srp, xp, sh, F, Hp, stp, w, wst, b, P), r0, r0 + P - 1)
        assert ROWS[r0] == (t, 1); r0 += P - 1
    assert r0 == NR
    D = {vn: nest(a_) for vn, a_ in acc.items()}; del acc
    names_ind = ["word", "position", "stream biases", "fc bias"] + [f"A{l}" for l in range(k + 1)] + [f"M{l}" for l in range(k)] + ["level rounding", "last rounding"]
    names_grp = ["word", "position", "biases (fc + stream)", f"newest write A{k}", f"depth {k - 1}'s writes" if k >= 1 else "depth -1's writes",
                 f"older writes (depth <= {k - 2})", "roundings"]
    names_split = ["line 447 (whole light)", "other 767 lines", "fc bias", "roundings"]
    rep = {}
    print(f"\n======== DEPTH {k} (fc.{k}): {NR} positions x 3,072 edges = {NR * 3072:,} edges;  line 447's light (stream value): "
          f"median {med / 65536:.2f}, 10%-90% {v447[len(v447) // 10] / 65536:.2f} .. {v447[9 * len(v447) // 10] / 65536:.2f}   ({time.time() - t0:.0f}s)")
    for vn in ("real", "shuf"):
        d = D[vn]; mH = float(np.median(d["absH"])); n = d["side"].size
        lab = "REAL LIGHT" if vn == "real" else "CONTROL (i): LIGHT SHUFFLED ACROSS ITS LINES"
        print(f"  --- {lab}: median |pile| at the edge {mH / 65536:.4f};  bright side {pct(d['side'].sum(), n):.1f}% of edges")
        tot_all = d["indiv"].sum(dtype=np.float64)
        rows = []
        for nm, arr in [("each source", (names_ind, d["indiv"])), ("by kind", (names_grp, d["groups"])), ("two ways by line", (names_split, d["split"]))]:
            nms, X = arr; T_ = X.sum(dtype=np.float64)
            if vn == "real" or nm != "each source": print(f"    {nm}: median |push| / median |pile|, share of all pushing")
            for i_, nn in enumerate(nms):
                r_ = (float(np.median(X[i_])) / mH, 100.0 * X[i_].sum(dtype=np.float64) / T_)
                rows.append((nm, nn) + r_)
                if vn == "real" or nm != "each source":
                    print(f"      {nn:28s} {r_[0]:7.3f}   {r_[1]:5.1f}%")
        hh = d["half"]; cc = d["cancel"]
        print(f"    sources to hold half of all pushing: median {np.median(hh):.0f}, mean {hh.mean():.2f} of {len(names_ind)};"
              f"  all pushing / |pile|: median {np.median(cc):.2f}, 90% {np.quantile(cc, 0.9):.1f}")
        tr = d["side"]; fx = d["fix"]; m = fx != tr
        line = (f"    THE FIXED PART: on the true side {pct((fx == tr).sum(), n):.2f}%  (true-bright edges it calls bright {pct((fx & tr).sum(), tr.sum()):.1f}%,"
                f" true-dark it calls dark {pct((~fx & ~tr).sum(), (~tr).sum()):.1f}%);  share within one table step {pct(d['within'].sum(), n):.2f}%;"
                f"  median |share error| {np.median(d['dshare']) / 65536:.4f}")
        print(line)
        R_ = dict(bright=pct(tr.sum(), n), fixed=pct((fx == tr).sum(), n), fix_bright=pct((fx & tr).sum(), tr.sum()),
                  fix_dark=pct((~fx & ~tr).sum(), (~tr).sum()), within=pct(d["within"].sum(), n), sizes=rows,
                  half_median=float(np.median(hh)), cancel_median=float(np.median(cc)))
        if vn == "real":
            ok = UNREL >= 0
            fu = (fx[ok] == tr[UNREL[ok]]).mean() * 100; tu = (tr[ok] == tr[UNREL[ok]]).mean() * 100
            fm, fo = pct((d["fix_med"] == tr).sum(), n), pct((d["fix_other"] == tr).sum(), n)
            base = 100 * (tr.mean() * fx.mean() + (1 - tr.mean()) * (1 - fx.mean()))
            print(f"    controls: (iii) fixed part vs an UNRELATED token's true side {fu:.2f}%;  two unrelated tokens agree {tu:.2f}%;"
                  f"  (ii) fixed part with another position's row {fo:.2f}%;  stone line at its median value {fm:.2f}%;  chance from the two rates {base:.2f}%")
            R_.update(fix_unrel=fu, unrel_agree=tu, fix_other=fo, fix_med=fm, chance=base)
        print(f"    misplaced by the fixed part: {pct(m.sum(), n):.2f}% of edges. For those, each piece of the token's own light:"
              f" alone with the fixed part reaches the true side | taken away from the whole, the edge leaves its true side")
        R_["misplaced"] = pct(m.sum(), n); R_["alone"] = {}; R_["needed"] = {}
        for c in d["alone"]:
            a_ = pct((d["alone"][c][m] == tr[m]).sum(), m.sum()); nd = pct((d["without"][c][m] != tr[m]).sum(), m.sum())
            print(f"      {c:16s} {a_:6.1f}%   {nd:6.1f}%"); R_["alone"][c] = a_; R_["needed"][c] = nd
        print(f"    THE WAVE: edges that leave their side when the push is taken away (all edges) | median |push| / median |rest of the pile|")
        R_["flips"] = {}
        for c in d["wave_side"]:
            fl = pct((d["wave_side"][c] != tr).sum(), n); rr = float(np.median(d["wave_arr"][c])) / float(np.median(d["wave_stand"][c]))
            print(f"      {c:16s} {fl:6.2f}%   {rr:6.3f}"); R_["flips"][c] = (fl, rr)
        rep[vn] = R_
    REPORT[k] = rep; del D
    print(f"  checks so far: {CHK};  float bound settled by whole numbers: sides {FALLBACK[0]}, table entries {FALLBACK[1]}", flush=True)

OLD = json.load(open("edges/edges.json")) if os.path.exists("edges/edges.json") else dict(report={}, checks={}, fallback={})
OLD["report"].update({str(k): v for k, v in REPORT.items()}); OLD["checks"][f"{DEP.start}-{DEP.stop - 1}"] = CHK
OLD["fallback"][f"{DEP.start}-{DEP.stop - 1}"] = FALLBACK
json.dump(OLD, open("edges/edges.json", "w"), indent=1)
REPORT = {int(k): v for k, v in OLD["report"].items()}
print("\n================ SUMMARY (positions >= 1, 30 texts) ================")
print("depth | bright% | FIXED on true side% | vs unrelated token% | unrelated tokens agree% | chance% | other pos row% | stone at median% | shuffled light: fixed% | within 1 step%")
for k in sorted(REPORT):
    r, s = REPORT[k]["real"], REPORT[k]["shuf"]
    print(f"  {k:2d}  | {r['bright']:6.1f} | {r['fixed']:8.2f} | {r['fix_unrel']:8.2f} | {r['unrel_agree']:8.2f} | {r['chance']:7.2f} | {r['fix_other']:8.2f} | {r['fix_med']:8.2f} | {s['fixed']:8.2f} | {r['within']:6.2f}")
print("depth | edges leaving their side when taken away (real | shuffled): newest write A_k, depth k-1's two writes, M_{k-1}+A_k, word row")
for k in sorted(REPORT):
    r, s = REPORT[k]["real"]["flips"], REPORT[k]["shuf"]["flips"]
    def f_(d_, c): return f"{d_[c][0]:5.2f}" if c in d_ else "  -  "
    print(f"  {k:2d}  | newest {f_(r, 'newest')} | {f_(s, 'newest')}   pair {f_(r, 'pair k-1')} | {f_(s, 'pair k-1')}   since last row {f_(r, 'since last row')} | {f_(s, 'since last row')}   word {f_(r, 'word')} | {f_(s, 'word')}")
print(f"\nchecks: {CHK}")
print(f"decisions the float bound could not settle, settled by Python whole numbers: sides {FALLBACK[0]}, table entries {FALLBACK[1]}")
import resource; print(f"largest memory held: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB;  {time.time() - t0:.0f}s")
