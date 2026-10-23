"""CHECKER: an independent split of every knife edge's pile, written apart from edges.py and check_edges.py.

His picture: at depth k the light (the stream) meets the fc plate after the water level; each neuron's pile, with
its bias, lands on its knife edge. The light is the sum of its parts: word row + position row + every arrival write
A_l (l <= k) and middle write M_l (l < k), each with its bias taken out, + the biases those writes carried. With the
water level's scale held, the pile x 2^sh is the sum of each part met with the plate, plus the water level's rounding.

What is different from the report's route:
  - my own column loop (the machine's own functions, base2.py executed from its source), stored separately;
  - every meeting in plain int64 numpy products (no float, no halves);
  - every side decided by an exact whole-number THRESHOLD on the push (N > floor(-b 2^s / E), N < ceil(H 2^s / E)),
    no float bound at all; table entries in Python whole numbers;
  - the shuffle control run twice: with the report's own orders (must match it to the last edge) and with fresh
    orders (must match it within scatter); the unrelated token and the other position drawn with fresh seeds.
    python3 edges/recheck_split.py forward            -> scratchpad cache of the column
    python3 edges/recheck_split.py exact 7,22         -> exactness on two texts, all 12 depths
    python3 edges/recheck_split.py depths 0,1,5        -> all 30 texts at those depths, compared with edges.json
"""
import sys, os, json, time, math, numpy as np
from fractions import Fraction
np.seterr(all="ignore")
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
CACHE = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/recheck_column.npz"
MODE, ARG = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)                                                        # the machine, untouched
refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]; NT = len(TEXTS)
STONE = 447
t0 = time.time()
def say(*a): print(*a, f"({time.time() - t0:.0f}s)", flush=True)

if MODE == "forward":
    out = {}; same = 0
    for t, ids in enumerate(TEXTS):
        P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; Rm, Aa, Mm, Hh = [], [], [], []
        for l in range(12):
            a = plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}"); R = R + a; Rm.append(R)
            h = gauged_bank(R, f"fc.{l}"); Hh.append(h)
            m = plain_bank(edge(h), f"mproj.{l}"); R = R + m; Aa.append(a); Mm.append(m)
        w_, we_, b_ = W["head"]; x16, ex, LS = gauge_log(R[-1:])
        last = race(power(product(x16, w_), (we_[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b_)
        same += last == forward(ids)
        for nm, X in (("R", Rm), ("A", Aa), ("M", Mm), ("H", Hh)):
            X = np.array(X); assert np.abs(X).max() < 2 ** 31; out[f"{nm}{t}"] = X.astype(np.int32)
    np.savez(CACHE, **out)
    FW = np.load("edges/forward.npz")
    sameH = sum(np.array_equal(out[f"H{t}"], FW[f"H{t}"]) and np.array_equal(out[f"A{t}"], FW[f"A{t}"]) and np.array_equal(out[f"M{t}"], FW[f"M{t}"]) for t in range(NT))
    say(f"my column: last word == base2.forward {same}/{NT};  H, A, M identical to edges/forward.npz {sameH}/{NT}")
    sys.exit()

C = np.load(CACHE)
COL = [dict(R=C[f"R{t}"].astype(np.int64), A=C[f"A{t}"].astype(np.int64), M=C[f"M{t}"].astype(np.int64), H=C[f"H{t}"].astype(np.int64)) for t in range(NT)]
BP = [W[f"proj.{l}"][2].astype(np.int64) for l in range(12)]; BM = [W[f"mproj.{l}"][2].astype(np.int64) for l in range(12)]
FC = {l: (M[f"fc.{l}.w"].astype(np.int64), M[f"fc.{l}.e"].astype(np.int64), M[f"fc.{l}.b"].astype(np.int64)) for l in range(12)}
del W                                                            # the float copies of the plates are not used below
WTE, WPE = M["wte"], M["wpe"]

def sources(t, k):
    """the light at depth k, part by part (each write with its bias taken out; the biases together as one part)"""
    ids = TEXTS[t]; P = len(ids); c = COL[t]
    sb = sum(BP[l] for l in range(k + 1)) + sum((BM[l] for l in range(k)), np.zeros(768, np.int64))
    names = ["word", "position", "stream biases"] + [f"A{l}" for l in range(k + 1)] + [f"M{l}" for l in range(k)]
    S = [WTE[ids], WPE[:P], np.broadcast_to(sb, (P, 768))] + [c["A"][l] - BP[l] for l in range(k + 1)] + [c["M"][l] - BM[l] for l in range(k)]
    return names, np.array(S, dtype=np.int64), sb

def scale(R, k):
    w, we, b = FC[k]
    x16, ex, LS = gauge_log(R); sh = ex + ES
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    I, E = F >> 16, EXP2[F & 0xFFFF].astype(np.int64)
    s = 16 - I + sh[:, None]                                     # a push N is worth N x E / 2^s at the edge
    return x16, sh, F, I, E, s

WF = {}
def meet(S, w):
    """exact meeting: float64 products when every row's sum of |S| x 2^15 stays below 2^53 (then every partial sum is
    a whole number below 2^53, so no rounding happens in any order); otherwise plain int64 products"""
    S = np.ascontiguousarray(S)
    if int(np.abs(S).sum(1).max()) * 32768 < 2 ** 53:
        key = id(w)
        if key not in WF: WF.clear(); WF[key] = (w, w.astype(np.float64))
        return (S.astype(np.float64) @ WF[key][1]).astype(np.int64)
    return S @ w

def thr_gt(B, E, s):
    """T with  N x E / 2^s + B > 0  <=>  N > T  (whole numbers, exact)"""
    sp, sn = np.maximum(s, 0), np.maximum(-s, 0)
    if int(np.abs(B).max()).bit_length() + int(sp.max()) < 62 and int(E.max()).bit_length() + int(sn.max()) < 62:
        return np.floor_divide(np.left_shift(-B, sp), np.left_shift(E, sn))
    Bo, Eo, so = (np.broadcast_to(x, E.shape).ravel() for x in (B, E, s))
    return np.array([((-int(b_)) << max(int(s_), 0)) // (int(e_) << max(-int(s_), 0)) for b_, e_, s_ in zip(Bo, Eo, so)], dtype=object).reshape(E.shape)

def thr_lt(B, E, s):
    """T with  B - N x E / 2^s > 0  <=>  N < T"""
    return -thr_gt(B, E, s)                                       # B - N E/2^s > 0 <=> (-N) E/2^s + B > 0 <=> -N > T' <=> N < -T'

def entry(N, E, s, b):
    """the share-table entry the machine's own arithmetic reads for the value N x E / 2^s + b (Python whole numbers)"""
    out = np.empty(N.shape, np.int64)
    Nf, Ef, sf = N.ravel(), np.broadcast_to(E, N.shape).ravel(), np.broadcast_to(s, N.shape).ravel(); bf = np.broadcast_to(b, N.shape).ravel()
    for i in range(Nf.size):
        x = int(Nf[i]) * int(Ef[i]); s_ = int(sf[i])
        h = ((x + (1 << (s_ - 1))) >> s_ if s_ > 0 else x << -s_) + int(bf[i])
        out.flat[i] = min(max(((h + 8) >> 4) + 32768, -1), 65536)
    return out

def entry_vec(N, E, s, b):
    """the same entry, vectorized and still exact: N = Nh 2^20 + Nl, every product held below 2^63"""
    s = np.broadcast_to(s, N.shape)
    if s.min() < 21 or s.max() > 60 or int(np.abs(N).max()).bit_length() > 61: return entry(N, E, s, b)
    Nh, Nl = N >> 20, N & ((1 << 20) - 1); X1 = Nh * E; d = s - 20
    q1 = X1 >> d; r1 = X1 - (q1 << d)
    h = q1 + (((r1 << 20) + Nl * E + (np.int64(1) << (s - 1))) >> s) + b
    return np.clip(((h + 8) >> 4) + 32768, -1, 65536)

def tidx(H): return np.clip(((H + 8) >> 4) + 32768, -1, 65536)

# ======================================================================== exactness on two texts, every depth
if MODE == "exact":
    TX = [int(x) for x in ARG.split(",")]; ok = dict(stream=0, H=0, pile_int64=0, pile_python=0, half=0, half_n=0, side_follow=0,
                                                     n=0, thr_vs_frac=0, thr_n=0, entry_vs_frac=0, order_src=0, order_lines=0, cnt=0)
    rng = np.random.default_rng(4242)
    for t in TX:
        for k in range(12):
            names, S, sb = sources(t, k); c = COL[t]; R = c["R"][k]; P = R.shape[0]; w, we, b = FC[k]
            ok["stream"] += int(np.array_equal(S.sum(0), R))
            x16, sh, F, I, E, s = scale(R, k)
            pile = product(x16, w.astype(np.float64)); H = power(pile, F) + b
            ok["H"] += int(np.array_equal(H, c["H"][k]))
            Sr = (x16 << sh[:, None]) - R
            N = meet(np.concatenate([S, Sr[None]]).reshape(-1, 768), w).reshape(len(S) + 1, P, 3072)
            ok["pile_int64"] += int(np.array_equal(N.sum(0), pile << sh[:, None]))
            No = N.astype(object).sum(0); po = pile.astype(object) * np.array([1 << int(x) for x in sh], dtype=object)[:, None]
            ok["pile_python"] += int((No == po).all())
            # the machine's last rounding at most half a unit: |(H - b) 2^s - sumN E| <= 2^(s-1)
            for i in range(P):
                for j in range(3072):
                    s_ = int(s[i, j]); lhs = (int(H[i, j]) - int(b[j])) * (1 << max(s_, 0)); rhs = int(No[i, j]) * int(E[i, j]) * (1 << max(-s_, 0))
                    ok["half"] += int(2 * abs(lhs - rhs) <= (1 << max(s_, 0))); ok["half_n"] += 1
            # exact value rebuilt from the pushes -> rounded as the machine rounds -> same side as the machine's
            Hb = rshift(np.array([[int(No[i, j]) >> int(sh[i]) for j in range(3072)] for i in range(P)], dtype=np.int64) * E, 16 - I) + b
            ok["side_follow"] += int(((Hb > 0) == (H > 0)).sum()); ok["n"] += H.size
            # the threshold sides against Fractions on random edges (fixed part and taken-away)
            L = WPE[:P] + sb[None, :]
            if k >= 3: L = L.copy(); L[:, STONE] = R[:, STONE]
            Nfix = meet(L, w); Tg = thr_gt(b[None, :], E, s); An = meet(c["A"][k] - BP[k], w); Tl = thr_lt(H, E, s)
            ce = entry(Nfix[:, :64], E[:, :64], s[:, :64], b[None, :64])
            for _ in range(400):
                i, j = int(rng.integers(P)), int(rng.integers(3072)); g = Fraction(int(E[i, j]), 1 << int(s[i, j])) if s[i, j] >= 0 else Fraction(int(E[i, j]) << int(-s[i, j]))
                vf = Fraction(int(Nfix[i, j])) * g + int(b[j]); vg = int(H[i, j]) - Fraction(int(An[i, j])) * g
                ok["thr_vs_frac"] += int((vf > 0) == (Nfix[i, j] > Tg[i, j])) + int((vg > 0) == (An[i, j] < Tl[i, j])); ok["thr_n"] += 2
                if j < 64:
                    hr = math.floor(vf - int(b[j]) + Fraction(1, 2)) + int(b[j])
                    ok["entry_vs_frac"] += int(min(max(((hr + 8) >> 4) + 32768, -1), 65536) == ce[i, j])
            # order-free: the parts added in a random order; the lines met in a random order
            acc = np.zeros_like(N[0])
            for q in rng.permutation(len(N)): acc = acc + N[q]
            ok["order_src"] += int(np.array_equal(acc, pile << sh[:, None]))
            pm = rng.permutation(768); ok["order_lines"] += int(np.array_equal(np.concatenate([S, Sr[None]])[:, :, pm].reshape(-1, 768) @ w[pm], N.reshape(-1, 3072)))
            ok["cnt"] += 1
        say(f"text {t}: {ok}")
    print(f"EXACTNESS on texts {TX}, all 12 depths: {ok}")
    sys.exit()

# ======================================================================== all 30 texts at the chosen depths
DEPTHS = [int(x) for x in ARG.split(",")]
ROWS = [(t, p) for t in range(NT) for p in range(1, len(TEXTS[t]))]; NR = len(ROWS); ROW = {tp: i for i, tp in enumerate(ROWS)}
rng = np.random.default_rng(90210)                               # fresh draws, not the report's
UNREL = np.full(NR, -1)
for i, (t, p) in enumerate(ROWS):
    cand = [u for u in range(NT) if u != t and len(TEXTS[u]) > p and TEXTS[u][p] != TEXTS[t][p]]
    if cand: UNREL[i] = ROW[(int(rng.choice(cand)), p)]
OTHER = [np.array([0] + [int(rng.choice([q for q in range(1, 24) if q != p])) for p in range(1, len(s))]) for s in TEXTS]
PERM_THEIRS = [np.array([np.random.default_rng(1000 * t + p + 11).permutation(768) for p in range(len(s))]) for t, s in enumerate(TEXTS)]
PERM_MINE = [np.array([rng.permutation(768) for p in range(len(s))]) for t, s in enumerate(TEXTS)]
J = json.load(open("edges/edges.json"))["report"]
OUT = {}

def run(k, perms):
    """one pass over the 30 texts at depth k; perms None = the real light, else one line order per position"""
    w, we, b = FC[k]; acc = dict(n=0, bright=0, fixed=0, fb=0, nb=0, fd=0, within=0, fother=0, flip_newest=0, flip_pair=0,
                                 flip_since=0, flip_word=0, alone={}, needed={}, mis=0, order_src=0, stream=0, Hsame=0)
    indiv_all, groups_all, split_all, absH_all, side_all, fix_all, half_all, cancel_all = [], [], [], [], [], [], [], []
    wave_arr, wave_rest = {"newest": [], "since": []}, {"newest": [], "since": []}
    for t in range(NT):
        names, S, sb = sources(t, k); c = COL[t]; R = c["R"][k]; P = R.shape[0]; ar = np.arange(P)[:, None]
        acc["stream"] += int(np.array_equal(S.sum(0), R))
        x16, sh, F, I, E, s = scale(R, k)
        Sr = (x16 << sh[:, None]) - R
        L = WPE[:P] + sb[None, :]
        Lo = WPE[OTHER[t]] + sb[None, :]
        if k >= 3: L = L.copy(); L[:, STONE] = R[:, STONE]; Lo = Lo.copy(); Lo[:, STONE] = R[:, STONE]
        if perms is not None:
            pm = perms[t]; S = S[:, ar, pm]; Sr = Sr[ar, pm]; x16 = x16[ar, pm]; L = L[ar, pm]; Lo = Lo[ar, pm]; R = R[ar, pm]
            stl = np.array([int(np.flatnonzero(pm[p] == STONE)[0]) for p in range(P)])
            x2, ex2, LS2 = gauge_log(R); assert np.array_equal(x2, x16)     # the water level is unchanged by a shuffle
        else: stl = np.full(P, STONE)
        pile = product(x16, w.astype(np.float64)); H = power(pile, F) + b
        if perms is None: acc["Hsame"] += int(np.array_equal(H, c["H"][k]))
        Sall = np.concatenate([S, Sr[None]])
        N = meet(Sall.reshape(-1, 768), w).reshape(len(Sall), P, 3072)
        tgt = pile << sh[:, None]
        sm = np.zeros_like(tgt)
        for q in np.random.default_rng(t + 100 * k).permutation(len(N)): sm = sm + N[q]
        acc["order_src"] += int(np.array_equal(sm, tgt))
        sl = slice(1, P); g = E.astype(np.float64) / np.ldexp(1.0, s)          # sizes only (float, not used for any side)
        rho = H - (pile.astype(np.float64) * E * np.ldexp(1.0, I - 16) + b)
        ia = lambda l: 3 + l; im = lambda l: 3 + (k + 1) + l; nS = len(S)
        V = N * g[None]                                                        # every push at the edge
        bb = np.broadcast_to(b.astype(np.float64), (P, 3072))
        indiv = [V[0], V[1], V[2], bb] + [V[ia(l)] for l in range(k + 1)] + [V[im(l)] for l in range(k)] + [V[nS], rho]
        z = np.zeros((P, 3072))
        groups = [V[0], V[1], V[2] + bb, V[ia(k)], (V[ia(k - 1)] + V[im(k - 1)]) if k >= 1 else z,
                  sum([V[ia(l)] + V[im(l)] for l in range(k - 1)], z), V[nS] + rho]
        v447 = R[ar[:, 0], stl].astype(np.float64)[:, None] * w[stl].astype(np.float64) * g     # the whole light on the stone line
        vall = (N[:nS].sum(0)) * g
        split = [v447, vall - v447, bb, V[nS] + rho]
        ia_ = np.abs(np.array(indiv))[:, sl]; srt = -np.sort(-ia_, axis=0); cs = np.cumsum(srt, axis=0); tt = cs[-1]
        half_all.append((cs < 0.5 * tt[None]).sum(0) + 1); cancel_all.append(tt / np.maximum(np.abs(H[sl]), 1)); del srt, cs
        indiv_all.append(ia_.astype(np.float32)); del ia_; groups_all.append(np.abs(np.array(groups))[:, sl].astype(np.float32))
        split_all.append(np.abs(np.array(split))[:, sl].astype(np.float32)); absH_all.append(np.abs(H[sl]).astype(np.float32))
        # sides, all by exact thresholds
        side = H[sl] > 0
        Nfix = meet(L, w); Tg = thr_gt(b[None, :], E, s)
        fix = (Nfix > Tg)[sl]; fo = (meet(Lo, w) > Tg)[sl]
        side_all.append(side); fix_all.append(fix)
        acc["n"] += side.size; acc["bright"] += int(side.sum()); acc["fixed"] += int((fix == side).sum())
        acc["fb"] += int((fix & side).sum()); acc["nb"] += int(side.sum()); acc["fd"] += int((~fix & ~side).sum()); acc["fother"] += int((fo == side).sum())
        ce = entry_vec(Nfix[sl], E[sl], s[sl], b[None, :])
        if t < 2: assert np.array_equal(ce, entry(Nfix[sl], E[sl], s[sl], b[None, :]))             # the loop agrees
        if t < 2: assert np.array_equal(N, (Sall.reshape(-1, 768) @ w).reshape(N.shape))           # int64 products agree
        acc["within"] += int((np.abs(ce - tidx(H[sl])) <= 1).sum())
        Tl = thr_lt(H, E, s)
        def gone(Nw): return (Nw < Tl)[sl]
        Nnew = N[ia(k)]; acc["flip_newest"] += int((gone(Nnew) != side).sum()); acc["flip_word"] += int((gone(N[0]) != side).sum())
        if k >= 1:
            acc["flip_pair"] += int((gone(N[ia(k - 1)] + N[im(k - 1)]) != side).sum())
            acc["flip_since"] += int((gone(N[im(k - 1)] + Nnew) != side).sum())
            for nm, Nw in (("newest", Nnew), ("since", N[im(k - 1)] + Nnew)):
                wave_arr[nm].append(np.abs(Nw * g)[sl].astype(np.float32)); wave_rest[nm].append(np.abs(H - Nw * g)[sl].astype(np.float32))
        # the pieces of the token's own light, the stone line's share left out from depth 3 on
        if k >= 3:
            def rest(q): return N[q] - Sall[q][ar[:, 0], stl][:, None] * w[stl]
        else:
            def rest(q): return N[q]
        comps = {"word": rest(0), "newest": rest(ia(k))}
        if k >= 1: comps["M(k-1)"] = rest(im(k - 1)); comps["A(k-1)"] = rest(ia(k - 1))
        if k >= 2: comps["older"] = sum(rest(ia(l)) + rest(im(l)) for l in range(k - 1))
        comps["all writes"] = sum(rest(ia(l)) for l in range(k + 1)) + sum((rest(im(l)) for l in range(k)), np.zeros_like(Nfix))
        mis = fix != side; acc["mis"] += int(mis.sum())
        for nm, v in comps.items():
            al = (Nfix + v > Tg)[sl]; nd = gone(v)
            acc["alone"][nm] = acc["alone"].get(nm, 0) + int((al[mis] == side[mis]).sum())
            acc["needed"][nm] = acc["needed"].get(nm, 0) + int((nd[mis] != side[mis]).sum())
    n = acc["n"]; P_ = lambda x: 100.0 * x / n
    ind = np.concatenate(indiv_all, axis=1); grp = np.concatenate(groups_all, axis=1); spl = np.concatenate(split_all, axis=1); aH = np.concatenate(absH_all)
    mH = float(np.median(aH))
    names_ind = ["word", "position", "stream biases", "fc bias"] + [f"A{l}" for l in range(k + 1)] + [f"M{l}" for l in range(k)] + ["level rounding", "last rounding"]
    names_grp = ["word", "position", "biases", "newest", "pair k-1", "older", "roundings"]
    sh_ind = {nm: 100 * ind[i].sum(dtype=np.float64) / ind.sum(dtype=np.float64) for i, nm in enumerate(names_ind)}
    sh_grp = {nm: 100 * grp[i].sum(dtype=np.float64) / grp.sum(dtype=np.float64) for i, nm in enumerate(names_grp)}
    sh_spl = {nm: 100 * spl[i].sum(dtype=np.float64) / spl.sum(dtype=np.float64) for i, nm in enumerate(["line 447", "other lines", "fc bias", "roundings"])}
    med_ind = {nm: float(np.median(ind[i])) / mH for i, nm in enumerate(names_ind)}
    half = np.concatenate(half_all); cancel = np.concatenate(cancel_all)
    side = np.concatenate(side_all); fix = np.concatenate(fix_all)
    res = dict(bright=P_(acc["bright"]), fixed=P_(acc["fixed"]), fix_bright=100 * acc["fb"] / acc["nb"], fix_dark=100 * acc["fd"] / (n - acc["nb"]),
               within=P_(acc["within"]), fix_other=P_(acc["fother"]), misplaced=P_(acc["mis"]),
               flips=dict(newest=P_(acc["flip_newest"]), word=P_(acc["flip_word"]), pair=P_(acc["flip_pair"]) if k else None, since=P_(acc["flip_since"]) if k else None),
               alone={c_: 100 * v / acc["mis"] for c_, v in acc["alone"].items()}, needed={c_: 100 * v / acc["mis"] for c_, v in acc["needed"].items()},
               share_each=sh_ind, share_kind=sh_grp, share_line=sh_spl, med_each=med_ind, half_median=float(np.median(half)), cancel_median=float(np.median(cancel)),
               checks=dict(stream=acc["stream"], Hsame=acc["Hsame"], order_src=acc["order_src"]))
    if k >= 1: res["wave_ratio"] = {nm: float(np.median(np.concatenate(wave_arr[nm]))) / float(np.median(np.concatenate(wave_rest[nm]))) for nm in wave_arr}
    if perms is None:
        ok = UNREL >= 0
        res["fix_unrel"] = 100 * (fix[ok] == side[UNREL[ok]]).mean(); res["unrel_agree"] = 100 * (side[ok] == side[UNREL[ok]]).mean()
        res["chance"] = 100 * (side.mean() * fix.mean() + (1 - side.mean()) * (1 - fix.mean()))
    return res

for k in DEPTHS:
    out = {"real": run(k, None)}; say(f"depth {k} real done")
    out["shuf_theirs"] = run(k, PERM_THEIRS); say(f"depth {k} shuffle (their orders) done")
    out["shuf_mine"] = run(k, PERM_MINE); say(f"depth {k} shuffle (fresh orders) done")
    OUT[k] = out
    r, jr, js = out["real"], J[str(k)]["real"], J[str(k)]["shuf"]
    def cmp(lbl, a, b_):
        d = None if (a is None or b_ is None) else a - b_
        print(f"    {lbl:34s} mine {a if a is None else round(a, 4)!s:>10}   report {b_ if b_ is None else round(b_, 4)!s:>10}   diff {d if d is None else round(d, 6)}")
    print(f"=== DEPTH {k}  checks {r['checks']}  shuffle checks {out['shuf_theirs']['checks']}")
    for q in ("bright", "fixed", "fix_bright", "fix_dark", "within", "misplaced", "half_median", "cancel_median"): cmp(q, r[q], jr[q])
    cmp("fix vs unrelated (fresh draw)", r["fix_unrel"], jr["fix_unrel"]); cmp("unrelated agree (fresh draw)", r["unrel_agree"], jr["unrel_agree"])
    cmp("chance", r["chance"], jr["chance"]); cmp("other position row (fresh draw)", r["fix_other"], jr["fix_other"])
    jmap = dict(newest="newest", word="word", pair="pair k-1", since="since last row")
    for q, jq in jmap.items():
        cmp(f"flip {q}", r["flips"][q], jr["flips"][jq][0] if jq in jr["flips"] else None)
    for q in ("newest", "since"):
        if "wave_ratio" in r: cmp(f"wave ratio {q}", r["wave_ratio"][q], jr["flips"]["newest" if q == "newest" else "since last row"][1])
    for c_ in r["alone"]: cmp(f"alone {c_}", r["alone"][c_], jr["alone"][c_]); cmp(f"needed {c_}", r["needed"][c_], jr["needed"][c_])
    je = {a[1]: (a[2], a[3]) for a in jr["sizes"] if a[0] == "each source"}
    jk = [a for a in jr["sizes"] if a[0] == "by kind"]; jl = [a for a in jr["sizes"] if a[0] == "two ways by line"]
    for nm in r["share_each"]: cmp(f"share each: {nm}", r["share_each"][nm], je[nm][1]);
    for nm in ("word", "A0"): cmp(f"median push/pile: {nm}", r["med_each"][nm], je[nm][0])
    for i, nm in enumerate(r["share_kind"]): cmp(f"share kind: {nm}", r["share_kind"][nm], jk[i][3])
    for i, nm in enumerate(r["share_line"]): cmp(f"share line: {nm}", r["share_line"][nm], jl[i][3])
    for vn in ("shuf_theirs", "shuf_mine"):
        s_ = out[vn]; jsk = [a for a in js["sizes"] if a[0] == "by kind"]; jsl = [a for a in js["sizes"] if a[0] == "two ways by line"]
        print(f"  -- {vn}")
        for q in ("bright", "fixed", "within"): cmp(q, s_[q], js[q])
        for q, jq in jmap.items(): cmp(f"flip {q}", s_["flips"][q], js["flips"][jq][0] if jq in js["flips"] else None)
        for i, nm in enumerate(s_["share_kind"]): cmp(f"share kind: {nm}", s_["share_kind"][nm], jsk[i][3])
        cmp("share line 447", s_["share_line"]["line 447"], jsl[0][3])
    sys.stdout.flush()
json.dump({str(k): v for k, v in OUT.items()}, open(CACHE.replace("recheck_column.npz", f"recheck_{ARG.replace(',', '_')}.json"), "w"), indent=1, default=float)
import resource; say(f"largest memory held: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
