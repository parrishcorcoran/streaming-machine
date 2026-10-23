"""THE COLLAPSING TOP READ ON FINE DIAGONALS WITH THE PILE-SCALE BOUND, ON QWEN3-0.6B'S 151,936-WORD TABLE.

base2_fringes2.py's top read (its level 4: the table and the light in signed digits, places 0..15; the stage list by place
value, the pairs with a + b >= 27, 25, ..., 1, then all; the bound started where the light says it can set a word aside;
each word's bound carried down; the set-aside test at the pile's own scale with the margin mu; the 5th-best low read
from the words in order of their highs only as far as needed; the words still in read to the end exactly), carried to
the all-base-2 Qwen3-0.6B's read (qwen_base2.py's gauged_slab(R, "head")): score_j = power(pile_j, (e_j + ex + 16) 65536 - LS).

WHAT IS DIFFERENT FROM GPT-2'S READ, AND WHY NOTHING ELSE IS
  - No standing amounts: Qwen's read has no offset (its water level takes nothing out and adds nothing), so beta = 0
    and the margin's gamma is the rounding half only (gamma = 1, as base2_fringes2's own formula gives with no offset).
  - Four powers of two among the words (e_j = -16..-13; GPT-2's table had one). Every word is carried at ONE common
    scale, A_j = pile_j x 2^(e_j - e_min), a shift (no photons). Then score_j = power(A_j, F0) with F0 the read's gain
    at e_min, exactly (rshift(Y, s - k) = rshift(Y 2^k, s)); checked here: every score row's sha256 equals the machine's
    own (capture.py). Each word's bound is carried at that scale too (its table values shifted by its k). The start
    rule's largest unread part is taken over the four shift classes: Rbar = max over k of sum over unread (a, b) of
    2^(a+b+k) min(n_a, M_bk), M_bk the largest lit count at place b among the words of class k (64 stored numbers).
  - The work is formed word chunk by word chunk (the table is three times GPT-2's), every stage's counted piece of every
    word written exactly (in float32, each piece divided by its stage's lowest diagonal 2^t is a whole number below
    2^16), then the collapse runs position by position on those pieces, as base2_fringes2 does.
  Top 5: by score, ties by word number (one fixed rule on both sides; capture.py's).

WORK, in the unit of all earlier results (base2_fringes2's rules): a meeting is a lit digit of the light x a lit digit of
the word on each line, for the pairs met by the words still in; an add counts the photons it adds; a divide (a product
by the gain) counts the number's photons x the gain's. The plain read: plain meetings (1s x 1s) of every word plus the
divide of every word. The cut: the staged meetings, the light written in signed digits, the bound (tables built from the
light, table values, carried, poured, the lows read), the start rule, and the words read to the end (bound taken off,
divide). Compares and table reads are counts, kept apart.

RUNS:  python3 qwen06b/top_read.py main          every position (613): exact checks, work
       python3 qwen06b/top_read.py shuffle       CONTROL: each position's light lines in a random order (every 4th position)
       python3 qwen06b/top_read.py wordperm      ORDER-FREE: the words in a random order (every 4th position)

PREDICTIONS (written before any run of this file)
  The task's: the top read's share is lower than GPT-2's (more words, same top 5).
  The session's:
  - Exact: the plain read's score rows equal the machine's own 613/613 (sha256); the cut's top 5 (set and order) and race
    winner equal the plain read's 613/613; bound broken 0; no word set aside that is not below the 5th best; pieces equal
    the plain read's piles at every word.
  - Share: 8-15% of the plain read with every add counted (GPT-2's 16.05%), meetings 7-13% (GPT-2's 14.47%); most likely
    about 11%.
  - The start at the top 14 at most positions (GPT-2: 595 of 648), the top 16 at most of the rest.
  - Survivors: after the top 14 a smaller fraction of the words than GPT-2's 62% (median 31,291 of 50,257); after the
    top 16 a few hundred; about 5-10 by the top 20.
  - Control (light shuffled across its lines): exact too; fewer survivors at the deciding stages and less work than the
    real light, as GPT-2's section 29 found (15.7% against 19.3%).
  - Order-free: top 5 identical at every position read (an exact tie at 5th place would be ordered by word number on
    both sides, so it cannot differ).
"""
import os, sys, json, time, resource, hashlib, warnings, numpy as np, torch
from fractions import Fraction
from types import SimpleNamespace
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen06b"
sys.path.insert(0, OUT)
import capture as CP
power, race, EXP2 = CP.B["power"], CP.B["race"], CP.B["EXP2"]; ES = CP.B["ES"]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def signed_digits(v):
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def top5(S): return np.array([np.lexsort((np.arange(len(s)), -s))[:5] for s in S])
torch.set_num_threads(10)
L_ = 1024; NNL = np.arange(L_ + 1); ONES_NN = ones(NNL); TPL = 16

def top_grid(tpl):                                            # base2_fringes2.top_grid, unchanged
    a_, b_ = np.arange(tpl)[:, None], np.arange(tpl)[None, :]
    THR = [27 - 2 * s for s in range(14)]
    TCS = [np.broadcast_to(a_ + b_ >= t, (tpl, tpl)).copy() for t in THR] + [np.ones((tpl, tpl), bool)]
    TNEW = [TCS[0]] + [TCS[s] & ~TCS[s - 1] for s in range(1, len(TCS))]
    assert (sum(n.astype(int) for n in TNEW) == 1).all()
    STAGE_OF = np.zeros((tpl, tpl), np.int64)
    for s, n in enumerate(TNEW): STAGE_OF[n] = s
    NMASK = np.stack([np.array([sum(1 << a for a in range(tpl) if n[a, b]) for b in range(tpl)], np.int64) for n in TNEW])
    return SimpleNamespace(TPL=tpl, TCS=TCS, TNEW=TNEW, STAGE_OF=STAGE_OF, POWA=(1 << np.arange(tpl)).astype(np.int64),
                           NMASK=NMASK, THR=THR, NS=len(THR))
G = top_grid(TPL); NSg = G.NS
TLOW = np.array([int(min(a + b for a in range(16) for b in range(16) if G.TNEW[s][a, b])) for s in range(NSg + 1)])   # each stage's lowest diagonal
WT = np.zeros((16, NSg + 1, 16), np.float32)                  # WT[b, st, a] = 2^(a + b - TLOW[st]) where (a, b) is in stage st
for a in range(16):
    for b in range(16):
        st = G.STAGE_OF[a, b]; WT[b, st, a] = 2.0 ** (a + b - TLOW[st])
assert WT.max() <= 8

class Table:
    def __init__(self, word_perm=None):
        t0 = time.time()
        T = np.ascontiguousarray(CP.member("head.w")); HE = np.asarray(CP.Mz["head.e"]).astype(np.int64)
        if word_perm is not None: T = np.ascontiguousarray(T[:, word_perm]); HE = HE[word_perm]
        self.wperm = np.arange(T.shape[1]) if word_perm is None else word_perm
        self.NW = T.shape[1]; self.EMIN = int(HE.min()); self.KW = HE - self.EMIN; assert self.KW.max() <= 3
        self.U = np.empty(T.shape, np.uint16); self.D = np.empty(T.shape, np.uint16)
        self.PPH = np.zeros(L_, np.int64); self.LPH = np.zeros(L_, np.int64)
        self.NT = np.zeros((self.NW, TPL), np.int64); self.CNT_ALL = np.zeros((TPL, L_), np.int64)
        for c in range(0, self.NW, 8192):
            u, d = signed_digits(T[:, c:c + 8192]); assert not ((u | d) >> TPL).any()
            self.U[:, c:c + 8192] = u; self.D[:, c:c + 8192] = d; lit = u | d
            self.PPH += ones(T[:, c:c + 8192]).sum(1); self.LPH += np.bitwise_count(lit).sum(1, dtype=np.int64)
            for b in range(TPL):
                lb = (lit >> b) & 1; self.NT[c:c + 8192, b] = lb.sum(0); self.CNT_ALL[b] += lb.sum(1)
        del T
        self.MBk = np.stack([self.NT[self.KW == k].max(0) for k in range(4)])          # (4, 16) stored numbers
        OFF = (L_ + 1) * np.arange(TPL)[None, :]; self.IX = self.NT + OFF
        self.HN = np.stack([np.bincount(self.NT[:, b], minlength=L_ + 1) for b in range(TPL)]).ravel()
        print(f"  table: {self.NW} words, shifts {np.bincount(self.KW).tolist()}, signed-digit sheets made ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    def T_chunk(self, c0, c1): return self.U[:, c0:c1].astype(np.int64) - self.D[:, c0:c1].astype(np.int64)
    def table(self, nvec, mask): return ((mask.T * G.POWA[None, :]) @ np.minimum(nvec[:, None], NNL[None, :]))
    def table_photons(self, nvec, mask):
        Q = np.array([ONES_NN[:n].sum() + (L_ + 1 - n) * ones(n) for n in nvec]); return int((mask * Q[:, None]).sum())
    def bound(self, nx, mask, idx=None):                      # each word's bound at the common scale, and the table photons
        Gm = self.table(nx, mask); OG = ones(Gm).ravel(); G2 = (Gm << np.arange(TPL)[:, None]).ravel()
        ix = self.IX if idx is None else self.IX[idx]; K = self.KW if idx is None else self.KW[idx]
        return G2.take(ix).sum(1) << K, OG, ix

def pieces(tb, X, cw=4096):
    """every stage's counted piece of every word (float32, divided by 2^TLOW[stage], at the word's own scale), the
    meetings each word would take at each stage (float32), and the direct piles (int64). X: (P, 1024) light."""
    P = len(X); NW = tb.NW
    ux, dx = signed_digits(X); litx = ux | dx; assert not (litx >> TPL).any()
    LA = torch.from_numpy(np.stack([((ux >> a) & 1) - ((dx >> a) & 1) for a in range(TPL)]).astype(np.float32).reshape(TPL * P, L_))
    LC = [torch.from_numpy(np.stack([np.bitwise_count(litx & G.NMASK[st, b]) for st in range(NSg + 1)]).astype(np.float32).reshape((NSg + 1) * P, L_))
          for b in range(TPL)]
    WTt = [torch.from_numpy(WT[b]) for b in range(TPL)]
    DCs = torch.zeros((NSg + 1, P, NW), dtype=torch.float32); MW = torch.zeros((NSg + 1, P, NW), dtype=torch.float32)
    PILE = np.empty((P, NW), np.int64); xf = X.astype(np.float64)
    for c0 in range(0, NW, cw):
        c1 = min(NW, c0 + cw); n_c = c1 - c0
        Uc = torch.from_numpy(tb.U[:, c0:c1].astype(np.int32)); Dc = torch.from_numpy(tb.D[:, c0:c1].astype(np.int32))
        for b in range(TPL):
            ub, db = (Uc >> b) & 1, (Dc >> b) & 1
            Pb = (ub - db).to(torch.float32); Lb = (ub | db).to(torch.float32)
            Cab = LA @ Pb                                                            # (16 P, n_c): counts, |.| <= 1,024
            DCs[:, :, c0:c1] += (WTt[b] @ Cab.reshape(TPL, P * n_c)).reshape(NSg + 1, P, n_c)
            MW[:, :, c0:c1] += (LC[b] @ Lb).reshape(NSg + 1, P, n_c)
        p = xf @ tb.T_chunk(c0, c1).astype(np.float64); assert np.abs(p).max(initial=0) < 2 ** 52; PILE[:, c0:c1] = p.astype(np.int64)
    assert DCs.abs().max().item() < 2 ** 24 and MW.max().item() < 2 ** 24
    return DCs.numpy(), MW.numpy(), PILE, litx

def read(tb, X, EX, LS, idx_pos, ref=None, B=32, log=print):
    """the collapsing read at every position given; checks against the plain read (and the machine's own rows if ref)"""
    N = len(X); C = {}; bad = {}
    def cadd(k, v): C[k] = C.get(k, 0) + int(np.sum(v))
    def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
    out = dict(top5=np.zeros((N, 5), np.int64), win=np.zeros(N, np.int64), ptop5=np.zeros((N, 5), np.int64), pwin=np.zeros(N, np.int64),
               surv=np.zeros((N, NSg), np.int64), visited=np.zeros((N, NSg), np.int64), start=np.full(N, -1, np.int64),
               work=np.zeros((N, NSg + 1), np.int64), plain=np.zeros(N, np.int64))
    KWv = tb.KW; t0 = time.time()
    for c0 in range(0, N, B):
        Xb = np.asarray(X[c0:c0 + B], np.int64); P = len(Xb); assert np.abs(Xb).max() < 1 << 15
        DCs, MW, PILE, litx = pieces(tb, Xb)
        F0 = (tb.EMIN + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
        A = PILE << KWv[None, :]; S = power(A, F0[:, None])                         # the plain read, every word
        cadd("cut|light written in signed digits", ones(Xb).sum())
        for p in range(P):
            q = c0 + p
            if ref is not None:
                nb("plain read's score row vs the machine's own (sha256)", hashlib.sha256(np.ascontiguousarray(S[p][np.argsort(tb.wperm)], np.int64).tobytes()).hexdigest() != ref["sha"][idx_pos[q]])
            DCi = [(DCs[st, p].astype(np.int64) << int(TLOW[st])) << KWv for st in range(NSg + 1)]
            nb("pieces vs the plain read's piles", (sum(DCi) != A[p]).sum())
            nx = ((litx[p][None, :] >> np.arange(TPL)[:, None]) & 1).sum(1)
            Fp = int(F0[p]); I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
            twos = Fraction(2) ** s; gamma = 1; delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator)))); mu = 2 * gamma + delta
            ge = int(ones(e))
            cadd("plain|meetings, plain", int(ones(Xb[p]) @ tb.PPH)); cadd("plain|divide (every word)", ones(PILE[p]).sum() * ge)
            out["plain"][q] = int(ones(Xb[p]) @ tb.PPH) + int(ones(PILE[p]).sum()) * ge
            alive = np.arange(tb.NW); started = False; Rw = np.zeros(tb.NW, np.int64); U = DCi[0].copy(); CUM = DCi[0].copy()
            work = np.zeros(NSg + 1, np.int64); work[0] = int(MW[0, p].sum(dtype=np.float64))
            MINT = [np.minimum(nx[:, None], tb.MBk[k][None, :]) for k in range(4)]
            VAL = [MINT[k] << (np.arange(TPL)[:, None] + np.arange(TPL)[None, :] + k) for k in range(4)]
            for st in range(NSg):
                if st > 0:
                    piece = DCi[st][alive]; CUM = CUM + piece
                    if started:
                        dR, OGn, ix = tb.bound(nx, G.TNEW[st], alive)
                        U = U + piece - dR; Rw = Rw - dR
                        cadd("cut|bound tables built from the light", tb.table_photons(nx, G.TNEW[st]))
                        cadd("cut|bound adds (table values)", OGn.take(ix).sum()); cadd("cut|bound carried down (high and bound)", 2 * ones(dR).sum())
                    else:
                        U = U + piece
                    work[st] = int(MW[st, p, alive].sum(dtype=np.float64))
                if not started:
                    unread = ~G.TCS[st]; Rbar = max(int(VAL[k][unread].sum()) for k in range(4))
                    na = len(alive); assert na == tb.NW
                    K5 = int(np.partition(U, na - 5)[na - 5]); Kmin = int(U.min())
                    cadd("cut|start rule: largest unread part from the light (adds)", sum(int(ones(MINT[k][unread]).sum()) for k in range(4)) + int(ones(2 * Rbar + mu)))
                    cadd("cut|start rule: the spread (5th-best less the lowest)", int(ones(K5 - Kmin))); cadd("cmp|compares", 2 * na + 1 + 3)
                    Rc, _, _ = tb.bound(nx, unread)
                    nb("start rule: a word's bound above the light's largest unread part", (Rc > Rbar).sum())
                    nb("bound broken", ((A[p] < CUM - Rc) | (A[p] > CUM + Rc)).sum())
                    if K5 - Kmin < 2 * Rbar + mu:
                        out["surv"][q, st] = na; continue
                    started = True; out["start"][q] = st
                    Rw, OGs, _ = tb.bound(nx, unread); U = U + Rw
                    cadd("cut|bound tables built from the light", tb.table_photons(nx, unread))
                    cadd("cut|bound adds (table values)", int((tb.HN * OGs).sum())); cadd("cut|bound poured into the highs", ones(Rw).sum())
                cnt = U - Rw
                nb("counted pile carried = the counted pile", (cnt != CUM).sum())
                nb("bound broken", ((A[p, alive] < cnt - Rw) | (A[p, alive] > cnt + Rw)).sum())
                na = len(alive); U5 = np.partition(U, na - 5)[na - 5]
                if U.min() > U5 - 2 * Rw.min() - mu:
                    cadd("cmp|compares", 3 * na); out["surv"][q, st] = na; continue
                K = min(64, na)
                while True:
                    top = np.argpartition(-U, K - 1)[:K] if K < na else np.arange(na)
                    ell = U[top] - 2 * Rw[top]; l5 = np.partition(ell, len(ell) - 5)[len(ell) - 5]
                    if K == na: break
                    rest_ = np.ones(na, bool); rest_[top] = False
                    if l5 >= U[rest_].max(): break
                    K = min(na, 2 * K)
                keepw = U > l5 - mu
                cadd("cut|lows of the words read for the 5th-best low", ones(Rw[top]).sum()); cadd("cmp|compares", na + K)
                out["visited"][q, st] = K
                s5 = np.partition(S[p], tb.NW - 5)[tb.NW - 5]
                nb("a word set aside that is not below the 5th-best", (S[p, alive[~keepw]] >= s5).sum())
                alive = alive[keepw]; U = U[keepw]; Rw = Rw[keepw]; CUM = CUM[keepw]
                out["surv"][q, st] = len(alive)
            work[NSg] = int(MW[NSg, p, alive].sum(dtype=np.float64))
            pile = U - Rw + DCi[NSg][alive]
            sc = power(pile, Fp)
            nb("words read to the end vs the plain read", (sc != S[p, alive]).sum())
            cadd("cut|words read to the end: bound taken off, divide", ones(Rw).sum() + ones(pile).sum() * ge)
            Sp = np.full(tb.NW, -(1 << 62), np.int64); Sp[alive] = sc
            out["top5"][q] = tb.wperm[top5(Sp[None])[0]]; out["win"][q] = tb.wperm[race(Sp)]
            out["ptop5"][q] = tb.wperm[top5(S[p:p + 1])[0]]; out["pwin"][q] = tb.wperm[race(S[p])]
            nb("all pairs' meetings = the whole read's", int(MW[:, p].sum(dtype=np.float64)) != int(np.bitwise_count(litx[p]).astype(np.int64) @ tb.LPH))
            out["work"][q] = work; cadd("cut|meetings, signed digits, staged", work.sum())
        del DCs, MW, PILE, A, S
        log(f"    {c0 + P} of {N} positions read ({time.time() - t0:.0f}s, {rss()} MB)")
    out["count"] = C; out["bad"] = bad
    return out

if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time()
    META = json.load(open(f"{OUT}/light/meta.json"))
    X = np.load(f"{OUT}/light/top_light.npy").astype(np.int64); EX = np.load(f"{OUT}/light/top_ex.npy"); LS = np.load(f"{OUT}/light/top_LS.npy")
    idx = np.arange(len(X)) if mode == "main" else np.arange(0, len(X), 4)
    X, EX, LS = X[idx], EX[idx], LS[idx]
    word_perm = None
    if mode == "shuffle":
        rng = np.random.default_rng([2026, 7]); X = rng.permuted(X, axis=1)
    if mode == "wordperm":
        word_perm = np.random.default_rng([2026, 11]).permutation(151936)
    print(f"TOP READ, mode {mode}: {len(X)} positions", flush=True)
    tb = Table(word_perm)
    res = read(tb, X, EX, LS, idx, ref=META if mode != "shuffle" else None)
    n = len(X); C = res["count"]
    cut = sum(v for k, v in C.items() if k.startswith("cut|")); plain = sum(v for k, v in C.items() if k.startswith("plain|"))
    meet = C["cut|meetings, signed digits, staged"]; plain_m = C["plain|meetings, plain"]
    ok5 = int((res["top5"] == res["ptop5"]).all(1).sum()); okw = int((res["win"] == res["pwin"]).sum())
    print(f"  top 5 (set and order) = the plain read's: {ok5}/{n}; race winner: {okw}/{n}")
    if mode != "shuffle":
        print(f"  against the machine's own read (capture.py): top 5 {int((res['top5'] == np.array(META['top5'])[idx]).all(1).sum())}/{n}, race winner {int((res['win'] == np.array(META['win'])[idx]).sum())}/{n}")
    print("  checks: " + ", ".join(f"{k} {v}" for k, v in res["bad"].items()))
    st_names = [f"a+b>={t:2d} (top {29 - t:2d})" for t in G.THR]           # GPT-2's labels (base2_fringes2: a+b>=15 is "top 14")
    starts = np.bincount(res["start"][res["start"] >= 0], minlength=NSg)
    print("  the bound started at (positions): " + ", ".join(f"{st_names[s]} {c}" for s, c in enumerate(starts) if c))
    for s in range(NSg):
        sv = res["surv"][:, s]
        print(f"    {st_names[s]}: words still in median {np.median(sv):8.0f} mean {sv.mean():9.1f} least {sv.min():6d} most {sv.max():6d}; "
              f"that stage's meetings {res['work'][:, s].mean() / 1e6:6.2f}M; words read for the 5th-best low (median) {np.median(res['visited'][:, s]):.0f}")
    print(f"  per position: plain read (meetings {plain_m / n:,.0f} + divide of every word) {plain / n:,.0f}; the cut {cut / n:,.0f} = {cut / plain * 100:.2f}%; "
          f"meetings alone {meet / n:,.0f} = {meet / plain_m * 100:.2f}% of the plain meetings")
    for k in sorted(C): print(f"      {k:75s} {C[k] / n:16,.0f}")
    json.dump(dict(mode=mode, n=n, count=C, bad=res["bad"], ok5=ok5, okw=okw, starts=starts.tolist(),
                   surv_median=np.median(res["surv"], 0).tolist(), surv_mean=res["surv"].mean(0).tolist(),
                   work_mean=res["work"].mean(0).tolist(), plain_per_position=res["plain"].tolist()),
              open(f"{OUT}/top_read_{mode}.json", "w"))
    print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
