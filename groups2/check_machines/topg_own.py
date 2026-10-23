"""CHECKER'S OWN TOP READ (ii) per place, g = 16, written here (no code from topg.py, topg.c or gtop.py; the set-aside
rules are taken as topg.py's docstring states them: GPT-2 topband's rule carried to the doubled scale, Qwen the per-shift-
class rule of qwen_fringes2.band_read; mu, beta, gamma and delta as those files define them). On the light at the top of
the checker's fresh base2 / qwen_base2 runs, texts 5 and 12. Per position: every word's pile from the checker's group
piles (checked against the direct product), the words read from the top place down, the survivors after each place, and
the work: table + light sums + reads into the words still in + bound photons + the words read to the end + standing
amounts; plus the ready piles written (the 1s of every half-table entry the word table reads) and the light written.
The plate side is laid word chunk by word chunk (3 GB limit).
    python3 topg_own.py model=gpt2|qwen -> topg_own_<model>.json
"""
import os, sys, json, time, zipfile, resource, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"; sys.path.insert(0, OUT)
from mycount import ones, sdc, expand, gray_turns, lit_places, NPL
MODEL = dict(a.split("=") for a in sys.argv[1:]).get("model", "gpt2"); G_ = 16; CH = 8192
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones1(v): return bin(abs(int(v))).count("1")
def sd1(v): return int(sdc(np.array([int(v)]))[0])
if MODEL == "gpt2":
    src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
    for a_, b_ in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"), ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
                   ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
        src = src.replace(a_, b_)
    B = {}; cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = argv; os.chdir(cwd)
    WT = np.asarray(B["M"]["head.w"]); HE = int(np.asarray(B["M"]["head.e"])[0]); BIAS = np.asarray(B["M"]["head.b"]).astype(np.int64)
    BIAS1 = ones(BIAS); BMAX = int(np.abs(BIAS).max()); power, race, EXP2, ES = B["power"], B["race"], B["EXP2"], B["ES"]
    REF = np.load(f"{OUT}/gpt2_ref.npz"); LIGHT = [(REF[f"t{t}_top_x16"], REF[f"t{t}_top_ex"], REF[f"t{t}_top_LS"]) for t in (5, 12)]
    KW = np.zeros(WT.shape[1], np.int64)
else:
    NPZ = f"{HERE}/qwen/qwen_base2.npz"
    def memb(name):
        zi = zipfile.ZipFile(NPZ).getinfo(name + ".npy")
        with open(NPZ, "rb") as f:
            f.seek(zi.header_offset); h = f.read(30); f.seek(zi.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
            v = np.lib.format.read_magic(f); shape, fo, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f); off = f.tell()
        return np.memmap(NPZ, dtype=dt, mode="r", offset=off, shape=shape, order="F" if fo else "C")
    src = open(f"{HERE}/qwen/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
    src = src.replace('M = np.load("qwen_base2.npz")', f'M = np.load("{NPZ}")').replace(
        'W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = {}')
    B = {}; argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = argv
    WT = np.asarray(memb("head.w")); HEw = np.asarray(memb("head.e")).astype(np.int64); EMIN = int(HEw.min()); KW = HEw - EMIN
    power, race, EXP2, ES = B["power"], B["race"], B["EXP2"], B["ES"]
    Z = np.load(f"{OUT}/qwen_ref.npz"); LIGHT = [(Z[f"r{t}__top__x16"], Z[f"r{t}__top__ex"], Z[f"r{t}__top__LS"]) for t in (5, 12)]
N, M = WT.shape
def chunk(j0): return np.asarray(WT[:, j0:j0 + CH]).astype(np.int64)
def top5(S):
    if MODEL == "gpt2":
        idx = np.argpartition(-S, 5)[:5]; return idx[np.argsort(-S[idx])]
    lab = np.arange(len(S)); return lab[np.lexsort((lab, -S))[:5]]

t0 = time.time()
# ---- the plate side, once (word chunks) ----
order = np.random.default_rng([1, 99, N, G_]).permutation(N); GR = [order[s:s + G_] for s in range(0, N, G_)]; NG = len(GR)
assert all(len(L) == G_ for L in GR)
Lb = np.zeros(NPL, np.int64); pph = np.zeros(N, np.int64)
for j0 in range(0, M, CH):
    Wc = chunk(j0); lit = lit_places(Wc); pph += ones(Wc).sum(1)
    for b in range(NPL): Lb[b] += int(((lit >> b) & 1).sum())
U = NG * M; mask = sum(1 << b for b in range(16) if not Lb[b] < U); upb = [b for b in range(16) if (mask >> b) & 1]
PAT = [np.zeros((len(upb), M), np.uint16) for _ in GR]; SDL = {}; kc = None
for j0 in range(0, M, CH):
    Wc = chunk(j0); m_ = Wc.shape[1]; E, Wsd, nlit, kcc = expand(Wc, np.full(m_, mask, np.int64))
    assert (kcc == kcc[0]).all(); kc = int(kcc[0]) if kc is None else kc; assert kc == int(kcc[0])
    for gi, L in enumerate(GR):
        Eg = E[L]
        for bi, b in enumerate(upb): PAT[gi][bi, j0:j0 + m_] = (((Eg >> b) & 1) << np.arange(G_)[:, None]).sum(0)
    rr = Wsd.copy(); nseen = 0                                         # the signed part's digits, place by place (the digit rule)
    for b in range(NPL):
        odd = (rr & 1) == 1; d = np.where(odd, 2 - (rr & 3), 0)
        if b in upb: assert not d.any(), "a signed digit at an up/hole place"
        nz = np.nonzero(d)
        if len(nz[0]): SDL.setdefault(b, []).append((nz[0], nz[1] + j0, d[nz])); nseen += len(nz[0])
        rr = (rr - d) >> 1
    assert not rr.any() and nseen == int(nlit.sum()), "the signed part's digits are not the expansion's"
    del E, Wsd, nlit, Wc
SDL = {b: tuple(np.concatenate(z) for z in zip(*v)) for b, v in SDL.items()}
REPS = []
for gi in range(NG):
    u = np.unique(PAT[gi]).astype(np.int64); top = 1 << (G_ - 1); REPS.append(np.unique(np.where(u & top, u, u ^ ((1 << G_) - 1))))
fmt = [(b in upb) or (b in SDL) for b in range(NPL)]; places = [b for b in range(NPL - 1, -1, -1) if fmt[b]]
def bound(b):                                    # the most the places below b can add per unit of |x| summed (doubled scale)
    v = sum(1 << q for q in upb if q < b); q = 0
    tn = lambda n: ((1 << (n + 1)) - 1) // 3 if n > 0 else 0
    while q < NPL:
        if q in upb: q += 1; continue
        a = q
        while q < NPL and q not in upb: q += 1
        if a < b: v += 2 * (1 << a) * tn(min(q, b) - a)
    return v
BND = {b: bound(b) for b in range(NPL + 1)}; GT = gray_turns(G_ - 1)
SG = np.where(((np.arange(1 << G_)[:, None] >> np.arange(G_)[None, :]) & 1) == 1, 1, -1).astype(np.int64)
print(f"plate side: signed places {[b for b in range(16) if b not in upb]}, kcorr {kc}, places {places}, signed digits {sum(len(v[0]) for v in SDL.values()):,} "
      f"({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
XA = np.concatenate([l[0] for l in LIGHT]).astype(np.int64); EXA = np.concatenate([l[1] for l in LIGHT]); LSA = np.concatenate([l[2] for l in LIGHT])
PILES = np.zeros((len(XA), M), np.int64)
for j0 in range(0, M, CH): PILES[:, j0:j0 + CH] = XA @ chunk(j0)
recs = []
for q in range(len(XA)):
    x = XA[q]; ex = int(EXA[q]); ls = int(LSA[q]); X = int(x.sum()); S1 = int(np.abs(x).sum()); sx = sdc(x); PILE = PILES[q]
    A = np.zeros((M, NPL), np.int64); H = np.zeros((M, NPL), np.int64); pw = 0; tab = 0
    for gi, L in enumerate(GR):
        V = SG @ x[L]; sV = sdc(V)
        for bi, b in enumerate(upb): A[:, b] += V[PAT[gi][bi]]; H[:, b] += sV[PAT[gi][bi]]
        pw += int(ones(V[REPS[gi]]).sum()); tab += int(sx[L[1:]].sum() + sx[L[:-1]] @ GT)
    for b, (ln, wd, sg) in SDL.items():
        A[:, b] += 2 * np.bincount(wd, weights=(sg * x[ln]).astype(np.float64), minlength=M).astype(np.int64)
        H[:, b] += np.bincount(wd, weights=sx[ln].astype(np.float64), minlength=M).astype(np.int64)
    CUM = np.cumsum((A << np.arange(NPL)[None, :])[:, ::-1], axis=1)[:, ::-1]
    Dbl = CUM[:, 0] + kc * X; assert np.array_equal(Dbl, 2 * PILE), "the checker's top piles differ from the direct product"
    pile = Dbl >> 1; light_sum = int(sx[1:].sum()); lsum = light_sum * (2 if kc else 1)
    if MODEL == "gpt2":
        Fp = (HE + ex + ES) * 65536 - ls; e = int(EXP2[Fp & 0xFFFF]); ge = ones1(e); F2 = -Fp; s = 16 - (Fp >> 16)
        e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - (F2 >> 16); ge2 = ones1(e2); beta = power(BIAS, F2)
        twos = Fraction(2) ** s; twot = Fraction(2) ** (s + s2)
        gg = Fraction(1, 2) + BMAX * twos * abs(e * e2 - twot) / (e * twot); gamma = -(-gg.numerator // gg.denominator)
        delta = max(1, -(-twos.numerator // (e * twos.denominator))); mu = 2 * gamma + delta
        S = power(PILE, Fp) + BIAS
        plain = int(ones(x) @ pph) + int(ones(PILE).sum()) * ge + int(BIAS1.sum()); stand = int(BIAS1.sum()) * ge2 + int(ones(beta).sum())
    else:
        Fp = (EMIN + ex + ES) * 65536 - ls; e = int(EXP2[Fp & 0xFFFF]); ge = ones1(e); s = 16 - (Fp >> 16)
        twos = Fraction(2) ** s; delta = max(1, -(-twos.numerator // (e * twos.denominator))); mu = 2 + delta
        S = power(PILE << KW, Fp); plain = int(ones(x) @ pph) + int(ones(PILE).sum()) * ge; stand = 0
    s5 = np.sort(S)[-5]
    alive = np.arange(M); rd = 0; bph = 0; surv = []
    for i, b in enumerate(places):
        rd += int(H[alive, b].sum())
        if i == len(places) - 1: break
        R2 = BND[b] * S1; assert (np.abs(CUM[:, 0] - CUM[:, b]) <= R2).all(), "the shared bound broken"
        if MODEL == "gpt2":
            V_ = CUM[alive, b] + 2 * beta[alive]; v5 = np.sort(V_)[-5]; thr = 2 * R2 + 2 * mu; keep = V_ > v5 - thr
            bph += ones1(S1) * ones1(BND[b]) + ones1(2 * mu) + ones1(thr)
        else:
            K = KW[alive]; Vs = (CUM[alive, b] + kc * X) << K; lows = []; ph = ones1(S1) * ones1(BND[b]); THR = np.zeros(4, np.int64)
            for k in range(4):
                Vk = Vs[K == k]
                if len(Vk) == 0: continue
                m = min(5, len(Vk)); t5 = np.sort(Vk)[-m:]; lows += list(t5 - (R2 << k)); ph += m * (ones1(R2 << k) + (sd1(kc * X << k) if kc else 0))
            L5 = sorted(lows)[-5]
            for k in range(4):
                if (K == k).any():
                    thr = (R2 << k) + 2 * mu; THR[k] = L5 - thr; ph += ones1(2 * mu) + ones1(thr) + (sd1(kc * X << k) if kc else 0)
            keep = Vs > THR[K]; bph += ph
        assert (S[alive[~keep]] < s5).all(), "a word set aside at or above the 5th-best"
        alive = alive[keep]; surv.append(len(alive))
    pa = pile[alive]
    if MODEL == "gpt2":
        sc = power(pa, Fp) + BIAS[alive]
        end = (sd1(kc * X) * len(alive) if kc else 0) + int(ones(beta[alive]).sum()) + int(ones(pa).sum()) * ge + int(BIAS1[alive].sum())
    else:
        Ve = pa << KW[alive]; sc = power(Ve, Fp); end = (sd1(kc * X) * len(alive) if kc else 0) + int(ones(Ve).sum()) * ge
    assert np.array_equal(sc, S[alive])
    Sp = np.full(M, -(1 << 62), np.int64); Sp[alive] = sc
    recs.append(dict(plain=plain, ii_total=tab + lsum + rd + bph + end + stand, table=tab, ii_reads=rd, ii_light=lsum, ii_bound=bph, ii_end=end,
                     ii_stand=stand, pwrite=pw, lwrite=int(ones(x).sum()), surv=surv, top5=[int(v) for v in top5(Sp)], win=int(race(Sp)),
                     ptop5=[int(v) for v in top5(S)], pwin=int(race(S))))
    if q % 10 == 0: print(f"  {q + 1}/{len(XA)} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
json.dump(dict(model=MODEL, mask=mask, kc=kc, places=places, recs=recs), open(f"{OUT}/topg_own_{MODEL}.json", "w"))
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
