"""DESIGNER 3: BOTH SIDES AT ONCE. A sample measurement for the cost relation (not the whole machine).
Predictions in PREDICTION.txt, written before this file ran.

WHAT IS MEASURED (the session's words; whole numbers only, every pile checked against base2.py's own pile)
  A. THE EXTRA IN THREE DIGIT FORMS, both sides in the same form, every place pair (light place a, plate place b):
     uh  up/hole both sides (every place a photon up or a hole; x' = 2x+1, D = 2w+1): N meetings a place pair;
         tip = agree - disagree, by popcount of the two planes' XOR over 64 lines a word.
     one the plain form, the sign and the 1s: lit x lit meetings; tip = same-sign - opposite-sign.
     sd  signed digits both sides: lit digit x lit digit meetings; tip = (+,+) + (-,-) - (+,-) - (-,+).
     For each: meetings, extra = sum |tip|, the tips added as whole numbers (sum of each tip's signed digits), and the
     pile rebuilt from the tips alone, checked exact.
  B. GROUP PILES IN THE PER-PLACE FORM (the confirmed machine's form) at a given g: plate places below h up/hole
     (addresses, centred: the bits of w_lo + 2^(h-1)), places h and up in signed digits met directly (rule: L_b < G x M);
     ready piles of each random group
     of g light lines, the half table by the flip. Counted: reads (each ready pile's signed digits), the plate's top
     digits met directly (the light value's signed digits), the half table (Gray steps), the ready piles read written
     in signed digits (their 1s), X and its take-away, the light written (its 1s).
  D. B WITH THE LIGHT'S SPARSE TOP PLACES MET DIRECTLY (the mirror rule, from the light alone each position: light
     places a and up where the light has fewer lit signed digits than groups, L^x_a < G); those digits meet each
     gate's whole weight (the weight's signed digits); the rest of the light goes into the ready piles.
  THE READERS RELATION: g* solves 2^(g-1) (g ln2 - 1) = R s_V / (s_x + o_V) from the g = 12 run; B and D rerun at g*.
  CONTROLS: run=shuffle (each position's light shuffled across its lines; piles checked against their own direct
  product), run=seed2 (a second random grouping). The tips' order-freeness is checked in every run (a random line
  order gives identical tips at one position a plate).
    python3 flat_design/3/measure.py run=main|shuffle|seed2   -> flat_design/3/out/<run>.json, .log
"""
import os, sys, json, time, math, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/3/out"
ARG = dict(a.split("=") for a in sys.argv[1:]); RUN = ARG.get("run", "main")
SEED = 2 if RUN == "seed2" else 1
POS = np.arange(5, 648, 40)[:16][:int(ARG.get("npos", 16))]
PLATES = [(k, l) for l in (3, 8) for k in ("qkv", "proj", "fc", "mproj")]
if "plates" in ARG: PLATES = [(p.split(".")[0], int(p.split(".")[1])) for p in ARG["plates"].split(",")]
KIND = ["qkv", "proj", "fc", "mproj"]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def sdg(A): a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def pack(B):
    B = np.ascontiguousarray(B.astype(np.uint8)); assert B.shape[-1] % 64 == 0
    return np.packbits(B, axis=-1, bitorder="little").view(np.uint64)
def naf_planes(v, n):
    """signed digits of signed whole numbers v: (up, down) planes, n places, as 0/1 arrays (..., n) -> moved to front"""
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a; up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    neg = v < 0; up, dn = np.where(neg, dn, up), np.where(neg, up, dn)
    assert not ((up | dn) >> n).any() and np.array_equal(up - dn, v)
    U = np.stack([(up >> b) & 1 for b in range(n)]); D = np.stack([(dn >> b) & 1 for b in range(n)])
    return U, D

# ---------- A. the extra in three digit forms ----------
def extra_forms(x, w, ref):
    """x (P, N), w (N, M), ref (P, M) the machine's piles. Returns the counts summed over positions."""
    P, N = x.shape; M = w.shape[1]; r = {}
    r["plain"] = int(ones(x).sum(0) @ ones(w).sum(1)); r["sdmeet"] = int(sdg(x).sum(0) @ sdg(w).sum(1))
    # up/hole
    nx = 16 if (x.max() < (1 << 15) and x.min() >= -(1 << 15)) else 17
    cx = x + (1 << (nx - 1)); cw = (w + (1 << 15)).T                                   # (P, N), (M, N)
    Xp = np.stack([pack((cx >> a) & 1) for a in range(nx)], 1)                         # (P, nx, words)
    Wp = np.stack([pack((cw >> b) & 1) for b in range(16)])                            # (16, M, words)
    Xs, Ws = x.sum(1), w.sum(0)
    ext = dig = 0; bad = 0; T0 = None
    for p in range(P):
        d = np.stack([np.bitwise_count(Xp[p, a][None, None, :] ^ Wp).sum(-1, dtype=np.int64) for a in range(nx)])
        T = N - 2 * d                                                                  # (nx, 16, M) tips
        ext += int(np.abs(T).sum()); dig += int(sdg(T).sum())
        S = sum((T[a, b] << (a + b)) for a in range(nx) for b in range(16))
        bad += int(((S - 2 * Xs[p] - 2 * Ws - N) != 4 * ref[p]).any())
        if p == 0: T0 = T
    r["uh"] = dict(meet=int(nx * 16 * N * M * P), extra=ext, tipdig=dig, places=nx, bad=bad)
    # order-free: a random line order, both sides, gives the same tips at position 0
    perm = np.random.default_rng(99).permutation(N)
    Xq = np.stack([pack(((cx[0:1, perm]) >> a) & 1) for a in range(nx)], 1); Wq = np.stack([pack((cw[:, perm] >> b) & 1) for b in range(16)])
    Tq = N - 2 * np.stack([np.bitwise_count(Xq[0, a][None, None, :] ^ Wq).sum(-1, dtype=np.int64) for a in range(nx)])
    r["uh"]["orderfree_same"] = bool(np.array_equal(Tq, T0))
    # the plain form: sign and 1s
    ax, aw = np.abs(x), np.abs(w).T
    XL = np.stack([pack((ax >> a) & 1) for a in range(16)], 1); SX = pack(x < 0)       # (P, 16, words), (P, words)
    WL = np.stack([pack((aw >> b) & 1) for b in range(16)]); SW = pack(w.T < 0)        # (16, M, words), (M, words)
    ext = dig = meet = 0; bad = 0
    for p in range(P):
        opp = SX[p][None, :] ^ SW                                                      # (M, words)
        lit = np.empty((16, 16, M), np.int64); dis = np.empty((16, 16, M), np.int64)
        for a in range(16):
            E = XL[p, a][None, None, :] & WL
            lit[a] = np.bitwise_count(E).sum(-1, dtype=np.int64); dis[a] = np.bitwise_count(E & opp[None]).sum(-1, dtype=np.int64)
        T = lit - 2 * dis; meet += int(lit.sum()); ext += int(np.abs(T).sum()); dig += int(sdg(T).sum())
        S = sum((T[a, b] << (a + b)) for a in range(16) for b in range(16)); bad += int((S != ref[p]).any())
    r["one"] = dict(meet=meet, extra=ext, tipdig=dig, bad=bad)
    # signed digits both sides
    XU, XD = naf_planes(x, 17); WU, WD = naf_planes(w.T, 17)                           # (17, P, N), (17, M, N)
    XU = np.stack([pack(XU[a]) for a in range(17)], 1); XD = np.stack([pack(XD[a]) for a in range(17)], 1)
    WU = np.stack([pack(WU[b]) for b in range(17)]); WD = np.stack([pack(WD[b]) for b in range(17)])
    ext = dig = meet = 0; bad = 0
    for p in range(P):
        T = np.empty((17, 17, M), np.int64); mm = 0
        for a in range(17):
            uu = np.bitwise_count(XU[p, a][None, None] & WU).sum(-1, dtype=np.int64); dd = np.bitwise_count(XD[p, a][None, None] & WD).sum(-1, dtype=np.int64)
            ud = np.bitwise_count(XU[p, a][None, None] & WD).sum(-1, dtype=np.int64); du = np.bitwise_count(XD[p, a][None, None] & WU).sum(-1, dtype=np.int64)
            T[a] = uu + dd - ud - du; mm += int((uu + dd + ud + du).sum())
        meet += mm; ext += int(np.abs(T).sum()); dig += int(sdg(T).sum())
        S = sum((T[a, b] << (a + b)) for a in range(17) for b in range(17)); bad += int((S != ref[p]).any())
    r["sd"] = dict(meet=meet, extra=ext, tipdig=dig, bad=bad)
    assert r["one"]["meet"] == r["plain"] and r["sd"]["meet"] == r["sdmeet"]
    return r

# ---------- B / D. group piles in the per-place form ----------
def grouping(N, g, key):
    perm = np.random.default_rng(key).permutation(N); return [perm[i:i + g] for i in range(0, N, g)]

def plate_form(w, g):
    """the per-place rule from the plate alone: h = 1 + the top place with L_b >= G x M (places h and up: signed digits)"""
    N, M = w.shape; G = -(-N // g); r = np.asarray(w, np.int64).copy(); L = []
    for b in range(18):
        odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); L.append(int((d != 0).sum())); r = (r - d) >> 1
    dense = [b for b in range(18) if L[b] >= G * M]; h = 1 + max(dense)
    C = 1 << (h - 1); wlo = ((w + C) & ((1 << h) - 1)) - C; whi = (w - wlo) >> h          # centred: wlo in [-C, C)
    assert np.array_equal(wlo + (whi << h), w)
    return h, wlo + C, whi, L                                  # the up/hole bits are those of wlo + C

def group_piles(x, w, ref2, g, key, light_split):
    """x (P, N); ref2 (P, M) = 2 x the machine's piles. Returns the counts summed over positions."""
    P, N = x.shape; M = w.shape[1]; groups = grouping(N, g, key); G = len(groups)
    h, wlo, whi, L = plate_form(w, g)
    PAT = []
    for L_ in groups:
        pk = np.zeros((M, h), np.int64)
        for t, line in enumerate(L_):
            pk |= ((wlo[line][:, None] >> np.arange(h)[None, :]) & 1) << t
        PAT.append(pk)
    nhi = sdg(whi).sum(1)                                     # the plate's top digits, a line, over all gates
    SWt = sdg(w).sum(1)                                       # a light digit meeting a line's whole weight row
    sh = (1 << np.arange(h)).astype(np.int64)
    c = dict(reads=0, reads_n=0, top_plate=0, build=0, build_n=0, write=0, write_n=0, X=0, light_write=0, top_light=0,
             top_light_digits=0, h=h, G=G, M=M, N=N, places_light_split=[])
    bad = 0
    for p in range(P):
        xp = x[p].astype(np.int64)
        if light_split:
            U, D = naf_planes(xp, 17); Lx = (U | D).sum(1)                                 # lit digits a light place
            sparse = [a for a in range(17) if Lx[a] < G]; dense = [a for a in range(17) if Lx[a] >= G]
            cpl = 1 + max(dense) if dense else 0
            top = sum(((U[a] - D[a]) << a) for a in range(cpl, 17)) if cpl < 17 else np.zeros(N, np.int64)
            xlo = xp - top; ntop = (U[cpl:] | D[cpl:]).sum(0) if cpl < 17 else np.zeros(N, np.int64)
            c["top_light"] += int(ntop @ SWt); c["top_light_digits"] += int(ntop.sum()); c["places_light_split"].append(cpl)
        else:
            xlo = xp; top = None
        sx = sdg(xlo)
        pile2 = np.zeros(M, np.int64)
        for k, L_ in enumerate(groups):
            gk = len(L_); xv = xlo[L_]; V = np.array([-xv.sum()], np.int64)
            for t in range(gk): V = np.concatenate([V, V + 2 * xv[t]])
            pk = PAT[k]; Vr = V[pk]                                                        # (M, h) reads
            pile2 += (Vr * sh[None, :]).sum(1)
            c["reads"] += int(sdg(Vr).sum()); c["reads_n"] += int(Vr.size)
            # the half table: the group total (gk - 1 adds) and 2^(gk-1) - 1 Gray steps over lines 0..gk-2
            c["build"] += int(sx[L_[1:]].sum()) + int(sum((1 << (gk - 2 - t)) * int(sx[L_[t]]) for t in range(gk - 1)))
            c["build_n"] += (gk - 1) + (1 << (gk - 1)) - 1
            last = 1 << (gk - 1); mask = (1 << gk) - 1
            keys = np.unique(np.where(pk & last, pk, (~pk) & mask))
            c["write"] += int(ones(V[keys]).sum()); c["write_n"] += int(keys.size)
        X = int(xlo.sum()); Y = X                               # sum_b 2^b (2 bit - 1) = 2 wlo + 1: reads = 2 x.wlo + X
        pile2 += -Y + (xlo @ whi) * (1 << (h + 1))
        c["top_plate"] += int(sx @ nhi)
        c["X"] += int(sx[1:].sum()) + int(sdg(Y)) * M
        c["light_write"] += int(ones(xp).sum())
        if light_split: pile2 += 2 * (top @ w)
        bad += int((pile2 != ref2[p]).any())
    c["bad"] = bad; c["sx_mean"] = None
    c["total"] = c["reads"] + c["top_plate"] + c["build"] + c["write"] + c["X"] + c["light_write"] + c["top_light"]
    return c

def readers_g(c, sx_mean):
    """g* from 2^(g-1) (g ln2 - 1) = R s_V / (s_x + o_V)"""
    R = c["h"] * c["M"]; sV = c["reads"] / c["reads_n"]; oV = c["write"] / c["write_n"]
    rhs = R * sV / (sx_mean + oV); f = lambda g: 2 ** (g - 1) * (g * math.log(2) - 1)
    lo, hi = 2.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if f(mid) < rhs else (lo, mid)
    return lo, R, sV, oV, rhs

if __name__ == "__main__":
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); TAG = RUN + ("_test" if ("npos" in ARG or "plates" in ARG) else ""); LOG = open(f"{OUT}/{TAG}.log", "w")
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"DESIGNER 3 sample, run {RUN}: plates {PLATES}, positions {POS.tolist()}")
    MZ = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    res = {}
    for kind, l in PLATES:
        tl = time.time()
        x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l][POS]).astype(np.int64)
        w = np.asarray(MZ[f"{kind}.{l}.w"]).astype(np.int64)
        ref = np.asarray(REF[f"{kind}_pile_{l}"][POS]).astype(np.int64)
        assert np.array_equal(x @ w, ref)                                               # the machine's light x plate
        if RUN == "shuffle":
            rng = np.random.default_rng([5, KIND.index(kind), l]); x = rng.permuted(x, axis=1); ref = x @ w
        rec = dict()
        if RUN != "seed2":
            rec["A"] = extra_forms(x, w, ref)
        sxm = float(sdg(x).mean())
        key = lambda g: [SEED, KIND.index(kind), l, g]
        b12 = group_piles(x, w, 2 * ref, 12, key(12), False); rec["B12"] = b12
        gs, R, sV, oV, rhs = readers_g(b12, sxm); g_star = int(round(gs))
        rec["relation"] = dict(g_real=gs, g_star=g_star, R=R, log2R=math.log2(R), sV=sV, oV=oV, sx=sxm, rhs=rhs)
        if RUN != "seed2":
            rec["D12"] = group_piles(x, w, 2 * ref, 12, key(12), True)
            if g_star != 12:
                rec["Bg"] = group_piles(x, w, 2 * ref, g_star, key(g_star), False)
                rec["Dg"] = group_piles(x, w, 2 * ref, g_star, key(g_star), True)
            else:
                rec["Bg"], rec["Dg"] = rec["B12"], rec["D12"]
        res[f"{kind}.{l}"] = rec
        pl = rec["A"]["plain"] if "A" in rec else int(ones(x).sum(0) @ ones(w).sum(1)); rec["plain"] = pl
        bads = {k: v["bad"] for k, v in rec.items() if isinstance(v, dict) and "bad" in v}
        if "A" in rec: bads.update({f"A.{f}": rec["A"][f]["bad"] for f in ("uh", "one", "sd")})
        assert all(v == 0 for v in bads.values()), bads
        s = f"  {kind}.{l}: exact {bads}; g* {gs:.2f} -> {g_star}"
        if "A" in rec:
            A = rec["A"]
            s += (f"; uh extra/meet {100 * A['uh']['extra'] / A['uh']['meet']:.3f}% extra/plain {100 * A['uh']['extra'] / pl:.2f}%"
                  f" tipdig/plain {100 * A['uh']['tipdig'] / pl:.2f}%; one extra/plain {100 * A['one']['extra'] / pl:.2f}%;"
                  f" sd meet/plain {100 * A['sd']['meet'] / pl:.2f}% extra/plain {100 * A['sd']['extra'] / pl:.2f}%")
        for v in ("B12", "D12", "Bg", "Dg"):
            if v in rec: s += f"; {v} {100 * rec[v]['total'] / pl:.2f}% (reads {100 * rec[v]['reads'] / pl:.2f})"
        log(s + f"  ({time.time() - tl:.0f}s, {rss()} MB)")
    json.dump(dict(run=RUN, positions=POS.tolist(), plates=res), open(f"{OUT}/{TAG}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
