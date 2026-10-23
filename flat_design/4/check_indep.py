"""flat_design/4/check_indep.py -- an independent recount of tips_units.py's numbers, with its own code: it imports
nothing from tips_units.py, gbank.py, uh_rebuild.py or uh_perplace.py, and reads only out/tips_units.json to compare.

 1. E1 (up/hole place-pair tips) by packed planes and popcounts (XOR of 64 lines at a time), not by matmuls;
    E2 and E3 (lit tips, place pairs and diagonals) by a loop over the light's places with whole-number sums.
    proj.5 and mproj.5, the same 24 positions, the real light and the same shuffle draw.
 2. The group work at proj.5 and mproj.5, g = 11 and g = 12, recounted from the rules as written in the delivered
    docstrings (groups2/gbank.py header): the per-place rule (signed digits at place p when the plate's lit signed digits
    there are fewer than groups x neurons), the grouping (numpy generator keyed [1, slab, layer, g], a random line order
    cut into runs of g), the digits (up/hole binary digits from the bottom with offset 2^(top up/hole place), signed
    digits elsewhere), reads (each up/hole place of each (neuron, group): the ready pile's signed digits), signed-digit
    reads (each lit signed digit: its light value's signed digits), the half table (line t of a group of k: 1 if t >= 1,
    plus 2^(k-2-t) if t <= k-2, times its light value's signed digits), writing (each half-table entry read, itself or
    its flip: the 1s of the entry with the group's last line up), X (lines 1.. of the light's signed digits), kcorr X,
    the light's 1s. Every pile rebuilt from the recount's own digits and checked against the direct product.
"""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/4/out"
R = json.load(open(f"{OUT}/tips_units.json")); POS = np.array(R["positions"]); LAYER = R["layer"]
MZ = np.load(f"{HERE}/base2_gpt2.npz")
def popc(a): return np.bitwise_count(a)
def sdc(a): a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def onec(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)
log = open(f"{OUT}/check_indep.log", "w")
def say(s): print(s, flush=True); log.write(s + "\n"); log.flush()

def pack_rows(B):                    # B (..., N) of 0/1 -> (..., words) uint64
    N = B.shape[-1]; W = (N + 63) // 64; pad = W * 64 - N
    if pad: B = np.concatenate([B, np.zeros(B.shape[:-1] + (pad,), B.dtype)], -1)
    return np.ascontiguousarray(np.packbits(np.ascontiguousarray(B, np.uint8), axis=-1, bitorder="little")).view(np.uint64)

def e1_popcount(x, w):
    P, N = x.shape; M = w.shape[1]
    U = x + (1 << 15); Uw = (w + (1 << 15)).T                                  # (M, N)
    LW = np.stack([pack_rows((Uw >> b) & 1) for b in range(16)], 0)          # (16, M, words)
    tot = 0
    for p in range(P):
        LX = np.stack([pack_rows((U[p] >> a) & 1) for a in range(16)], 0)    # (16, words)
        for a in range(16):
            d = popc(LX[a][None, None, :] ^ LW).sum(-1, dtype=np.int64)     # (16, M): lines that disagree
            tot += int(np.abs(N - 2 * d).sum())
    return tot

def e2_e3_loop(x, w):
    P, N = x.shape; M = w.shape[1]; ax, aw = np.abs(x), np.abs(w); sx, sw = np.sign(x), np.sign(w)
    WB = [(sw * ((aw >> b) & 1)).astype(np.float64) for b in range(15)]      # (N, M) each; +-1/0, sums below 2^53
    e2 = 0; diag = np.zeros((P, 29, M), np.int64)
    for a in range(15):
        xa = (sx * ((ax >> a) & 1)).astype(np.float64)                         # (P, N)
        for b in range(15):
            t = np.rint(xa @ WB[b]).astype(np.int64); e2 += int(np.abs(t).sum()); diag[:, a + b] += t
    return e2, int(np.abs(diag).sum())

# ---------------- 1 ----------------
say("INDEPENDENT RECOUNT 1: the tips by packed popcounts and whole-number loops")
for ki, kind in [(1, "proj"), (3, "mproj")]:
    w = MZ[f"{kind}.{LAYER}.w"].astype(np.int64)
    x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[LAYER][POS]).astype(np.int64)
    xs = np.random.default_rng([4, LAYER, ki]).permuted(x, axis=1)
    for tag, xx in (("real", x), ("shuffle", xs)):
        e1 = e1_popcount(xx, w); e2, e3 = e2_e3_loop(xx, w); ref = R["A"][kind][tag]
        say(f"  {kind}.{LAYER} {tag:7s}: E1 {e1:,} (tips_units {ref['E1']:,}) {'SAME' if e1 == ref['E1'] else 'DIFFERENT'};"
            f" E2 {e2:,} ({ref['E2']:,}) {'SAME' if e2 == ref['E2'] else 'DIFFERENT'}; E3 {e3:,} ({ref['E3']:,}) {'SAME' if e3 == ref['E3'] else 'DIFFERENT'}")

# ---------------- 2 ----------------
def naf_digits(v):                    # signed digits of whole numbers, place by place: (16+, shape) in {-1, 0, 1}
    v = np.asarray(v, np.int64).copy(); out = []
    for _ in range(18):
        odd = (v & 1) == 1; d = np.where(odd, 2 - (v & 3), 0); out.append(d); v = (v - d) >> 1
    assert not v.any(); return np.stack(out, 0)

def recount(kind, slab, g, x):
    w = MZ[f"{kind}.{LAYER}.w"].astype(np.int64); N, M = w.shape; P = len(x); G = -(-N // g)
    nd = naf_digits(w); lit = (nd != 0).reshape(18, -1).sum(1)
    assert not lit[16:].any()
    upl = [p for p in range(16) if not lit[p] < G * M]; assert upl == list(range(len(upl))), upl
    top = len(upl) - 1; C = 1 << top; K = sum(1 << b for b in upl)
    r = w + C; E = np.zeros_like(w); sdig = []                                  # digits bottom up
    for b in range(18):
        if b <= top:
            e = r & 1; E |= e << b; r = (r - e) >> 1; sdig.append(np.zeros_like(w))
        else:
            odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); sdig.append(d); r = (r - d) >> 1
    assert not r.any()
    order = np.random.default_rng([1, slab, LAYER, g]).permutation(N); groups = [order[s:s + g] for s in range(0, N, g)]
    kcorr = K - 2 * C
    # the piles from the recount's own digits: 2 x.w = sum_b 2^b V + 2 sum signed + kcorr X
    X = x.sum(1); D2 = np.repeat((kcorr * X)[:, None], M, 1).astype(np.int64)
    cnt = dict(light=int(onec(x).sum()), table=0, pwrite=0, xsum=int(sdc(x[:, groups[0][0]:groups[0][0]]).sum()), reads=0, sdreads=0, xoff=0)
    sx = sdc(x)                                                                 # (P, N)
    for gr in groups:
        k = len(gr); xg = x[:, gr]                                              # (P, k)
        ch = np.array([(1 if t >= 1 else 0) + ((1 << (k - 2 - t)) if t <= k - 2 else 0) for t in range(k)])
        cnt["table"] += int((sx[:, gr] @ ch).sum())
        pats = np.zeros((M, top + 1), np.int64)
        for t in range(k): pats |= (((E[gr[t]][:, None] >> np.arange(top + 1)[None, :]) & 1) << t)
        n = 1 << k; allp = np.arange(n)
        bits = ((allp[:, None] >> np.arange(k)[None, :]) & 1) * 2 - 1           # (n, k) up +1 / hole -1
        V = xg @ bits.T                                                         # (P, n): every pattern's ready pile
        D2 += np.einsum("pjb,b->pj", V[:, pats], (1 << np.arange(top + 1)).astype(np.int64))
        cnt["reads"] += int(sdc(V[:, pats]).sum())
        full = n - 1; hi = 1 << (k - 1)
        keys = np.unique(np.where(pats & hi, pats, pats ^ full))                # the half-table entries read (last line up)
        cnt["pwrite"] += int(onec(V[:, keys]).sum())
    for b in range(top + 1, 18):
        d = sdig[b]; D2 += 2 * (x @ d) << b
        cnt["sdreads"] += int((sx @ (d != 0).astype(np.int64)).sum())
    cnt["xsum"] = int(sx[:, 1:].sum())
    for kv in np.unique(np.broadcast_to(kcorr, (M,))):
        if kv == 0: continue
        cnt["xoff"] += M * int(sdc(kv * X).sum()) + (M * (int(sdc(kv)) - 1) * int(sdc(X).sum()) if int(sdc(kv)) > 1 else 0)
    assert not (D2 & 1).any(); pile = D2 >> 1
    bad = int((pile != x @ w).any(1).sum())
    plain = int((onec(x) @ onec(w).sum(1)).sum())
    return cnt, bad, plain

say("\nINDEPENDENT RECOUNT 2: the group work at the relation's g and at 12, from the written rules")
for kind, slab in (("proj", 1), ("mproj", 3)):
    x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[LAYER][POS]).astype(np.int64)
    for g in (11, 12):
        cnt, bad, plain = recount(kind, slab, g, x); ref = R["C"][kind]["runs"][str(g)]["items"]
        mine = dict(light=cnt["light"], table=cnt["table"], pwrite=cnt["pwrite"], xsum=cnt["xsum"], xoff=cnt["xoff"],
                    **{f"reads|{kind}": cnt["reads"], f"sdreads|{kind}": cnt["sdreads"]})
        diff = {k: (mine[k], ref.get(k, 0)) for k in mine if mine[k] != ref.get(k, 0)}
        say(f"  {kind}.{LAYER} g = {g}: piles differing from the direct product {bad}; whole {sum(mine.values()):,} = "
            f"{100 * sum(mine.values()) / plain:.2f}% of plain (tips_units: {sum(ref.values()):,}); items differing: {diff if diff else 'none'}")
