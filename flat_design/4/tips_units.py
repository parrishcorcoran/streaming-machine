"""flat_design/4/tips_units.py -- DESIGNER 4, FROM THE TIPS BACKWARDS. Predictions: PREDICTION.txt (written first).

Part A, THE TIPS IN TWO UNITS (GPT-2, base2's own light and plates; layer 5's qkv, proj, fc, mproj; 24 positions):
  E1  up/hole place-pair tips, sum of |N - 2d| over the 16 x 16 place pairs (FINDINGS sec. 5; offset 2^15, D = 2v + 1)
  E1w the same tips written as whole numbers: their signed digits
  E2  lit form (sign and 1s, 15 places): sum over place pairs of |up meetings - down meetings|; E2w their signed digits
  E3  lit tips paired along each diagonal a + b = d first (E3u: the up/hole ones)
  E4  the gate's place sums (whole light against plate place b), their signed digits (lit; E4u up/hole)
  E5  the pile's own signed digits
  Each against: the up/hole meetings (N x 256 a pile) and plain (lit light x lit plate, binary 1s).
  Checks: both tip forms rebuild every pile (asserted). Controls: SHUFFLE (each position's light across its lines,
  piles checked against the shuffled direct product); ORDER-FREE (one line order, the same on both sides: every number
  identical, asserted).
Part B, THE FLOOR RELATION's parts at each plate: the up/hole places the per-place rule reads, reads a weight, photons
  a read (the confirmed machine's layer-5 counts, groups2/machine/gpt2_refs.json), the plate's bits a weight.
Part C, THE METHOD's sample: the confirmed group bank (groups2/gbank.py, unchanged) at g = 12 and at the relation's g
  (written in PREDICTION.txt before this ran: 11 at proj and mproj; 12 at qkv and fc). Every pile checked against the
  direct product; work counted item by item (light, half table, writing, reads, signed-digit reads, X, kcorr X).
Whole numbers only in the machine path (float32 matmuls carry only +-1/0 planes; every tip is below 2^24, so exact).
    python3 flat_design/4/tips_units.py  -> flat_design/4/out/tips_units.json, .log
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/4/out"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import gbank as GB
import uh_perplace as PP
sys.argv = _argv
LAYER = 5; KINDS = ["qkv", "proj", "fc", "mproj"]; POS = np.linspace(0, 647, 24).astype(int)
LOG = open(f"{OUT}/tips_units.log", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def sd(A): a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def planes_uh(v, axis_last):            # v whole numbers, |v| < 2^15: 16 up/hole places (+1 photon, -1 hole), offset 2^15
    U = np.asarray(v, np.int64) + (1 << 15); assert U.min() >= 0 and U.max() < (1 << 16)
    return [(2 * ((U >> a) & 1) - 1).astype(np.float32) for a in range(16)]

def planes_lit(v):                      # sign and 1s, 15 places
    v = np.asarray(v, np.int64); a = np.abs(v); s = np.sign(v)
    assert a.max() < (1 << 15)
    return [(s * ((a >> q) & 1)).astype(np.float32) for q in range(15)]

def tips(x, w):
    """x (P, N), w (N, M) whole numbers -> T1 (P, 16, M, 16) up/hole tips N - 2d, T2 (P, 15, M, 15) lit tips"""
    P, N = x.shape; M = w.shape[1]
    X1 = np.stack(planes_uh(x, True), 1).reshape(P * 16, N)
    W1 = np.stack(planes_uh(w, True), 2).reshape(N, M * 16)
    T1 = (X1 @ W1).reshape(P, 16, M, 16); del X1, W1
    X2 = np.stack(planes_lit(x), 1).reshape(P * 15, N)
    W2 = np.stack(planes_lit(w), 2).reshape(N, M * 15)
    T2 = (X2 @ W2).reshape(P, 15, M, 15)
    PL = (np.abs(X2) @ np.abs(W2)).sum(dtype=np.float64)          # every lit meeting: must equal plain
    del X2, W2
    T1 = np.rint(T1).astype(np.int64); T2 = np.rint(T2).astype(np.int64)
    return T1, T2, int(PL)

def measure(x, w):
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); P, N = x.shape; M = w.shape[1]
    pile = x @ w
    T1, T2, PL = tips(x, w)
    pw1 = (1 << (np.arange(16)[:, None] + np.arange(16)[None, :])).astype(np.int64)
    pw2 = (1 << (np.arange(15)[:, None] + np.arange(15)[None, :])).astype(np.int64)
    S1 = np.einsum("pajb,ab->pj", T1, pw1)
    V = S1 - (2 * x.sum(1) + N)[:, None] - (2 * w.sum(0) + N)[None, :] + N
    assert not (V % 4).any() and np.array_equal(V // 4, pile), "up/hole tips do not rebuild the piles"
    S2 = np.einsum("pajb,ab->pj", T2, pw2)
    assert np.array_equal(S2, pile), "lit tips do not rebuild the piles"
    plain = int((ones(x) @ ones(w).sum(1)).sum())
    assert plain == PL, (plain, PL)
    r = dict(P=P, N=N, M=M, piles=P * M, piles_rebuilt_uh=P * M, piles_rebuilt_lit=P * M, plain=plain,
             signed=int((sd(x) @ sd(w).sum(1)).sum()), adds_plain=int(((x != 0).astype(np.int64) @ ones(w).sum(1)).sum()),
             uh_meetings=P * M * N * 256)
    r["E1"] = int(np.abs(T1).sum()); r["E1w"] = int(sd(T1).sum())
    r["E2"] = int(np.abs(T2).sum()); r["E2w"] = int(sd(T2).sum())
    D1 = np.zeros((P, 31, M), np.int64); D2 = np.zeros((P, 29, M), np.int64)
    for a in range(16):
        for b in range(16): D1[:, a + b] += T1[:, a, :, b]
    for a in range(15):
        for b in range(15): D2[:, a + b] += T2[:, a, :, b]
    r["E3u"] = int(np.abs(D1).sum()); r["E3"] = int(np.abs(D2).sum()); r["E3w"] = int(sd(D2).sum())
    A2 = np.einsum("pajb,a->pjb", T2, (1 << np.arange(15)).astype(np.int64))
    A1 = np.einsum("pajb,a->pjb", T1, (1 << np.arange(16)).astype(np.int64))
    r["E4"] = int(sd(A2).sum()); r["E4u"] = int(sd(A1).sum())
    r["E5"] = int(sd(pile).sum())
    # the tips' size by diagonal (lit form), share of the plain meetings on that diagonal
    lit = np.zeros(29, np.int64); ex = np.zeros(29, np.int64)
    ox = np.stack([((np.abs(x) >> q) & 1) for q in range(15)], 1)            # (P, 15, N)
    ow = np.stack([((np.abs(w) >> q) & 1) for q in range(15)], 2)            # (N, M, 15)
    cnt = np.einsum("pan,nb->pab", ox, ow.sum(1))                             # lit meetings per place pair
    for a in range(15):
        for b in range(15): lit[a + b] += cnt[:, a, b].sum(); ex[a + b] += np.abs(T2[:, a, :, b]).sum()
    r["lit_by_diag"] = lit.tolist(); r["E2_by_diag"] = ex.tolist()
    del T1, T2, D1, D2, A1, A2
    return r, pile

def share(r):
    p, u = r["plain"], r["uh_meetings"]
    return {k: (100 * r[k] / u, 100 * r[k] / p) for k in ("E1", "E1w", "E2", "E2w", "E3u", "E3", "E3w", "E4u", "E4", "E5")}

if __name__ == "__main__":
    t0 = time.time(); MZ = np.load(f"{HERE}/base2_gpt2.npz"); RES = dict(layer=LAYER, positions=POS.tolist(), A={}, B={}, C={})
    J = json.load(open(f"{HERE}/groups2/machine/gpt2_refs.json")); NP = J["npos"]; CJ = J["counts"]
    def jget(key): return CJ.get(key, 0) / NP
    log(f"DESIGNER 4: the tips in two units, the floor, the method's sample. GPT-2 layer {LAYER}, {len(POS)} positions {POS.tolist()}")
    # ---------------- Part A ----------------
    for ki, kind in enumerate(KINDS):
        tk = time.time()
        w = MZ[f"{kind}.{LAYER}.w"].astype(np.int64)
        x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[LAYER][POS]).astype(np.int64)
        real, pile = measure(x, w)
        rng = np.random.default_rng([4, LAYER, ki]); xs = rng.permuted(x, axis=1)
        shuf, ps = measure(xs, w); assert np.array_equal(ps, xs @ w)
        perm = np.random.default_rng([44, LAYER, ki]).permutation(x.shape[1])
        ordf, po = measure(x[:, perm], w[perm]); assert np.array_equal(po, pile)
        same = all(ordf[k] == real[k] for k in real if not isinstance(real[k], list)) and ordf["E2_by_diag"] == real["E2_by_diag"]
        assert same, "order-free check failed"
        RES["A"][kind] = dict(real=real, shuffle=shuf, orderfree_identical=same)
        sr, ss = share(real), share(shuf)
        log(f"\n[{kind}.{LAYER}] N = {real['N']}, M = {real['M']}: every pile rebuilt from the up/hole tips and from the lit tips "
            f"({real['piles']:,} of {real['piles']:,}); order-free identical: {same}  ({time.time() - tk:.0f}s, {rss()} MB)")
        log(f"   plain {real['plain']:,}; up/hole meetings {real['uh_meetings']:,} = {real['uh_meetings'] / real['plain']:.2f}x plain;"
            f" signed digits {100 * real['signed'] / real['plain']:.2f}% of plain")
        log(f"   {'':5s} {'what':58s} {'% of up/hole':>12s} {'% of plain':>11s} {'shuffle, % of plain':>20s}")
        names = dict(E1="place-pair tips, up/hole (sum |N - 2d|)", E1w="  the same tips written as whole numbers (signed digits)",
                     E2="place-pair tips, lit (sign and 1s)", E2w="  the same written as whole numbers", E3u="diagonal tips, up/hole",
                     E3="diagonal tips, lit", E4u="the gate's place sums, up/hole, signed digits", E4="the gate's place sums, lit, signed digits",
                     E5="the pile's own signed digits")
        for k in ("E1", "E1w", "E2", "E2w", "E3u", "E3", "E4u", "E4", "E5"):
            log(f"   {k:5s} {names[k]:58s} {sr[k][0]:11.3f}% {sr[k][1]:10.3f}% {ss[k][1]:19.3f}%")
        g = jget(f"ctl|L{LAYER}|{kind}|group work, the real light"); gp = jget(f"plain|L{LAYER}|{kind}|meetings, plain")
        log(f"   the confirmed group machine at {kind}.{LAYER} (all 648 positions): {100 * g / gp:.2f}% of plain")
        lb, eb = np.array(real["lit_by_diag"]), np.array(real["E2_by_diag"])
        log("   lit tips by diagonal, % of that diagonal's lit meetings (top diagonal first): "
            + " ".join(f"{100 * e / l:.1f}" if l else "-" for l, e in zip(lb[::-1], eb[::-1])))
        del x, xs, w, pile, ps, po
    # ---------------- Part B ----------------
    log("\nTHE FLOOR RELATION'S PARTS (g = 12, the per-place rule; photons from the confirmed machine's layer-5 counts)")
    for kind in KINDS:
        w = MZ[f"{kind}.{LAYER}.w"].astype(np.int64); N, M = w.shape; G = -(-N // 12)
        up, chosen = PP.rule(w, [(kind, 0, M)], 12); upl = bin(int(up[0])).count("1"); assert (up == up[0]).all()
        reads = M * G * upl
        ph_up = jget(f"fr|L{LAYER}|{kind}|reads, up/hole places") + jget(f"fcw|L{LAYER}|{kind}|reads, up/hole places")
        ph_sd = jget(f"fr|L{LAYER}|{kind}|reads, signed-digit places") + jget(f"fcw|L{LAYER}|{kind}|reads, signed-digit places")
        tab = jget(f"fr|L{LAYER}|{kind}|half table (by the flip)"); wr = jget(f"fr|L{LAYER}|{kind}|ready piles written in signed digits")
        plain = jget(f"plain|L{LAYER}|{kind}|meetings, plain")
        vals, cnts = np.unique(w, return_counts=True); p = cnts / cnts.sum(); H = float(-(p * np.log2(p)).sum())
        Hs = []
        for j in range(0, M, max(1, M // 64)):
            q = np.unique(w[:, j], return_counts=True)[1] / N; Hs.append(float(-(q * np.log2(q)).sum()))
        Hn = float(np.mean(Hs))
        rec = dict(N=N, M=M, groups=G, uh_places=upl, signed_places=chosen[kind]["signed_places"], reads=reads,
                   reads_per_weight=reads / (N * M), photons_per_read=ph_up / reads, read_photons=ph_up, sd_photons=ph_sd,
                   table=tab, write=wr, plain=plain, bits_per_weight_plate=H, bits_per_weight_column_sample=Hn)
        RES["B"][kind] = rec
        log(f"  {kind:5s}: up/hole places {upl} of 16 (signed digits at {chosen[kind]['signed_places']}); reads a weight {reads / (N * M):.3f} "
            f"(= {upl}/12); photons a read {ph_up / reads:.2f}; reads {100 * ph_up / plain:.2f}% + signed-digit reads {100 * ph_sd / plain:.2f}% "
            f"+ table {100 * tab / plain:.2f}% + writing {100 * wr / plain:.2f}% of plain")
        log(f"         the plate's bits a weight: {H:.2f} (whole plate's values), {Hn:.2f} (one neuron's column; it has its own power of two)")
    # ---------------- Part C ----------------
    log("\nTHE METHOD'S SAMPLE: the confirmed group bank at g = 12 and at the relation's g (fixed in PREDICTION.txt)")
    GREL = dict(qkv=12, proj=11, fc=12, mproj=11)
    for kind in KINDS:
        w = MZ[f"{kind}.{LAYER}.w"].astype(np.int64); M = w.shape[1]
        x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[LAYER][POS]).astype(np.int64)
        direct = x @ w; plain = int((ones(x) @ ones(w).sum(1)).sum()); out = {}
        for g in sorted({12, GREL[kind]}):
            gp = GB.GPlate(w, [(kind, 0, M)], g, [1, GB.R.SLABS.index(kind), LAYER, g])
            pile, C = gp.read(x); bad = int((pile != direct).any(1).sum()); assert bad == 0, (kind, g, bad)
            tot = {k: int(v.sum()) for k, v in C.items()}; whole = sum(tot.values())
            nreads = M * gp.G * bin(int(gp.up[0])).count("1") * len(POS)
            adds_tab = int(gp.ch.sum()) * len(POS) if gp.anyup else 0
            adds = nreads + int(((x[:, gp.tline] != 0)).sum()) + adds_tab
            out[g] = dict(whole=whole, share=100 * whole / plain, items=tot, piles_differ=bad, reads=nreads, adds=adds,
                          adds_share=100 * adds / int(((x != 0).astype(np.int64) @ ones(w).sum(1)).sum()))
            log(f"  {kind:5s} g = {g:2d}: every pile identical ({len(POS) * M:,} of {len(POS) * M:,}); work {100 * whole / plain:.2f}% of plain"
                f" (items, % of plain: " + ", ".join(f"{k} {100 * v / plain:.2f}" for k, v in tot.items() if v) + f"); adds {out[g]['adds_share']:.2f}% of plain adds")
            del gp, pile
        RES["C"][kind] = dict(plain=plain, runs={str(k): v for k, v in out.items()}, relation_g=GREL[kind])
    json.dump(RES, open(f"{OUT}/tips_units.json", "w"))
    log(f"\ndone in {time.time() - t0:.0f}s, {rss()} MB")
