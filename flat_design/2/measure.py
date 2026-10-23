"""FLAT_DESIGN/2/MEASURE: designer 2, the light side first. A SAMPLE, not the whole machine. Whole numbers only.
Predictions are in PREDICTION.txt (written before this file ran).

Part A, the excess (the tips), every 8th position of the 648 refs positions, plates qkv/proj/fc/mproj at layers 3 and 9:
  three writings of the same pile, each place pair (a of the light, b of the plate) one tip = sum over the lines of the
  two places' signs:  up/hole (every place a photon up or a hole; 16 x 16 places),  sign and 1s (plain's own lit
  photons),  signed digits (17 x 17 places). Each writing must rebuild every pile exactly (asserted).
  Reported: sum |tip| as a share of that writing's own meetings, and as a share of plain photon meetings.
  Shuffle control: each position's light shuffled across its lines (every value kept).

Part B, the group machine and the one method, every position of texts 15-29 (the test texts), same 8 plates:
  - the group machine's form rebuilt (up/hole, g = 12, groupmachine.py's own groups: seed 1, key [1, kind, layer, 12]),
    counted as gm_uh.c counts it (half table by Gray steps, every read = the ready pile's signed digits at all 16
    places, X summed once a position and taken from each pile) plus the ready piles written once in signed digits
    (their 1s, as base2_groups.py rule 6 counts it);
  - the plate's sign run read once (count only): places above the group's largest weight hold the same pattern up to
    the flip, so they collapse to one read;
  - THE FLAT HELD ON BOTH SIDES: S = each line's rounded median of the light over texts 0-14 (fixed before texts 15-29
    are read); e = x - S met through the same group table; S . w_j stored, added once a neuron (its signed digits);
  - where the cancelling left after the table sits: per neuron and plate place, the photons of the group reads
    against the signed digits of their own sum; per neuron, the place sums against the pile;
  - shuffle control: the light shuffled across lines at every position (texts 0-14 too), S rebuilt the same way;
  - order-free: the groups and the lines inside each group in a random order: every pile and every read count the
    same (the Gray build may differ; reported).
Every pile is checked against the direct product x @ w (int64).
    python3 flat_design/2/measure.py   -> flat_design/2/measure.json, measure.log
"""
import os, sys, json, time, resource, subprocess, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/2"
G = 12; KINDS = ["qkv", "proj", "fc", "mproj"]; LAYERS = [3, 9]
LOG = None                                          # opened only when run as the main script
def log(s):
    print(s, flush=True)
    if LOG is not None: LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def sd(A): a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def wait_turn():                                   # one heavy python at a time (another over 1.5 GB: wait)
    while True:
        out = subprocess.run(["ps", "-axo", "pid,rss,comm"], capture_output=True, text=True).stdout.split("\n")[1:]
        big = [l for l in out if l.strip() and "python" in l.lower() and int(l.split()[0]) != os.getpid()
               and int(l.split()[1]) > 1_500_000]
        if not big: return
        log(f"  waiting: another heavy python is running ({big[0].strip()})"); time.sleep(60)

def grouping(N, g, key, order_rng=None):           # groupmachine.py's own rule, copied
    perm = np.random.default_rng(key).permutation(N); groups = [perm[i:i + g] for i in range(0, N, g)]
    if order_rng is not None:
        groups = [order_rng.permutation(gr) for gr in groups]; groups = [groups[i] for i in order_rng.permutation(len(groups))]
    return groups

# ---------------- Part A: the excess in three writings ----------------
def planes_uh(v, n):          # up/hole: v + 2^15 in binary; place a: +1 where 1, -1 where 0.  D = 2v + 1 = sum 2^a s_a
    c = np.asarray(v, np.int64) + (1 << 15); assert c.min() >= 0 and c.max() < (1 << 16)
    return [(2 * ((c >> a) & 1) - 1).astype(np.float32) for a in range(n)]
def planes_sm(v, n):          # sign and 1s: sign(v) x bit a of |v|
    v = np.asarray(v, np.int64); a_ = np.abs(v); s = np.sign(v)
    return [(s * ((a_ >> a) & 1)).astype(np.float32) for a in range(n)]
def planes_sd(v, n):          # signed digits (non-adjacent): +1 / -1 / 0 at each place
    v = np.asarray(v, np.int64); a_ = np.abs(v); t = 3 * a_; up, dn = (t & ~a_) >> 1, (a_ & ~t) >> 1
    neg = v < 0; up, dn = np.where(neg, dn, up), np.where(neg, up, dn)
    return [(((up >> a) & 1) - ((dn >> a) & 1)).astype(np.float32) for a in range(n)]

def excess(x, w, form):
    """sum |tip| over every pile and place pair; the pile rebuilt from the tips alone, checked exact"""
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); T, N = x.shape; Mn = w.shape[1]
    n = {"uh": 16, "sm": 16, "sd": 17}[form]; P = {"uh": planes_uh, "sm": planes_sm, "sd": planes_sd}[form]
    XP = P(x, n); WP = np.concatenate(P(w, n), axis=1)            # N x (n Mn)
    tot = 0; sdt = 0; nz = 0; acc = np.zeros((T, Mn), np.int64)
    for a in range(n):
        with np.errstate(all='ignore'): R = XP[a] @ WP             # exact: whole numbers below 2^24 (Accelerate's
        # matmul raises spurious floating warnings on this Mac; the pile rebuilt from the tips is asserted below)
        Ri = np.rint(R).astype(np.int64); tot += int(np.abs(Ri).sum()); sdt += int(sd(Ri).sum()); nz += int((Ri != 0).sum())
        for b in range(n): acc += Ri[:, b * Mn:(b + 1) * Mn] << (a + b)
    direct = x @ w
    if form == "uh":   # D_x . D_w = 4 x.w + 2 X + 2 Wsum + N
        X = x.sum(1)[:, None]; Ws = w.sum(0)[None, :]; back = (acc - 2 * X - 2 * Ws - N) // 4
    else: back = acc
    assert np.array_equal(back, direct), f"{form}: the tips do not rebuild the pile"
    meet = {"uh": 256 * N * T * Mn, "sm": int(ones(x).sum(0) @ ones(w).sum(1)), "sd": int(sd(x).sum(0) @ sd(w).sum(1))}[form]
    return dict(tips_abs=tot, tips_sd=sdt, tips_nonzero=nz, meetings=meet)

# ---------------- Part B: the group table ----------------
SMAT = None
def smat(g):                  # g x 2^g: +1 where bit t of the pattern is 1 (line up), -1 where 0 (hole)
    q = np.arange(1 << g); return (2 * ((q[None, :] >> np.arange(g)[:, None]) & 1) - 1).astype(np.int64)

def patterns(w, groups):      # (G, Mn, 16) up/hole patterns: bit t = 1 where line t's weight + 2^15 has a 1 at place b
    c = np.asarray(w, np.int64) + (1 << 15); Mn = c.shape[1]; PT = np.zeros((len(groups), Mn, 16), np.int64)
    for k, L in enumerate(groups):
        for t, i in enumerate(L):
            ci = c[i]
            for b in range(16): PT[k, :, b] |= ((ci >> b) & 1) << t
    return PT

def run_collapse_places(PT, gsz):
    """per (group, neuron): the lowest place c such that places c..14 all hold the complement of place 15's pattern
    (the sign run). Reads with the run read once = c + 1 (places 0..c-1, then one read of place 15's pile at place c)."""
    Gn, Mn, _ = PT.shape; c = np.full((Gn, Mn), 15, np.int64)
    for k in range(Gn):
        mask = (1 << gsz[k]) - 1; top = PT[k, :, 15]; comp = (~top) & mask; run = np.ones(Mn, bool)
        for b in range(14, -1, -1):
            run &= PT[k, :, b] == comp; c[k] = np.where(run, b, c[k])
    return c

def group_machine(x, w, groups, PT, cplace, want_cancel=False):
    """the up/hole group machine on light x: every pile exact; the counts in photons (signed digits) and in adds"""
    x = np.asarray(x, np.int64); T, N = x.shape; Mn = w.shape[1]
    pile2 = np.zeros((T, Mn), np.int64); pile2r = np.zeros((T, Mn), np.int64); cnt = dict(build=0, build_n=0, write=0, write_n=0, read=0, read_n=0,
                                                  read_run=0, read_run_n=0, xsum=0, xsum_n=0, off=0, off_n=0)
    Pb = np.zeros((T, Mn, 16), np.int64) if want_cancel else None; readsum_by_place = 0
    for k, L in enumerate(groups):
        g = len(L); S_ = smat(g); xl = x[:, L]; V = xl @ S_                       # T x 2^g, every pattern's pile
        sdl = sd(xl)                                                              # T x g
        # the half table (patterns with the group's last line up), as gm_uh.c: its first entry (all up) g-1 adds,
        # then 2^(g-1)-1 Gray steps over lines 0..g-2 (line t is stepped 2^(g-2-t) times)
        cnt["build"] += int(sdl[:, 1:].sum()) + int(sum((1 << (g - 2 - t)) * sdl[:, t].sum() for t in range(g - 1)))
        cnt["build_n"] += T * ((g - 1) + (1 << (g - 1)) - 1)
        half = V[:, (1 << (g - 1)):]                                              # the written half (last line up)
        cnt["write"] += int(ones(half).sum()); cnt["write_n"] += half.size
        R = V[:, PT[k]]                                                           # T x Mn x 16: every read
        sR = sd(R); cnt["read"] += int(sR.sum()); cnt["read_n"] += R.size
        # the sign run read once: places 0..c-1 one by one, then place 15's pile once at place c
        c = cplace[k]; below = np.arange(16)[None, :] < c[:, None]                # Mn x 16
        cnt["read_run"] += int((sR * below[None]).sum() + sR[:, :, 15].sum()); cnt["read_run_n"] += T * int(below.sum() + Mn)
        pile2 += (R << np.arange(16)[None, None, :]).sum(2)
        pile2r += ((R << np.arange(16)[None, None, :]) * below[None]).sum(2) + (R[:, :, 15] << c[None, :])
        if want_cancel: Pb += R
    X = x.sum(1); cnt["xsum"] += int(sd(x[:, 1:]).sum()); cnt["xsum_n"] += T * (N - 1)
    cnt["off"] += int(sd(X).sum()) * Mn; cnt["off_n"] += T * Mn
    assert np.array_equal(pile2r, pile2), "the sign run read once does not give the same pile"
    D = pile2 - X[:, None]; assert not (D & 1).any(); pile = D // 2
    return pile, cnt, Pb

def total(c, extra=0): return c["build"] + c["write"] + c["read"] + c["xsum"] + c["off"] + extra
def total_n(c, extra=0): return c["build_n"] + c["write_n"] + c["read_n"] + c["xsum_n"] + c["off_n"] + extra

if __name__ == "__main__":
    LOG = open(f"{OUT}/measure.log", "w"); t0 = time.time(); wait_turn()
    M = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    npos = REF["npos"].astype(int); off = np.concatenate([[0], np.cumsum(npos)]); NP = int(off[-1])
    hold = np.arange(off[0], off[15]); test = np.arange(off[15], off[30]); every8 = np.arange(0, NP, 8)
    log(f"positions: {NP}; held-out (texts 0-14) {len(hold)}; test (texts 15-29) {len(test)}; part A every 8th: {len(every8)}")
    LIGHT = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in KINDS}
    rng_sh = np.random.default_rng(11); res = dict(A={}, B={})
    for l in LAYERS:
        for ki, kind in enumerate(KINDS):
            name = f"{kind}.{l}"; tl = time.time()
            w = np.asarray(M[f"{kind}.{l}.w"]).astype(np.int64); xa = np.asarray(LIGHT[kind][l]).astype(np.int64)
            N, Mn = w.shape; assert xa.shape == (NP, N)
            perm = np.argsort(rng_sh.random((NP, N)), axis=1); xs = np.take_along_axis(xa, perm, 1)   # shuffled light
            # ---- part A ----
            x8, xs8 = xa[every8], xs[every8]; plain8 = int(ones(x8).sum(0) @ ones(w).sum(1)); A = dict(plain=plain8)
            for form in ("uh", "sm", "sd"):
                A[form] = excess(x8, w, form); A[form + "_shuffle"] = excess(xs8, w, form)
            A["plain_shuffle"] = int(ones(xs8).sum(0) @ ones(w).sum(1)); res["A"][name] = A
            log(f"{name}  A: excess as share of own meetings / of plain:  " + "  ".join(
                f"{f} {100 * A[f]['tips_abs'] / A[f]['meetings']:.2f}% / {100 * A[f]['tips_abs'] / plain8:.2f}%"
                f" (shuffled {100 * A[f + '_shuffle']['tips_abs'] / A[f + '_shuffle']['meetings']:.2f}% / "
                f"{100 * A[f + '_shuffle']['tips_abs'] / A['plain_shuffle']:.2f}%)" for f in ("uh", "sm", "sd")))
            # ---- part B ----
            groups = grouping(N, G, [1, ki, l, G]); gsz = [len(L) for L in groups]; PT = patterns(w, groups)
            cpl = run_collapse_places(PT, gsz)
            xt = xa[test]; plain_t = int(ones(xt).sum(0) @ ones(w).sum(1)); direct = xt @ w
            S = np.rint(np.median(xa[hold], axis=0)).astype(np.int64); e = xt - S
            Sw = S @ w                                                             # stored: one number a neuron
            pile_x, cx, Pb = group_machine(xt, w, groups, PT, cpl, want_cancel=True)
            assert np.array_equal(pile_x, direct), "group machine pile differs"
            pile_e, ce, _ = group_machine(e, w, groups, PT, cpl)
            assert np.array_equal(pile_e + Sw[None, :], direct), "the flat held on both sides: pile differs"
            stored_add = int(sd(Sw).sum()) * len(test)                            # S . w_j added once a neuron a token
            # where the cancelling sits (real light): per (pos, neuron, place) the reads' photons against sd(P_b);
            # per (pos, neuron) sum_b sd(P_b) against sd(x . D)
            sdPb = int(sd(Pb).sum()); xD = (Pb << np.arange(16)[None, None, :]).sum(2); sdxD = int(sd(xD).sum())
            nG = len(groups); del Pb
            # shuffle control: S from the shuffled held-out light, the same rule
            xst = xs[test]; Ss = np.rint(np.median(xs[hold], axis=0)).astype(np.int64); es = xst - Ss
            pile_xs, cxs, _ = group_machine(xst, w, groups, PT, cpl); assert np.array_equal(pile_xs, xst @ w)
            pile_es, ces, _ = group_machine(es, w, groups, PT, cpl); assert np.array_equal(pile_es + (Ss @ w)[None, :], xst @ w)
            # order-free: groups and lines inside each group in a random order
            og = grouping(N, G, [1, ki, l, G], order_rng=np.random.default_rng([7, ki, l]))
            PTo = patterns(w, og); cplo = run_collapse_places(PTo, [len(L) for L in og])
            pile_o, co, _ = group_machine(xt, w, og, PTo, cplo); assert np.array_equal(pile_o, direct)
            energy = lambda v: float((v.astype(np.float64) ** 2).sum())
            B = dict(N=N, M=Mn, T=len(test), groups=nG, plain=plain_t, plain_signed=int(sd(xt).sum(0) @ sd(w).sum(1)),
                     adds_plain=int((xt != 0).sum(0) @ ones(w).sum(1)), gm=cx, flat=ce, stored_add=stored_add,
                     gm_shuffle=cxs, flat_shuffle=ces, stored_add_shuffle=int(sd(Ss @ w).sum()) * len(test), order=co,
                     sd_light=int(sd(xt).sum()), sd_dep=int(sd(e).sum()), sd_light_sh=int(sd(xst).sum()), sd_dep_sh=int(sd(es).sum()),
                     energy_light=energy(xt), energy_dep=energy(e), energy_light_sh=energy(xst), energy_dep_sh=energy(es),
                     sd_place_sums=sdPb, sd_xD=sdxD, run_places_mean=float(cpl.mean()) + 1.0)
            res["B"][name] = B
            gmT, flT = total(cx), total(ce, stored_add)
            log(f"{name}  B: gm {100 * gmT / plain_t:.2f}% (read {100 * cx['read'] / plain_t:.2f}, table {100 * cx['build'] / plain_t:.2f}, "
                f"write {100 * cx['write'] / plain_t:.2f}; {cx['read'] / cx['read_n']:.2f} photons a read) | sign run read once: reads "
                f"{100 * cx['read_run'] / plain_t:.2f}% ({B['run_places_mean']:.2f} reads a neuron and group) | FLAT HELD BOTH SIDES "
                f"{100 * flT / plain_t:.2f}% = {100 * flT / gmT:.2f}% of gm (reads {100 * ce['read'] / cx['read']:.2f}% of gm's; "
                f"energy left {100 * B['energy_dep'] / B['energy_light']:.1f}%, digits left {100 * B['sd_dep'] / B['sd_light']:.1f}%) | "
                f"shuffle: flat/gm {100 * total(ces, B['stored_add_shuffle']) / total(cxs):.2f}%, energy left "
                f"{100 * B['energy_dep_sh'] / B['energy_light_sh']:.1f}% | order-free reads {'same' if co['read'] == cx['read'] else 'DIFFER'} | "
                f"cancel: group reads {cx['read']:,} vs their place sums {sdPb:,} ({100 * (1 - sdPb / cx['read']):.1f}% cancel), "
                f"place sums vs piles {sdxD:,}  ({time.time() - tl:.0f}s, {rss()} MB)")
            json.dump(res, open(f"{OUT}/measure.json", "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
