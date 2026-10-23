"""BASEDOWN/RECHECK_INDEP: an independent checker's recount of lineup.py at four plates it did not use for check.py
(qkv.8, proj.2, fc.6, mproj.11), in (a) each text and (b) all 648 together. Whole numbers only.
  - signed digits counted by writing each number's signed digits out one by one (not the a ^ 3a trick);
  - the gaps formed by sorting each side's magnitudes with zeros left in (a sort over the whole column, then a diff);
  - the landings counted by grouping equal light values on a line (each distinct value x multiplicity), not token by token;
  - plain counted with a shift-and-mask popcount for the light and the plate;
  - piles built by the walk (running product = previous + w x gap, landed into the token's pile) for BOTH soups,
    asserted equal to plain_refs and to x @ w; every running product asserted = w x the token's value;
  - order-free: the tokens shown in a random order (own seed), piles and counts asserted identical;
  - shuffle control at proj.2: lineup.py's own shuffle (seed [7,1,2]) recounted exactly, and a second shuffle with
    this checker's own seed.
    python3 basedown/recheck_indep.py -> basedown/out/recheck_indep.log
"""
import json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/basedown/out"
LOG = open(f"{OUT}/recheck_indep.log", "w"); T0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def pop_shift(A):
    """1s of |A| by shift and mask (whole numbers)"""
    a = np.abs(np.asarray(A, np.int64)); c = np.zeros(a.shape, np.int64)
    while a.any(): c += a & 1; a = a >> 1
    return c

def sd_digits(A):
    """count of nonzero signed digits of |A|, written out digit by digit (non-adjacent form)"""
    a = np.abs(np.asarray(A, np.int64)).copy(); c = np.zeros(a.shape, np.int64)
    while a.any():
        odd = (a & 1) == 1
        d = np.where(odd, 2 - (a & 3), 0)          # +1 if a = 1 mod 4, -1 if a = 3 mod 4
        a = (a - d) >> 1; c += odd
    return c

# self-test of the two counters against bin()
_t = np.arange(-70000, 70001)
assert all(int(pop_shift(v)) == bin(abs(int(v))).count("1") for v in _t[::97])
def _naf_ref(v):
    v = abs(v); n = 0
    while v:
        if v & 1: d = 2 - (v % 4); v -= d; n += 1
        v //= 2
    return n
assert all(int(sd_digits(v)) == _naf_ref(int(v)) for v in _t[::89])

REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = REF["npos"].astype(int); P = int(NPOS.sum())
st = np.concatenate([[0], np.cumsum(NPOS)])
SOUPS = {"a": [np.arange(st[i], st[i + 1]) for i in range(30)], "b": [np.arange(P)]}
Wz = np.load(f"{HERE}/base2_gpt2.npz")
LJ = {r["bank"]: r for r in json.load(open(f"{OUT}/lineup.json"))["rec"]}
log(f"RECHECK_INDEP: P = {P}, texts {len(NPOS)} ({NPOS.min()}-{NPOS.max()} tokens)")

def counts(x, nrow, soups):
    """cancel share parts, walk, nonzero gaps, by sorting each side's magnitudes per column (zeros kept in)"""
    v1 = g1 = vsd = gsd = walk = lit = nz = 0
    for rows in soups:
        V = x[rows]                                   # (p, N)
        for side in (np.where(V > 0, V, 0), np.where(V < 0, -V, 0)):
            S = np.sort(side, axis=0)                 # zeros first, then the magnitudes upward
            G = np.diff(np.vstack([np.zeros((1, S.shape[1]), np.int64), S]), axis=0)   # gaps (0 for zeros and repeats)
            sdg = sd_digits(G)
            walk += int((sdg.sum(0) * nrow).sum())
            g1 += int(pop_shift(G).sum()); gsd += int(sdg.sum())
            v1 += int(pop_shift(S).sum()); vsd += int(sd_digits(S).sum())
            lit += int((S != 0).sum()); nz += int((G != 0).sum())
    return dict(v1=v1, g1=g1, vsd=vsd, gsd=gsd, walk=walk, entries=lit, nzgap=nz)

def landings(x, w):
    """1s of every product w[j,n] x x[t,j], grouped by equal |x| on each line"""
    tot = 0; tot_sd = 0
    for j in range(x.shape[1]):
        u, c = np.unique(np.abs(x[:, j]), return_counts=True)
        keep = u != 0; u, c = u[keep], c[keep]
        if not len(u): continue
        prod = u[:, None] * np.abs(w[j])[None, :]
        tot += int((np.bitwise_count(prod).sum(1) * c).sum())
    return tot

def walk_piles(x, w, soups):
    """the walk: per soup, per line, per side, tokens in lined-up order; running = previous + w x gap; landed"""
    Pn, N = x.shape; M = w.shape[1]; pile = np.zeros((Pn, M), np.int64); bad = 0
    for rows in soups:
        V = x[rows]
        for j in range(N):
            v = V[:, j]
            for sgn in (1, -1):
                m = (v * sgn) > 0
                if not m.any(): continue
                t = rows[m]; mag = v[m] * sgn
                o = np.argsort(mag, kind="mergesort"); t = t[o]; mag = mag[o]
                gap = np.diff(np.concatenate([[0], mag])) * sgn            # signed gap along this side
                run = np.cumsum(gap[:, None] * w[j][None, :], axis=0)       # previous product + w x gap
                bad += int((run != x[t, j][:, None] * w[j][None, :]).any(1).sum())
                np.add.at(pile, t, run)
    return pile, bad

def plain(x, w):
    wrow = pop_shift(w).sum(1); return int((pop_shift(x) @ wrow).sum())

def pct(a, b): return 100.0 * a / b
FAIL = []
def same(label, mine, theirs):
    ok = mine == theirs
    if not ok: FAIL.append(label)
    return f"{label} {mine:,} vs {theirs:,} {'SAME' if ok else 'DIFFERENT'}"

for name in ("qkv.8", "proj.2", "fc.6", "mproj.11"):
    tb = time.time(); kind, l = name.split("."); l = int(l)
    x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l], np.int64)
    w = np.asarray(Wz[f"{name}.w"], np.int64); N, M = w.shape
    ref = np.asarray(REF[f"{kind}_pile_{l}"], np.int64)
    nrow = sd_digits(w).sum(1)
    dir_ = x @ w
    log(f"\n{name}: light {x.shape}, plate {w.shape}, |x| max {np.abs(x).max()}; x @ w rows differing from plain_refs: {int((dir_ != ref).any(1).sum())}")
    if (dir_ != ref).any(): FAIL.append(f"{name} direct")
    pl = plain(x, w); land = landings(x, w)
    r = LJ[name]
    log("  " + same("plain", pl, r["b"]["plain"]) + "; " + same("landings", land, r["b"]["land"]) + "; " + same("landings (a)", land, r["a"]["land"]))
    for sm in ("a", "b"):
        c = counts(x, nrow, SOUPS[sm]); t = r[sm]
        log(f"  ({sm}) cancel 1s {pct(c['v1'] - c['g1'], c['v1']):.2f}% (theirs {pct(t['v1'] - t['g1'], t['v1']):.2f}%), signed {pct(c['vsd'] - c['gsd'], c['vsd']):.2f}%;"
            f" walk {pct(c['walk'], pl):.2f}% of plain; landings {pct(land, pl):.2f}%; walk+landings {pct(c['walk'] + land, pl):.2f}%;"
            f" equal neighbours {pct(c['entries'] - c['nzgap'], c['entries']):.1f}%")
        for k in ("v1", "g1", "vsd", "gsd", "walk", "entries", "nzgap"):
            log("     " + same(f"({sm}) {k}", c[k], t[k]))
        pile, bad = walk_piles(x, w, SOUPS[sm])
        pd_ref = int((pile != ref).any(1).sum()); pd_dir = int((pile != dir_).any(1).sum())
        log(f"     ({sm}) walk piles differing from plain_refs {pd_ref}, from x @ w {pd_dir}; running products differing from w x x {bad}")
        if pd_ref or pd_dir or bad: FAIL.append(f"{name} ({sm}) piles")
        # order-free: tokens shown in a random order
        rng = np.random.default_rng([2026, 10, 7, l, len(kind)])
        perm = np.concatenate([rng.permutation(rows) for rows in SOUPS[sm]])
        xp = x[perm]; pp, badp = walk_piles(xp, w, SOUPS[sm]); back = np.empty_like(pp); back[perm] = pp
        cp = counts(xp, nrow, SOUPS[sm])
        of_ok = (back == ref).all() and badp == 0 and all(cp[k] == c[k] for k in c)
        log(f"     ({sm}) order-free (tokens in a random order): piles and counts {'IDENTICAL' if of_ok else 'DIFFERENT'}")
        if not of_ok: FAIL.append(f"{name} ({sm}) order-free")
    if name == "proj.2":
        # their shuffle, reproduced: rng = default_rng([7, SLAB.index(kind), l]); xs = rng.permuted(x, axis=1)
        for lab, seed in (("lineup.py's shuffle", [7, 1, l]), ("own shuffle", [99, 1, l])):
            xs = np.random.default_rng(seed).permuted(x, axis=1)
            assert (np.sort(xs, 1) == np.sort(x, 1)).all()
            pls = plain(xs, w); lands = landings(xs, w); dss = xs @ w
            msg = f"  SHUFFLE CONTROL ({lab}): plain {pls:,}"
            for sm in ("a", "b"):
                c = counts(xs, nrow, SOUPS[sm]); ps, bads = walk_piles(xs, w, SOUPS[sm])
                pdiff = int((ps != dss).any(1).sum())
                msg += (f" | ({sm}) cancel {pct(c['v1'] - c['g1'], c['v1']):.2f}% walk {pct(c['walk'], pls):.2f}% landings {pct(lands, pls):.2f}%"
                        f" whole(walk+land) {pct(c['walk'] + lands, pls):.2f}% piles vs own direct differ {pdiff}, bad {bads}")
                if pdiff or bads: FAIL.append(f"shuffle {lab} {sm} piles")
                if lab == "lineup.py's shuffle":
                    t = r[sm + "_ctl"]
                    for k in ("v1", "g1", "walk"):
                        if c[k] != t[k]: FAIL.append(f"ctl {sm} {k}")
                    msg += f" [theirs: cancel {pct(t['v1'] - t['g1'], t['v1']):.2f}% walk {pct(t['walk'], t['plain']):.2f}% land {t['land'] == lands} plain {t['plain'] == pls}]"
                    if t["land"] != lands or t["plain"] != pls: FAIL.append(f"ctl {sm} land/plain")
            log(msg)
    log(f"  ({time.time() - tb:.0f}s)")

# position 0: identical light across texts
x0 = np.asarray(np.load(f"{HERE}/groups/light_gpt2/qkv.npy", mmap_mode="r")[:, st[:-1]], np.int64)    # (12, 30, N)
for l in (0, 6, 11):
    _, inv, cnt = np.unique(x0[l], axis=0, return_inverse=True, return_counts=True)
    log(f"position 0 at qkv.{l}: {len(cnt)} distinct rows among 30 texts, largest group {cnt.max()}")
log(f"\nRECHECK {'PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} ({time.time() - T0:.0f}s)")
