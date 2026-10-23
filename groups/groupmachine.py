"""THE GROUP MACHINE: WORK SHARED ACROSS LIGHT LINES, BY A TABLE OF GROUP PILES (the one thing the quick search of
2026-10-06 said was missed; PRIOR_ART.md's last section). Whole numbers only; every pile checked digit for digit.

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  Every exact sharing tried so far goes ACROSS OUTPUT NEURONS (the line-up, way 2: one line's weights lined up across
  its neurons). This one goes ACROSS LIGHT LINES:
   1. a token's light reaches a plate; take a RANDOM group of g light lines (random groups only, never by likeness);
   2. at one plate place, each neuron's plate lights some subset of those g lines; there are only 2^g subsets;
   3. by the flip (Babinet), each subset's pile is the group total minus its complement's pile, so only half need
      building;
   4. build those piles once per token (each from a neighbour by one add, Gray-code order); every neuron, at every
      place, then adds ONE ready pile (shifted to that place) instead of meeting its lit pixels one at a time.
  Plate signs, two ways, both run: (SM) the plate as a sign and its 1s: at each place an up-subset (lines whose weight
  is positive and lit there) and a down-subset (negative and lit there); (SD) the plate in signed digits: at each place
  an up-subset (+1 digits) and a down-subset (-1 digits). Where both are lit: two lookups, combined by one take-away.
  The light enters the table as whole numbers, as to16 gives them.

THE WORK COUNTED (photon meetings, the unit of every earlier result: lit places added)
  - table: each Gray step adds or takes away one light value: that value's signed digits. Full table: 2^g - 1 steps.
    Half table (the flip): the subsets without the group's last line by Gray steps over the other g-1 lines, then the
    group total by one more add (2^(g-1) adds).
  - reads: each (neuron, place, group) with any lit line: the ready pile added into the neuron's pile, shifted to the
    place: the ready pile's signed digits.
  - combines: where both up and down are lit: the down pile taken from the up pile: the down pile's signed digits.
  - flips (half table only): a lookup of a subset holding the group's last line (not the whole group) is made by
    T[G] - T[G\\S]: one more take-away, T[G\\S]'s signed digits.
  Beside the photons, the same items as a plain count of additions (each add or take-away of one whole number = 1).
  Against: plain photon meetings (lit light x lit plate per pair: ones(x) x ones(w)), signed digits (naf(x) x naf(w)),
  way 2 (lineup2/refs_main_counts.npz; qwen06b/banks_*.json; the slab line-up qwen_fringes2/report.log); and for the
  adds, the direct machine's adds: each lit plate place met by a lit light value adds that value once (1s, signed
  digits).

THE BASES
  GPT-2: base2.py's own plates (base2_gpt2.npz) and the machine's own light at every bank, every position of the 30
    refs texts (groups/capture_gpt2.py; checked against recording/C_ref/plain_refs.npz, 0 differ). EXACT here: every
    pile against plain_refs.npz's piles (base2's own run).
  Qwen3-0.6B: qwen/qwen_base2.npz's plates and qwen06b/light/ (capture.py, 613 positions). EXACT here: every slab's
    piles (q|k|v, o, gate|up, down) against capture's sha256 of qwen_base2's own piles, text by text. q, k, v share one
    light and so one table a token (one grouping per slab); gate and up likewise.
  The shuffled light's piles are checked against the direct product of the shuffled light (int64, gm.c's `direct`).

CONTROLS (each a separate run of this file)
  seed2    a second random grouping (another seed)
  shuffle  each position's light shuffled across its lines (same values, the pairing with the plate broken)
  order    order-free: the same groups taken in a random order and the lines in each group in a random order
RULE 13, CHECKED BEFORE THE RUN
  13.1 the light's 1s are not the measure; the work of the finished piles is, and every pile is checked exact.
  13.2 the base is the machine's own (base 2, every part as it is; the light is the machine's own light).
  13.3 the shares are not statistics that must return chance; the shuffle control is the test of whether any of the
       saving comes from the pairing of light and plate (predicted: none of it does).
  13.4 the definitions decide part of the answer: the empty-cell share and the pile sizes are fixed by the plate's
       digits and g alone, the read costs by the light's sizes. The run measures their size; the exactness check and
       the controls test the rest. The table's share is fixed by 2^g over the neurons that share one table.
  13.5 the prediction below marked THE SESSION'S PREDICTION is the task's text, the session's own.
  13.6 every plate, every position, both g, both plate forms; nothing cut short.

THE SESSION'S PREDICTION (written first, uncertain; the task's text)
  GPT-2 banks at g = 8 about 55-65% of plain photons, at g = 12 about 40-50% where the plate is wide enough to pay for
  its table (fc, qkv), table cost under 2% at g = 8; Qwen gate/up at g = 12 about 40%; adds fall to about a quarter at
  g = 8; the second seed within 1 point; estimate at 14B widths and g = 16 about 30%.

ANALYSIS BEFORE THE RUN (the session's, written before any count was made)
  The plate's digits: GPT-2 about 4.7 signed digits a weight over 16 places (6.3 1s); each neuron has its own power of
  two, so the top places are seldom lit and the low and middle places are lit about 1 time in 3 (signed digits) or 1
  in 2 (1s). A group of 8 lines then leaves almost no low or middle place empty: about 13-14 of 16 places are read a
  (neuron, group), 14-15 at g = 12. With signed digits, up and down each lit about 1 in 6 a line, so both are lit
  (a combine) at about 57% of the low places at g = 8, 78% at g = 12. A ready pile holds 1.7-2.2 lines' light, and
  the light's values are mostly far below 2^15 (the stream's few large lines set each position's power of two), so a
  ready pile's signed digits are about 4-5.
  So per (neuron, group of 8): about 13.5 reads x 4.3 + 8 combines x 3.8 = about 88 photons, 11 a pair, against the
  plain 32.3 a pair: about 30-40% of plain at g = 8 (lower than the session's 55-65%). At g = 12 the reads saturate
  (every place read, and combines at nearly every place), so g = 12 gains only a little: 25-35%, the table 4,096 x
  about 3.5 photons a group, over 768-3,072 neurons: 1-5% of plain (proj and mproj, 768 neurons, pay most).
  Plate as sign + 1s (SM): more lit places, more combines: about 5-10 points above signed digits.
  Adds: about 21 adds a (neuron, group of 8) against the direct machine's 8 x 4.7 = 38 (signed digits) or 8 x 6.3 = 50
  (1s): 40-57%, not a quarter.
  The flip (half table): the build saving is 2^(g-1) light values a group a token, shared by all the neurons, while
  about 1 lookup in 5 holds the last line and must be flipped, each a take-away of a whole pile: about 20 photons a
  (neuron, group). The half table costs more than it saves wherever a table serves more than about 2^(g-1)/5 neurons:
  at every plate here. (It would pay only on narrow plates at large g.)
  Second seed and shuffled light within 1 point (nothing here uses the pairing of a line with a plate row).
  Order-free: piles, reads and combines identical (a subset is a set); the table's count and the flips can change,
  since the Gray order decides which line is added most often and which line is the flip line.
  Qwen gate/up at g = 12: about 25-35%, the table under 1% (6,144 neurons share it).
  14B, g = 16: the reads per (neuron, group) saturate, so about the same share as g = 12 (25-35%); the table pays only
  where many neurons share it (gate/up), and costs several points at 5,120-wide outputs (q, o, down).

    python3 groups/groupmachine.py model=gpt2|qwen g=8|12|16 run=main|seed2|shuffle|order [layers=a-b] [forms=sd,sm]
    -> groups/out/<model>_g<g>_<run>[_L<a>-<b>].json
"""
import os, sys, json, time, ctypes, subprocess, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; GD = f"{HERE}/groups"; OUTD = f"{GD}/out"
ARG = dict(a.split("=") for a in sys.argv[1:])
MODEL, G_, RUN = ARG.get("model", "gpt2"), int(ARG.get("g", 8)), ARG.get("run", "main")
FORMS = ARG.get("forms", "sd,sm").split(",")
SEED = {"main": 1, "seed2": 2, "shuffle": 1, "order": 1}[RUN]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
NB = 16                                                         # places of the plate (|w| <= 2^15: signed digits fit in 16)

# ---------- gm.c ----------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{GD}/gm.c", f"{GD}/gm.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_ = ctypes.c_void_p, ctypes.c_int64
        _lib.group_machine.argtypes = [I_, I_, I_, I_, I_, P_, P_, P_, P_, P_, P_, P_]; _lib.group_machine.restype = None
        _lib.direct.argtypes = [I_, I_, I_, P_, P_, P_]; _lib.direct.restype = None
        _lib.nc.restype = ctypes.c_int
    return _lib
CNT = ["build_full", "build_full_n", "build_half", "build_half_n", "flipbuild", "flipbuild_n", "read", "read_n",
       "combine", "combine_n", "flip", "flip_n"]

def direct(x, w):
    x = np.ascontiguousarray(x, np.int32); w = np.ascontiguousarray(w, np.int16); P, N = x.shape; M = w.shape[1]
    out = np.empty((P, M), np.int64); lib().direct(P, N, M, x.ctypes.data, w.ctypes.data, out.ctypes.data); return out

# ---------- the plate's subsets (weight-only) ----------
def plate_masks(w, form):
    """(up, down) place masks of every weight: SD signed digits (no two lit side by side), SM the sign and the 1s"""
    w = np.asarray(w, np.int64); a = np.abs(w)
    if form == "sd":
        t = 3 * a; up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = w < 0
        up, dn = np.where(neg, dn, up), np.where(neg, up, dn)
    else:
        up, dn = np.where(w > 0, a, 0), np.where(w < 0, a, 0)
    assert not ((up | dn) >> NB).any() and not (up & dn).any()
    back = up - dn; assert np.array_equal(back, w), "the masks do not give the plate back"
    return up, dn

def grouping(N, g, key, order_rng=None):
    """a random grouping of the N light lines (a random line order cut into runs of g; the last run may be short).
    order_rng: the same groups, taken in a random order, each group's lines in a random order"""
    perm = np.random.default_rng(key).permutation(N); groups = [perm[i:i + g] for i in range(0, N, g)]
    if order_rng is not None:
        groups = [order_rng.permutation(gr) for gr in groups]; groups = [groups[i] for i in order_rng.permutation(len(groups))]
    goff = np.concatenate([[0], np.cumsum([len(gr) for gr in groups])]).astype(np.int64)
    return np.concatenate(groups).astype(np.int32), goff

def cells_of(up, dn, glines, goff):
    """every (group, neuron, place) with a lit line: u, d as masks over the group's lines (bit t = the group's line t).
    Returns the packed cells (by group, neuron, place), their offsets, and the weight-only counts."""
    N, M = up.shape; G = len(goff) - 1; U = np.zeros((G, M, NB), np.uint32); D = np.zeros((G, M, NB), np.uint32)
    sizes = np.diff(goff)
    for gk in np.unique(sizes):
        ks = np.flatnonzero(sizes == gk); lines = np.stack([glines[goff[k]:goff[k + 1]] for k in ks])   # (n, gk)
        for t in range(gk):
            ut, dt = up[lines[:, t]], dn[lines[:, t]]                                                    # (n, M)
            for b in range(NB):
                U[ks, :, b] |= (((ut >> b) & 1).astype(np.uint32) << t); D[ks, :, b] |= (((dt >> b) & 1).astype(np.uint32) << t)
    lit = (U | D) != 0
    q = np.nonzero(lit); u, d, b = U[q], D[q], q[2].astype(np.uint64)
    cells = np.ascontiguousarray(u.astype(np.uint64) | (d.astype(np.uint64) << 16) | (b << 32))
    coff = np.concatenate([[0], np.cumsum(lit.sum(2).ravel())]).astype(np.int64)
    pu, pd = np.bitwise_count(U), np.bitwise_count(D); gmax = int(sizes.max())
    hist = np.zeros((gmax + 1, gmax + 1), np.int64); np.add.at(hist, (pu[lit], pd[lit]), 1)
    W = dict(cells_all=int(G * M * NB), cells_lit=int(lit.sum()), both=int(((U != 0) & (D != 0)).sum()),
             up_lit=int((U != 0).sum()), dn_lit=int((D != 0).sum()), up_lines=int(pu.sum()), dn_lines=int(pd.sum()),
             lit_by_place=lit.sum((0, 1)).tolist(), hist_ud=hist.tolist())
    return cells, coff, W

def run_machine(x, glines, goff, cells, coff, M):
    x = np.ascontiguousarray(x, np.int32); P, N = x.shape; G = len(goff) - 1; gmax = int(np.diff(goff).max())
    pile = np.empty((P, M), np.int64); nc = lib().nc(); cnt = np.empty((P, nc), np.int64)
    lib().group_machine(P, N, M, G, gmax, x.ctypes.data, glines.ctypes.data, goff.ctypes.data, cells.ctypes.data,
                        coff.ctypes.data, pile.ctypes.data, cnt.ctypes.data)
    return pile, cnt

def baselines(x, w):
    """plain photon meetings, signed digits, pairs (lit light x neurons), the direct machine's adds"""
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); M = w.shape[1]
    ox, nx, lx = ones(x).sum(0), naf(x).sum(0), (x != 0).sum(0)                  # per line, summed over positions
    W1, WN = ones(w).sum(1), naf(w).sum(1)
    return dict(plain=int(ox @ W1), signed=int(nx @ WN), pairs=int(lx.sum() * M), adds_plain=int(lx @ W1), adds_signed=int(lx @ WN),
                light_sd=int(nx.sum()), light_lit=int(lx.sum()))

def plate_run(x, w, glines, goff, forms, ref_check):
    """both plate forms: the machine's piles checked (ref_check(pile) -> positions differing, or None when the check is
    made on the whole slab afterwards), its counts summed over positions. Returns the record and the piles."""
    r = dict(base=baselines(x, w)); M = w.shape[1]; piles = {}
    for form in forms:
        up, dn = plate_masks(w, form); cells, coff, Wst = cells_of(up, dn, glines, goff); del up, dn
        pile, cnt = run_machine(x, glines, goff, cells, coff, M); del cells, coff
        tot = cnt.sum(0); bad = ref_check(pile)
        r[form] = dict(counts={k: int(tot[i]) for i, k in enumerate(CNT)}, naf_by_size=tot[len(CNT):].tolist(), weight=Wst,
                       piles_differ=bad, pile_digest=digest(pile))
        assert bad in (0, None), f"{form}: {bad} positions differ"
        piles[form] = pile
    return r, piles

# ---------- the bases ----------
def gpt2_plates():
    M = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    LIGHT = {k: np.load(f"{GD}/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
    for l in range(12):
        for kind in ("qkv", "proj", "fc", "mproj"):
            yield l, kind, [kind], np.asarray(LIGHT[kind][l]), [np.asarray(M[f"{kind}.{l}.w"])], [REF[f"{kind}_pile_{l}"]]

QKINDS = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
          "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
def qwen_plates():
    Mz = np.load(f"{HERE}/qwen/qwen_base2.npz"); META = json.load(open(f"{HERE}/qwen06b/light/meta.json"))
    LIGHT = {s: np.load(f"{HERE}/qwen06b/light/{s}_light.npy", mmap_mode="r") for s in QKINDS}
    for l in range(28):
        for slab, kinds in QKINDS.items():
            w = np.asarray(Mz[f"{slab}.{l}.w"])
            yield l, slab, [k for k, _, _ in kinds], np.asarray(LIGHT[slab][l]), [w[:, a:b] for _, a, b in kinds], (META["rows"], META["dig"][f"{slab}.{l}"])

if __name__ == "__main__":
    t0 = time.time(); os.makedirs(OUTD, exist_ok=True)
    LAY = ARG.get("layers"); L0, L1 = (int(v) for v in LAY.split("-")) if LAY else (0, 99)
    tag = f"{MODEL}_g{G_}_{RUN}" + (f"_L{L0}-{L1}" if LAY else "")
    LOG = open(f"{OUTD}/{tag}.log", "w")
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"THE GROUP MACHINE: {MODEL}, g = {G_}, run {RUN} (grouping seed {SEED}), plate forms {FORMS}, layers {L0}..{min(L1, 27)}")
    res = {}; srng = np.random.default_rng([77, G_])
    for l, slab, kinds, x, ws, ref in (gpt2_plates() if MODEL == "gpt2" else qwen_plates()):
        if not (L0 <= l <= L1): continue
        tl = time.time(); N = x.shape[1]; key = [SEED, ["qkv", "proj", "fc", "mproj", "o", "gu", "down"].index(slab), l, G_]
        glines, goff = grouping(N, G_, key, order_rng=np.random.default_rng([7] + key) if RUN == "order" else None)
        if RUN == "shuffle":
            x = srng.permuted(x, axis=1); assert x.shape[1] == N
        slab_piles = {f: [] for f in FORMS}
        for ki, (kind, w) in enumerate(zip(kinds, ws)):
            if RUN == "shuffle":                                    # the shuffled light: against its own direct product
                dref = direct(x, w); check = lambda pile, dref=dref: int((pile != dref).any(1).sum())
            elif MODEL == "gpt2":                                   # against base2's own run (plain_refs.npz)
                check = lambda pile, rr=ref[ki]: int((pile != rr).any(1).sum())
            else:                                                   # Qwen: the whole slab against capture's digests, below
                check = lambda pile: None
            r, piles = plate_run(x, w, glines, goff, FORMS, check); res[f"{kind}.{l}"] = r
            for f in FORMS: slab_piles[f].append(piles[f])
            del piles
        if MODEL == "qwen" and RUN != "shuffle":
            rows, dig = ref
            for form in FORMS:
                sp = np.concatenate(slab_piles[form], 1)
                bad = sum(digest(sp[a:b]) != dig[t] for t, (a, b) in enumerate(rows)); del sp
                for kind in kinds: res[f"{kind}.{l}"][form]["piles_differ"] = int(bad)      # texts differing
                assert bad == 0, (slab, l, form, bad)
        del slab_piles
        msg = " ".join(f"{k}: " + "/".join(f"{100 * (res[f'{k}.{l}'][f]['counts']['read'] + res[f'{k}.{l}'][f]['counts']['combine']) / res[f'{k}.{l}']['base']['plain']:.1f}"
                                          for f in FORMS) for k in kinds)
        log(f"  layer {l:2d} {slab:5s} exact; reads + combines, % of plain ({'/'.join(FORMS)}): {msg}  ({time.time() - tl:.0f}s, {rss()} MB)")
    json.dump(dict(model=MODEL, g=G_, run=RUN, seed=SEED, forms=FORMS, plates=res), open(f"{OUTD}/{tag}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
