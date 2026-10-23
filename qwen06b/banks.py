"""THE EXACT CUTS FOUND ON GPT-2, MEASURED AT QWEN3-0.6B'S PLATES AS SHARES OF THE PLAIN WORK.

THE BASE. The all-base-2 Qwen3-0.6B (qwen/qwen_base2.py, its whole numbers in qwen/qwen_base2.npz) and its own light,
captured by qwen06b/capture.py on the 30 refs texts in Qwen's words (613 positions; the harness gave check30.log's words
exactly). Every plate: 16-place whole numbers, one power of two per neuron, the water level's gain folded in (q, k, v,
gate, up). Per layer seven plates (lines x neurons): q 1024 x 2048, k and v 1024 x 1024, o 2048 x 1024, gate and up
1024 x 3072, down 3072 x 1024. Every count is in the unit of all GPT-2 results: photon meetings, on each line the
light's photons x the plate's photons, summed over the lines (plain: the 1s of |x| times the 1s of |w|).

WHAT IS MEASURED AT EVERY PLATE, EVERY POSITION (exact checks first: every cut's piles = the direct piles, digit for digit)
 1. UP/DOWN FORM (updown_machine.py, fringes/fringe_map.py): every place of every value is a photon up (+1) or a hole
    (-1); offset 2^(n-1), D = 2v + 1, n = 16 places (17 for the one light value at exactly 2^15, see capture.log). For
    every meeting (position x neuron x light place a x plate place b), d = lines where they disagree; the straight-
    through part is N - |N - 2d| (the pixels that pair off), the fringe |N - 2d|. CHECK: the pile from the counts,
    1/4 [sum 2^(a+b)(N - 2d) - token stack - neuron stack + N], equals the direct pile. CONTROL: each position's light
    lines in their own random order (same values, pairing broken). ORDER-FREE (layers 0, 14, 27): the lines in one
    random order on both sides, every count the same.
 2. SIGNED DIGITS (no two lit side by side) on the weights and on the light: meetings = lit x lit per line. CHECK: the
    pile summed place pair by place pair from the signed planes equals the direct pile.
 3. WAY 2 (lineup2/way2.py): each line's weights lined up across its neurons once; per token, the walk r_k = r_(k-1) +
    x gap_k, poured into the neuron at rank k. Gap meetings: lit digits of the light x lit digits of each gap (the
    first rank's whole lowest weight included); pours: ones(r_k) for every running product; the light written in
    signed digits, ones(x). CHECK: the walk's piles equal the direct piles at every position. ORDER-FREE (every third
    layer): lines walked in a random order and DOWN from the top weight, the walk closing at 0: identical piles and
    pours. SHUFFLE CONTROL: the gaps' lit digits with each line's neurons in a random order (count only); and the light
    shuffled across its lines (walked, piles checked against its own direct product).
 5. WEIGHT-ONLY, per plate: lit digits per weight (1s, signed digits), dark weights, way 2's gap lit digits per weight
    and its zero gaps (ties), way 1's (each neuron's weights lined up down its lines), and the gaps against neurons a
    line: each line's neurons drawn at random, 128, 256, ..., up to the plate's own count, lined up, gap lit digits per
    weight.

RULE 13, CHECKED BEFORE THE RUN
  - 13.1 nothing here counts the light's 1s as the measure; the work counts are the unit, and the exactness of every
    cut is checked on the piles.
  - 13.2 the base is the machine's own (base 2, every part as it is); no part changed.
  - 13.3 the straight-through share is a statistic whose shuffle control keeps every value: it MUST return chance (the
    coin-toss value C(N, N/2)/2^N), and section 5 showed it does on GPT-2. It is measured as asked, as a law of the
    width, not as agreement.
  - 13.4 two definitions decide part of the answer: the signed-digit share follows from each number's own digits, and
    the pours are fixed by the light and the plate alone (r_k = x w whatever the order). The run measures their size;
    only the walk depends on the line-up, and its shuffle control is the test.
  - 13.5 the task's prediction is labelled as the task's; the session's own below.

PREDICTIONS (written before any run of this file)
  The task's: signed digits about 58-60% (fixed per weight); way 2's walk per pair falls with the neurons per line (1,024
  and 3,072 here, so about GPT-2's), the pour stays about 10; the straight-through share follows the coin-toss value.
  The session's:
  - Exact: every cut's piles equal the direct piles at all 613 positions at all 196 plates (up/down, signed place
    pairs, the walk, the walk down in a random line order); the slab piles equal the capture's digests.
  - 1: straight-through = 1 - coin toss at N lines: q, k, v, gate, up (1,024 lines) 97.51%; o (2,048) 98.24%; down
    (3,072) 98.56%. The fringe share within 0.97-1.10 x the coin-toss value at every kind, the control within 0.1 point
    of the real light.
  - 2: signed digits 56-62% of plain at every kind (GPT-2: 57.6-59.6%).
  - 3: Qwen's stored weights are bf16, 8 significant places. Every line's weights are then 8-place numbers (o, down)
    or 8-place numbers times one gain (q, k, v, gate, up: the gain is one number per line), so neighbours in rank are
    often tied and the nonzero gaps are short. So the walk falls BELOW GPT-2's at the same width (GPT-2: 9.2-9.9 at 768,
    7.0 at 2,304, 6.4 at 3,072): walk per pair o, down 2-5; k, v 4-8; q 3-6; gate, up 2.5-5. Pours per pair 7-10.5.
    Plain per pair lower than GPT-2's 32 (the weights have fewer 1s): 18-28. Way 2 40-60% of plain, below signed digits
    at gate, up and q by 3-10 points, about equal at k, v, o, down.
  - Shuffle control: gaps in a random neuron order 55-70% of plain; the light shuffled across lines leaves every share
    within 1-2 points (the saving is the line-up, not which line the light meets).
  - 5: 1s per weight: folded plates 6-7.5, o and down 3.5-4.5; signed/plain per weight 0.70-0.80; way 2's zero gaps
    20-60% at 2,048-3,072 neurons a line; gap lit digits per weight fall by 0.2-0.4 per doubling of the neurons a line.
    python3 qwen06b/banks.py L0 L1      (layers L0..L1-1; writes qwen06b/banks_L0_L1.json)
"""
import os, sys, json, time, ctypes, subprocess, resource, hashlib, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")           # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen06b"; NPZ = f"{HERE}/qwen/qwen_base2.npz"
KINDS = [("q", "qkv", 0, 2048), ("k", "qkv", 2048, 3072), ("v", "qkv", 3072, 4096), ("o", "o", 0, 1024),
         ("gate", "gu", 0, 3072), ("up", "gu", 3072, 6144), ("down", "down", 0, 1024)]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def signed_digits(v):                                          # (up, down) place masks; no two lit side by side
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def direct(x, w):                                              # the direct pile, whole numbers (exact: below 2^47)
    p = np.asarray(x, np.float64) @ np.asarray(w, np.float64); assert np.abs(p).max(initial=0) < 2 ** 52
    return p.astype(np.int64)

# ---------- the walk (walk.c) ----------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/walk.c", f"{OUT}/walk.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O2", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.walk.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, i_, I_, P_, P_, P_]; _lib.walk.restype = None
    return _lib
class Lineup:
    """way 2's record of one plate: each line's weights in order of size across its neurons (ties in neuron order)"""
    def __init__(self, w):
        w = np.asarray(w); L, N = w.shape; self.L, self.N = L, N
        self.NR = np.empty((L, N), np.uint32); self.G = np.zeros((L, N), np.uint16)
        self.LOW = np.empty(L, np.int64); self.TOP = np.empty(L, np.int64); self.GN = np.empty(L, np.int64); self.ZG = np.empty(L, np.int64)
        for i0 in range(0, L, 256):
            wb = w[i0:i0 + 256].astype(np.int64); o = np.argsort(wb, axis=1, kind="stable")
            ws = np.take_along_axis(wb, o, 1); g = np.diff(ws, axis=1)
            assert (g >= 0).all() and g.max(initial=0) < 1 << 16
            back = np.empty_like(wb); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
            assert np.array_equal(back, wb), "the record does not give the plate back"
            self.NR[i0:i0 + 256] = o; self.G[i0:i0 + 256, 1:] = g; self.LOW[i0:i0 + 256] = ws[:, 0]; self.TOP[i0:i0 + 256] = ws[:, -1]
            self.GN[i0:i0 + 256] = naf(ws[:, 0]) + naf(g).sum(1); self.ZG[i0:i0 + 256] = (g == 0).sum(1)
    def walk(self, x, line_order=None, down=False, blk=4):
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P, L = x.shape; assert L == self.L and np.abs(x).max(initial=0) <= 1 << 15
        LO = np.ascontiguousarray(np.arange(L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        pile = np.zeros((P, self.N), np.int64); p1 = np.zeros(P, np.int64); op = np.zeros(P, np.int64)
        lib().walk(P, L, self.N, x.ctypes.data, self.LOW.ctypes.data, self.G.ctypes.data, self.NR.ctypes.data, self.TOP.ctypes.data,
                   LO.ctypes.data, -1 if down else 1, 1, blk, pile.ctypes.data, p1.ctypes.data, op.ctypes.data)
        return pile, p1, int(op.sum())

# ---------- up/down form and signed planes, place pair by place pair ----------
def updown(x, w, pc=96):
    """for every meeting: d (lines where light place a and plate place b disagree). Returns the fringe sum per (a, b)
    (17 x 16; place 16 of the light is used only by a value at exactly 2^15), the meetings, the piles from the counts."""
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); P, N = x.shape; out = w.shape[1]
    Uw = w + (1 << 15); assert Uw.min() >= 0 and Uw.max() < 1 << 16
    Wp = np.empty((16, N, out), np.float32)
    for b in range(16): Wp[b] = (Uw >> b) & 1
    popW = Wp.sum(1, dtype=np.float64).astype(np.float32); del Uw
    ws_ = 2 * w.sum(0) + N
    FR = np.zeros((17, 16), np.float64); meet = 0; PILE = np.zeros((P, out), np.int64)
    big = np.abs(x).max(1) >= 1 << 15
    for rows in ([np.flatnonzero(~big)[i:i + pc] for i in range(0, int((~big).sum()), pc)] + [np.flatnonzero(big)] * bool(big.any())):
        xr = x[rows]; n = 17 if big[rows].any() else 16
        U = xr + (1 << (n - 1)); assert U.min() >= 0 and U.max() < 1 << n
        Xp = np.empty((n, len(rows), N), np.float32)
        for a in range(n): Xp[a] = (U >> a) & 1
        popX = Xp.sum(2)                                                                    # (n, rows)
        Xs = Xp.reshape(n * len(rows), N); base = (N - 2 * popX).reshape(-1, 1).astype(np.float32)
        pow2 = (2.0 ** np.arange(n))[None, :]; S = np.zeros((len(rows), out), np.float64)
        for b in range(16):
            T = Xs @ Wp[b]; T *= 4; T += base; T -= 2 * popW[b][None, :]                    # N - 2d, exact in float32
            FR[:n, b] += np.abs(T).reshape(n, len(rows), out).sum(axis=(1, 2), dtype=np.float64)
            S += (pow2 @ T.reshape(n, -1)).reshape(len(rows), out) * float(1 << b)
        assert np.abs(S).max() < 2 ** 53
        V = S.astype(np.int64) - (2 * xr.sum(1) + N)[:, None] - ws_[None, :] + N
        assert (V % 4 == 0).all(); PILE[rows] = V // 4; meet += n * 16 * len(rows) * out
    return FR, meet, PILE
def signed_pile(x, w, pc=96):
    """the pile summed place pair by place pair from the signed-digit planes of the light and of the plate"""
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); P, N = x.shape; out = w.shape[1]
    up, dn = signed_digits(w); assert not ((up | dn) >> 16).any()
    Wp = np.empty((16, N, out), np.float32); back = np.zeros((N, out), np.int64)
    for b in range(16):
        Wp[b] = ((up >> b) & 1) - ((dn >> b) & 1); back += Wp[b].astype(np.int64) << b
    assert np.array_equal(back, w); del back, up, dn
    PILE = np.zeros((P, out), np.int64); pow2 = (2.0 ** np.arange(16))[None, :]
    for r0 in range(0, P, pc):
        xr = x[r0:r0 + pc]; ux, dx = signed_digits(xr); assert not ((ux | dx) >> 16).any()
        Xp = np.empty((16, len(xr), N), np.float32); back = np.zeros(xr.shape, np.int64)
        for a in range(16):
            Xp[a] = ((ux >> a) & 1) - ((dx >> a) & 1); back += Xp[a].astype(np.int64) << a
        assert np.array_equal(back, xr)
        Xs = Xp.reshape(16 * len(xr), N); S = np.zeros((len(xr), out), np.float64)
        for b in range(16):
            S += (pow2 @ (Xs @ Wp[b]).reshape(16, -1)).reshape(len(xr), out) * float(1 << b)
        assert np.abs(S).max() < 2 ** 53; PILE[r0:r0 + pc] = S.astype(np.int64)
    return PILE

def subsample_gaps(w, sizes, rng):
    """weight-only: each line's neurons drawn at random (S of them), lined up; gap lit digits per weight (first rank's
    whole weight included) and the share of zero gaps"""
    w = np.asarray(w, np.int64); L, N = w.shape; res = {}
    for S in sizes:
        if S > N: continue
        lit = 0; zero = 0
        for i0 in range(0, L, 256):
            wb = w[i0:i0 + 256]; idx = rng.permuted(np.tile(np.arange(N), (len(wb), 1)), axis=1)[:, :S]
            ws = np.sort(np.take_along_axis(wb, idx, 1), 1); g = np.diff(ws, axis=1)
            lit += int(naf(ws[:, 0]).sum() + naf(g).sum()); zero += int((g == 0).sum())
        res[S] = dict(lit_per_weight=lit / (L * S), zero_gaps=zero / (L * (S - 1)))
    return res

if __name__ == "__main__":
    L0, L1 = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    Mz = np.load(NPZ); META = json.load(open(f"{OUT}/light/meta.json")); ROWS = META["rows"]
    LIGHT = {s: np.load(f"{OUT}/light/{s}_light.npy", mmap_mode="r") for s in ("qkv", "o", "gu", "down")}
    P = META["npos"]; res = {}
    print(f"BANKS layers {L0}..{L1 - 1}: {P} positions", flush=True)
    for l in range(L0, L1):
        tl = time.time(); slab_ok = {}
        for slab in ("qkv", "o", "gu", "down"):                    # the slab's direct piles against the capture's digests
            x = np.asarray(LIGHT[slab][l]).astype(np.int64); pile = direct(x, Mz[f"{slab}.{l}.w"])
            slab_ok[slab] = all(digest(pile[a:b]) == META["dig"][f"{slab}.{l}"][t] for t, (a, b) in enumerate(ROWS)); del pile
        for kind, slab, c0, c1 in KINDS:
            rng = np.random.default_rng([2026, l, ["q", "k", "v", "o", "gate", "up", "down"].index(kind)])
            w = np.asarray(Mz[f"{slab}.{l}.w"][:, c0:c1]); x = np.asarray(LIGHT[slab][l]).astype(np.int64)
            N, out = w.shape; dp = direct(x, w); r = dict(slab_digest_ok=slab_ok[slab], N=N, out=out)
            ox, nx = ones(x), naf(x); W1, WN = ones(w), naf(w)
            r["pairs"] = int(((x != 0).sum(1) * out).sum()); r["plain"] = int((ox @ W1.sum(1)).sum()); r["signed"] = int((nx @ WN.sum(1)).sum())
            r["writing"] = int(ox.sum())
            # weight-only (5)
            r["w_ones"] = int(W1.sum()); r["w_naf"] = int(WN.sum()); r["w_dark"] = int((w == 0).sum()); r["w_count"] = int(w.size)
            lu = Lineup(w); r["w_gap_lit"] = int(lu.GN.sum()); r["w_zero_gaps"] = int(lu.ZG.sum())
            os_ = rng.permuted(np.tile(np.arange(out), (N, 1)), axis=1); wsh = np.take_along_axis(w.astype(np.int64), os_, 1)
            GNS = naf(wsh[:, 0]) + naf(np.diff(wsh, axis=1)).sum(1); del os_, wsh
            r["w_gap_lit_shuffled"] = int(GNS.sum())
            wc = np.sort(w.astype(np.int64), 0); g1 = np.diff(wc, axis=0)
            r["w1_gap_lit"] = int(naf(wc[0]).sum() + naf(g1).sum()); r["w1_zero_gaps"] = int((g1 == 0).sum()); del wc, g1
            r["subsample"] = subsample_gaps(w, [128, 256, 512, 1024, 2048, 3072], rng)
            # way 2 (3): the walk, every position
            r["gaps"] = int((nx @ lu.GN).sum()); r["gaps_shuffled"] = int((nx @ GNS).sum())
            pw, pours, _ = lu.walk(x); r["walk_piles_differ"] = int((pw != dp).any(1).sum()); r["pours"] = int(pours.sum()); del pw
            if l % 3 == 0:                                         # order-free: random line order, walked down
                pw2, pours2, opn = lu.walk(x, line_order=rng.permutation(N), down=True)
                r["orderfree_walk_piles_differ"] = int((pw2 != dp).any(1).sum()); r["orderfree_pours_equal"] = bool(pours2.sum() == pours.sum()); r["orderfree_open"] = opn; del pw2
            xs = rng.permuted(x, axis=1); assert (np.sort(xs, 1) == np.sort(x, 1)).all()        # CONTROL: light shuffled across lines
            oxs, nxs = ones(xs), naf(xs)
            r["ctl_plain"] = int((oxs @ W1.sum(1)).sum()); r["ctl_signed"] = int((nxs @ WN.sum(1)).sum()); r["ctl_gaps"] = int((nxs @ lu.GN).sum())
            dps = direct(xs, w); pws, pourss, _ = lu.walk(xs); r["ctl_walk_piles_differ"] = int((pws != dps).any(1).sum()); r["ctl_pours"] = int(pourss.sum()); del pws
            del lu
            # up/down (1): real, control, order-free
            FR, meet, pu = updown(x, w); r["ud_fringe_ab"] = FR.tolist(); r["ud_fringe"] = float(FR.sum()); r["ud_meetings"] = meet
            r["ud_piles_differ"] = int((pu != dp).any(1).sum()); del pu
            FRc, meetc, puc = updown(xs, w); r["ctl_ud_fringe_ab"] = FRc.tolist(); r["ctl_ud_fringe"] = float(FRc.sum()); r["ctl_ud_meetings"] = meetc
            r["ctl_ud_piles_differ"] = int((puc != dps).any(1).sum()); del puc, dps
            if l in (0, 14, 27):
                perm = rng.permutation(N); FRo, meeto, puo = updown(x[:, perm], w[perm])
                r["orderfree_ud_same"] = bool(np.array_equal(FRo, FR) and meeto == meet and np.array_equal(puo, dp)); del puo
            # signed digits (2): the pile place pair by place pair
            r["signed_piles_differ"] = int((signed_pile(x, w) != dp).any(1).sum())
            res[f"{kind}.{l}"] = r
            del w, x, xs, dp
        print(f"  layer {l:2d}: slabs = capture {all(slab_ok.values())}; "
              + " ".join(f"{k}: ud {res[f'{k}.{l}']['ud_piles_differ']}/sd {res[f'{k}.{l}']['signed_piles_differ']}/walk {res[f'{k}.{l}']['walk_piles_differ']}" for k, *_ in KINDS)
              + f"  ({time.time() - tl:.0f}s, {rss()} MB)", flush=True)
    json.dump(res, open(f"{OUT}/banks_{L0}_{L1}.json", "w"))
    print(f"done layers {L0}..{L1 - 1} ({time.time() - t0:.0f}s, max memory {rss()} MB)")
