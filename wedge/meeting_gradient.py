"""THE MEETING AS A GRADIENT (exact).

A bank's pile is the light meeting the plate, line by line. Write every value in base 2 with its sign as its
direction: a 1 of the light at place a is a photon going the light's way, a 1 of the plate at place b a photon going
the plate's way. A light photon and a plate photon on the same line meet as a photon pair on diagonal k = a + b; the
pair goes up when the two directions agree and down when they differ. Then, for every pile,

    pile = sum over diagonals k of 2^k x B_k,   B_k = up pairs - down pairs on diagonal k, summed over all lines

The triangle: diagonal k holds the place pairs a + b = k, and each diagonal counts double the one below it.

Checked against the machine's own piles (captured from the machine's own product call, nothing recomputed) on
fc.0 (layer 0, into the 3,072 neurons) and mproj.6 (layer 6, out of the 3,072 neurons), every position of all 30
refs texts (prompt + the 16 reference words), with np.array_equal.

Counted:
  - diagonals that carry a nonzero balance B_k (per pile, and anywhere in the bank), and diagonals that hold any
    pair at all (U_k + D_k > 0)
  - the work in photon pairs: line by line (ones of the light x ones of the plate, per line) and regrouped by
    diagonal (sum over k of U_k + D_k); these must be equal, each pair sits on exactly one diagonal
  - what the balance leaves after up and down cancel on each diagonal: sum over k of |B_k|
CONTROL: the same light against the same plate with its lines shuffled (every plate value kept, only which line meets
which light changes); the same counts.

    python3 wedge/meeting_gradient.py
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # spurious flags from the float matmul; every sum is checked whole below
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M, W = J.M, J.W
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
BANKS = ("fc.0", "mproj.6")
WTE, WPE = M["wte"], M["wpe"]
PLACES = 16                                           # light on 16 bits: |x16| <= 2^15; plate int16: |w| <= 2^15
NDIAG = 2 * PLACES - 1                                # k = 0 .. 30

# ---- capture the machine's own piles: the light it put on 16 bits and the pile its own product returned ----
_orig_product = J.product; CAP = {}
def _recording_product(x16, w):
    p = _orig_product(x16, w)
    for n in BANKS:
        if w is W[n][0]: CAP[n] = (np.asarray(x16).astype(np.int64).copy(), p.copy())
    return p
J.product = _recording_product                        # gauged_bank / plain_bank look product up in J's globals
def run_through_layer6(ids):                          # J.scores_all, step for step, stopped after layer 6's out-bank
    CAP.clear(); P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(7):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return {n: CAP[n] for n in BANKS}

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):                                          # photons per value, direction apart
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
def planes(A):                                        # place a: +1 photon going up, -1 going down, 0 dark
    A = np.asarray(A, np.int64); s = np.sign(A); m = np.abs(A)
    assert m.max() < (1 << PLACES)
    return [(s * ((m >> a) & 1)).astype(np.float32) for a in range(PLACES)]

def diagonals(x16, wpl):
    """B[k] = up - down pairs on diagonal k, T[k] = up + down pairs on diagonal k, for every pile.
    Each entry is a whole number no bigger than lines x 16 (< 2^24): exact in float32."""
    xp = planes(x16); P, N = x16.shape[0], wpl[0].shape[1]
    B = np.zeros((NDIAG, P, N), np.int64); T = np.zeros((NDIAG, P, N), np.int64)
    for a in range(PLACES):
        if not xp[a].any(): continue
        xa = np.abs(xp[a])
        for b in range(PLACES):
            if not wpl[b].any(): continue
            B[a + b] += (xp[a] @ wpl[b]).astype(np.int64)
            T[a + b] += (xa @ np.abs(wpl[b])).astype(np.int64)
    return B, T

def line_by_line_pairs(x16, w):                       # up pairs, down pairs, counted line by line
    ox, ow = ones(x16).astype(np.float64), ones(w).astype(np.float64)
    xp, xn = ox * (x16 > 0), ox * (x16 < 0); wp, wn = ow * (w > 0), ow * (w < 0)
    return (xp @ wp + xn @ wn).astype(np.int64), (xp @ wn + xn @ wp).astype(np.int64)

def rebuild(B):
    out = np.zeros(B.shape[1:], np.int64)
    for k in range(B.shape[0]): out += B[k] << k
    return out

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(20261004)
    caps = [run_through_layer6(ids) for ids in TEXTS]
    J.product = _orig_product
    print(f"machine run on {len(TEXTS)} refs texts ({sum(len(t) for t in TEXTS)} positions), its own piles captured on "
          f"{', '.join(BANKS)} ({time.time() - t0:.0f}s)", flush=True)
    for n in BANKS:
        w = M[n + ".w"].astype(np.int64); wpl = planes(w)
        perm = rng.permutation(w.shape[0]); wsh = w[perm]; wpl_sh = planes(wsh)
        S = dict(same=0, pairs_ok=True, bal_ok=True, up=0, down=0, diag_pairs=0, absbal=0, piles=0,
                 nzB=np.zeros(NDIAG + 1, np.int64), nzT=np.zeros(NDIAG + 1, np.int64), anyB=np.zeros(NDIAG, bool), anyT=np.zeros(NDIAG, bool),
                 c_same=0, c_pairs=0, c_absbal=0, c_nzB=np.zeros(NDIAG + 1, np.int64), c_anyB=np.zeros(NDIAG, bool))
        xmax = 0
        for cap in caps:
            x16, pile = cap[n]; xmax = max(xmax, int(np.abs(x16).max()))
            B, T = diagonals(x16, wpl)
            S["same"] += np.array_equal(rebuild(B), pile)
            up, down = line_by_line_pairs(x16, w)
            S["bal_ok"] &= np.array_equal(B.sum(0), up - down)
            S["pairs_ok"] &= np.array_equal(T.sum(0), up + down)
            S["up"] += int(up.sum()); S["down"] += int(down.sum()); S["diag_pairs"] += int(T.sum()); S["absbal"] += int(np.abs(B).sum())
            S["piles"] += pile.size
            S["nzB"] += np.bincount((B != 0).sum(0).ravel(), minlength=NDIAG + 1)
            S["nzT"] += np.bincount((T != 0).sum(0).ravel(), minlength=NDIAG + 1)
            S["anyB"] |= (B != 0).reshape(NDIAG, -1).any(1); S["anyT"] |= (T != 0).reshape(NDIAG, -1).any(1)
            # CONTROL: plate lines shuffled
            Bc, Tc = diagonals(x16, wpl_sh)
            S["c_same"] += np.array_equal(rebuild(Bc), _orig_product(x16, wsh.astype(np.float64)))
            S["c_pairs"] += int(Tc.sum()); S["c_absbal"] += int(np.abs(Bc).sum())
            S["c_nzB"] += np.bincount((Bc != 0).sum(0).ravel(), minlength=NDIAG + 1); S["c_anyB"] |= (Bc != 0).reshape(NDIAG, -1).any(1)
            del B, T, Bc, Tc
        wmax = int(np.abs(w).max())
        def stats(h): j = np.flatnonzero(h); return (h * np.arange(len(h))).sum() / h.sum(), j.min(), j.max()
        mB, loB, hiB = stats(S["nzB"]); mT, loT, hiT = stats(S["nzT"]); mC, loC, hiC = stats(S["c_nzB"])
        lbl = S["up"] + S["down"]
        print(f"\n{n}: plate {w.shape[0]} lines x {w.shape[1]} neurons; {S['piles']:,} piles (positions x neurons)")
        print(f"  light places 0..{xmax.bit_length() - 1} (largest |light| {xmax}), plate places 0..{wmax.bit_length() - 1} "
              f"(largest |plate| {wmax}) -> diagonals k = 0..{NDIAG - 1} ({NDIAG} possible)")
        print(f"  pile = sum_k 2^k B_k equals the machine's own pile (np.array_equal): {S['same']}/{len(caps)} texts")
        print(f"  sum_k B_k = up - down pairs line by line, every pile: {S['bal_ok']};  sum_k (U_k + D_k) = up + down line by line, every pile: {S['pairs_ok']}")
        print(f"  diagonals carrying a nonzero balance: anywhere in the bank {int(S['anyB'].sum())} of {NDIAG} "
              f"(k = {np.flatnonzero(S['anyB']).min()}..{np.flatnonzero(S['anyB']).max()}); per pile mean {mB:.2f}, fewest {loB}, most {hiB}")
        print(f"     piles by number of nonzero-balance diagonals: " + ", ".join(f"{j}: {c:,}" for j, c in enumerate(S['nzB']) if c))
        print(f"  diagonals holding any pair at all: anywhere {int(S['anyT'].sum())}; per pile mean {mT:.2f}, fewest {loT}, most {hiT}")
        print(f"  work in photon pairs: line by line up {S['up']:,} + down {S['down']:,} = {lbl:,}  ({lbl / S['piles']:,.0f} per pile)")
        print(f"     regrouped by diagonal: {S['diag_pairs']:,}  -> unchanged: {S['diag_pairs'] == lbl} (each pair sits on exactly one diagonal a + b)")
        print(f"  left after up and down cancel on each diagonal, sum_k |B_k|: {S['absbal']:,} = {100 * S['absbal'] / lbl:.2f}% of the pairs")
        print(f"  CONTROL, plate lines shuffled (same plate values, same light): rebuilt = shuffled machine product {S['c_same']}/{len(caps)}; "
              f"nonzero-balance diagonals anywhere {int(S['c_anyB'].sum())}, per pile mean {mC:.2f} (fewest {loC}, most {hiC}); "
              f"pairs {S['c_pairs']:,}; sum |B_k| = {100 * S['c_absbal'] / S['c_pairs']:.2f}% of the pairs", flush=True)
        del wpl, wpl_sh
    print(f"\nmax memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
