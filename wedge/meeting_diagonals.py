"""THE MEETING AS A GRADIENT (exact). A bank's pile is light meeting plate, line by line. Write every value in base 2
with its sign as its direction: each 1 of the light at place a is a photon going the light's way, each 1 of the plate
at place b a photon going the plate's way. A meeting of a light photon and a plate photon is a photon pair at
diagonal k = a + b, going up when the two directions agree and down when they differ. Then

    pile = sum over diagonals k of 2^k x B_k,     B_k = (up pairs - down pairs) on diagonal k, summed over all 768
                                                   (or 3,072) lines

The triangle: diagonal k holds the pairs a + b = k, and each diagonal counts double the one below.

Checked against the machine's own piles (captured from the machine's own product call) on fc.0 (neurons in, layer 0)
and mproj.6 (neurons out, layer 6), every position of all 30 refs texts (prompt + 16 reference words), np.array_equal.
Counted: how many diagonals carry a nonzero balance, per pile and per bank; the work in photon pairs (up + down),
which is the line-by-line meeting regrouped, so unchanged; and how much of it the up/down balance on each diagonal
leaves (sum of |B_k|).
CONTROL: the same light against the same plate with its lines shuffled (every plate value kept, only which line meets
which light changes); same counts.
    python3 wedge/meeting_diagonals.py
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum is a whole number checked below
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M, W = J.M, J.W
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
BANKS = ("fc.0", "mproj.6")
WTE, WPE = M["wte"], M["wpe"]

# capture the machine's own piles: the light it put on 16 bits and the pile its product returned
_orig_product = J.product; CAP = {}
def _recording_product(x16, w):
    p = _orig_product(x16, w)
    for n in BANKS:
        if w is W[n][0]: CAP[n] = (np.asarray(x16).astype(np.int64).copy(), p.copy())
    return p
J.product = _recording_product
def run_to_mproj6(ids):                              # J.scores_all, step for step, stopped after layer 6's neurons out
    P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(7):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return {n: CAP[n] for n in BANKS}

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A): return _POP[np.abs(np.asarray(A, np.int64))]   # photons per value (every value here is below 2^16 in size)
PLACES = 16
def planes(A):                                       # place a: +1 where a photon goes up, -1 where it goes down, 0 dark
    A = np.asarray(A, np.int64); s = np.sign(A); m = np.abs(A)
    return [(s * ((m >> a) & 1)).astype(np.float32) for a in range(PLACES)]

def diagonals(x16, wplanes):                         # B_k for every pile: sum over a + b = k of (light place a) . (plate place b)
    xp = planes(x16); P, N = x16.shape[0], wplanes[0].shape[1]
    B = np.zeros((2 * PLACES - 1, P, N), np.int64)
    for a in range(PLACES):
        if not xp[a].any(): continue
        for b in range(PLACES):
            if not wplanes[b].any(): continue
            B[a + b] += (xp[a] @ wplanes[b]).astype(np.int64)    # each entry a whole number of size <= lines: exact in float32
    return B

def pair_counts(x16, w):                             # up pairs and down pairs, counted line by line
    ox, ow = ones(x16).astype(np.float64), ones(w).astype(np.float64)
    xp, xn = ox * (x16 > 0), ox * (x16 < 0); wp, wn = ow * (w > 0), ow * (w < 0)
    up = (xp @ wp + xn @ wn).astype(np.int64); down = (xp @ wn + xn @ wp).astype(np.int64)
    return up, down

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(11)
    caps = []
    for ids in TEXTS: caps.append(run_to_mproj6(ids))
    print(f"machine run on {len(TEXTS)} refs texts ({sum(len(t) for t in TEXTS)} positions), piles captured ({time.time() - t0:.0f}s)", flush=True)
    for n in BANKS:
        w = M[n + ".w"].astype(np.int64); wpl = planes(w)
        perm = rng.permutation(w.shape[0]); wsh = w[perm]; wpl_sh = planes(wsh)
        same = 0; nz_hist = np.zeros(2 * PLACES, np.int64); nz_any = np.zeros(2 * PLACES - 1, bool)
        nz_hist_c = np.zeros(2 * PLACES, np.int64); nz_any_c = np.zeros(2 * PLACES - 1, bool)
        tot_up = tot_down = tot_bal = tot_bal_c = tot_pairs_c = 0; npiles = 0; amax = 0; balsum_ok = True
        for cap in caps:
            x16, pile = cap[n]
            B = diagonals(x16, wpl)
            rebuilt = np.zeros_like(pile)
            for k in range(B.shape[0]): rebuilt += B[k] << k
            same += np.array_equal(rebuilt, pile)
            up, down = pair_counts(x16, w); balsum_ok &= np.array_equal(B.sum(0), up - down)
            nz = (B != 0).sum(0); nz_hist += np.bincount(nz.ravel(), minlength=2 * PLACES); nz_any |= (B != 0).reshape(B.shape[0], -1).any(1)
            tot_up += int(up.sum()); tot_down += int(down.sum()); tot_bal += int(np.abs(B).sum()); npiles += pile.size
            amax = max(amax, int(np.abs(x16).max()).bit_length())
            # CONTROL: plate lines shuffled
            Bc = diagonals(x16, wpl_sh); rc = np.zeros_like(pile)
            for k in range(Bc.shape[0]): rc += Bc[k] << k
            assert np.array_equal(rc, _orig_product(x16, wsh.astype(np.float64)))   # the shuffled meeting, rebuilt the same way
            nzc = (Bc != 0).sum(0); nz_hist_c += np.bincount(nzc.ravel(), minlength=2 * PLACES); nz_any_c |= (Bc != 0).reshape(Bc.shape[0], -1).any(1)
            uc, dc = pair_counts(x16, wsh); tot_pairs_c += int(uc.sum() + dc.sum()); tot_bal_c += int(np.abs(Bc).sum())
        bmax = int(np.abs(w).max()).bit_length()
        line_by_line = tot_up + tot_down
        mean_nz = (nz_hist * np.arange(2 * PLACES)).sum() / npiles; mean_nz_c = (nz_hist_c * np.arange(2 * PLACES)).sum() / npiles
        print(f"\n{n}: plate {w.shape[0]} lines x {w.shape[1]} neurons; {npiles:,} piles (positions x neurons)")
        print(f"  pile = sum_k 2^k B_k, identical to the machine's pile (np.array_equal): {same}/{len(caps)} texts")
        print(f"  sum over k of B_k = up pairs - down pairs, line by line, every pile: {balsum_ok}")
        print(f"  places: light up to place {amax - 1}, plate up to place {bmax - 1}  ->  diagonals k = 0..{amax + bmax - 2} ({amax + bmax - 1} possible)")
        print(f"  diagonals with a nonzero balance anywhere in the bank: {int(nz_any.sum())}  (k = {np.flatnonzero(nz_any).min()}..{np.flatnonzero(nz_any).max()})")
        print(f"  per pile, diagonals with a nonzero balance: mean {mean_nz:.2f}, fewest {np.flatnonzero(nz_hist).min()}, most {np.flatnonzero(nz_hist).max()}")
        print(f"     how many piles have j nonzero diagonals: " + ", ".join(f"{j}: {c:,}" for j, c in enumerate(nz_hist) if c))
        print(f"  work in photon pairs: up {tot_up:,} + down {tot_down:,} = {line_by_line:,}  ({line_by_line / npiles:,.0f} per pile)")
        print(f"     the same pairs, regrouped by diagonal: unchanged (each pair sits on exactly one diagonal a + b)")
        print(f"  left after up and down cancel on each diagonal, sum of |B_k|: {tot_bal:,} = {100 * tot_bal / line_by_line:.2f}% of the pairs  ({tot_bal / npiles:,.0f} per pile)")
        print(f"  CONTROL, plate lines shuffled (same values, same light): nonzero diagonals per pile mean {mean_nz_c:.2f} "
              f"(fewest {np.flatnonzero(nz_hist_c).min()}, most {np.flatnonzero(nz_hist_c).max()}), anywhere {int(nz_any_c.sum())}; "
              f"pairs {tot_pairs_c:,} (other photons meet now, so a different count); sum |B_k| {100 * tot_bal_c / tot_pairs_c:.2f}% of the pairs", flush=True)
    print(f"\nmax memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
