"""EXCESS ON A SAMPLE: what "the tips" (the excess above the flat) come to, counted in plain's own unit, on real GPT-2
plates and the machine's own light. Designer 1 (plate side first). Whole numbers only in every count; the tips are
formed by float32 sums of +1/-1/0 entries over at most 3,072 lines, which are whole numbers below 2^24 and so exact.

PREDICTION (written before the first run of this file; the session's, not his):
  - up/hole tips (FINDINGS sec. 5's count, every place pair alike): 2.9% of the 256 x N meetings at the 768-line
    plates, 1.44% at mproj (3,072 lines). Counted as photons a weight that is 0.029 x 256 = 7.4 a weight at 768 lines
    (about 23% of plain's ~32 a weight) and 3.7 a weight at mproj (about 11-12% of plain).
  - sign-and-1s tips (plain's own meetings, up and down paired inside each place pair): about 8% of plain at 768 lines,
    about 4% at 3,072.
  - signed-digit tips (both sides in signed digits, paired inside each place pair): about 6% of plain at 768 lines.
  - the same, paired inside each diagonal (all place pairs of one size together): about 2% of plain.
  - the shuffle control (each position's light shuffled across its lines, every value kept) within 0.1 point of the
    real light at every count, since these are coin-toss sizes set by the number of lit meetings.
  - every pile rebuilt from its tips exactly, in all three writings (asserted).

    python3 flat_design/1/excess_sample.py  -> flat_design/1/excess_sample.json
"""
import json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/1"
LAYER = 5; NPOS = 48
PLATES = ["qkv", "proj", "fc", "mproj"]
def ones(a): return np.bitwise_count(np.abs(a)).astype(np.int64)
def naf(a): a = np.abs(a); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def places_uh(*vs):
    """16 places, or 17 when a value sits at exactly 2^15 (the bend's output can)"""
    return 16 if max(int(np.abs(v).max()) for v in vs) < (1 << 15) else 17
def planes_uh(v, Q):
    """two's complement up/hole: place q in 0..Q-1 is +1 (photon up) or -1 (hole); the top place weighs -2^(Q-1)"""
    u = (v.astype(np.int64) & ((1 << Q) - 1))
    return [(2 * ((u >> q) & 1) - 1).astype(np.float32) for q in range(Q)]
def planes_sm(v):
    """sign and 1s: place q is sign x (bit q of |v|)"""
    a = np.abs(v.astype(np.int64)); s = np.sign(v).astype(np.int64)
    return [(s * ((a >> q) & 1)).astype(np.float32) for q in range(16)]
def planes_sd(v):
    """signed digits (non-adjacent): digit q in {-1,0,+1}; up to 17 places for |v| <= 2^15"""
    a = np.abs(v.astype(np.int64)); s = np.sign(v).astype(np.int64); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    return [(s * (((up >> q) & 1) - ((dn >> q) & 1))).astype(np.float32) for q in range(17)]

def tips(LX, LW):
    """all place-pair tips at once: (Pa x P x N) @ (N x Pb x M) -> tip[a, pos, b, neuron]"""
    A = np.concatenate(LX, 0); B = np.concatenate(LW, 1)
    T = A @ B
    return T.reshape(len(LX), LX[0].shape[0], len(LW), LW[0].shape[1])

def measure(x, w):
    P, N = x.shape; M = w.shape[1]
    pile = x.astype(np.int64) @ w.astype(np.int64)
    r = dict(P=P, N=N, M=M)
    r["plain"] = int(ones(x).sum(0) @ ones(w).sum(1))
    r["signed"] = int(naf(x).sum(0) @ naf(w).sum(1))
    r["pile_sd_digits"] = int(naf(pile).sum())
    # up/hole
    Q = places_uh(x, w); r["uh_places"] = Q
    T = tips(planes_uh(x, Q), planes_uh(w, Q)); assert np.abs(T).max() <= N
    c = np.array([1 << q for q in range(Q - 1)] + [-(1 << (Q - 1))], np.int64)
    acc = np.einsum("a,apbm,b->pm", c, T.astype(np.int64), c)
    U = 2 * x.astype(np.int64) + 1; V = 2 * w.astype(np.int64) + 1
    again = (acc - U.sum(1)[:, None] - V.sum(0)[None, :] + N) // 4
    assert np.array_equal(again, pile), "up/hole tips do not rebuild the piles"
    r["uh_meetings"] = int(Q * Q * N * P * M); r["uh_excess"] = int(np.abs(T).sum())
    # sign and 1s
    T = tips(planes_sm(x), planes_sm(w))
    c = np.array([1 << q for q in range(16)], np.int64)
    assert np.array_equal(np.einsum("a,apbm,b->pm", c, T.astype(np.int64), c), pile), "sign-and-1s tips do not rebuild"
    r["sm_excess"] = int(np.abs(T).sum())
    # signed digits, by place pair and by diagonal
    T = tips(planes_sd(x), planes_sd(w)).astype(np.int64)
    c = np.array([1 << q for q in range(17)], np.int64)
    assert np.array_equal(np.einsum("a,apbm,b->pm", c, T, c), pile), "signed-digit tips do not rebuild"
    r["sd_excess"] = int(np.abs(T).sum())
    D = np.zeros((33, P, M), np.int64)
    for a in range(17):
        for b in range(17): D[a + b] += T[a, :, b, :]
    r["sd_diag_excess"] = int(np.abs(D).sum())
    r["sd_diag_digits"] = int(naf(D).sum())          # each diagonal's net count added as one number
    r["sd_pair_digits"] = int(naf(T).sum())          # each place pair's net count added as one number
    return r

def main():
    t0 = time.time(); Z = np.load(f"{HERE}/base2_gpt2.npz"); out = {}
    rng = np.random.default_rng(20261008)
    for k in PLATES:
        Lall = np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r")
        pos = np.linspace(0, Lall.shape[1] - 1, NPOS).round().astype(int)
        x = np.asarray(Lall[LAYER, pos]).astype(np.int64); w = Z[f"{k}.{LAYER}.w"].astype(np.int64)
        real = measure(x, w)
        xs = np.stack([row[rng.permutation(len(row))] for row in x])        # shuffle control: same values, pairing broken
        shuf = measure(xs, w)
        out[f"{k}.{LAYER}"] = dict(real=real, shuffle=shuf)
        for tag, r in (("real", real), ("shuffle", shuf)):
            pl = r["plain"]; wts = r["P"] * r["N"] * r["M"]
            print(f"{k}.{LAYER} {tag:7s} N={r['N']:5d} plain {pl / wts:6.2f}/weight | signed {100 * r['signed'] / pl:5.2f}% | "
                  f"uh tips {100 * r['uh_excess'] / r['uh_meetings']:5.3f}% of uh meetings = {r['uh_excess'] / wts:5.2f}/weight = "
                  f"{100 * r['uh_excess'] / pl:5.2f}% of plain | sm tips {100 * r['sm_excess'] / pl:5.2f}% | "
                  f"sd tips {100 * r['sd_excess'] / pl:5.2f}% | sd diag {100 * r['sd_diag_excess'] / pl:5.2f}% | "
                  f"pair counts as numbers {100 * r['sd_pair_digits'] / pl:5.2f}% | diag counts as numbers "
                  f"{100 * r['sd_diag_digits'] / pl:5.3f}% | pile digits {100 * r['pile_sd_digits'] / pl:5.3f}%", flush=True)
    json.dump(out, open(f"{OUT}/excess_sample.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
