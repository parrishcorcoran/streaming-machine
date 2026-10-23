"""The flat is the balanced places. Each gate's weights in size + sign; from the top place down, the first place where
the gate's 1s are balanced (share of 1s within coin-flip of 50%: 3 x 0.5/sqrt(n)) starts the flat. That place and
everything below is removed, keeping its zero point (the cut is rounded at that place, as quantization does). A weight
with nothing left above the flat is n/a. Fixed rule, no search. Every plate cut together; the word table untouched.
Control: the flat replaced by fresh coin flips with the same balance (the tips kept)."""
import sys, json, time, numpy as np
src = open("../base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
sys.argv = ["x"]; import os; os.chdir(".."); NS = {}; exec(src, NS); os.chdir("flatcut")
M, W0 = NS["M"], dict(NS["W"])
rng = np.random.default_rng(1)
def cut(w16, control=False):
    A = np.abs(w16.astype(np.int64)); S = np.sign(w16.astype(np.int64)); n = A.shape[0]
    band = 3 * 0.5 / np.sqrt(n)
    share = np.stack([((A >> p) & 1).mean(0) for p in range(15)])          # (15 places, gates)
    flat = np.zeros(A.shape[1], np.int64)                                  # first balanced place from the top, per gate
    for g in range(A.shape[1]):
        p = 14
        while p >= 0 and abs(share[p, g] - 0.5) > band: p -= 1
        flat[g] = p + 1                                                    # places 0..flat-1 are removed
    step = np.left_shift(1, flat)[None, :]
    if control:   # keep tips exactly, replace the flat by fresh coin flips
        tip = A - (A % step); out = tip + rng.integers(0, 2**31, A.shape) % step
    else:         # remove the flat, keep its zero point: round at the flat's top place
        out = ((A + step // 2) // step) * step
    return (S * out), flat
def run(Wcut):
    NS["W"].clear(); NS["W"].update(Wcut); fw = NS["forward"]
    refs = json.load(open("../refs_many.json")); same = 0; first = 0
    for r in refs:
        for i in range(16):
            got = fw(r["ids"] + r["ref"][:i]); same += got == r["ref"][i]; first += (i == 0 and got == r["ref"][0])
    return same, first
t = time.time(); base = run(W0); print(f"uncut machine: {base[0]}/480 words, first word {base[1]}/30  ({time.time()-t:.0f}s)", flush=True)
for control in [False, True]:
    Wc = dict(W0); kept_bits = []; na = []; flats = []
    for k in W0:
        if k == "head": continue
        w, e, b = W0[k]; c, flat = cut(w.astype(np.int64), control); Wc[k] = (c.astype(np.float64), e, b)
        if not control:
            A = np.abs(c) >> flat[None, :]                                # what is stored: the tip above the flat
            nz = A > 0; L = np.where(nz, np.floor(np.log2(np.maximum(A, 1))) + 1, 0)
            kept_bits.append(np.where(nz, L + 1, 0).ravel()); na.append((~nz).ravel()); flats.append(flat)
    if not control:
        kb = np.concatenate(kept_bits); n = np.concatenate(na); f = np.concatenate(flats)
        print(f"flat starts at place {np.median(f):.0f} of 15 (median over gates; middle half {np.percentile(f,25):.0f}-{np.percentile(f,75):.0f})")
        print(f"tips: n/a {n.mean()*100:.1f}% of weights | kept weights {kb[~n].mean():.2f} places each (sign included) | "
              f"whole plates {kb.mean():.2f} places a weight + 1 flag")
    t = time.time(); s = run(Wc)
    print(f"{'CONTROL flat -> fresh coin flips' if control else 'FLAT REMOVED'}: {s[0]}/480 words, first word {s[1]}/30  ({time.time()-t:.0f}s)", flush=True)
