"""A LOOK (not a test of his sentence): WHICH WRITES CANCEL WHICH, at the same position. 2026-10-06.

Why: totals.py part B found the finished machine's own writes add to Q = 0.980 (70m) and 0.742 (160m) of their summed
squares, against the blank's 0.980 and 0.976, while the attention writes and the MLP writes line up (r +0.70 and +0.46),
and taking each write from another position raises Q to 1.342 and 1.139. So something cancels at the same position.
Each write w_k (the word row k = 0, then attention and MLP at every layer) is split into its STANDING part s_k (its
average over the 622 positions, lane by lane: what every position gets) and its WORD part t_k = w_k - s_k. Summed over
positions the cross terms of s with t vanish, so
     Q - 1 = [ sum over pairs k != k' of 622 <s_k, s_k'>  +  sum over pairs of sum over positions <t_k, t_k'> ] / sum |w|^2
and the look reports both parts, every pair's share of Q - 1, each write's share of the summed squares, and the
largest cancelling and lining-up pairs. Products in float64 (a look in real numbers; the exact Q is totals.py's and
check.py's), from capture.py's whole-number writes.

GUESS, written before the first run: the word parts cancel mostly between the late layers' MLP writes and the earlier
writes (the late layers take back what the early ones wrote for the word); the standing parts line up mostly among the
MLP writes. The blank: both parts within +-0.03 of 0.

    python3 look_writes.py size=160m
"""
import os, sys, json, numpy as np
A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "160m")
OUT = os.path.dirname(os.path.abspath(__file__))
LOGF = open(os.path.join(OUT, f"look_writes_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
res = {}
for run in ("start", "finished"):
    W = np.load(os.path.join(OUT, f"light_{SIZE}_{run}.npz"))["writes"].astype(np.float64); P, K, d = W.shape; L_ = (K - 1) // 2
    names = ["word"] + [f"{k}{l}" for l in range(L_) for k in ("A", "M")]
    S = W.mean(0); T = W - S[None]
    CS = P * (S @ S.T); CT = np.einsum("pkd,pjd->kj", T, T); C = CS + CT
    tot = np.trace(C); off = lambda M: (M.sum() - np.trace(M)) / tot
    Q = C.sum() / tot
    log(f"\n== {SIZE} {run}: Q {Q:.3f} = 1 + standing pairs {off(CS):+.3f} + word pairs {off(CT):+.3f};  standing share of the summed squares {np.trace(CS) / tot:.3f}")
    log("  each write's share of the summed squares: " + " ".join(f"{n} {C[i, i] / tot:.3f}" for i, n in enumerate(names)))
    pr = [(2 * CT[i, j] / tot, 2 * CS[i, j] / tot, names[i], names[j]) for i in range(K) for j in range(i + 1, K)]
    log("  word parts, most cancelling pairs (share of Q - 1): " + "  ".join(f"{a}-{b} {x:+.3f}" for x, _, a, b in sorted(pr)[:8]))
    log("  word parts, most lining-up pairs:                   " + "  ".join(f"{a}-{b} {x:+.3f}" for x, _, a, b in sorted(pr)[::-1][:5]))
    log("  standing parts, most lining-up pairs:               " + "  ".join(f"{a}-{b} {y:+.3f}" for _, y, a, b in sorted(pr, key=lambda t: -t[1])[:6]))
    log("  standing parts, most cancelling pairs:              " + "  ".join(f"{a}-{b} {y:+.3f}" for _, y, a, b in sorted(pr, key=lambda t: t[1])[:5]))
    # each write against everything written before it at the same position (word parts), as a share of Q - 1
    before = [2 * CT[k, :k].sum() / tot for k in range(K)]
    log("  word part of each write against all earlier writes:  " + " ".join(f"{n} {b:+.3f}" for n, b in zip(names, before)))
    res[run] = dict(Q=Q, standing_pairs=off(CS), word_pairs=off(CT), share=[C[i, i] / tot for i in range(K)], before=before, names=names)
json.dump(res, open(os.path.join(OUT, f"look_writes_{SIZE}.json"), "w"))
