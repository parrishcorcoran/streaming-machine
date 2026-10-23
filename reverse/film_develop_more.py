"""THE NEWEST FILM, STEP 4 ADDENDUM -- NOT EXACT: THIS TEST CHANGES VALUES. film_develop.log: with half of each layer's
newest bends developed from the light (largest |part| or at random) and the rest from the token-alone column, the
answer stays in the top 5 in 15/30 and the machine's top-5 set is kept in 0/30. Is the answer on nearly all the grains?
Same test (film_develop.py's lines, copied), k = 2765 (90%) and 3041 (99%) per layer, largest |part| and at random.

PREDICTION (Claude's, written before running): answer in top 5 -- 2765: 24/30, 3041: 29/30; top-5 set kept --
2765: 8/30, 3041: 22/30; random a little lower.
    python3 reverse/film_develop_more.py     -> reverse/film_develop_more.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def develop(ids, chosen, share_al):
    P = len(ids); t = P - 1; R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = gauged_bank(R, f"fc.{l}")
        idx = rshift(H, ES - 12) + 32768
        share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
        share[t] = np.where(chosen[l], share[t], share_al[l])
        R = R + plain_bank(rshift(H * share, 16), f"mproj.{l}")
    x16, ex, LS = gauge_log(R[t:t + 1])
    return power(product(x16, HEADW), (HEADE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + HEADB

def rank_of(V, i): return int((V > V[i]).sum())

t0 = time.time()
D = np.load("/Users/abundancemachine/streaming-machine/reverse/film_split.npz")
GN, BO, SHA = D["GboN"], D["Bout"], D["share_al"]
NT = GN.shape[0]; KS = [3041, 2765]
rng = np.random.default_rng(20261006)
R = {(m, k): dict(ans=[], first=[], set5=[]) for m in ("largest", "random") for k in KS}
for i in range(NT):
    ids = TEXTS[i]; S = machine_scores(ids); a = race(S); top5 = set(int(x) for x in np.argsort(-S, kind="stable")[:5])
    order = [sorted(range(3072), key=lambda n: -abs(int(GN[i, l, n]) * int(BO[i, l, n]))) for l in range(12)]
    line = []
    for k in KS:
        for m in ("largest", "random"):
            chosen = []
            for l in range(12):
                c = np.zeros(3072, bool)
                c[order[l][:k] if m == "largest" else rng.choice(3072, k, replace=False)] = True; chosen.append(c)
            V = develop(ids, chosen, SHA[i])
            r = rank_of(V, a); R[(m, k)]["ans"].append(r); R[(m, k)]["first"].append(r == 0)
            R[(m, k)]["set5"].append(set(int(x) for x in np.argsort(-V, kind="stable")[:5]) == top5)
            line.append(f"{k}{'r' if m == 'random' else ''}:{r}{'*' if R[(m, k)]['set5'][-1] else ''}")
    print(f"  text {i:2d} {word(a)!r:13s} answer's rank (* = top-5 set kept)  " + "  ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)

print("\nNOT EXACT -- values change. Newest column's bends: k per layer developed from the light, the rest from the token-alone column")
print("   k per layer (of 3,072) | chosen by       | answer in top 5 | answer first | top-5 set kept | answer's median rank")
for k in KS:
    for m in ("largest", "random"):
        d = R[(m, k)]; an = np.array(d["ans"])
        print(f"   {k:>22d} | {'largest |part|' if m == 'largest' else 'at random':15s} | {int((an < 5).sum()):>9d}/{NT}   |"
              f" {sum(d['first']):>7d}/{NT}   | {sum(d['set5']):>8d}/{NT}    | {np.median(an):8.0f}")
print(f"({time.time() - t0:.0f}s)")
