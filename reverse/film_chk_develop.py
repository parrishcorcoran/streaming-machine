"""CHECKER, own code -- NOT EXACT: CHANGES VALUES. k of the newest column's bends per layer developed from the light
(the machine's own bend at this run's own pile), the rest from the token-alone column's shares (film_chk_grains.npz,
built as one position alone); every water level, arrival share, every other bend set by this run's own light.
Chosen by the checker's exact |part| (film_chk_split.npz) and, separately, by the report's 16-place |part|.
k = 3072 (must be the machine), 768, 307, 0 (must be swap_alone.log's ranks); random control with own seed.
Also a look at the dark grains with the exact light (shut share of the 31 smallest-|part|).
    python3 reverse/film_chk_develop.py   -> reverse/film_chk_develop.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

t0 = time.time()
C = np.load("/Users/abundancemachine/streaming-machine/reverse/film_chk_split.npz")
G = np.load("/Users/abundancemachine/streaming-machine/reverse/film_chk_grains.npz")
D = np.load("/Users/abundancemachine/streaming-machine/reverse/film_split.npz")
PA, SA, SH = C["parts50"], G["share_al"], C["share"]; NT = PA.shape[0]; GN16, BO16 = D["GboN"], D["Bout"]
ALONE_LOG = [2, 2, 1063, 275, 74, 317, 9, 22, 0, 34, 10, 5243, 27, 3, 5, 75, 1, 1, 772, 6419, 43, 438, 1477, 171, 312, 0, 9, 53, 15968, 24]

def develop(ids, chosen):
    P = len(ids); t = P - 1; R = WTE[ids] + WPE[:P]; i = TEXTS.index(ids)
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = gauged_bank(R, f"fc.{l}"); idx = rshift(H, ES - 12) + 32768
        s = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
        s[t] = np.where(chosen[l], s[t], SA[i, l])
        R = R + plain_bank(rshift(H * s, 16), f"mproj.{l}")
    x16, ex, LS = gauge_log(R[t:t + 1])
    return power(product(x16, HEADW), (HEADE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + HEADB

rng = np.random.default_rng(4242)
RES = {}
def tally(key, V, a, top5):
    r = int((V > V[a]).sum()); RES.setdefault(key, []).append((r < 5, r == 0, set(np.argsort(-V, kind="stable")[:5].tolist()) == top5, r))
dark_shut, dark_shut_rand = [], []
for i in range(NT):
    ids = TEXTS[i]; S = machine_scores(ids); a = race(S); top5 = set(np.argsort(-S, kind="stable")[:5].tolist())
    ordE = [np.argsort(-np.abs(PA[i, l]), kind="stable") for l in range(12)]
    ord16 = [np.array(sorted(range(3072), key=lambda n: -abs(int(GN16[i, l, n]) * int(BO16[i, l, n])))) for l in range(12)]
    V = develop(ids, [np.ones(3072, bool)] * 12); assert (V == S).all()
    V = develop(ids, [np.zeros(3072, bool)] * 12); r0 = int((V > V[a]).sum()); assert r0 == ALONE_LOG[i], (i, r0)
    tally(("none", 0), V, a, top5)
    line = []
    for k in (768, 307):
        for m, od in (("exact", ordE), ("16place", ord16), ("random", None)):
            ch = []
            for l in range(12):
                c = np.zeros(3072, bool); c[od[l][:k] if od is not None else rng.choice(3072, k, replace=False)] = True; ch.append(c)
            V = develop(ids, ch); tally((m, k), V, a, top5); line.append(f"{k}{m[0]}:{RES[(m, k)][-1][3]}")
    for l in range(12):
        dark_shut.append(float((SH[i, l][ordE[l][-31:]] < 655).mean())); dark_shut_rand.append(float((SH[i, l][rng.choice(3072, 31, replace=False)] < 655).mean()))
    print(f"  text {i:2d} {word(a)!r:13s} k=3072 is the machine; k=0 rank {r0} (= swap_alone.log); " + " ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)
print("\nNOT EXACT -- values change.  answer in top 5 / answer first / top-5 set kept, out of 30")
for key, v in RES.items():
    v = np.array([x[:3] for x in v]); print(f"   {str(key):22s}: {int(v[:, 0].sum())} / {int(v[:, 1].sum())} / {int(v[:, 2].sum())}")
print(f"dark look (exact light): 31 smallest-|part| grains shut {np.mean(dark_shut):.1%}; 31 random {np.mean(dark_shut_rand):.1%}; all {np.mean(SH < 655):.1%}")
print(f"({time.time() - t0:.0f}s)")
