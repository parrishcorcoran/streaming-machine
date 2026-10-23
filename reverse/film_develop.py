"""THE NEWEST FILM, STEP 4 -- NOT EXACT: THIS TEST CHANGES VALUES. How many grains of the newest film carry the answer?

For each of the 30 refs texts (5 tokens), the machine is run with its own lines (base2.py's gauged_bank, attention,
plain_bank and its bend), except at the newest column's bends: at layer l, only the k neurons with the largest |part_n|
(film_split.npz, 16-place light x bend output, chosen from the full run of this same text) are developed from the light
-- the machine's own bend reading the pile that arrives in THIS run -- and the other 3,072 - k take the token-alone
column's shares (swap_alone.py's run: position t's arrival over only itself). Every other dimming, every water level,
every arrival share and every earlier column is set by this run's own light. k = 3072, 1536, 768, 307, 31 (and 0 = all
token-alone, which must equal swap_alone.log). Judged out of 30: the answer in this run's top 5; the machine's top-5
set kept (same five words). Choosing the k largest uses the full run, so this only shows how many grains carry the
answer, not a way to find them in advance.
CONTROL: the same k, chosen at random within each layer (same counts, no knowledge of the parts).
Checks: k = 3072 gives the machine's own scores exactly (30/30); k = 0 gives swap_alone.py's scores exactly.

PREDICTION (Claude's, written before running, after film_grains.log showed the signed part settles only after ~2/3 of
a layer's neurons): answer in top 5 -- k=1536: 20/30, 768: 15/30, 307: 11/30, 31: 8/30 (token-alone 7/30); random k a
few texts fewer at each k (16, 12, 9, 7). Top-5 set kept: k=1536 ~5/30, smaller k <= 2/30; random fewer.
    python3 reverse/film_develop.py     -> reverse/film_develop.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def develop(ids, chosen, share_al):
    """base2.py's forward, its own lines; at the newest column's bend of layer l a neuron's share is read by the machine's
    own bend from this run's pile where chosen[l], else it is the token-alone column's share"""
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
NT = GN.shape[0]; KS = [3072, 1536, 768, 307, 31, 0]
rng = np.random.default_rng(20261005)
R = {(m, k): dict(ans=[], first=[], set5=[]) for m in ("largest", "random") for k in KS}
for i in range(NT):
    ids = TEXTS[i]; S = machine_scores(ids); a = race(S); top5 = set(int(x) for x in np.argsort(-S, kind="stable")[:5])
    assert a == int(D["ans"][i])
    order = [sorted(range(3072), key=lambda n: -abs(int(GN[i, l, n]) * int(BO[i, l, n]))) for l in range(12)]
    line = []
    for k in KS:
        for m in ("largest", "random"):
            if m == "random" and k in (3072, 0): continue
            chosen = []
            for l in range(12):
                c = np.zeros(3072, bool)
                c[order[l][:k] if m == "largest" else rng.choice(3072, k, replace=False)] = True; chosen.append(c)
            V = develop(ids, chosen, SHA[i])
            if k == 3072: assert (V == S).all(), "k = 3072 must be the machine itself"
            if k == 0:                                                  # swap_alone.py's run, through record()
                rec = record(ids); sh = []
                for l in range(12):
                    s = rec["layers"][l]["bend"]["share"].copy(); s[-1] = SHA[i, l]; sh.append(s)
                assert (record(ids, held=dict(layers=[dict(bend=dict(share=s)) for s in sh]))["top"]["scores"] == V).all()
            r = rank_of(V, a); R[(m, k)]["ans"].append(r); R[(m, k)]["first"].append(r == 0)
            R[(m, k)]["set5"].append(set(int(x) for x in np.argsort(-V, kind="stable")[:5]) == top5)
            line.append(f"{k}{'r' if m == 'random' else ''}:{r}")
    print(f"  text {i:2d} {word(a)!r:13s} answer's rank  " + "  ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)

print("\nchecks: k = 3072 equals the machine's own scores 30/30; k = 0 equals swap_alone.py's run (record with held shares) 30/30")
print("NOT EXACT -- values change. Newest column's bends: k per layer developed from the light, the rest from the token-alone column")
print("   k per layer (of 3,072) | chosen by       | answer in top 5 | answer first | top-5 set kept | answer's median rank")
for k in KS:
    for m in ("largest", "random"):
        if m == "random" and k in (3072, 0): continue
        d = R[(m, k)]; an = np.array(d["ans"])
        print(f"   {k:>22d} | {('largest |part|' if m == 'largest' else 'at random') if k not in (3072, 0) else ('all' if k else 'none (token-alone)'):15s} |"
              f" {int((an < 5).sum()):>9d}/{NT}   | {sum(d['first']):>7d}/{NT}   | {sum(d['set5']):>8d}/{NT}    | {np.median(an):8.0f}")
print(f"({time.time() - t0:.0f}s)")
