"""CHECKER, own code: concentration and shared grains from the checker's EXACT parts (film_chk_split.npz), with own
controls: (i) the light's lines shuffled within the layer (own random order), (ii) the bend outputs of the token-alone
column, built here independently as ONE position run alone at position t (base2.py's own lines), compared with
film_split.npz's token-alone shares. Also the same measures on film_split.npz's 16-place parts.
    python3 reverse/film_chk_grains.py   -> reverse/film_chk_grains.log
"""
import sys, time, itertools; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

t0 = time.time()
C = np.load("/Users/abundancemachine/streaming-machine/reverse/film_chk_split.npz")
D = np.load("/Users/abundancemachine/streaming-machine/reverse/film_split.npz")
PA, LI, BO, ANS, LEAD, PS = C["parts50"], C["light50"], C["Bout"], C["ans"], C["lead"], C["partsum"]
NT = PA.shape[0]
assert (BO == D["Bout"]).all() and (ANS == D["ans"]).all() and (C["second"] == D["second"]).all() and (LEAD == D["lead"]).all()

# ---- token-alone column, own construction: the last token alone at position t
BA = np.zeros((NT, 12, 3072), np.int64); SA = np.zeros((NT, 12, 3072), np.int64)
for i in range(NT):
    ids = TEXTS[i]; t = len(ids) - 1; R = WTE[ids[t:t + 1]] + WPE[t:t + 1]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = gauged_bank(R, f"fc.{l}"); idx = rshift(H, ES - 12) + 32768
        sh = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
        Bo = rshift(H * sh, 16); BA[i, l], SA[i, l] = Bo[0], sh[0]
        R = R + plain_bank(Bo, f"mproj.{l}")
print(f"token-alone column built alone == film_split.npz's: shares {(SA == D['share_al']).all()}, bend outputs {(BA == D['Bout_al']).all()}")

# ---- exact parts vs the report's 16-place parts
P16 = [[[int(g) * int(b) for g, b in zip(D["GboN"][i, l], D["Bout"][i, l])] for l in range(12)] for i in range(NT)]
sgn = np.mean([np.mean([(np.sign(PA[i, l, n]) == np.sign(P16[i][l][n])) for n in range(3072)]) for i in range(NT) for l in range(12)])
t100 = [len(set(np.argsort(-np.abs(PA[i, l]), kind="stable")[:100]) & set(np.argsort(-np.abs(np.array(P16[i][l], dtype=float)), kind="stable")[:100]))
        for i in range(NT) for l in range(12)]
print(f"exact vs 16-place parts (30 texts x 12 layers): same sign {sgn:.4%}; top-100 by |part| shared: mean {np.mean(t100):.1f}, min {min(t100)}")
r = PS / LEAD[:, None]
print("newest part / lead per layer (exact): median " + " ".join(f"{np.median(r[:, l]):+.2f}" for l in range(12)))
print("   negative in: " + " ".join(f"{int((r[:, l] < 0).sum())}" for l in range(12)) + f" of {NT};  10th-90th spans: " +
      " ".join(f"[{np.percentile(r[:, l], 10):.1f},{np.percentile(r[:, l], 90):.1f}]" for l in range(12)))

def meas(p):
    p = [int(x) for x in p]; n = len(p); a = [abs(x) for x in p]; order = sorted(range(n), key=lambda k: -a[k])
    A = sum(a); S = sum(p); cum = 0
    for k, j in enumerate(order):
        cum += a[j]
        if 2 * cum >= A: hab = (k + 1) / n; break
    sg = 1 if S > 0 else -1; cum = 0
    for k, v in enumerate(sorted((sg * x for x in p), reverse=True)):
        cum += v
        if 2 * cum >= sg * S: hsg = (k + 1) / n; break
    cum = 0; stay = 1
    for k, j in enumerate(order):
        cum += p[j]
        if 10 * abs(cum - S) > abs(S): stay = k + 2
    return dict(hab=hab, hsg=hsg, mx=a[order[0]] / A, t31=sum(a[j] for j in order[:31]) / A, t307=sum(a[j] for j in order[:307]) / A,
                stay=stay, net=abs(S) / A, top=frozenset(order[:100]))
rng = np.random.default_rng(777)
K = ["real", "shuf", "alone", "real16"]; Mx = {k: [[None] * 12 for _ in range(NT)] for k in K}
TG = [[None] * 12 for _ in range(NT)]; TB = [[None] * 12 for _ in range(NT)]
for i in range(NT):
    for l in range(12):
        g = [int(x) for x in LI[i, l]]; b = [int(x) for x in BO[i, l]]; perm = rng.permutation(3072)
        Mx["real"][i][l] = meas(PA[i, l]); Mx["shuf"][i][l] = meas([g[perm[n]] * b[n] for n in range(3072)])
        Mx["alone"][i][l] = meas([g[n] * int(BA[i, l, n]) for n in range(3072)]); Mx["real16"][i][l] = meas(P16[i][l])
        TG[i][l] = frozenset(np.argsort(-np.abs(LI[i, l]), kind="stable")[:100].tolist())
        TB[i][l] = frozenset(np.argsort(-np.abs(BO[i, l]), kind="stable")[:100].tolist())
def col(k, f, l=None):
    ls = range(12) if l is None else [l]; return np.array([Mx[k][i][x][f] for i in range(NT) for x in ls], float)
print("\nmean over texts: real exact | shuffled light | token-alone outputs | real 16-place (report's parts)")
for f, lab in [("hab", "half sum|part| (share of neurons)"), ("hsg", "half signed (share)"), ("t31", "top 31 share of |part|"),
               ("t307", "top 307 share"), ("mx", "largest single")]:
    print(f"  {lab:34s}: " + " | ".join(f"{col(k, f).mean():.2%}" for k in K))
print(f"  {'stay within 10% (median neurons)':34s}: " + " | ".join(f"{np.median(col(k, 'stay')):.0f}" for k in K))
print(f"  {'net/gross (median)':34s}: " + " | ".join(f"{np.median(col(k, 'net')):.3f}" for k in K))
print("  per layer, half sum|part| real / shuffle / alone; largest single real / shuffle; stay median real:")
for l in range(12):
    print(f"   layer {l:2d}: {col('real', 'hab', l).mean():.1%} / {col('shuf', 'hab', l).mean():.1%} / {col('alone', 'hab', l).mean():.1%};"
          f"  {col('real', 'mx', l).mean():.2%} / {col('shuf', 'mx', l).mean():.2%};  {np.median(col('real', 'stay', l)):.0f}")
h, hs = col("real", "hab"), col("shuf", "hab"); st = col("real", "stay")
print(f"  real below shuffle in {int((h < hs).sum())}/{h.size}; median ratio {np.median(h / hs):.2f};  stay quartiles "
      f"{np.percentile(st, 25):.0f}/{np.median(st):.0f}/{np.percentile(st, 75):.0f}; <=307 in {int((st <= 307).sum())}, <=31 in {int((st <= 31).sum())} of {st.size}")

pairs = list(itertools.combinations(range(NT), 2)); same = [(a, b) for a, b in pairs if ANS[a] == ANS[b]]
def shared(get, ls, prs): return np.array([len(get(a, l) & get(b, l)) for l in ls for a, b in prs], float)
src = {"real": lambda i, l: Mx["real"][i][l]["top"], "shuf": lambda i, l: Mx["shuf"][i][l]["top"], "alone": lambda i, l: Mx["alone"][i][l]["top"],
       "light": lambda i, l: TG[i][l], "Bout": lambda i, l: TB[i][l]}
print("\nshared top-100 across texts (435 pairs; chance 3.3): " + " | ".join(src))
for l in list(range(12)) + [None]:
    ls = range(12) if l is None else [l]
    print(f"   {'all' if l is None else l:>3}: " + " | ".join(f"{shared(g, ls, pairs).mean():5.1f} (max {int(shared(g, ls, pairs).max())})" for g in src.values())
          + f" | same answer ({len(same)} pairs) {shared(src['real'], ls, same).mean():.1f}")
print(f"  texts 4, 9 answer/2nd: {word(ANS[4])!r}/{word(C['second'][4])!r}, {word(ANS[9])!r}/{word(C['second'][9])!r}; brightest-light top 100 shared at layer 11: {len(TG[4][11] & TG[9][11])}")
wg = np.mean([len(Mx["real"][i][l]["top"] & TG[i][l]) for i in range(NT) for l in range(12)])
wb = np.mean([len(Mx["real"][i][l]["top"] & TB[i][l]) for i in range(NT) for l in range(12)])
print(f"  within text and layer: top-100 |part| among brightest-100 light {wg:.1f}, among largest-100 outputs {wb:.1f}")
np.savez("/Users/abundancemachine/streaming-machine/reverse/film_chk_grains.npz", share_al=SA, Bout_al=BA)
print(f"({time.time() - t0:.0f}s)")
