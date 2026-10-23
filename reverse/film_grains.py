"""THE NEWEST FILM, STEPS 2 AND 3: HOW CONCENTRATED, AND ARE THE SAME GRAINS USED ACROSS TEXTS?

From film_split.npz (16-place style, 30 refs texts, 5 tokens): at layer l, neuron n of the newest column,
    part_n = (reverse lead-light at n's bend output) x (n's bend output),
and the layer's part S_l = sum over the 3,072 neurons (exact; film_split.log: adds back 360/360). All sums here are
Python whole numbers (each part is a whole number over the layer's one power of two, which no share below depends on).

STEP 2, per layer and text:
  half|part|  the fewest neurons, largest |part| first, whose |part| reaches half of sum |part_n| (share of 3,072)
  half S      the fewest neurons, largest first in the direction of S_l, whose sum reaches half of S_l (share of 3,072)
  top 1%/10%  share of sum |part_n| in the largest 31 / 307 neurons
  largest     the largest single neuron's |part| / sum |part_n|
  stay 10%    neurons (largest |part| first) after which the running sum reaches S_l and stays within 10% of it
  net/gross   S_l / sum |part_n|
CONTROLS: (i) the light's lines put in a random order within the layer (same values; pairing with the bend outputs
broken); (ii) the bend outputs of the token-alone column (swap_alone.py's run) met with the same light.
STEP 3: per layer, the 100 largest-|part| neurons of text A vs text B (all 435 pairs): how many shared, against
100 x 100 / 3,072 = 3.3 by chance; the same for the controls, for the 100 brightest lines of the light alone and the 100
largest bend outputs alone (which of the two carries any sharing), and for pairs of texts with the same answer word.

PREDICTION (Claude's, written before running): the lead is spread, half of it in about a tenth of the neurons per
layer, only modestly above the shuffle; the top-100 grains share little across texts. In numbers: half|part| ~ 10%
of the neurons (shuffle ~ 13%, token-alone ~ 10%); stay 10% needs most of the layer (> 1,500 neurons) because the
parts cancel (net/gross small); top-100 shared across texts ~ 5-10 (chance 3.3), and the bend outputs alone share more
than the light alone.
    python3 reverse/film_grains.py      -> reverse/film_grains.log
"""
import sys, time, itertools
import numpy as np
from fractions import Fraction

t0 = time.time()
D = np.load("/Users/abundancemachine/streaming-machine/reverse/film_split.npz")
GN, BO, BA, ANS = D["GboN"], D["Bout"], D["Bout_al"], D["ans"]
NT, NL, NN = GN.shape
rng = np.random.default_rng(20261004)
K1, K10 = round(NN / 100), round(NN / 10)

def parts_of(G, B): return [int(g) * int(b) for g, b in zip(G, B)]

def measures(p):
    n = len(p); a = [abs(x) for x in p]; order = sorted(range(n), key=lambda k: -a[k]); A = sum(a); S = sum(p)
    cum = 0; h_abs = n
    for k, i in enumerate(order):
        cum += a[i]
        if 2 * cum >= A: h_abs = k + 1; break
    h_sig = np.nan
    if S != 0:
        sg = 1 if S > 0 else -1; cum = 0
        for k, v in enumerate(sorted((sg * x for x in p), reverse=True)):
            cum += v
            if 2 * cum >= sg * S: h_sig = k + 1; break
    cum = 0; last_out = -1
    for k, i in enumerate(order):
        cum += p[i]
        if 10 * abs(cum - S) > abs(S): last_out = k
    return dict(h_abs=h_abs / n, h_sig=h_sig / n, top1=float(Fraction(sum(a[i] for i in order[:K1]), A)),
                top10=float(Fraction(sum(a[i] for i in order[:K10]), A)), mx=float(Fraction(a[order[0]], A)),
                k10=last_out + 2, net=float(Fraction(S, A)), top=frozenset(order[:100]))

def top100(v): return frozenset(sorted(range(len(v)), key=lambda k: -abs(int(v[k])))[:100])

KEYS = ["own", "shuf", "alone"]
M = {k: [[None] * NL for _ in range(NT)] for k in KEYS}
TG = [[None] * NL for _ in range(NT)]; TB = [[None] * NL for _ in range(NT)]
for i in range(NT):
    for l in range(NL):
        G, B, Ba = GN[i, l], BO[i, l], BA[i, l]
        M["own"][i][l] = measures(parts_of(G, B))
        M["shuf"][i][l] = measures(parts_of(G[rng.permutation(NN)], B))
        M["alone"][i][l] = measures(parts_of(G, Ba))
        TG[i][l] = top100(G); TB[i][l] = top100(B)
    print(f"  text {i:2d} done ({time.time() - t0:.0f}s)", flush=True)

def col(key, f, l=None):
    if l is None: return np.array([M[key][i][l_][f] for i in range(NT) for l_ in range(NL)], float)
    return np.array([M[key][i][l][f] for i in range(NT)], float)

print("\n================ STEP 2: HOW CONCENTRATED (mean over 30 texts; real | lines shuffled | token-alone bend outputs) ================")
print(" layer |  half|part| (share of neurons)  |  half S (share of neurons)       |  top 1% holds of |part|          |  top 10% holds of |part|"
      "         |  largest single neuron           |  stay 10%: neurons (median)      |  net/gross (median |.|)")
F = [("h_abs", "{:6.1%}"), ("h_sig", "{:6.1%}"), ("top1", "{:6.1%}"), ("top10", "{:6.1%}"), ("mx", "{:6.2%}")]
for l in list(range(NL)) + [None]:
    cells = []
    for f, fmt in F:
        cells.append(" | ".join(fmt.format(np.nanmean(col(k, f, l))) for k in KEYS))
    cells.append(" | ".join(f"{np.median(col(k, 'k10', l)):6.0f}" for k in KEYS))
    cells.append(" | ".join(f"{np.median(np.abs(col(k, 'net', l))):6.3f}" for k in KEYS))
    print(f" {'all' if l is None else l:>5} |  " + "  |  ".join(cells))
print("  (half S: in 'all', texts where S_l = 0 are left out; none expected)")
h_own, h_sh, h_al = col("own", "h_abs"), col("shuf", "h_abs"), col("alone", "h_abs")
print(f"  half|part| real below its shuffle in {int((h_own < h_sh).sum())}/{h_own.size} (text, layer); below token-alone in "
      f"{int((h_own < h_al).sum())}/{h_own.size};  median ratio real/shuffle {np.median(h_own / h_sh):.2f}")
k_own = col("own", "k10")
print(f"  stay 10% (real): quartiles {np.percentile(k_own, 25):.0f} / {np.median(k_own):.0f} / {np.percentile(k_own, 75):.0f} neurons; "
      f"<= 307 (a tenth) in {int((k_own <= 307).sum())}/{k_own.size}; <= 31 in {int((k_own <= 31).sum())}/{k_own.size}")

print("\n================ STEP 3: THE SAME GRAINS ACROSS TEXTS? (top 100 of 3,072 per layer; chance 100 x 100 / 3,072 = 3.3) ================")
pairs = list(itertools.combinations(range(NT), 2)); same_ans = [(a, b) for a, b in pairs if ANS[a] == ANS[b]]
def shared(get, l, prs): return np.array([len(get(a, l) & get(b, l)) for a, b in prs], float)
SRC = [("|part| real", lambda i, l: M["own"][i][l]["top"]), ("|part| lines shuffled", lambda i, l: M["shuf"][i][l]["top"]),
       ("|part| token-alone", lambda i, l: M["alone"][i][l]["top"]), ("light alone |G|", lambda i, l: TG[i][l]),
       ("bend output alone |Bout|", lambda i, l: TB[i][l])]
print(" layer | " + " | ".join(f"{n:>24s}" for n, _ in SRC) + f" | real, same answer ({len(same_ans)} pairs)")
for l in list(range(NL)) + [None]:
    ls = range(NL) if l is None else [l]
    row = []
    for n, g in SRC:
        v = np.concatenate([shared(g, l_, pairs) for l_ in ls]); row.append(f"{v.mean():6.1f} (max {int(v.max()):3d})")
    v = np.concatenate([shared(SRC[0][1], l_, same_ans) for l_ in ls]) if same_ans else np.array([np.nan])
    print(f" {'all' if l is None else l:>5} | " + " | ".join(f"{x:>24s}" for x in row) + f" | {v.mean():6.1f} (max {int(np.nanmax(v)):3d})")
print(f"  same-answer pairs: {[(a, b) for a, b in same_ans]}")
within_G = np.array([len(M["own"][i][l]["top"] & TG[i][l]) for i in range(NT) for l in range(NL)])
within_B = np.array([len(M["own"][i][l]["top"] & TB[i][l]) for i in range(NT) for l in range(NL)])
print(f"  within one text and layer: the top-100 |part| neurons among the top-100 brightest light lines {within_G.mean():.1f}, among the top-100 "
      f"largest bend outputs {within_B.mean():.1f} (chance 3.3)")
print(f"({time.time() - t0:.0f}s)")
