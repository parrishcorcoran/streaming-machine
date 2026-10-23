"""THE PAGE-SHUFFLE TEST. Each middle neuron is one page: its row in (the fc plate's column) is the reference, its row
out (the mproj plate's row) the object. The bend sets each page's brightness (its share, 0..65,536 at 2^16) from the
light arriving there. Are the pages interchangeable?

At the LAST position (position 4) of each of the 30 refs texts (first 5 tokens of ids+ref), each layer's 3,072 bend
shares are shuffled among that layer's own pages: the same 3,072 values, the same spread, only which page gets which
brightness moves (page n gets the share page perm[n] had). Positions 0..3 are never touched. The water levels and the
arrival shares are always set by the run's own light (the machine's own arithmetic, base2.py's lines). The bends at the
other places are done two ways, both reported and labelled:
  HELD   every bend share not shuffled is held at the run's own recorded share (the bends after the shuffled layer
         are NOT re-set by the changed light). All layers together: every layer's own recorded last-position shares,
         shuffled.
  LIGHT  every bend share not shuffled is set by the light as it now is (the machine's bend table on the changed
         light). All layers together: at each layer the light sets the shares (the light already changed by the
         shuffles below), then that layer's shuffle moves them.
(Layers below the shuffled one are the same both ways: the change only reaches upward. Layer 11 alone is identical
both ways: nothing comes after it -- checked.)
Reported: the answer's rank among 50,257 (rank = how many words score strictly higher; 0 = first) and in-top-5
counts, one layer shuffled alone (12) and all layers together, both ways. The shuffle is drawn once per text, per
layer, per seed (seed 0 is the reported run, seeds 1 and 2 repeat it with other shuffles).
Control: no shuffle, both ways: must be 30/30 first and the machine's own scores exactly.
Also counted: how many of the 3,072 pages actually got a different brightness (two pages both fully closed or both
fully open swap to no change).

PREDICTION (from the task, written before running): with all layers shuffled, the answer leaves the top 5 in at least
28/30 texts. If instead it stays, the pages are interchangeable and that is real redundancy.
PREDICTION (Claude's own, also written before running):
  - one layer alone: the answer stays in the top 5 in FEWER texts than with another text's real bends at that same
    layer (swap_bends.log, B at layer l only: 21..30/30), at every layer, both ways; layer 0 alone drives it out of the
    top 5 in at least 25/30.
  - the two ways agree within 4 texts at every layer.
  - all layers: leaves the top 5 in at least 28/30 both ways (same as the task's).

Work: the shuffle is 0 photon meetings, 0 adds, 0 bounds -- it only re-wires which page receives which brightness.
This is a test of what the pages carry, not a way to cut work; the shuffled runs' meetings are not counted.
    python3 reverse/pageshuf_test.py [seeds=0,1,2] [n=30]      -> reverse/pageshuf_test.log
"""
import sys, time, resource; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

ARGS = dict(a.split("=") for a in sys.argv[1:])
SEEDS = [int(s) for s in ARGS.get("seeds", "0,1,2").split(",")]
NT = int(ARGS.get("n", len(TEXTS)))

def light_share(H):                                   # the machine's own bend table (base2.py edge(), its lines)
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))

def run(ids, perms, way, own):
    """base2.py's forward, its own lines, the bend written out. perms: {layer: permutation of 3,072} moving that layer's
    last-position shares. way 'light': shares set by the light, then moved; way 'held': shares held at `own` (the run's
    own recorded shares, a (P, 3072) array per layer), then moved."""
    P = len(ids); t = P - 1; R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = gauged_bank(R, f"fc.{l}")
        share = light_share(H) if way == "light" else own[l].copy()
        if l in perms: share[t] = share[t][perms[l]]
        R = R + plain_bank(rshift(H * share, 16), f"mproj.{l}")
    w, we, b = W["head"]
    x16, ex, LS = gauge_log(R[t:t + 1])
    return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b

def rank_of(V, i): return int((V > V[i]).sum())

WAYS = ["held", "light"]
CONF = [f"layer {l} alone" for l in range(12)] + ["all layers"]
def perms_for(c, perm): return {l: perm[l] for l in range(12)} if c == "all layers" else {int(c.split()[1]): perm[int(c.split()[1])]}

t0 = time.time()
RES = {(s, w, c): [] for s in SEEDS for w in WAYS for c in CONF}; CTL = {(s, w, c): [] for s in SEEDS for w in WAYS for c in CONF}
NOSH = {w: [] for w in WAYS}; MOVED = np.zeros(12, np.int64); MIRROR = 0; L11 = 0
for i in range(NT):
    ids = TEXTS[i]; rec = record(ids); own = [Ly["bend"]["share"] for Ly in rec["layers"]]; S0 = rec["top"]["scores"]
    assert np.array_equal(S0, machine_scores(ids))                                   # the recorded run is the machine
    wA = race(S0); top5 = [int(x) for x in np.argsort(-S0, kind="stable")[:5]]
    for w in WAYS:                                                                   # control: no shuffle
        S = run(ids, {}, w, own); assert np.array_equal(S, S0); NOSH[w].append(rank_of(S, wA))
    for s in SEEDS:
        rng = np.random.default_rng(1000 * s + i)
        perm = [rng.permutation(3072) for _ in range(12)]
        if s == SEEDS[0]:
            for l in range(12): MOVED[l] += int((own[l][4][perm[l]] != own[l][4]).sum())
        for w in WAYS:
            for c in CONF:
                S = run(ids, perms_for(c, perm), w, own)
                RES[(s, w, c)].append(rank_of(S, wA)); CTL[(s, w, c)] += [rank_of(S, x) for x in top5[1:]]
                if s == SEEDS[0] and w == "held" and c == "all layers":             # mirror: machine.py's record, bends held
                    sh = [o.copy() for o in own]
                    for l in range(12): sh[l][4] = own[l][4][perm[l]]
                    Sr = record(ids, held=dict(layers=[dict(bend=dict(share=x)) for x in sh]))["top"]["scores"]
                    assert np.array_equal(Sr, S); MIRROR += 1
        assert RES[(s, "held", "layer 11 alone")][-1] == RES[(s, "light", "layer 11 alone")][-1]; L11 += 1
    s = SEEDS[0]
    print(f"  text {i:2d} {word(wA)!r:14s} HELD alone: " + " ".join(str(RES[(s, 'held', c)][-1]) for c in CONF[:12])
          + f" | all {RES[(s, 'held', 'all layers')][-1]}   LIGHT alone: " + " ".join(str(RES[(s, 'light', c)][-1]) for c in CONF[:12])
          + f" | all {RES[(s, 'light', 'all layers')][-1]}  ({time.time() - t0:.0f}s)", flush=True)

def mid(a): a = sorted(a); return f"{a[len(a) // 2 - 1]},{a[len(a) // 2]}"
print(f"\nCONTROL, no shuffle: HELD answer first {sum(r == 0 for r in NOSH['held'])}/{NT}, LIGHT answer first "
      f"{sum(r == 0 for r in NOSH['light'])}/{NT}; scores equal to the machine's own, every word, both ways: {NT}/{NT}")
print(f"mirror: HELD all-layers run equal to machine.py's record() with those bends held: {MIRROR}/{NT};  "
      f"layer 11 alone the same rank both ways: {L11}/{NT * len(SEEDS)}")
print(f"pages that got a different brightness at the last position (seed {SEEDS[0]}, summed over {NT} texts, of {NT * 3072:,}): "
      + ", ".join(f"L{l} {MOVED[l]:,}" for l in range(12)))
print(f"\nthe answer at the last position, bend shares shuffled among the layer's own pages there. Columns: seed {SEEDS[0]}: answer in"
      f" top 5 / first / middle two ranks of {NT} | the 2nd..5th words in top 5 (of {4 * NT}) | answer in top 5 with seeds "
      + ", ".join(str(s) for s in SEEDS[1:]))
for w in WAYS:
    print(f"  {'HELD (bends not shuffled held at the run own shares)' if w == 'held' else 'SET BY THE LIGHT (bends not shuffled set by the changed light)'}")
    for c in CONF:
        a = RES[(SEEDS[0], w, c)]; k = CTL[(SEEDS[0], w, c)]
        print(f"    {c:16s} {sum(r < 5 for r in a):2d}/{NT}  {sum(r == 0 for r in a):2d}/{NT}  {mid(a):>13s}  | {sum(r < 5 for r in k):3d}/{4 * NT}"
              f"  | " + "  ".join(f"{sum(r < 5 for r in RES[(s, w, c)]):2d}/{NT}" for s in SEEDS[1:]))
print("  for comparison (earlier logs): another text's real bends at one layer, all positions (swap_bends.log): 21..30/30;"
      " another text's bends at the last position, all layers: 1/30; the answer in another text's own top 5 (coincidence): 1/30")
print(f"({time.time() - t0:.0f}s, max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6:,} MB)")
