"""THE PAGE-SHUFFLE TEST'S SHUFFLE CONTROL: ARE THE ANSWERS THAT STAY THE PAGES, OR THE COMMON-WORD FLOOR?
pageshuf_test.log: with every layer's last-position bend shares shuffled among the layer's own pages, the answer stays
in the top 5 in 8/30 texts (bends not shuffled held) and 6/30 (set by the changed light). The texts that stay are
',', ' the', '.', '\n', '-': the most common words. A scrambled machine may simply say common words whatever the text.

Control (the answers shuffled among the texts: every answer stays in the list, only which text it is checked against
moves): for each text A and each shuffled run of A (the same seed-0 shuffles as pageshuf_test.py, rebuilt from the same
seeds), how many of the OTHER 29 texts' answers sit in A's shuffled top 5 (of 870 pairs)? If the pages carried nothing,
A's own answer would sit there at the same rate as another text's answer. Also the same count for the unshuffled
machine (the real machine's own coincidence), and the shuffled machine's first word, text by text, all layers.

PREDICTION (Claude's, written before running):
  - all layers shuffled: other texts' answers sit in A's shuffled top 5 at no less than half A's own rate (own 8/30
    held, 6/30 light, i.e. at least 58 of 870 held, 44 of 870 light): the stays are the common-word floor.
  - the shuffled machine's first word, all layers, is one of ',' ' the' '.' ' and' '\n' in at least 20/30 texts.
  - unshuffled machine: other texts' answers in A's top 5 well below A's own 30/30 (about the 1/30 of swap_kinds.log,
    scaled: under 120 of 870).
    python3 reverse/pageshuf_floor.py       -> reverse/pageshuf_floor.log
"""
import sys, time, resource; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def light_share(H):                                   # the machine's own bend table (base2.py edge(), its lines)
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))

def run(ids, perms, way, own):                        # the same forward as pageshuf_test.py
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

def top5(S): return [int(x) for x in np.argsort(-S, kind="stable")[:5]]
WAYS = ["held", "light"]
CONF = [f"layer {l} alone" for l in range(12)] + ["all layers"]
def perms_for(c, perm): return {l: perm[l] for l in range(12)} if c == "all layers" else {int(c.split()[1]): perm[int(c.split()[1])]}

t0 = time.time(); NT = len(TEXTS)
recs = [record(ids) for ids in TEXTS]; OWN = [[Ly["bend"]["share"] for Ly in r["layers"]] for r in recs]
ANS = [race(r["top"]["scores"]) for r in recs]
T5 = {(w, c): [] for w in WAYS for c in CONF}; FIRST = {w: [] for w in WAYS}
base_t5 = [top5(r["top"]["scores"]) for r in recs]
for i, ids in enumerate(TEXTS):
    rng = np.random.default_rng(1000 * 0 + i)                         # seed 0, as in pageshuf_test.py
    perm = [rng.permutation(3072) for _ in range(12)]
    for w in WAYS:
        for c in CONF:
            S = run(ids, perms_for(c, perm), w, OWN[i]); T5[(w, c)].append(top5(S))
        FIRST[w].append(T5[(w, "all layers")][-1][0])
    print(f"  text {i:2d} answer {word(ANS[i])!r:14s} all layers shuffled, top 5 HELD: "
          + " ".join(repr(word(x)) for x in T5[('held', 'all layers')][-1]) + "   LIGHT: "
          + " ".join(repr(word(x)) for x in T5[('light', 'all layers')][-1]) + f"  ({time.time() - t0:.0f}s)", flush=True)

def own_and_others(t5s):
    own = sum(ANS[i] in t5s[i] for i in range(NT))
    oth = sum(ANS[j] in t5s[i] for i in range(NT) for j in range(NT) if j != i)
    nxt = sum(ANS[(i + 1) % NT] in t5s[i] for i in range(NT))
    return own, oth, nxt
o, x, n = own_and_others(base_t5)
print(f"\nUNSHUFFLED machine: A's own answer in A's top 5 {o}/{NT};  other texts' answers in A's top 5 {x}/{NT * (NT - 1)}"
      f" (as a rate over {NT}: {x * NT // (NT * (NT - 1))}/{NT} rounded down); the next text's answer {n}/{NT}")
print(f"\nshuffled (seed 0): A's own answer in A's shuffled top 5 | other texts' answers in A's shuffled top 5 (of {NT * (NT - 1)}) "
      f"| the next text's answer (of {NT})")
for w in WAYS:
    print(f"  {'HELD (bends not shuffled held at the run own shares)' if w == 'held' else 'SET BY THE LIGHT (bends not shuffled set by the changed light)'}")
    for c in CONF:
        o, x, n = own_and_others(T5[(w, c)])
        print(f"    {c:16s} own {o:2d}/{NT}  | others {x:3d}/{NT * (NT - 1)}  | next {n:2d}/{NT}")
print("\nthe shuffled machine's first word, all layers:")
for w in WAYS:
    words_ = {}
    for f in FIRST[w]: words_[f] = words_.get(f, 0) + 1
    print(f"  {w:5s}: " + ", ".join(f"{word(k)!r} x{v}" for k, v in sorted(words_.items(), key=lambda kv: -kv[1])))
print(f"answers of the 30 texts: " + ", ".join(repr(word(a)) for a in ANS))
print(f"({time.time() - t0:.0f}s, max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6:,} MB)")
