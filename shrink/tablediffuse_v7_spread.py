"""TABLEDIFFUSE v7 -- cancellation WITHOUT extra grain: the hypothesis itself, isolated.
v1-v5 mixed two things: the carry makes a word's grain cancel over its lines, and it also multiplies every value's
grain power (x2, x6, x1.33). Here the owed leftover is spread evenly over ALL the word's remaining lines (the k-th line
from the end repays 1/k of what is owed; a reciprocal table, whole numbers), so a word's grain still adds to less than
half a step over its 768 lines (nearest: about 6.4 steps), while each value's grain power stays within 1% of nearest
(0.084 vs 0.083 step^2; checked before judging).
The relation says the cancellation lands only in each word's white level, which the gauge removes from every read at
the entry, and which the top sees only against x - mean(x) (T's rows add to 0). So:
Predicted: equal to nearest within the judge's luck -- c=32 29-30, c=34 29-30, c=37 23-28 (nearest and its screen
phases: {30,30,29,29,30}, {29,29,29,30,29}, {28,27,24,23,27}); natural and random line orders alike; bytes = nearest
+-0.1%. If the hypothesis held, c=37 should climb above 28.
    python3 shrink/tablediffuse_v7_spread.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v7 spread the owed grain over the word's remaining lines, c={c}", c=c, how="spread") for c in (32, 34, 37)]
       + [dict(name=f"v7 spread, random line order seed={s}, c={c}", c=c, how="spread", line_seed=s)
          for s in (1, 2) for c in (32, 34, 37)])
