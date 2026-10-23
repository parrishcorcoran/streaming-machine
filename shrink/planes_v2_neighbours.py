"""PLANES family, version 2: neighbour context (JBIG / JPEG2000 style). A pixel still dark above also knows
  - its line (column) number: each line keeps its own chances (each line is its own film stock), and
  - how many of its 4 neighbours are already lit: the line before / after in the same word, the word before / after
    in the same line (table order; the left and upper ones are known down to this place, the others down to the place
    above).
Lit values keep the version 1 context. Sign: one chance.
Three runs + two shuffle controls (fixed seed 0, random -- nothing sorted by value):
  a) line only            b) line + neighbours
  c) CONTROL: each line's 50,257 words put in a random order (same values on every line, no neighbours across words)
  d) CONTROL: every value of the table in a random place (same values, no lines, no neighbours)
Prediction: the line is worth what each line's own histogram of top places is worth (E: 2.567 -> 2.459 bits for the
top place, about -0.1; T: 2.768 -> 2.452, about -0.3, because T carries the last gain gf per line). Neighbours add
nothing: the word order is the tokenizer's order, and neighbouring lines are different directions of the light -- no
reason for the grain of one to predict the grain of the next. c) should equal b); d) should fall back to version 1.
    python3 planes_v2_neighbours.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v2a line", tb, A, M_HIGH | M_LINE)
    run("v2b line + 4 neighbours", tb, A, M_HIGH | M_LINE | M_NEIGH)
    rng = np.random.default_rng(0)
    S = np.ascontiguousarray(rng.permuted(A, axis=0))
    run("v2c CONTROL line + 4 neighbours, each line's words shuffled", tb, S, M_HIGH | M_LINE | M_NEIGH); del S
    S = rng.permutation(A.ravel()).reshape(A.shape)
    run("v2d CONTROL line + 4 neighbours, all values shuffled", tb, S, M_HIGH | M_LINE | M_NEIGH); del S
    del A
