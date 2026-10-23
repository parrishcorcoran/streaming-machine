"""PLANES family, shuffle controls for the best system (version 7b: flat field + higher places + line + word meter +
line-aware places below the top + sign context line x top place). Same coder, same values, order broken at random
(numpy default_rng, fixed seeds), nothing sorted by value:
  c1) each line's 50,257 words put in a random order (seed 0): every line keeps exactly its values, but a "word" is
      now a random mix, so the word meter can only read noise
  c2) every value of the table in a random place (seed 1): same values, no lines, no words
Prediction: c1 loses only the word meter's share (E ~0.02, T ~0.03 bits); c2 falls back to version 1 (E 15.24,
T 11.85) -- everything versions 2-7 found is each line's own histogram and each word's own brightness, nothing else.
    python3 planes_v7_controls.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
MODE = M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE
def flat(A):
    off = ((A[:32768].astype(np.int64).sum(0) + (1 << 14)) >> 15).astype(np.int32)
    return A - off[None, :], A.shape[1] * (int(np.abs(off).max()).bit_length() + 1)
for tb in ["E", "T"]:
    A = load(tb)
    S = np.ascontiguousarray(np.random.default_rng(0).permuted(A, axis=0)); F, side = flat(S); del S
    run("v7b CONTROL c1 each line's words shuffled", tb, F, MODE, extra_side=side); del F
    S = np.random.default_rng(1).permutation(A.ravel()).reshape(A.shape); F, side = flat(S); del S
    run("v7b CONTROL c2 all values shuffled", tb, F, MODE, extra_side=side); del F
    del A
