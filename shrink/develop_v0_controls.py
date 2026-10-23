"""DEVELOP family, version 0: the frame of reference, before any photon is taken away.
  same    : E and T exactly as they are -> must give 30/30 (checks the judge), and the photon coder run on every photon
            (lossless; the coder used for every developed table below, here with nothing removed).
  shuffle : CONTROL. Each word keeps its exact 768 values (its whole histogram, every photon), but they are put on
            random lines (seed fixed). The distribution stays, the placement goes. Whatever words this gets is what the
            30 inputs give with no word-specific light at all: the floor every developed table is read against.
Predicted before running: same 30/30 (both); E shuffled 0-4/30; T shuffled 0-2/30 (the read at the top is then noise
over 50,257 words). The lossless photon coder: about 15.2 bits/value for E and 11.8 for T (the order-0 floors), since
with nothing removed the photons sit about a third of all places and naming where each is costs about a bit a place.
    python3 develop_v0_controls.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [
    dict(name="v0 same E+T (no change)", E=dict(how="same"), T=dict(how="same")),
    dict(name="v0 CONTROL E words' values on random lines", E=dict(how="wordshuffle", seed=0)),
    dict(name="v0 CONTROL T words' values on random lines", T=dict(how="wordshuffle", seed=1)),
]
if __name__ == "__main__": run(SPECS, sys.argv[1])
