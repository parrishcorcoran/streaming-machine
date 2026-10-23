"""PLANES family, version 6: exposure meters made of already-coded pixels (a camera's metering: how bright is this
frame, read from the pixels already developed). For a value still dark above place k:
  a) + the word's meter: how many of this word's 768 lines are already lit above place k (bit length of the count:
     0, 1, 2-3, 4-7, ... 512-768) -- a word whose light is brighter lights more of its lines early;
  b) a) + the two places right below a value's top lit place also know their line (each line's own shape of the
     grain just under the top).
Everything else as version 3a (higher places, line, sign context line x top place). No side information: the reader
counts the same lit pixels.
Prediction: small. Words of E are nearly equally bright (bit length of the word's sum of |v| is 23 for 49,714 of
50,257 words); the top place's histogram per word is worth 2.567 -> 2.540 bits, so a) can give at most ~0.03 on E
and on T (each word of T is already scaled to its own power of two); b) a hundredth or two.
    python3 planes_v6_exposure.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v6a + word exposure meter", tb, A, M_HIGH | M_LINE | M_SIGNC | M_WEXP)
    run("v6b + word meter + line-aware places below the top", tb, A, M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE)
    del A
