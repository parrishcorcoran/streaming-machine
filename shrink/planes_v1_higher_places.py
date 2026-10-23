"""PLANES family, version 1: higher-place context. A pixel of place k knows what the higher places of the SAME value
already said: still dark above (the value has not lit yet) or lit, and then how far below its top lit place this pixel
is (1..7+) and whether the place just above is lit. One adaptive chance per (place, that state). Sign: one chance.
Prediction: this is the chain rule of the value's own histogram, so it lands on the order-0 floor (E 15.23, T 11.84)
plus learning cost -- the dependence between a value's places is almost all in its top lit place.
    python3 planes_v1_higher_places.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v1 higher places of the same value", tb, A, M_HIGH)
    del A
