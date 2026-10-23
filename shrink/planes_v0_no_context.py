"""PLANES family, version 0: no context. Each place (bit plane) of the table is one black-and-white image; every pixel
of that image shares ONE adaptive chance (the share of lit pixels in that place, learned as the coder goes). The sign
is one more image, one chance for all. Lossless; decoded and checked equal.
Prediction before running (from planes_look.py): the cost is the sum over places of the binary entropy of each place's
lit share: E: 0.02 + 0.40 + 0.89 + 0.98 + 0.99 + 11 places at ~1.00 + sign 1.00 = about 15.27 bits/value;
T: about 11.9. Both a little above the order-0 floor (15.23 / 11.84) because a place alone does not know whether the
value already lit higher up.
    python3 planes_v0_no_context.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v0 no context (one chance per place)", tb, A, 0)
    del A
