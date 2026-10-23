"""PLANES family, version 4: change the writing before coding the places.
  a) single wave (up / hole, no two lit places side by side): each place of v is dark, up (+1) or a hole (-1); a lit
     place right above forces this one dark and is not coded. Per place: lit? (context as version 3a: higher places,
     line), then up or hole? (the top lit place's direction is the sign: context line x place; a lower lit place:
     context = how far below the top, and the direction of the nearest lit place above).
  b) the +-1 writing with no zeros at all (every place up or a hole, D = 2v + 1 = offset binary): the sign place first
     as its own image (context: line), then every lower place read against the sign place (m = v, or -v - 1 for
     holes), and the dark-above context knows the sign as well as the line.
Prediction: neither can beat version 3a. Both are one-to-one rewritings of the same numbers, so the information is the
same (15.0 / 11.4 bits); the single wave has fewer photons (5.01 instead of 6.69 a value in E) but every photon now
carries a direction bit, and on the coin-flip places a single-wave place costs exactly 1 bit on average too:
2/3 of places are free to light, each costs 1 bit (lit?) + 1/2 bit (direction) = 1.5, and 2/3 x 1.5 = 1.
    python3 planes_v4_single_wave.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v4a single wave (up/hole) places, line + sign ctx", tb, A, M_HIGH | M_LINE | M_SIGNC, naf=True)
    run("v4b +-1 writing (no zeros), sign place first", tb, A, M_HIGH | M_LINE | M_SIGNC | M_SIGNTOP)
    del A
