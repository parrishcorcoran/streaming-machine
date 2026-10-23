"""PLANES family, version 3: the sign plane coded separately, with its own context. On top of version 2a (line):
  a) sign as the LAST image, context = line x the value's top lit place (a few lines have their whole light shifted
     one way -- a dark level -- so on those lines the sign of a big value is nearly known), and
  b) the same + the signs of the left and upper neighbour (0 / up / hole),
  c) the sign coded the moment a value first lights (inside its place), context line x that place x neighbours
     already lit: the same information as b) but in a different order.
Prediction: a) is worth what the per-line asymmetry is worth -- small, only the few lines with a large dark level
carry it (E: largest |line mean| 25,258 against a typical |v| of 6,176); b) adds nothing (neighbours were empty in
version 2); c) = b).
    python3 planes_v3_sign_plane.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    run("v3a sign last, context line x top place", tb, A, M_HIGH | M_LINE | M_SIGNC)
    run("v3b sign last, line x top place + neighbour signs", tb, A, M_HIGH | M_LINE | M_SIGNC | M_SIGNN)
    run("v3c sign when the value lights, line x place + neighbours", tb, A, M_HIGH | M_LINE | M_SIGNC | M_SIGNN | M_SIGNLIT)
    del A
