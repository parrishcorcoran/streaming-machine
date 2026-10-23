"""DEVELOP family, version 8: one negative, two prints. The top table T is not a second picture: it is the entry table
E seen through a lens (the last water level's 768 gains, then each word's centring across its lines), at 2^-12. So store
ONLY the developed E; print T from it with whole numbers (develop_gain.py, print_top): the stored T is just the lens,
768 gains (~3 KB). Printed from the undeveloped E it matches the stored T at 97.7% of values, never off by more than 1.
  check : E unchanged, T printed                                               -> must be 30
  cut   : E with one place cut per word, b = 12, 10, 8 (nearest), T printed
  lens  : E exposed for the lens: line i keeps places >= c - (its gain's place), c = 31, 33, 35 (nearest), T printed.
          A line read through a strong gain is exposed finely, a faint-gain line coarsely; c = 31/33/35 gives the top
          the same step as T's own word cut at b = 12/10/8 (versions 3: 30, 28, 28 at 41.2, 31.4, 21.7 MB for T alone).
Predicted before running:
  check 30.  cut: b=12 29-30, b=10 28-30, b=8 25-28 (the cut ignores the lens: the two strongest-gain lines (x17) get
  E's coarse step multiplied by 17 at the top).  lens: c=31 28-30, c=33 27-29, c=35 25-28 (as T's own cut, minus what the
  entry loses on faint-gain lines, which the entry has tolerated so far).
  Stored (E only + 3 KB): cut b=12 ~56 MB, b=10 46.4, b=8 36.5; lens c=31 ~40, c=33 ~30, c=35 ~21 MB -- i.e. BOTH
  tables for about the price of T alone, against v5a's 58.3 MB for the pair at 30/30.
    python3 develop_v8_one_negative.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name="v8 check: E unchanged, T printed from E", E=dict(how="same"), T=dict(how="print"))]
SPECS += [dict(name=f"v8 E cut b={b} nearest, T printed", E=dict(how="cut", b=b), T=dict(how="print")) for b in (12, 10, 8)]
SPECS += [dict(name=f"v8 E lens exposure c={c}, T printed", E=dict(how="lines", c=c), T=dict(how="print")) for c in (31, 33, 35)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
