"""DEVELOP family, version 6: subtract the white plate first (a camera's dark/flat frame), develop only each word's
departure from it, add the plate back. The white plate = each line's average over all 50,257 words (768 whole numbers,
stored once, ~3 KB).
Why (from versions 1-4): at the top, only DIFFERENCES between words decide the race -- light every word shares adds the
same amount to every word's pile and drops out. A value written as (shared part + own part) and cut by its own top
photons spends its photons on the shared part, where they decide nothing; the own part is then cut coarsely. And
dithering made T collapse (v2: top-1 by chance 0/30, top-2 7-8/30) because chance gives every word its OWN error even on
lines where all words carry the same light: errors that a fixed rounding would have made identical for all words (and
so cancelling in the race) became independent. Subtracting the plate puts every photon on what differs between words.
Ladders: top-k nearest on the departure, k = 3, 2, 1 (E and T); word-wide cut nearest on the departure, E b = 6, 4, 2;
T b = 10, 8, 6.
Predicted before running: T improves at the same k over v1 (v1: k=3 29, k=2 25, k=1 21) -> k=2 27-29, k=1 23-27, if the
plate is a large part of T's loud lines; E stays 29-30 (it was already). Stored: fewer photons a value at the same
precision of the departure; the departure's top places are lower.
    python3 develop_v6_white_plate.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v6 plate + top-{k} nearest {tab}", **{tab: dict(how="dark", inner=dict(how="topk", k=k))})
         for tab in ("E", "T") for k in (3, 2, 1)]
SPECS += [dict(name=f"v6 plate + cut b={b} nearest E", E=dict(how="dark", inner=dict(how="cut", b=b))) for b in (6, 4, 2)]
SPECS += [dict(name=f"v6 plate + cut b={b} nearest T", T=dict(how="dark", inner=dict(how="cut", b=b))) for b in (10, 8, 6)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
