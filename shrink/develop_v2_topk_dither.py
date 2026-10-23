"""DEVELOP family, version 2: as version 1 (each value keeps its top k photons, single-wave writing), but what is below
the last kept place q is rounded BY CHANCE: a random whole number in [0, q) is added and the value is cut at q (floor).
So each value is exact on average (the phone camera's tremor-shifted frames: the cut lands at a random spot each time).
The result never has more than k photons (an odd number +-1 at the last place never needs more photons).
Seed fixed per table (E: 100 + k, T: 200 + k); k = 2 also with a second seed, to see how much the chance itself moves
the verdict.
What version 1 measured: E 30 30 29 29 and T 29 29 25 21 for k = 4 3 2 1.
Predicted before running: dithering is WORSE than the nearest at the same k. The nearest error is at most q/3 (the
single-wave tail); the dithered error is a whole photon at q with chance |r|/q, about 3x the error power. Dithering only
pays where a hard cut throws a whole class of values away together (a dead zone: values under the cut all go dark, a
bias that does not average out over 768 lines). Top-k per value has no dead zone -- every value keeps its brightest
photon -- so there is nothing for the chance to rescue. Expected: E 30, 29-30, 27-29, 24-28; T 28-29, 27-29, 20-25, 12-20.
    python3 develop_v2_topk_dither.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v2 top-{k} dither {tab}", **{tab: dict(how="topk", k=k, seed=(100 if tab == 'E' else 200) + k)})
         for tab in ("E", "T") for k in (4, 3, 2, 1)]
SPECS += [dict(name=f"v2 top-2 dither {tab} (second seed)", **{tab: dict(how="topk", k=2, seed=(100 if tab == 'E' else 200) + 50)})
          for tab in ("E", "T")]
if __name__ == "__main__": run(SPECS, sys.argv[1])
