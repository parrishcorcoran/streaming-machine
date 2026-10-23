"""DEVELOP family, version 1: each value keeps only its top k photons in the single-wave writing (up +1 / hole -1,
never two lit side by side); what is below the last kept place is rounded to the nearest. Because a single-wave tail
is never more than a third of the last kept place, the nearest IS the top k photons exactly (checked in code).
Applied to E alone and to T alone, k = 4, 3, 2, 1. Stored as photon rounds (the brightest photon of every value, then
the next, ...), range-coded and lzma'd, decoded and checked.
Before: E has 5.01 photons a value, T 3.82.
Predicted before running (words/30):
  E: k=4 30, k=3 30, k=2 30, k=1 26-29   (only 109 of E's 50,257 words are ever lit by the 30 inputs)
  T: k=4 30, k=3 30, k=2 28-30, k=1 15-24 (every one of T's words races at the top on every input)
Stored: k photons cost about (top place ~3 bits + sign) + (k-1) x (gap ~2 bits + up/hole) -> E k=2 ~6.4 bits/value
(measured on a slice), k=1 ~3.5-4.
    python3 develop_v1_topk_nearest.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v1 top-{k} nearest {tab}", **{tab: dict(how="topk", k=k)}) for tab in ("E", "T") for k in (4, 3, 2, 1)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
