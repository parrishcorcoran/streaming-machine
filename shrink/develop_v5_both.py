"""DEVELOP family, version 5: both tables developed at once, at the levels versions 1-4 measured for each alone (no
search: the three pairs are fixed from the table of single-table results).
  a: E cut b=4 nearest (alone 30, 17.08 MB) + T cut b=12 nearest (alone 30, 41.24 MB)    -> the smallest pair at 30+30
  b: E cut b=4 nearest (alone 30)          + T cut b=10 nearest (alone 28, 31.41 MB)
  c: E top-1 nearest (alone 29, 16.31 MB)  + T cut b=8 nearest (alone 28, 21.72 MB)      -> the smallest pair at 29+28
Predicted before running: a 29-30, b 27-29, c 26-28. The misses at the top are near-ties (the closest input's first two
words are 0.053 apart in GPT-2's own score; the next 0.157, 0.168, 0.224), so errors entering at the bottom (E) and at
the top (T) add their pushes on the same few knife-edge inputs.
    python3 develop_v5_both.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [
    dict(name="v5a both: E cut b=4 + T cut b=12 (nearest)", E=dict(how="cut", b=4), T=dict(how="cut", b=12)),
    dict(name="v5b both: E cut b=4 + T cut b=10 (nearest)", E=dict(how="cut", b=4), T=dict(how="cut", b=10)),
    dict(name="v5c both: E top-1 + T cut b=8 (nearest)", E=dict(how="topk", k=1), T=dict(how="cut", b=8)),
]
if __name__ == "__main__": run(SPECS, sys.argv[1])
