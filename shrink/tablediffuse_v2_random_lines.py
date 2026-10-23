"""TABLEDIFFUSE v2 -- the carry of v1 along a word's lines taken in a RANDOM line order (fixed seeds 1 and 2; the same
order for every word). Line i and line i+1 are not neighbours for any reader, so natural order has no meaning the
random order lacks: this is v1's order control.
Predicted: same grain power as v1 (2/12 step^2), words equal to v1 within the judge's luck (+-1-2 at c=32/34, +-3 at
c=37), both seeds alike; stored bytes equal to v1 within 0.1%.
    python3 shrink/tablediffuse_v2_random_lines.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v2 carry to next line, random order seed={s}, c={c}", c=c, how="lines", line_seed=s)
        for s in (1, 2) for c in (32, 34, 37)])
