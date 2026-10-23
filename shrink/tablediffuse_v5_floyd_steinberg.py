"""TABLEDIFFUSE v5 -- two-dimensional error diffusion, Floyd-Steinberg weights: a value's leftover goes 7/16 to the
next line of the same word, 3/16, 5/16, 1/16 to the previous, same and next line of the next word (vocabulary order),
whole numbers ((7u) >> 4 etc., the last share takes what is left so nothing is lost).
The relation: a value's grain is its own leftover minus 7/16, 1/16, 5/16, 3/16 of four neighbours' leftovers: power
1 + (49 + 1 + 25 + 9)/256 = 1.33 times plain rounding -- the gentlest carry, still more grain in every direction a
reader sees. Predicted: c=32 29-30, c=34 28-30, c=37 22-28; bytes as nearest +0-1%.
    python3 shrink/tablediffuse_v5_floyd_steinberg.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v5 Floyd-Steinberg 2-D, c={c}", c=c, how="fs") for c in (32, 34, 37)])
