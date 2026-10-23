"""DEVELOP family, version 3: one exposure per word instead of one per value. The word's loudest value reaches place
top-1; every value of that word keeps only the places >= top - b (one fixed place cut for the whole word, from a fixed
rule on the word's own loudest value), rounded to the nearest. Values below half the cut go dark.
Stored: the whole numbers above the cut (range coder, the line as context) + the 50,257 cut places.
E's words reach places 15-17 (typical value 13-14 places); T's all reach 15 (typical value 10).
Ladders: E b = 10, 8, 6, 4; T b = 12, 10, 8, 6.
Predicted before running:
  E: b=10 30, b=8 30, b=6 29-30, b=4 20-28 ; stored ~8.5, 6.5, 4.5, 2.5 bits/value
  T: b=12 29-30, b=10 28-29, b=8 20-26, b=6 5-15 ; stored ~7.5, 5.5, 3.5, 1.8 bits/value
Per value, a word-wide cut spends its places where the value is loud (uniform steps); top-k spends them on each value's
own size (steps proportional to the value). Which serves the read better is the question this version asks.
    python3 develop_v3_cut_nearest.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v3 cut b={b} nearest E", E=dict(how="cut", b=b)) for b in (10, 8, 6, 4)]
SPECS += [dict(name=f"v3 cut b={b} nearest T", T=dict(how="cut", b=b)) for b in (12, 10, 8, 6)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
