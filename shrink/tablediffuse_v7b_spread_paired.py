"""TABLEDIFFUSE v7b -- v7 paired with the screen phases. v7's spread carry changes only 1.3% of values (one step
each), so each v7 table is nearly the nearest table it started from, and its words largely inherit that table's luck.
The fair test lays the spread on each of v0's four screen phases (independent grain realizations) and compares PAIR BY
PAIR: phase s nearest vs phase s with each word's grain cancelled over its lines.
Predicted (the cancellation lands in the white level that the gauge removes): every pair equal within +-1, no common
sign. c=37 nearest phases 27, 24, 23, 27 -> spread 26-28, 23-25, 22-24, 26-28; c=34 29, 29, 30, 29 -> 28-30 each.
If cancelling a word's grain over its lines mattered, the spread column would sit above the nearest column.
    python3 shrink/tablediffuse_v7b_spread_paired.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run_phase((34, 37), (1, 2, 3, 4), how="spread", label="v7b spread on screen phase")
