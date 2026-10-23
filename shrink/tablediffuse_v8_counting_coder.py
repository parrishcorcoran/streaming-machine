"""TABLEDIFFUSE v8 -- keep shrinking where the negatives point. v1-v7 showed that nothing in the arrangement of the grain
reaches a reader: every read sees the grain's power, line by line. At a fixed grain the bits are set by the stored
numbers themselves, and the per-line contexts already hold everything (the developed values are independent across
lines and words: a word's loudness, the previous line, the previous word give 0.1-0.2%). What is left in the coder is
its own jitter: round one's coder nudges each chance by 1/32 after every bit and never settles -- 2.6% (c=32) to 3.3%
(c=37) above the per-line floor. The counting coder (tablediffuse_rc.c) learns each chance as a running share, nudging
by 1/(n+2) until n = 255, then by 1/257 (a slow window: the statistics drift along the vocabulary, so a pure count,
n up to 65,535, does worse -- measured +1.8% to +3.4%).
The SAME developed tables as round one (nearest, lens-matched, cap 12): the words cannot change; only the bytes.
Also: the cube's luck at c=33 and c=35 (screen phases 1-4, as v0 did for 32, 34, 37), to finish the reference curve.
Predicted: c=32 33.95 MB (30/30), c=33 ~29.1 (28), c=34 24.36 (29), c=35 ~19.6 (28), c=37 14.57 (28) -- words exactly
round one's. Phases: c=33 27-30, c=35 26-29.
    python3 shrink/tablediffuse_v8_counting_coder.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v8 counting coder, nearest c={c}", c=c, how="nearest") for c in (32, 33, 34, 35, 37)], coder="rc2")
td.run_phase((33, 35), (1, 2, 3, 4))
