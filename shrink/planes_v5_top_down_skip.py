"""PLANES family, version 5: top places nearly empty -- skip them. JPEG2000's "zero bit planes": the table is cut into
fixed 64 x 64 blocks (64 words x 64 lines, table order, no look at the values to choose them); each block stores how
many places can be lit in it (5 bits of side information a block), and every place above that is known dark and not
coded at all. Contexts otherwise as version 3a (higher places, line, sign context line x top place).
Prediction: nothing to gain. A block of 4,096 values almost always holds at least one value lit in place 15 (E has
66,849 such values over 9,432 blocks), so at best place 16 is skipped -- and place 16 already costs only 0.002 bits a
value, because an adaptive coder pays -log2(1 - p) ~ p x 1.44 bits for a dark pixel in a place with lit share p. The
5-bit side information (0.0012 bits a value) eats what is left.
    python3 planes_v5_top_down_skip.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
for tb in ["E", "T"]:
    A = load(tb)
    bt = blocktops(A); print(f"   {tb}: blocks {bt.size}, lit-place counts per block: {np.bincount(bt.ravel()).tolist()}")
    run("v5 empty top places skipped per 64x64 block", tb, A, M_HIGH | M_LINE | M_SIGNC | M_SKIP, skip=True)
    del A
