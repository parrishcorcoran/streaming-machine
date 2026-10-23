"""COMBINED v3 -- per depth exposure, on the whole machine (tables c=34; the other plates v12 A=8).
v2b negative, read: one development's words count is the exposure PLUS the one grain pattern that fell. The same
exposures on shifted grids (same step, same grain power, another pattern) gave c=34: 28,28,29,30 (nearest 30); c=35:
28,27,27,28 (nearest 30); c=36: 23,25,26,24 (nearest 25); c=37: 21,23,28,26 (nearest 28). A single count is the
level +- 2 words. So every comparison here is made on 5 frames of the same exposure (nearest + 4 shifted grids), and
the level is their average; the nearest frame is the one that would be stored.
Picture: the stream is a medium that keeps what is added to it, but every later crossing reads it through a gauge
(divides by its own size) and the stream grows with depth; grain laid early is a smaller share of the light by the
time it reaches the top, and its memory is about 8 crossings. Grain laid in the last 4 layers reaches the race with
no time to fade.
Test: one band of 4 layers three half places coarser (A=5: grain power x8 there), the other bands at A=8.
Each band at A=5 saves ~5.4 MB. Prediction before running (base level 29.0): layers 0-3 at A=5: ~28;
4-7: ~27; 8-11: ~26 -- the far end needs the finer exposure, the near end tolerates the coarser."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
BANDS = {"0-3": range(0, 4), "4-7": range(4, 8), "8-11": range(8, 12)}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(BANDS)
for b in sel:
    cfg = cfg_by(lambda n: ("v12", 5 if layer(n) in BANDS[b] else 8))
    r = judge(f"v3 layers {b} at A=5, rest A=8, tables c=34", cfg, 34, note="per depth", predicted={"0-3": "~28", "4-7": "~27", "8-11": "~26"}[b])
    w = judge_phase(f"v3 layers {b} at A=5, rest A=8, tables c=34", cfg, 34)
    print(f"band {b}: nearest {r['words']}, phases {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB", flush=True)
