"""COMBINED v5 -- the depth profile at finer grain: two layers at a time four half places coarser (A=4, grain power x16
there), the rest A=8, tables c=34, 5 frames each (nearest + 4 shifted grids).
v3 read: the middle band (4-7) took three half places for 0.6 words of level; both ends lost ~3. My prediction (the
near end fades, so it tolerates more) was wrong at the near end. Reading in the picture: v12 exposes every plate
relative to its OWN light, so its grain is a fixed share of what the plate adds. What decides is that share of the
stream at the race. Early plates make most of the stream's light while it is still small (their grain is a big share,
and the medium keeps it ~8 crossings -- long enough to reach layer 8+); the last plates add the largest pieces and
reach the race with no fade; the middle adds small pieces to an already bright stream.
Prediction before running (levels): a U across depth: pairs 0-1 and 10-11 lowest (~24-25), 4-5 and 6-7 highest
(~27-28), 2-3 and 8-9 between."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
sel = [int(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(6)
for i in sel:
    p = PAIRS[i]
    cfg = cfg_by(lambda n: ("v12", 4 if layer(n) in p else 8))
    r = judge(f"v5 layers {p[0]}-{p[1]} at A=4, rest A=8, tables c=34", cfg, 34, note="depth pairs, 5 frames",
              predicted={0: "24-25", 5: "24-25", 2: "27-28", 3: "27-28"}.get(i, "25-27"))
    w = judge_phase(f"v5 layers {p[0]}-{p[1]} at A=4, rest A=8, tables c=34", cfg, 34)
    print(f"layers {p}: nearest {r['words']}, phases {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB", flush=True)
