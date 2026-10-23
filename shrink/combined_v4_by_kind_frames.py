"""COMBINED v4 -- per bank kind again, now read on 5 frames (nearest + 4 shifted grids), whole machine (tables c=34).
v1 negative, read: every one-kind change lost 1-2 words in ONE frame, among the knife-edge texts; v2b showed one frame
is the level +- 2, so v1 could not tell the kinds apart. v3 (per depth, 5 frames) did separate: layers 4-7 three half
places coarser kept level 28.4 (base 29.0), layers 0-3 26.0, layers 8-11 26.2.
Picture per kind (where the grain goes after the plate):
  fc   : through the bend -- a neuron below its edge passes almost nothing, so most of its grain is shaded.
  qkv  : q, k grain moves arrival times a little; v grain is averaged over positions by the arrival.
  proj : straight into the stream, but the arrival's output that lights it is an average (dim, smooth light).
  mproj: lit by the bent light (few neurons, bright) and straight into the stream: no shade at all.
Test: one kind three half places coarser (A=5), the others A=8. Saves: qkv 4.27, proj 1.33, fc 4.63, mproj 5.37 MB.
Predictions before running (level of 5 frames; base 29.0): proj 28.5, fc 27.5, qkv 27, mproj 26."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
PRED = dict(proj="28.5", fc="27.5", qkv="27", mproj="26")
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else ["proj", "fc", "qkv", "mproj"]
for k in sel:
    cfg = cfg_by(lambda n: ("v12", 5 if kind(n) == k else 8))
    r = judge(f"v4 {k} at A=5, rest A=8, tables c=34", cfg, 34, note="per kind, 5 frames", predicted=PRED[k])
    w = judge_phase(f"v4 {k} at A=5, rest A=8, tables c=34", cfg, 34)
    print(f"kind {k}: nearest {r['words']}, phases {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB", flush=True)
