"""COMBINED v7 -- push down from v6 P3 (69.51 MB, nearest 29, level 27.2: layers 4-7 A=5, mproj of the end layers A=5,
the rest A=8, tables c=35), one lever at a time, 5 frames each.
v6 read: the tables' c=35 cost only 0.6 words of level on top of P2 (alone on A=8 plates it cost 1.4): the race grain
from the tables and from the plates add in power, and the words lost grow slower than the grain once the knife-edge
texts (12, 16, 7) are already coins -- the next texts lead by 0.15-0.3, so the loss curve flattens between ~0.1 and
~0.2 of grain. Middle layers at A=4 (P4) cost 1.8 words: the middle's cheap grain is used up at A=5.
Levers and predictions (5-frame level) before running:
  Q1 + qkv of the end layers A=6              (-2.1 MB)  ~26.4
  Q2 tables c=36                              (-4.7 MB)  ~24.5
  Q3 + mproj of layers 4-7 A=3                (-1.2 MB)  ~27
  Q4 the same exposures under round one's v14 rule (fc and v neurons exposed for the next plate's line they light)
                                              (-0.3 MB)  ~27.5
  Q5 + mproj of the end layers A=4            (-1.1 MB)  ~26.8"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])
RUNS = {
    "Q1": ("P3 + end qkv A=6", alloc(5, 5, qkv_end=6), 35, "~26.4"),
    "Q2": ("P3 with tables c=36", alloc(5, 5), 36, "~24.5"),
    "Q3": ("P3 + mid mproj A=3", alloc(5, 5, mproj_mid=3), 35, "~27"),
    "Q4": ("P3 under rule v14", alloc(5, 5, rule="v14"), 35, "~27.5"),
    "Q5": ("P3 + end mproj A=4", alloc(5, 4), 35, "~26.8"),
}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(RUNS)
for key in sel:
    name, cfg, c, pred = RUNS[key]
    r = judge(f"v7 {key} {name}", cfg, c, note="push from P3, 5 frames", predicted=pred)
    w = judge_phase(f"v7 {key} {name}", cfg, c)
    print(f"{key}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB "
          f"(plates {r['plates_MB']}, tables {r['tables_MB']}, rest {r['rest_MB']})", flush=True)
