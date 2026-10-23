"""COMBINED v9 -- the last cheap levers, and the 30/30 lottery stated as one.
v8 read, words of level per MB saved at the Q1 point (5 frames each, SE ~0.6):
   qkv end 6->5 0.4/1.1 = 0.36/MB   fc end 8->7 0.4/0.73 = 0.55/MB   tables c 35->35 1/2 2.2/2.3 = 0.95/MB
   mproj end 5->4 1.6/MB            proj end 8->5 1.6/0.9 = 1.8/MB
The cheap grain left is in qkv and in the end layers' fc; proj, mproj and the tables are past their knees.
proj, read in photons: it is only 7.1M weights, so coarsening it saves little, but every one of its rows is lit (the
arrival's output is an average of v over positions -- never dark) and its output goes straight into the stream:
paying the full grain for a small saving.
Levers (on R1 = 66.63 MB, nearest 29, level 26.6), predictions of the 5-frame level before running:
  T1 R1 + end fc A=7                       (65.9 MB)  ~26.2
  T2 R1 + all qkv A=4                      (65.6 MB)  ~26.2
  T3 T1 + T2                               (64.9 MB)  ~25.8
  T4 R1 + end fc A=6                       (64.3 MB)  ~25.5
30/30 needs text 12 (lead 0.0074) to survive: a coin at any grain above ~0.01, so a stored 30/30 below 76 MB is a
lucky frame at a level near 27-28. Three tries, each predicted 'level ~27-27.5, nearest 30 with chance ~1/5':
  U1 layers 4-7 A=5, end mproj A=6, rest A=8, c=34 (~75.3 MB)
  U2 S1 + end fc A=7 (~75.4 MB)        U3 S2 + end fc A=7 (~73.1 MB)"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])
def qkv_all(cfg, A): return {n: ((cfg[n][0], A) if kind(n) == "qkv" else cfg[n]) for n in NAMES}
RUNS = {
    "T1": ("R1 + end fc A=7", alloc(5, 5, qkv_end=5, fc_end=7), 35, "~26.2"),
    "T2": ("R1 + all qkv A=4", qkv_all(alloc(5, 5, qkv_end=5), 4), 35, "~26.2"),
    "T3": ("R1 + end fc A=7 + all qkv A=4", qkv_all(alloc(5, 5, qkv_end=5, fc_end=7), 4), 35, "~25.8"),
    "T4": ("R1 + end fc A=6", alloc(5, 5, qkv_end=5, fc_end=6), 35, "~25.5"),
    "U1": ("layers 4-7 A=5, end mproj A=6, rest A=8, c=34", alloc(5, 6), 34, "~27.5, 30 by luck"),
    "U2": ("S1 + end fc A=7", alloc(5, 8, qkv_end=6, fc_end=7), 34, "~27, 30 by luck"),
    "U3": ("S2 + end fc A=7", alloc(5, 6, qkv_end=6, fc_end=7), 34, "~27, 30 by luck"),
}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(RUNS)
for key in sel:
    name, cfg, c, pred = RUNS[key]
    r = judge(f"v9 {key} {name}", cfg, c, note="last cheap levers / 30 lottery, 5 frames", predicted=pred)
    w = judge_phase(f"v9 {key} {name}", cfg, c)
    print(f"{key}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB "
          f"(plates {r['plates_MB']}, tables {r['tables_MB']}, rest {r['rest_MB']})", flush=True)
