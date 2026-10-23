"""COMBINED v6 -- spend the grain where the light does not carry it: per kind AND per depth, with the tables, on the
whole machine, 5 frames each.
Read from v3/v4/v5 (levels = mean of 5 frames; base A=8 + c=34 = 29.0), words of level lost per MB saved:
   layers 4-7 (all kinds) 8->5: 0.12/MB   mproj 8->5: 0.19/MB   tables c 34->35: 0.29/MB   qkv 8->5: 0.33/MB
   proj 8->5: 0.45/MB   fc 8->5: 0.48/MB   layers 0-3 / 8-11 8->5: 0.55/MB
v4 negative (I predicted mproj the MOST sensitive; it is the least per MB), read in the picture: a plate's grain only
reaches the stream where light falls on its lines-in. mproj's lines-in carry the BENT light -- for any one word most
fc neurons sit below their edge and are dark, so most of mproj's rows are unlit and their grain is never seen. fc and
qkv are lit on all 768 lines-in by the gauged stream. The shade I placed after fc is real, but it shades mproj's input.
Information theory (reverse water-filling): the least bytes for a given grain at the race is where every source's
marginal cost is equal -- coarsen the cheapest first, stop when they meet.
Path (cumulative), prediction of the 5-frame level before running:
  P2 layers 4-7 all A=5, mproj of layers 0-3/8-11 A=5, the rest A=8, c=34          ~27.5
  P3 P2 + tables c=35                                                              ~26
  P4 layers 4-7 all A=4, mproj elsewhere A=5, rest A=8, c=34                       ~27
  P5 P2 + qkv of layers 0-3/8-11 A=6                                               ~27
(P1 = layers 4-7 all A=5 alone, measured in v3: nearest 30, frames 27 29 28 28, level 28.4, 77.96 MB.)"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
MID = range(4, 8)
def alloc(mid, mproj_end, qkv_end=8, proj_end=8, fc_end=8, mproj_mid=None, rule="v12"):
    def f(n):
        k, L = kind(n), layer(n)
        if L in MID: return (rule, mproj_mid if (k == "mproj" and mproj_mid is not None) else mid)
        return (rule, dict(qkv=qkv_end, proj=proj_end, fc=fc_end, mproj=mproj_end)[k])
    return cfg_by(f)
RUNS = {
    "P2": ("layers 4-7 A=5, end mproj A=5, rest A=8, c=34", alloc(5, 5), 34, "~27.5"),
    "P3": ("layers 4-7 A=5, end mproj A=5, rest A=8, c=35", alloc(5, 5), 35, "~26"),
    "P4": ("layers 4-7 A=4, end mproj A=5, rest A=8, c=34", alloc(4, 5), 34, "~27"),
    "P5": ("layers 4-7 A=5, end mproj A=5, end qkv A=6, rest A=8, c=34", alloc(5, 5, qkv_end=6), 34, "~27"),
}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(RUNS)
for key in sel:
    name, cfg, c, pred = RUNS[key]
    r = judge(f"v6 {key} {name}", cfg, c, note="allocation by where the light falls, 5 frames", predicted=pred)
    w = judge_phase(f"v6 {key} {name}", cfg, c)
    print(f"{key}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB "
          f"(plates {r['plates_MB']}, tables {r['tables_MB']}, rest {r['rest_MB']})", flush=True)
