"""COMBINED v10 -- the 30/30 lottery at the tables' knee (c=35), stated as a lottery before it runs.
Reading of v2b/v3-v9: 30/30 hangs on text 12, whose right word leads the next by 0.0074 of a unit at the race; any
grain at the race above ~0.01 makes it a coin. Every point below 77 MB has a 5-frame level of 26-28.4, so a stored
30/30 there is a frame that happened to fall on the right side of text 12 (and of 16, 7). v8 S1 (75.53 MB, level 27.2)
was such a frame. With the final table coder c=35 tables cost 19.55 MB (c=34: 24.31), 4.76 MB less.
Three tries, each predicted 'level ~26.5-27.5, nearest 30 with chance about 1 in 5' (so most likely none):
  W1 layers 4-7 A=5, rest A=8, c=35            (~72.5 MB)
  W2 W1 + end qkv A=6 (S1 at c=35)              (~70.8 MB)
  W3 W1 + end mproj A=6 (U1 at c=35)            (~70.2 MB)"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])
RUNS = {
    "W1": ("layers 4-7 A=5, rest A=8, c=35", alloc(5, 8), 35),
    "W2": ("layers 4-7 A=5, end qkv A=6, rest A=8, c=35", alloc(5, 8, qkv_end=6), 35),
    "W3": ("layers 4-7 A=5, end mproj A=6, rest A=8, c=35", alloc(5, 6), 35),
}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(RUNS)
for key in sel:
    name, cfg, c = RUNS[key]
    r = judge(f"v10 {key} {name}", cfg, c, note="30/30 tries at c=35, 5 frames", predicted="level ~27, 30 by luck (~1/5)")
    w = judge_phase(f"v10 {key} {name}", cfg, c)
    print(f"{key}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB "
          f"(plates {r['plates_MB']}, tables {r['tables_MB']}, rest {r['rest_MB']})", flush=True)
