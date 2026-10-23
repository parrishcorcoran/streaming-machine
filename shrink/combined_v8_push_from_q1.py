"""COMBINED v8 -- push down from v7 Q1 (67.74 MB, nearest 29, level 27.0: layers 4-7 A=5, end-layer mproj A=5,
end-layer qkv A=6, end-layer proj and fc A=8, tables c=35), and look for a smaller 30/30 near level 28.
v7 negatives, read in photons:
  - mproj below A=5 (Q3 middle A=3: 24.8; Q5 ends A=4: 25.4) and tables c=36 (Q2: 23.6) are cliffs. mproj's
    tolerance at A=5 came from its DARK rows (fc neurons below their edge for that word); the rows that ARE lit carry
    all of mproj's light into the stream, and at A=4 they keep about one photon a weight -- the lit rows' grain is
    then no longer small next to their light. A cut that is free on the dark rows is paid in full on the lit ones.
  - tables c=36 puts 711 of 768 lines at the cap (step 2^12, 1/16 on values of ~0.14 rms): one more place on the
    bulk of the lines at once doubles the race grain from the tables (4.7 MB for x4 in grain power).
  - v14 (fc and v neurons exposed for the next plate's line) was 2 words of level WORSE than v12 at the same size
    (25.2 vs 27.2, 5 frames each); round one's single frames had it the other way (26 vs 22 at A=5). One frame
    misled round one.
  - end-layer qkv 8->6 was nearly free (-0.2 words of level for 1.8 MB).
Levers (cumulative on Q1 unless said) and predictions of the 5-frame level before running:
  R1 + end qkv A=5                                     (-1.1 MB) ~26.5
  R2 + end proj A=5                                    (-0.9 MB) ~26.5
  R3 tables "c=35 1/2": a random half of the lines (fixed seed) one place coarser, the exposure halftoned across
     lines between c=35 and c=36                       (-2.4 MB) ~25.5
  R4 + end fc A=7                                      (-0.7 MB) ~26.5
  S1 layers 4-7 A=5, end qkv A=6, rest A=8, c=34       (76.2 MB) ~28.2
  S2 S1 + end mproj A=6                                (74.5 MB) ~27.8"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])
D0, _ = dl.develop_lines(E0[:8192], 35, 12); assert np.array_equal(develop_e(E0[:8192], exposure(35, 12)), D0); del D0
RUNS = {
    "R1": ("Q1 + end qkv A=5", alloc(5, 5, qkv_end=5), 35, None, "~26.5"),
    "R2": ("Q1 + end proj A=5", alloc(5, 5, qkv_end=6, proj_end=5), 35, None, "~26.5"),
    "R3": ("Q1 with tables c=35 1/2 (halftoned lines, seed 35)", alloc(5, 5, qkv_end=6), 35, 35, "~25.5"),
    "R4": ("Q1 + end fc A=7", alloc(5, 5, qkv_end=6, fc_end=7), 35, None, "~26.5"),
    "S1": ("layers 4-7 A=5, end qkv A=6, rest A=8, c=34", alloc(5, 8, qkv_end=6), 34, None, "~28.2"),
    "S2": ("layers 4-7 A=5, end qkv A=6, end mproj A=6, rest A=8, c=34", alloc(5, 6, qkv_end=6), 34, None, "~27.8"),
}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(RUNS)
for key in sel:
    name, cfg, c, half, pred = RUNS[key]
    r = judge(f"v8 {key} {name}", cfg, c, note="push from Q1 / near 30, 5 frames", predicted=pred, half=half)
    w = judge_phase(f"v8 {key} {name}", cfg, c, half=half)
    print(f"{key}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB "
          f"(plates {r['plates_MB']}, tables {r['tables_MB']}, rest {r['rest_MB']})", flush=True)
