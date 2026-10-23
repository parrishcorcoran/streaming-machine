"""Every row collapsed as before (layers 0-10's buckets, the streams into layers 1-10), and now layer 11's row of 3,072
too: as one line, or split into groups (3 x 1,024, 12 x 256), each group its own line. Right next word out of 30.
    python3 row_collapse_last.py
"""
import sys, json, numpy as np
exec(open("row_collapse_all.py").read().split("def run(")[0])
def as_lines(v, g):                                    # the row split into groups of g, each group struck into its own line
    B = bank(g); return np.concatenate([as_line(v[:, i:i + g], B) for i in range(0, v.shape[1], g)], 1)
def run(ids, last=None):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        if 0 < l < 11: R = as_line(R, B768)
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = edge(gauged_bank(R, f"fc.{l}"))
        if l < 11: H = as_line(H, B3072)
        elif last: H = as_lines(H, last)                                     # the 12th row
        R = R + plain_bank(H, f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]                     # (everything but the 12th row collapsed: 30/30 before)
for name, g in [("12th row as one line of 3,072", 3072), ("12th row in 3 groups of 1,024", 1024), ("12th row in 12 groups of 256", 256)]:
    plain_truth = truth
    print(f"  {name:32s}: right next word {sum(run(ids, g) == t for ids, t in zip(inputs, plain_truth))} / 30", flush=True)
