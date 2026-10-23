"""COMBINED look: how far apart the two leading words are at the race, for each of the 30 texts, in the UNCHANGED
machine (whole numbers at 2^16 of the head's output; 65536 = one unit). Not a judge of anything developed: it tells
which texts sit on a knife edge, so a lost word there is a coin toss of the grain, not a trend."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
t = tb
def race_vals(ids, wte):
    P = len(ids); R = wte[ids] + t.M["wpe"][:P]
    for l in range(12):
        R = R + t.plain_bank(t.attention(t.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + t.plain_bank(t.edge(t.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = t.W["head"]; x16, ex, LS = t.gauge_log(R[-1:])
    return t.power(t.product(x16, w), (we[None, :] + ex[:, None] + t.ES) * 65536 - LS[:, None])[0] + b
t.W["head"] = (np.ascontiguousarray(t.M["head.w"]).astype(np.float64), t.W["head"][1], t.W["head"][2])
rows = []
for i, ids in enumerate(t.inputs):
    v = race_vals(ids, E0); o = np.argsort(-v)[:2]
    rows.append((i, int(v[o[0]] - v[o[1]]), int(o[0]) == t._TRUTH[i]))
for i, m, ok in sorted(rows, key=lambda r: r[1]):
    print(f"text {i:2d}: lead of the right word over the next = {m:8d} / 65536 = {m/65536:.4f}", flush=True)
