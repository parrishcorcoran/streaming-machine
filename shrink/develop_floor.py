"""DEVELOP family: reference only (not a codec). For the developed tables on the frontier, the Shannon floor of the
stored whole numbers with each LINE its own histogram (what an ideal coder with the line as context needs, the model
itself not counted), next to the real range-coder bytes recorded in develop_results.json. Says whether the sizes are set
by the coder or by the information in the developed table.     python3 develop_floor.py
"""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import *
E, T = load_tables()
rows = {r["name"]: r for r in json.load(open(OUT))}
def line_floor(Q):                                      # bits, each line its own histogram
    tot = 0.0
    for i in range(Q.shape[1]):
        _, c = np.unique(Q[:, i], return_counts=True); p = c / c.sum(); tot += float(-(c * np.log2(p)).sum())
    return tot
for c, name in ((32, "v9 E lens exposure c=32 cap 12, T printed"), (34, "v9 E lens exposure c=34 cap 12, T printed"),
                (37, "v9 E lens exposure c=37 cap 12, T printed")):
    D, e = develop_lines(E, c, 12); Q = D.astype(np.int64) >> e[None, :]
    f = line_floor(Q) / 8e6; rc = rows[name]["store_E"]["rc_bytes"] / 1e6
    print(f"c={c}: floor {f:.3f} MB  range coder {rc:.3f} MB  ({100 * (rc / f - 1):+.1f}%)")
D, P = develop_cut(T, 8); Q = D.astype(np.int64) >> P[:, None]
print(f"T cut b=8: floor {line_floor(Q) / 8e6:.3f} MB  range coder {rows['v3 cut b=8 nearest T']['store_T']['rc_bytes'] / 1e6:.3f} MB")
D, P = develop_cut(E, 4); Q = D.astype(np.int64) >> P[:, None]
print(f"E cut b=4: floor {line_floor(Q) / 8e6:.3f} MB  range coder {rows['v3 cut b=4 nearest E']['store_E']['rc_bytes'] / 1e6:.3f} MB")
