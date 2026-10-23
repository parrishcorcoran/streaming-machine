"""CHECKER'S OWN COUNT of the Qwen3-0.6B group machine's plates on refs texts 5 and 12 (32 positions), from qwen_base2's
fresh run (qwen_ref.npz: the light and the pile of every slab), by mycount.py: every slab in the per-place group form,
g = 12, the grouping key [1, slab, layer, 12], each plate's rule from the plate alone; every pile checked against
qwen_base2's pile. Writes count_qwen.json (per text, per slab and plate, every item).
"""
import os, sys, json, time, zipfile, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"; sys.path.insert(0, OUT)
from mycount import slab_work
NPZ = f"{HERE}/qwen/qwen_base2.npz"
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
QSEG = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
        "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
Mz = np.load(NPZ); Z = np.load(f"{OUT}/qwen_ref.npz"); t0 = time.time(); res = {}; masks = {}
for t in (5, 12):
    for l in range(28):
        for slab, segs in QSEG.items():
            name = f"{slab}.{l}"; x = Z[f"r{t}__light__{name}"]; ref = Z[f"r{t}__pile__{name}"]
            w = np.asarray(Mz[name + ".w"]).astype(np.int64)
            pile, c, ch = slab_work(x, w, segs, 12, [1, SLABS.index(slab), l, 12])
            assert np.array_equal(pile, ref), f"{name}: the checker's group pile differs from qwen_base2's"
            res[f"r{t}|{name}"] = {k: int(v.sum()) for k, v in c.items()}
            for k_, m in ch.items(): masks[f"{k_}.{l}"] = [b for b in range(16) if not (m >> b) & 1]
    print(f"text {t} done ({time.time() - t0:.0f}s)", flush=True)
json.dump(dict(res=res, masks=masks), open(f"{OUT}/count_qwen.json", "w"))
print(f"done ({time.time() - t0:.0f}s)")
