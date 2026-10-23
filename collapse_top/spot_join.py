"""Join the parts of a spot_top.py run (spot_<tag>_p<i>of<n>.npz, i = 1..n) into spot_<tag>.npz: per-position arrays
put end to end, the kept words' and the union's offsets carried on, the check counts added. Nothing is recomputed.
    python3 collapse_top/spot_join.py <tag> <n>        e.g.  signed_real_s1 6
"""
import sys, numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
tag, n = sys.argv[1], int(sys.argv[2])
Z = [np.load(f"{D}/spot_{tag}_p{i}of{n}.npz") for i in range(1, n + 1)]
keys = Z[0].files
SAME = ("gs", "diag_t", "light_c", "pl", "bad_keys", "store", "parts_names")
assert all(z.files == keys for z in Z) and all(np.array_equal(z[k], Z[0][k]) for z in Z for k in SAME)
out = {k: Z[0][k] for k in SAME}; out["bad"] = sum(z["bad"] for z in Z)
for k in keys:
    if k in out: continue
    if k == "uoff" or k.startswith("koff_"):
        offs = [Z[0][k]]
        for z in Z[1:]: offs.append(z[k][1:] + offs[-1][-1])
        out[k] = np.concatenate(offs)
    else: out[k] = np.concatenate([z[k] for z in Z])
sel = out["sel"]; assert len(np.unique(sel)) == len(sel) and (np.diff(sel) > 0).all()
assert out["uoff"][-1] == len(out["uids"]) and len(out["uoff"]) == len(sel) + 1
for k in keys:
    if k.startswith("koff_"): assert out[k][-1] == len(out["kept_" + k[5:]]) and len(out[k]) == len(sel) + 1
np.savez(f"{D}/spot_{tag}.npz", **out)
print(f"spot_{tag}.npz: {len(sel)} positions from {n} parts; checks " +
      ", ".join(f"{k} {v}" for k, v in zip(out["bad_keys"].tolist(), out["bad"].tolist())))
