"""Join the parts of a fine_staged_top.py run (fine_<tag>_p<i>of<n>.npz, i = 1..n) into fine_<tag>.npz: per-position
arrays put end to end, the saved final words' offsets carried on, the check counts added. Nothing is recomputed.
    python3 collapse_top/fine_join.py <tag> <n>        e.g.  diag_real 4
"""
import sys, numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
tag, n = sys.argv[1], int(sys.argv[2])
Z = [np.load(f"{D}/fine_{tag}_p{i}of{n}.npz") for i in range(1, n + 1)]
keys = Z[0].files
assert all(z.files == keys for z in Z) and all(np.array_equal(z["bad_keys"], Z[0]["bad_keys"]) and np.array_equal(z["ks"], Z[0]["ks"]) for z in Z)
out = {"ks": Z[0]["ks"], "bad_keys": Z[0]["bad_keys"], "bad": sum(z["bad"] for z in Z)}
for sx in ("", "_s"):
    offs = [Z[0][f"fin_off{sx}"]]
    for z in Z[1:]: offs.append(z[f"fin_off{sx}"][1:] + offs[-1][-1])
    out[f"fin_off{sx}"] = np.concatenate(offs)
for k in keys:
    if k not in out: out[k] = np.concatenate([z[k] for z in Z])
sel = out["sel"]; assert np.array_equal(sel, np.arange(len(sel))), "the parts must cover every position once, in order"
for sx in ("", "_s"): assert out[f"fin_off{sx}"][-1] == len(out[f"fin_idx{sx}"]) and len(out[f"fin_off{sx}"]) == len(sel) + 1
np.savez(f"{D}/fine_{tag}.npz", **out)
print(f"fine_{tag}.npz: {len(sel)} positions from {n} parts; checks " + ", ".join(f"{k} {v}" for k, v in zip(out["bad_keys"].tolist(), out["bad"].tolist())))
