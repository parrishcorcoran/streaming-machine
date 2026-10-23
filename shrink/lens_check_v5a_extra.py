"""CHECKER for LENS v5a, part 3.
 (a) the T film coder's 8 shared frequency tables are chosen per line by the line's own mean size. The rules allow only
     random groups or a fixed rule that does not look at values; so re-code T with (i) one table for all lines and
     (ii) 8 tables on random groups of lines (seed 0), everything else the same; bytes counted, decode checked.
 (b) the words(...) test reads only the ~150 rows of E that the 30 prompts use: compare E^'s error on those rows
     with its error on all 50,257 rows (is the tested sample typical?).
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/lenscheck"
src = open("/Users/abundancemachine/streaming-machine/shrink/lens_v10_rans_film.py").read()
src = src[:src.index("E, T, TE, HB, WPE = load()")]
OLD = "grp = ((m >> k.astype(np.uint64)) >> np.uint64(1)).astype(np.int64)"
assert OLD in src
M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
T = np.ascontiguousarray(M["head.w"].T); E = M["wte"]
r = {}
for tag, new in (("one_table", "grp = np.zeros(Ln, np.int64)"),
                 ("random_8_groups_seed0", "grp = np.random.default_rng(0).integers(0, 8, Ln).astype(np.int64)")):
    ns = {}; exec(src.replace(OLD, new), ns)
    P = ns["encode_film"](T)
    b = sum(len(v) for k, v in P.items() if k != "shape") + 8
    ok = bool(np.array_equal(ns["decode_film"](P).astype(np.int16), T))
    r[f"T_film_bytes_{tag}"] = b; r[f"T_roundtrip_{tag}"] = ok
    print(tag, b, ok, flush=True)
    del P

refs = json.load(open("/Users/abundancemachine/streaming-machine/refs_many.json"))
rows = np.unique(np.concatenate([np.array((x["ids"] + x["ref"])[:5]) for x in refs]))
Eh = np.load(f"{SP}/Eh.npy")
d = np.abs(Eh - E)
r["tested_rows"] = int(rows.size)
r["mean_abs_dE_tested_rows"] = float(d[rows].mean()); r["mean_abs_dE_all_rows"] = float(d.mean())
r["max_abs_dE_tested_rows"] = int(d[rows].max()); r["max_abs_dE_all_rows"] = int(d.max())
rowmean = d.mean(1)
r["row_mean_abs_dE_pct_rank_of_worst_tested_row"] = float((rowmean <= rowmean[rows].max()).mean() * 100)
r["row_mean_abs_dE_quantiles_all"] = [float(np.percentile(rowmean, q)) for q in (50, 90, 99, 100)]
r["row_mean_abs_dE_quantiles_tested"] = [float(np.percentile(rowmean[rows], q)) for q in (50, 90, 100)]
print(json.dumps(r, indent=1))
json.dump(r, open(f"{SP}/part3.json", "w"), indent=1)
