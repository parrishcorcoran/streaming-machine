"""Checker: the order-free check again, on texts 12 and 4 with seeds the builder did not use (order 11; words relabelled
with seed 13): every line order random, every walk down, the arrival's lines random; compared with the checker's
reference (digests, read) and with the delivered machine's main-order counts on the same texts (machine10_counts.npz)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import qwen_fringes as QF
OUT = "/Users/abundancemachine/streaming-machine/qwen_fringes/indep_check"
meta = json.load(open("/Users/abundancemachine/streaming-machine/qwen06b/light/meta.json")); RF = json.load(open(f"{OUT}/ref10.json")); MJ = json.load(open(f"{OUT}/machine10.json"))
T10 = RF["T10"]; TS = [12, 4]
res = QF.forward([meta["texts"][t] for t in TS], "fr", order=11, light_cmp=False, log=lambda s: None)
print("checks:", QF.ST["chk"])
bad = {}
for k, v in res["dig"].items():
    p = k.split("|"); p[-1] = str(TS[int(p[-1])]); bad.setdefault(p[0], [0, 0]); bad[p[0]][0] += 1; bad[p[0]][1] += v != RF["dig"]["|".join(p)]
print("order-free vs the reference:", {k: f"{d} of {n} differ" for k, (n, d) in bad.items()})
M10 = np.load(f"{OUT}/machine10_counts.npz"); loc = np.concatenate([np.arange(*MJ["rows"][T10.index(t)]) for t in TS])
cd = [k for k, v in res["cnt"].items() if not np.array_equal(v, M10[k.replace("|", "__")][loc])]
print(f"order-free counts vs the main order's: {len(cd)} of {len(res['cnt'])} arrays differ {cd[:5]}")
wperm = np.random.default_rng([13, 151936]).permutation(151936)
tab = QF.Table(word_perm=wperm, log=lambda s: None)
R = QF.band_read(tab, res["x16"], res["ex"], res["LS"], line_rng=np.random.default_rng([13, 1024]), down=True, log=lambda s: None)
print("band read checks:", R["bad"])
rt5 = sum((RF["read"][str(t)]["top5"] for t in TS), []); rw = sum((RF["read"][str(t)]["win"] for t in TS), [])
print(f"order-free band read: top 5 = reference {int((R['top5'] == np.array(rt5)).all(1).sum())}/{len(rt5)}, race winner {int((R['win'] == np.array(rw)).sum())}/{len(rw)}")
TM = np.load(f"{OUT}/machine10_top.npz")
print("band read count arrays differing from the main order's:", [k for k in ("walk", "writing", "pours", "hall", "bound", "end", "plain_meet", "plain_div", "surv", "cmp", "n", "mu") if not np.array_equal(R[k], TM[k][loc])])
print(f"max memory {QF.rss()} MB")
