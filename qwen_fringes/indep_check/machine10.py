"""Checker: qwen_fringes.py AS DELIVERED (imported, nothing changed) run on the 10 drawn texts.
phase 'fr': its forward in mode "fr" (way 2 + the 3x walk at every plate, the arrival in signed digits); saves its digests,
its counts per position and its own in-machine checks, and the top light.
phase 'top': its Table and band_read on that top light; saves the read and its counts."""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import qwen_fringes as QF
OUT = "/Users/abundancemachine/streaming-machine/qwen_fringes/indep_check"
T10 = [3, 4, 5, 7, 8, 9, 12, 16, 18, 22]
meta = json.load(open("/Users/abundancemachine/streaming-machine/qwen06b/light/meta.json"))
ph = sys.argv[1]; t0 = time.time()
if ph == "fr":
    res = QF.forward([meta["texts"][t] for t in T10], "fr", light_cmp=False)
    print("in-machine checks:", QF.ST["chk"])
    dig = {}
    for k, v in res["dig"].items():
        p = k.split("|"); p[-1] = str(T10[int(p[-1])]); dig["|".join(p)] = v
    json.dump(dict(dig=dig, chk=QF.ST["chk"], rows=res["rows"], zg=QF.ST["zg"]), open(f"{OUT}/machine10.json", "w"))
    QF.save_counts(f"{OUT}/machine10_counts.npz", res["cnt"])
    np.savez(f"{OUT}/machine10_top_light.npz", x16=res["x16"], ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
else:
    Z = np.load(f"{OUT}/machine10_top_light.npz")
    tab = QF.Table()
    R = QF.band_read(tab, Z["x16"], Z["ex"], Z["LS"])
    print("band read checks:", R["bad"])
    np.savez(f"{OUT}/machine10_top.npz", **{k: v for k, v in R.items() if k not in ("bad", "sha")})
    json.dump(dict(bad=R["bad"], sha=R["sha"]), open(f"{OUT}/machine10_top.json", "w"))
print(f"done ({time.time() - t0:.0f}s, max memory {QF.rss()} MB)")
