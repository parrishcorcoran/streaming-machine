"""Checker: is the delivered machine's output really made by its own path? The CHECK PATH is broken on purpose (the exact
product at the plates and the word table's exact pile both return zeros), qwen_fringes.py is otherwise as delivered.
If the piles, streams and the read still equal the reference, they come from the walks, the band parts and the
signed-digit arrival, not from the check path. Texts 12 and 9."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import qwen_fringes as QF
OUT = "/Users/abundancemachine/streaming-machine/qwen_fringes/indep_check"
QF.direct = lambda x16, w: np.zeros((len(x16), np.asarray(w).shape[1]), np.int64)
QF.Table.pile = lambda self, X: np.zeros((len(X), self.N), np.int64)
def no_head(x16): raise RuntimeError("the machine asked for the plain head product")
QF.head_product = no_head
meta = json.load(open("/Users/abundancemachine/streaming-machine/qwen06b/light/meta.json")); RF = json.load(open(f"{OUT}/ref10.json"))
TS = [12, 9]
res = QF.forward([meta["texts"][t] for t in TS], "fr", light_cmp=False, log=lambda s: None)
print("in-machine checks (expected to fire, the check path is broken):", QF.ST["chk"])
bad = {}
for k, v in res["dig"].items():
    p = k.split("|"); p[-1] = str(TS[int(p[-1])]); bad.setdefault(p[0], [0, 0]); bad[p[0]][0] += 1; bad[p[0]][1] += v != RF["dig"]["|".join(p)]
print("machine with the check path broken vs the checker's reference:", {k: f"{d} of {n} differ" for k, (n, d) in bad.items()})
tab = QF.Table(log=lambda s: None); R = QF.band_read(tab, res["x16"], res["ex"], res["LS"], log=lambda s: None)
print("band read checks (expected to fire):", R["bad"])
rt5 = sum((RF["read"][str(t)]["top5"] for t in TS), []); rw = sum((RF["read"][str(t)]["win"] for t in TS), [])
print(f"band read with the check path broken: top 5 = reference {int((R['top5'] == np.array(rt5)).all(1).sum())}/{len(rt5)}, race winner {int((R['win'] == np.array(rw)).sum())}/{len(rw)}")
print(f"max memory {QF.rss()} MB")
