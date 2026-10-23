"""JUDGE5'S OWN CHECK of the spot-bound top read. For each writing, G and schedule of the main runs
(spot_<write>_real_s1.npz from spot_top.py), every one of the 300 judge5 windows gets back, at its 20 judged
positions, the exact scores of the words that run kept to the end (all of them, or their best 256; every other word
below them all). Then
  judge5.judge(forward=...)        -> in the reference top 5, same top word, out of 6,000
  judge5.top5 of the same scores   -> top 5 (set and order) equal to judge5's reference (judge5_ref.npz)
    python3 collapse_top/spot_judge.py
"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
NW, FLOOR, NREF = 50257, -(1 << 62), 480
REF = J._ref()
for write in ("signed", "plain"):
    p = f"{HERE}/collapse_top/spot_{write}_real_s1.npz"
    if not os.path.exists(p): continue
    Z = np.load(p); sel = Z["sel"]; assert np.array_equal(sel, np.arange(len(sel)))
    uids, usc, uoff = Z["uids"], Z["usc"], Z["uoff"]
    for G in Z["gs"]:
        for s in ("diag", "light"):
            G = int(G); kept, koff = Z[f"kept_{G}_{s}"], Z[f"koff_{G}_{s}"]; it = iter(range(300)); cnt = dict(ref=0, top=0)
            def fwd(ids):
                i = next(it); assert ids == J.WINDOWS[i]["ids"]
                S = np.full((20, NW), FLOOR, np.int64)
                for j in range(20):
                    q = NREF + 20 * i + j
                    u = uids[uoff[q]:uoff[q + 1]]; sc = dict(zip(u.tolist(), usc[uoff[q]:uoff[q + 1]].tolist()))
                    k = kept[koff[q]:koff[q + 1]]; S[j, k] = [sc[w] for w in k.tolist()]
                t5 = J.top5(S)
                cnt["ref"] += int((t5 == REF[i]).all(1).sum())
                cnt["top"] += int((t5 == Z[f"top_{G}_{s}"][NREF + 20 * i:NREF + 20 * i + 20]).all(1).sum())
                return S
            inn, same, tot = J.judge(forward=fwd, n=300)
            print(f"{write} G={G:2d} {s:5s}: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; "
                  f"top 5 (set and order) = judge5's reference {cnt['ref']}/{tot}; = the run's own top 5 {cnt['top']}/{tot}", flush=True)
