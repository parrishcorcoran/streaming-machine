"""JUDGE5'S OWN CHECK of the staged top read. For each schedule (staged_<sched>_real.npz from staged_top.py), every one
of the 300 judge5 windows gets back, at its 20 judged positions, the exact scores of the words the staged read kept to
the end (all of them, or their best 256 when more were kept; every other word below them all). Then
  judge5.judge(forward=...)                 -> in the reference top 5, same top word, out of 6,000
  judge5.top5 of the same scores            -> top 5 (set and order) equal to judge5's reference (judge5_ref.npz)
    python3 collapse_top/judge_check.py
"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
NW, FLOOR = 50257, -(1 << 62)
REF = J._ref()
for sc in ("stamp", "light", "diag"):
    Z = np.load(f"{HERE}/collapse_top/staged_{sc}_real.npz")
    sel, off, fi, fs, top_st = Z["sel"], Z["fin_off"], Z["fin_idx"], Z["fin_sc"], Z["top_st"]
    NREF = 480; q_of = {int(g): q for q, g in enumerate(sel)}
    it = iter(range(300)); same_ref = same_saved = 0
    def fwd(ids):
        global same_ref, same_saved
        i = next(it); assert ids == J.WINDOWS[i]["ids"]
        S = np.full((20, NW), FLOOR, np.int64)
        for j in range(20):
            q = q_of[NREF + 20 * i + j]; S[j, fi[off[q]:off[q + 1]]] = fs[off[q]:off[q + 1]]
        t5 = J.top5(S)
        same_ref += int((t5 == REF[i]).all(1).sum())
        same_saved += int((t5 == top_st[[q_of[NREF + 20 * i + j] for j in range(20)]]).all(1).sum())
        return S
    inn, same, tot = J.judge(forward=fwd, n=300)
    print(f"{sc}: judge5.judge  in the reference top 5 {inn}/{tot}, same top word {same}/{tot};  "
          f"judge5.top5 = judge5's reference (set and order) {same_ref}/{tot};  = the staged run's own top 5 {same_saved}/{tot}")
