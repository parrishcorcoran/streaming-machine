"""JUDGE5'S OWN CHECK of the fine-grid staged top read (collapse_top/judge_check.py, for the fine_ results and both bound
rows). For each schedule (fine_<sched>_real.npz from fine_staged_top.py) and each bound (plain, signed), every one of
the 300 judge5 windows gets back, at its 20 judged positions, the exact scores of the words the staged read kept to the
end (all of them, or their best 256 when more were kept; every other word below them all). Then
  judge5.judge(forward=...)                 -> in the reference top 5, same top word, out of 6,000
  judge5.top5 of the same scores            -> top 5 (set and order) equal to judge5's reference (judge5_ref.npz)
    python3 collapse_top/fine_judge_check.py
"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
NW, FLOOR = 50257, -(1 << 62)
REF = J._ref()
for sc in ("diag", "light"):
    Z = np.load(f"{HERE}/collapse_top/fine_{sc}_real.npz")
    for row, sx in (("plain bound", ""), ("signed bound", "_s")):
        sel, off, fi, fs, top_st = Z["sel"], Z["fin_off" + sx], Z["fin_idx" + sx], Z["fin_sc" + sx], Z["top_st" + sx]
        NREF = 480; q_of = {int(g): q for q, g in enumerate(sel)}
        it = iter(range(300)); cnt = dict(ref=0, saved=0)
        def fwd(ids):
            i = next(it); assert ids == J.WINDOWS[i]["ids"]
            S = np.full((20, NW), FLOOR, np.int64)
            for j in range(20):
                q = q_of[NREF + 20 * i + j]; S[j, fi[off[q]:off[q + 1]]] = fs[off[q]:off[q + 1]]
            t5 = J.top5(S)
            cnt["ref"] += int((t5 == REF[i]).all(1).sum())
            cnt["saved"] += int((t5 == top_st[[q_of[NREF + 20 * i + j] for j in range(20)]]).all(1).sum())
            return S
        inn, same, tot = J.judge(forward=fwd, n=300)
        print(f"{sc:5s} [{row}]: judge5.judge  in the reference top 5 {inn}/{tot}, same top word {same}/{tot};  "
              f"judge5.top5 = judge5's reference (set and order) {cnt['ref']}/{tot};  = the staged run's own top 5 {cnt['saved']}/{tot}", flush=True)
