"""JUDGE5'S OWN CHECK of the combined machine (as collapse_top/judge_check.py does for the staged read alone).
Every one of the 300 judge5 windows gets back, at its 20 judged positions, the exact scores of the words the combined
machine read to the end (all of them, or their best 256; every other word below them all), from
combined/windows_top.npz. Then:
  judge5.judge(forward=...)     in the reference top 5, same top word, out of 6,000
  judge5.top5 of those scores   = judge5's reference top 5 (set and order)
    python3 combined/judge_check.py
"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
NW, FLOOR = 50257, -(1 << 62)
REF = J._ref(); Z = np.load(f"{HERE}/combined/windows_top.npz")
off, fi, fs, top5_saved = Z["fin_off"], Z["fin_idx"], Z["fin_sc"], Z["top5"]; N = int(Z["n"])
it = iter(range(N)); same_ref = same_saved = 0
def fwd(ids):
    global same_ref, same_saved
    i = next(it); assert ids == J.WINDOWS[i]["ids"]
    S = np.full((20, NW), FLOOR, np.int64)
    for j in range(20):
        q = 20 * i + j; S[j, fi[off[q]:off[q + 1]]] = fs[off[q]:off[q + 1]]
    t5 = J.top5(S)
    same_ref += int((t5 == REF[i]).all(1).sum()); same_saved += int((t5 == top5_saved[20 * i:20 * i + 20]).all(1).sum())
    return S
inn, same, tot = J.judge(forward=fwd, n=N)
print(f"combined machine: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; "
      f"judge5.top5 = judge5's reference (set and order) {same_ref}/{tot}; = the run's own top 5 {same_saved}/{tot}")
