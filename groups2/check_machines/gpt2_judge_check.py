"""CHECKER: judge5 on the 50 drawn windows, base2_groups.py AS DELIVERED end to end: its forward on each window (two
calls of 25 windows), the light at the top at positions 3..22 against base2.py's fresh run (gpt2_ref.npz), then the
top read (gtop.run_top, as phase_top), then judge5's own reference (judge5_ref.npz) and rule (in the top 5; same top
word), and the top 5 (set and order) and race winner against base2's fresh top read.
    python3 gpt2_judge_check.py -> gpt2_judge_check.log
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/groups2")
argv = sys.argv; sys.argv = ["base2_groups.py"]
import base2_groups as BG
sys.argv = argv
FM = BG.FM
LOGF = open(f"{OUT}/gpt2_judge_check.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
REF = np.load(f"{OUT}/gpt2_ref.npz"); WS = [int(w) for w in REF["windows"]]
WIN = json.load(open(f"{HERE}/judge5_texts.json")); t0 = time.time()
X, EX, LS = [], [], []; BAD = {}
def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
for part in (WS[:25], WS[25:]):
    ids = [WIN[i]["ids"] for i in part]; assert all(len(i) == 24 for i in ids)
    (x16, ex, ls), _ = BG.forward(ids, None, None, chk, log=lambda s: None)
    rows = np.concatenate([24 * j + np.arange(3, 23) for j in range(len(ids))])
    X.append(x16[rows]); EX.append(ex[rows]); LS.append(ls[rows])
    log(f"  windows {part[0]}..{part[-1]} done ({time.time() - t0:.0f}s, {BG.rss()} MB)")
X, EX, LS = np.concatenate(X).astype(np.int64), np.concatenate(EX), np.concatenate(LS)
log(f"light at the top of the {len(X)} judged positions = base2's fresh run: "
    f"{bool(np.array_equal(X, REF['w_x16']) and np.array_equal(EX, REF['w_ex']) and np.array_equal(LS, REF['w_LS']))}")
log("the machine's own check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
TG = BG.top_module(False)
rec, bad, pwrite, lwrite, ppmask, lab = BG.run_top(TG, X, EX, LS, False, lambda s: None)
r = rec["pp"]; WINr, T5 = r["ii_win"], r["ii_top5"]
log("top read checks: " + ", ".join(f"{k} {v}" for k, v in sorted(bad["pp"].items())))
ref = np.load(f"{HERE}/judge5_ref.npz")["top5"]; inn = same = tot = 0; same5 = 0
for j, i in enumerate(WS):
    pick = WINr[20 * j:20 * j + 20]
    inn += int((ref[i] == pick[:, None]).any(1).sum()); same += int((ref[i][:, 0] == pick).sum()); tot += len(pick)   # judge5.judge's two lines
    same5 += int((T5[20 * j:20 * j + 20] == ref[i]).all(1).sum())
log(f"JUDGE5 on the 50 drawn windows: in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; machine's top 5 = judge5_ref (set and order) {same5}/{tot}")
log(f"against base2's fresh top read: top 5 (set and order) {int((T5 == REF['w_top5']).all(1).sum())}/{tot}, race winner {int((WINr == REF['w_win']).sum())}/{tot}")
pl = rec["plain_meet"] + rec["plain_div"] + rec["plain_stand"]
log(f"top read: plain {pl.sum() / tot:,.0f} a position; (ii) + writings {100 * (r['ii_total'] + pwrite + lwrite).sum() / pl.sum():.2f}% of the plain read")
log(f"done ({time.time() - t0:.0f}s, max memory {BG.rss()} MB)")
