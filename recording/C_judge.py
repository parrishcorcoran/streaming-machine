"""PHASE C, STEP 2: judge5.judge WITH forward = THE RECORDING MACHINE.
The machine ran every window from the files alone (C_windows.py). Here forward(ids) gives, for that window's 20
judged positions, the top read on the machine's own light at the top: base2's plain read, every word, with the top
table rebuilt from the recording's top-table sheets (recording/C/head.*), never from base2_gpt2.npz. judge5 itself
(imported here) loads base2_gpt2.npz for its own reference machine, which forward never calls.
PREDICTION: in the reference top 5 6000/6000, the same top word 6000/6000; judge5.top5 of the scores = judge5's
reference (set and order) 6000/6000.
    python3 recording/C_judge.py   -> recording/C_judge.log
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = "/Users/abundancemachine/streaming-machine"
LOG = open(f"{HERE}/recording/C_judge.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import C_machine as K
Z = np.load(f"{HERE}/recording/C_logs/windows_top_light.npz"); X, EX, LS = Z["x16"], Z["ex"], Z["LS"]
K.top_table(); log(f"top table rebuilt from the recording's sheets ({time.time() - t0:.0f}s, {K.rss()} MB)")
sys.path.insert(0, HERE); _cwd = os.getcwd(); os.chdir(HERE)
import judge5 as J
os.chdir(_cwd)
REF = J._ref(); it = iter(range(len(J.WINDOWS))); same_ref = [0]
def fwd(ids):
    i = next(it); assert ids == J.WINDOWS[i]["ids"]
    r = slice(20 * i, 20 * i + 20); S = K.top_scores(X[r], EX[r], LS[r])
    same_ref[0] += int((J.top5(S) == REF[i]).all(1).sum())
    return S
inn, same, tot = J.judge(forward=fwd)
log(f"recording machine: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; "
    f"judge5.top5 of the scores = judge5's reference (set and order) {same_ref[0]}/{tot}  "
    f"({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
