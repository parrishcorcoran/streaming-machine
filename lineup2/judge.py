"""judge5.judge WITH forward = WAY 2.
lineup2/run_windows.py ran every window by the walk (every array identical to base2.py's own run) and walked the top
table at the 20 judged positions of each. Here forward(ids) gives, for that window, the machine's word at each judged
position: the race winner of its whole-number scores. judge5 itself (imported here) loads base2_gpt2.npz for its own
reference; forward never calls it.
PREDICTION: in the reference top 5 6000/6000, the same top word 6000/6000; the machine's top 5 = judge5's reference
(set and order) 6000/6000.
    python3 lineup2/judge.py   -> lineup2/judge.log
"""
import sys, os, glob, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"
LOG = open(f"{OUT}/judge.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
parts = sorted(glob.glob(f"{OUT}/windows_*_*.npz"))
Z = [np.load(p) for p in parts]; W_ = np.concatenate([z["windows"] for z in Z]); assert np.array_equal(W_, np.arange(300)), "windows missing"
WIN = np.concatenate([z["win"] for z in Z]); T5 = np.concatenate([z["top5"] for z in Z])
P = np.load(f"{HERE}/fringes_machine/plain_windows_top_light.npz")
light_same = sum(int(all(np.array_equal(np.concatenate([z[k] for z in Z])[20 * i:20 * i + 20], P[k][20 * i:20 * i + 20])
                         for k in ("x16", "ex", "LS"))) for i in range(300))
sys.path.insert(0, HERE); _cwd = os.getcwd(); os.chdir(HERE)
import judge5 as J
os.chdir(_cwd)
REF = J._ref(); it = iter(range(len(J.WINDOWS))); same_ref = [0]
def fwd(ids):
    i = next(it); assert ids == J.WINDOWS[i]["ids"]
    r = slice(20 * i, 20 * i + 20); same_ref[0] += int((T5[r] == REF[i]).all(1).sum())
    return WIN[r]
inn, same, tot = J.judge(forward=fwd)
log(f"way 2: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; the machine's top 5 = judge5's "
    f"reference (set and order) {same_ref[0]}/{tot}; the light at the top of the judged positions = base2's saved light "
    f"{light_same}/300 windows  (parts {len(parts)}; {time.time() - t0:.0f}s)")
