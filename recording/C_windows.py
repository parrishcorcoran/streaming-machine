"""PHASE C, STEP 2 (for the judge): THE RECORDING MACHINE ON THE 300 judge5 WINDOWS, FROM THE FILES ONLY.
Each window is 24 tokens (judge5_texts.json); every position goes through every layer; each window's arrivals read
only its own positions. Kept: the light at the top (x16, ex, LS) of the 20 judged positions (3..22) of every window,
for C_judge.py, where judge5.judge reads the machine's words through the recording's top table.
Checked as well: that light against collapse_top/light_top.npz (saved earlier from the plain machine, src = 1).
PREDICTION: identical in 300 of 300 windows (6,000 positions).
    python3 recording/C_windows.py a=0 b=60     windows a..b-1 -> recording/C_logs/windows_part_a_b.npz
    python3 recording/C_windows.py join         all parts -> recording/C_logs/windows_top_light.npz, and the check
"""
import sys, os, glob, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = "/Users/abundancemachine/streaming-machine"; os.makedirs(f"{HERE}/recording/C_logs", exist_ok=True)
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a); JOIN = "join" in sys.argv[1:]
a, b = int(ARGS.get("a", 0)), int(ARGS.get("b", 300))
LOG = open(f"{HERE}/recording/C_windows{'_join' if JOIN else f'_{a:03d}_{b:03d}'}.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); FIRST, LAST = 3, 23
if not JOIN:
    import C_machine as K
    ids = [w["ids"] for w in WINDOWS[a:b]]; assert all(len(i) == 24 for i in ids)
    x16, ex, ls = K.run(ids, log=log)
    rows = np.concatenate([24 * j + np.arange(FIRST, LAST) for j in range(len(ids))])
    np.savez(f"{HERE}/recording/C_logs/windows_part_{a:03d}_{b:03d}.npz", x16=x16[rows], ex=ex[rows], LS=ls[rows], windows=np.arange(a, b))
    log(f"windows {a}..{b - 1} done ({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
else:
    Z = [np.load(p) for p in sorted(glob.glob(f"{HERE}/recording/C_logs/windows_part_*.npz"))]
    W_ = np.concatenate([z["windows"] for z in Z]); assert np.array_equal(W_, np.arange(len(WINDOWS))), "windows missing"
    X, EX, LS = (np.concatenate([z[k] for z in Z]) for k in ("x16", "ex", "LS"))
    np.savez(f"{HERE}/recording/C_logs/windows_top_light.npz", x16=X, ex=EX, LS=LS)
    L = np.load(f"{HERE}/collapse_top/light_top.npz"); same = 0
    for i in range(len(WINDOWS)):
        gi = np.flatnonzero((L["src"] == 1) & (L["win"] == i)); r = slice(20 * i, 20 * i + 20)
        same += int(np.array_equal(X[r], L["x16"][gi]) and np.array_equal(EX[r], L["ex"][gi]) and np.array_equal(LS[r], L["LS"][gi]))
    log(f"the light at the top of the 6,000 judged positions = the plain machine's saved light (collapse_top/light_top.npz): "
        f"{same}/{len(WINDOWS)} windows")
    log(f"saved recording/C_logs/windows_top_light.npz ({time.time() - t0:.0f}s)")
