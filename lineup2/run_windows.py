"""WAY 2 ON judge5's 300 WINDOWS, IN PARTS: EVERY ARRAY AGAINST base2.py'S OWN RUN, THE TOP TABLE BY THE WALK.

Each window is 24 tokens; every position goes through every layer by the walk (each window's arrivals read only its
own positions). Every array is compared with base2.py's own run by its sha256, per window
(fringes_machine/plain_digests_windows.json, made by fringes_machine/plain_ref.py in a separate process; its row
conventions: at layer 0 the proj pile, stream a, fc pile and mproj pile over positions 1 on, everything else over all
24). The top table is walked at the 20 judged positions (3..22): its piles are checked against base2's own product of
the same light (CHECK path, never used by the machine), and the race winner and top 5 kept for lineup2/judge.py.
No count here (the work is measured on the refs texts).
    python3 lineup2/run_windows.py a=0 b=50      windows a..b-1 -> lineup2/windows_a_b.npz / .log
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"; sys.path.insert(0, OUT)
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a); a, b = int(ARGS.get("a", 0)), int(ARGS.get("b", 300))
LOG = open(f"{OUT}/windows_{a:03d}_{b:03d}.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import way2 as W2
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); FIRST, LAST = 3, 23
PD = json.load(open(f"{HERE}/fringes_machine/plain_digests_windows.json"))
ids = [w["ids"] for w in WINDOWS[a:b]]; assert all(len(i) == 24 for i in ids); nw = len(ids)
rec = W2.Record(order="sorted", log=log, keep_shuffled_counts=False)
mach = W2.Machine(rec, count=False)
SUB0 = {"proj pile 0", "stream a 0", "fc pile 0", "mproj pile 0"}
DIFF = {}; NCMP = [0]
def check(key, value):
    v = np.asarray(value)
    for j in range(nw):
        part = v[24 * j:24 * j + 24]
        if key in SUB0: part = part[1:]
        NCMP[0] += 1
        if W2.digest(part) != PD[a + j][key]: DIFF[key] = DIFF.get(key, 0) + 1
x16, ex, LS = mach.forward(ids, check=check, log=log)
top_same = sum(int(W2.digest(np.concatenate([x16[24 * j:24 * j + 24].ravel(), ex[24 * j:24 * j + 24], LS[24 * j:24 * j + 24]]))
                   == PD[a + j]["top light"]) for j in range(nw))
rows = np.concatenate([24 * j + np.arange(FIRST, LAST) for j in range(nw)])
HW = np.asarray(W2.M["head.w"]).astype(np.float64); TOPBAD = [0, 0]
def top_check(pile, xr):
    ref = W2.product(xr, HW); TOPBAD[0] += int((pile != ref).any(1).sum()); TOPBAD[1] += len(pile)
mach.top_check = top_check
SHA, T5, WIN = mach.top(x16[rows], ex[rows], LS[rows])
np.savez(f"{OUT}/windows_{a:03d}_{b:03d}.npz", windows=np.arange(a, b), top5=T5, win=WIN, x16=x16[rows], ex=ex[rows], LS=LS[rows])
log(f"windows {a}..{b - 1}: {NCMP[0]} window-arrays compared with base2.py's own run; differing: {sum(DIFF.values())} {DIFF}; "
    f"the light at the top identical in {top_same}/{nw} windows; the top table's piles = base2's product "
    f"{TOPBAD[1] - TOPBAD[0]}/{TOPBAD[1]} ({time.time() - t0:.0f}s, max memory {W2.rss()} MB)")
