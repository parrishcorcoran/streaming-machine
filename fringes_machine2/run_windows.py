"""BASE2_FRINGES2 ON judge5's 300 WINDOWS, IN PARTS: EVERY ARRAY AGAINST base2.py'S OWN RUN (fringes_machine/run_windows.py,
unchanged, with the level of additions ADD).

Each window is 24 tokens; every array the machine forms is compared with base2.py's own run by its sha256, per window
(fringes_machine/plain_digests_windows.json, base2.py's own digests, read only). The check path's counts (the knife's
intervals, position 0, and at ADD >= 3 the arrival's signed-digit products against base2's) must all be 0. Kept: the
light at the top of the 20 judged positions (3..22) for fringes_machine2/top_read.py set=windows.
    python3 fringes_machine2/run_windows.py add=N a=0 b=40   windows a..b-1 -> fringes_machine2/windows_a<N>_part_a_b.npz/.log
    python3 fringes_machine2/run_windows.py add=N join        all parts -> fringes_machine2/windows_top_light_a<N>.npz
"""
import sys, os, glob, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a); JOIN = "join" in sys.argv[1:]
os.environ["FR2_ADD"] = ARGS["add"]; ADD = int(ARGS["add"])
a, b = int(ARGS.get("a", 0)), int(ARGS.get("b", 300)); OUT = f"{HERE}/fringes_machine2"; REFD = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/windows_a{ADD}{'_join' if JOIN else f'_{a:03d}_{b:03d}'}.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); FIRST, LAST = 3, 23
if not JOIN:
    import base2_fringes2 as FM
    assert FM.ADD == ADD
    PD = json.load(open(f"{REFD}/plain_digests_windows.json"))
    ids = [w["ids"] for w in WINDOWS[a:b]]; assert all(len(i) == 24 for i in ids)
    nw = len(ids); DIFF = {}; NCMP = [0]
    def check(key, value, rows=None):
        v = np.asarray(value); per = 24 if rows is None else 23
        assert len(v) == nw * per, (key, v.shape)
        for j in range(nw):
            NCMP[0] += 1
            if FM.digest(v[j * per:(j + 1) * per]) != PD[a + j][key]: DIFF[key] = DIFF.get(key, 0) + 1
    BAD = {}
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    (x16, ex, ls), STOPS, GATES, MEAS, LPLS = FM.forward(ids, None, check, chk, log=log)
    top_same = sum(int(FM.digest(np.concatenate([x16[24 * j:24 * j + 24].ravel(), ex[24 * j:24 * j + 24], ls[24 * j:24 * j + 24]]))
                       == PD[a + j]["top light"]) for j in range(nw))
    rows = np.concatenate([24 * j + np.arange(FIRST, LAST) for j in range(nw)])
    np.savez(f"{OUT}/windows_a{ADD}_part_{a:03d}_{b:03d}.npz", x16=x16[rows], ex=ex[rows], LS=ls[rows], windows=np.arange(a, b))
    log(f"ADD={ADD} windows {a}..{b - 1}: {NCMP[0]} window-arrays compared with base2.py's own run; differing: {sum(DIFF.values())} {DIFF}; "
        f"the light at the top identical in {top_same}/{nw} windows; check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items()))
        + f"; fallbacks {FM.FALLBACK} ({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
else:
    Z = [np.load(p) for p in sorted(glob.glob(f"{OUT}/windows_a{ADD}_part_*.npz"))]
    W_ = np.concatenate([z["windows"] for z in Z]); assert np.array_equal(W_, np.arange(len(WINDOWS))), "windows missing"
    X, EX, LS = (np.concatenate([z[k] for z in Z]) for k in ("x16", "ex", "LS"))
    np.savez(f"{OUT}/windows_top_light_a{ADD}.npz", x16=X, ex=EX, LS=LS)
    P = np.load(f"{REFD}/plain_windows_top_light.npz")
    same = sum(int(np.array_equal(X[20 * i:20 * i + 20], P["x16"][20 * i:20 * i + 20]) and np.array_equal(EX[20 * i:20 * i + 20], P["ex"][20 * i:20 * i + 20])
                   and np.array_equal(LS[20 * i:20 * i + 20], P["LS"][20 * i:20 * i + 20])) for i in range(len(WINDOWS)))
    log(f"ADD={ADD}: joined {len(Z)} parts: the light at the top of the 6,000 judged positions = base2.py's own: {same}/{len(WINDOWS)} windows "
        f"({time.time() - t0:.0f}s)")
