"""THE FRINGES MACHINE'S TOP READ (base2_fringes.TopRead) ON THE SAVED LIGHT AT THE TOP, WITH ITS CHECKS AND COUNT.

set=refs     the 648 positions of the 30 refs texts, light from fringes_machine/run_refs.py (refs_top_light.npz),
             checked against base2's plain read of every word (made here, CHECK ONLY) and against the top 5 and race
             winner base2.py's own run kept (recording/C_ref/plain_refs.npz).
set=windows  the 6,000 judged positions of judge5's 300 windows, light from fringes_machine/run_windows.py, checked
             against the plain read and against judge5's reference top 5 (judge5_ref.npz).
light=shuffled (CONTROL): each position's light shuffled across its lines (every value kept); counts only.
order=SEED (ORDER-FREE): the words and the lines (light and table together) in one shared random order.
part=i/n     the i-th of n equal slices of the positions (each run under the 10 minute foreground limit).
    python3 fringes_machine/top_read.py set=refs|windows [part=i/n] [light=real|shuffled] [order=SEED] [tag=...]
      -> fringes_machine/top_<set>[_<light>][_order][_p<i>of<n>].npz and .log
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
SET, LIGHT = ARG.get("set", "refs"), ARG.get("light", "real"); ORDER = int(ARG["order"]) if "order" in ARG else None
PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/")); SRC_TAG = ARG.get("tag", "")
TAG = f"{SET}" + ("" if LIGHT == "real" else f"_{LIGHT}") + ("" if ORDER is None else f"_order{ORDER}") + \
      ("" if PN == 1 else f"_p{PI}of{PN}") + (f"_{SRC_TAG}" if SRC_TAG else "")
OUT = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/top_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes as FM

if SET == "refs":
    Z = np.load(f"{OUT}/refs_top_light{('_' + SRC_TAG) if SRC_TAG else ''}.npz"); X, EX, LS, ROWS = Z["x16"], Z["ex"], Z["LS"], Z["rows"]
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); RT5, RWIN = REF["top5"][ROWS], REF["win"][ROWS]
    assert np.array_equal(X, REF["x16"][ROWS]) and np.array_equal(EX, REF["ex"][ROWS]) and np.array_equal(LS, REF["LS"][ROWS])
else:
    Z = np.load(f"{OUT}/windows_top_light.npz"); X, EX, LS = Z["x16"], Z["ex"], Z["LS"]
    RT5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); RWIN = None
N_ALL = len(X); cut = np.linspace(0, N_ALL, PN + 1).astype(np.int64); SEL = np.arange(cut[PI - 1], cut[PI])
X, EX, LS = X[SEL].astype(np.int64), EX[SEL], LS[SEL]
if LIGHT == "shuffled":
    rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])
wperm = lperm = None
if ORDER is not None:
    rng = np.random.default_rng(ORDER); wperm = rng.permutation(50257); lperm = rng.permutation(768)
log(f"top read: set={SET} light={LIGHT} order={ORDER} part={PI}/{PN}: {len(SEL)} positions")
TR = FM.TopRead(check=True, word_perm=wperm, line_perm=lperm, log=log)
C = FM.Count(); BAD = {}
R = TR.read(X, EX, LS, C=C, bad=BAD)
same5 = int((R["top5"] == R["ptop5"]).all(1).sum()); samew = int((R["win"] == R["pwin"]).sum())
log(f"  top 5 (set and order) = base2's plain read of every word: {same5}/{len(SEL)}; race winner: {samew}/{len(SEL)}")
if LIGHT == "real":
    s5r = int((R["ptop5"] == RT5[SEL]).all(1).sum()); f5r = int((R["top5"] == RT5[SEL]).all(1).sum())
    log(f"  against the saved reference top 5 ({'base2.py run, plain_refs' if SET == 'refs' else 'judge5_ref'}): "
        f"fringes {f5r}/{len(SEL)}, the plain read here {s5r}/{len(SEL)}")
    if RWIN is not None: log(f"  race winner = base2.py's saved winner: {int((R['win'] == RWIN[SEL]).sum())}/{len(SEL)}")
log("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
sv = R["surv"]
for s, k in enumerate(FM.KS):
    log(f"    after top {k:2d}: words still in median {int(np.median(sv[:, s])):6d} mean {sv[:, s].mean():9.1f} "
        f"least {sv[:, s].min():6d} most {sv[:, s].max():6d}; that stage's meetings {R['work'][:, s].mean() / 1e6:7.2f}M; "
        f"words read for the 5th-best low (median) {int(np.median(R['visited'][:, s]))}")
n = len(SEL); agg = {}
for k, v in C.items():
    who, lay, part, item = k.split("|"); agg[(who, item)] = agg.get((who, item), 0) + v
pl_ = sum(v for (w, i), v in agg.items() if w == "plain"); fr_ = sum(v for (w, i), v in agg.items() if w == "fr")
log(f"  per position: plain top read (meetings, divide, standing amounts) {pl_ / n:,.0f}; fringes {fr_ / n:,.0f} = {FM.pct2(fr_, pl_)}")
for (w, i), v in sorted(agg.items()):
    if w in ("plain", "fr", "cmp"): log(f"      {w:5s} {i:85s} {v / n:16,.0f}")
np.savez(f"{OUT}/top_{TAG}.npz", sel=SEL, top5=R["top5"], win=R["win"], ptop5=R["ptop5"], pwin=R["pwin"], surv=R["surv"],
         work=R["work"], visited=R["visited"], bad_keys=np.array(sorted(BAD)), bad=np.array([BAD[k] for k in sorted(BAD)]))
json.dump(dict(counts=dict(C), n=int(n), bad=BAD), open(f"{OUT}/top_{TAG}.json", "w"), indent=0)
log(f"({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
