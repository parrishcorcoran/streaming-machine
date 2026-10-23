"""judge5.judge WITH forward = THE BAND-BY-BAND TOP READ.
The light at the top of every judged position is base2.py's own (fringes_machine/windows_top_light.npz, equal to
lineup2's saved light); topband/topband.py set=windows read the top band by band there. Here forward(ids) gives, for
that window, the read's word at each judged position: the race winner among the words still in after band 7, scored
exactly. judge5 itself (imported here) loads base2_gpt2.npz for its own reference; forward never calls it.
PREDICTION: in the reference top 5 6000/6000, the same top word 6000/6000; the read's top 5 = judge5's reference (set and
order) 6000/6000.
    python3 topband/judge.py   -> topband/judge.log
"""
import sys, os, glob, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/topband"
LOG = open(f"{OUT}/judge.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
parts = sorted(glob.glob(f"{OUT}/top_windows_p*of*.npz"))
Z = [np.load(p) for p in parts]; sel = np.concatenate([z["sel"] for z in Z]); assert np.array_equal(sel, np.arange(6000))
WIN = np.concatenate([z["win"] for z in Z]); T5 = np.concatenate([z["top5"] for z in Z])
sys.path.insert(0, HERE); _cwd = os.getcwd(); os.chdir(HERE)
import judge5 as J
os.chdir(_cwd)
REF = J._ref(); it = iter(range(len(J.WINDOWS))); same_ref = [0]
def fwd(ids):
    i = next(it); assert ids == J.WINDOWS[i]["ids"]
    r = slice(20 * i, 20 * i + 20); same_ref[0] += int((T5[r] == REF[i]).all(1).sum())
    return WIN[r]
inn, same, tot = J.judge(forward=fwd)
log(f"band-by-band top read: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; "
    f"the read's top 5 = judge5's reference (set and order) {same_ref[0]}/{tot}  (parts {len(parts)}; {time.time() - t0:.0f}s)")
