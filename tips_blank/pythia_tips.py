"""PARTS A AND C ON PYTHIA: the paired-off share by diagonal, finished against blank against a fresh bell draw, on the
finished light; the blank on its own light; and every tip size against the same plate met by the light shuffled
across its lines.

HIS WORDS (2026-10-06), the only words quoted as his: "I really feel like the tips are the only thing the model learned
and needs and it's because all the non tips are the blank slate somehowz"
The analysis and the predictions (the task's and the builder's) are in PREDICTION.txt, written before this ran.

THE MACHINE AND THE LIGHT, as they are (no part changed):
  plates  ../foundation0/base2_pythia-<size>.npz: every plate on one grid shared by the finished (.w, step 143000) and
          the blank (.s, step 0), one power of two per neuron (.e); 16 places.
  light   ../lineblanks/light_<size>_finished.npz (the finished machine's own light at every plate, x16 exactly as the
          product meets it; capture.py checked the head piles against the machine's own run, 30 of 30 texts) and
          light_<size>_start.npz (the blank machine's own light on the same words). 622 positions.
  plates read: qkv, proj, fc, mproj of every layer, and the word table (head), every position, every neuron.
THE COUNT: engine.tipcount (the up/down form of qwen06b/banks.py, the same counts, checked equal to banks.updown on a
  test plate): for every meeting, the tip T = N - 2d; paired off = N - |T|. Per diagonal d = a + b: the paired-off
  share; with 2^d weights over all diagonals; and the size of the diagonal's signed part of the pile.
  EXACT: at every plate and every call, the pile from the tips equals the direct pile at every position (bad = 0).
CALLS per plate (each one a full count of every meeting):
  fin    finished plate, finished light          blank   blank plate, finished light
  fresh  a fresh bell draw of each finished neuron's own spread (lineblanks/common.fresh: through fp16 onto the
         neuron's power of two, as the stored plates were), finished light
  own    blank plate, the blank machine's own light
  fin_sh, blank_sh   the same plates, the finished light shuffled across its lines (each position its own random
         order; the same draw for both): CONTROL, the pairing of light and plate broken, every value kept
  own_sh blank plate, its own light shuffled the same way
ORDER-FREE CHECK: at qkv.0, fc.(L/2) and mproj.(L-1), the finished count with the lines in one random order on both
  sides: every tip sum, net and diagonal sum identical.
    python3 tips_blank/pythia_tips.py size=70m      -> tips_blank/pythia_<size>.json
"""
import os, sys, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")             # Accelerate's spurious flags; every pile checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tips_blank"
sys.path.insert(0, OUT); sys.path.insert(0, f"{HERE}/lineblanks")
import engine
from common import fresh
A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m")
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()
Z = np.load(f"{HERE}/foundation0/base2_pythia-{SIZE}.npz"); L = int(Z["L"])
LF = np.load(f"{HERE}/lineblanks/light_{SIZE}_finished.npz"); LS = np.load(f"{HERE}/lineblanks/light_{SIZE}_start.npz")
PLATES = [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
ORDERFREE = {"qkv.0", f"fc.{L // 2}", f"mproj.{L - 1}"}
if "only" in A: PLATES = A["only"].split(",")                                   # a test run on named plates (not the result)
code = {"14m": 14, "70m": 70, "160m": 160}[SIZE]
res = dict(size=SIZE, positions=None, plates={})
print(f"PYTHIA-{SIZE}: {len(PLATES)} plates", flush=True)
for i, pl in enumerate(PLATES):
    tp = time.time()
    xf = LF[f"light.{pl}"].astype(np.int64); xo = LS[f"light.{pl}"].astype(np.int64); res["positions"] = len(xf)
    W = Z[f"{pl}.w"].astype(np.int64); S = Z[f"{pl}.s"].astype(np.int64); E = Z[f"{pl}.e"]
    F = fresh(W, E, 1, np.random.default_rng([2026, 10, 6, code, i])).astype(np.int64)
    rng = np.random.default_rng([2026, 1006, code, i])
    xf_sh = engine.shuffled_light(xf, rng); xo_sh = engine.shuffled_light(xo, rng)
    r = dict(N=W.shape[0], M=W.shape[1], coin=engine.coin(W.shape[0]))
    for name, x, w in (("fin", xf, W), ("blank", xf, S), ("fresh", xf, F), ("own", xo, S),
                       ("fin_sh", xf_sh, W), ("blank_sh", xf_sh, S), ("own_sh", xo_sh, S)):
        r[name] = engine.tojson(engine.tipcount(x, w))
    if pl in ORDERFREE:
        perm = np.random.default_rng([2026, 7, code, i]).permutation(W.shape[0])
        R2 = engine.tipcount(xf[:, perm], W[perm])
        r["orderfree_same"] = bool(all(np.array_equal(np.asarray(r["fin"][k]), R2[k]) for k in ("FR", "SG", "DS")) and R2["bad"] == 0)
    res["plates"][pl] = r
    bad = {k: r[k]["bad"] for k in ("fin", "blank", "fresh", "own", "fin_sh", "blank_sh", "own_sh")}
    dd = {k: engine.diag(r[k]) for k in ("fin", "blank", "fresh", "own", "fin_sh")}
    top = int(np.flatnonzero(dd["fin"]["used"])[-1])
    print(f"  {pl:8s} N={r['N']:5d} M={r['M']:6d}  piles differing {sum(bad.values())}"
          + (f"  order-free {'same' if r['orderfree_same'] else 'DIFFERENT'}" if "orderfree_same" in r else "")
          + "  top diagonal paired off: " + " ".join(f"{k} {100 * dd[k]['paired'][top]:.2f}" for k in dd)
          + f"  (coin {100 * (1 - r['coin']):.2f}; {time.time() - tp:.0f}s, {rss()} MB)", flush=True)
    assert sum(bad.values()) == 0, "a pile from the tips differs from the direct pile"
    json.dump(res, open(f"{OUT}/pythia_{SIZE}{'_test' if 'only' in A else ''}.json", "w"))
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
