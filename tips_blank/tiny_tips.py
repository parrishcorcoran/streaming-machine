"""PART B (AND C) ON THE TINY MODELS BUILT FROM A KNOWN BLANK: the paired-off share by diagonal for the finished plate,
the blank as it stands in it, and the pushes alone, all on the finished machine's own light, at every snapshot.

HIS WORDS (2026-10-06), the only words quoted as his: "I really feel like the tips are the only thing the model learned
and needs and it's because all the non tips are the blank slate somehowz"
The analysis and the predictions (the task's and the builder's) are in PREDICTION.txt, written before this ran.

THE MACHINES, as they are: ../blankgo/machines/w128_L4_<recipe>_step<t>.npz (convert.py) run by ../blankgo/bmachine.py
(machine.py's own machine, copied part for part, which keeps the light each plate meets: x16 and its power). Every
plate holds the finished (.w) and the blank as it stands (.s) on one grid, one power of two per neuron; finished =
blank + pushes digit for digit (pushes = .w - .s). Recipes: A plain push, shave off; B scaled push, shave off; C scaled
push, shave on (the blank worn to 0.848 of itself by step 3000). Snapshots 0, 10, 30, 100, 300, 1000, 2000, 3000.
TEXT: the first 32 of blankgo's 128 fixed held-out windows (model.held_windows), 128 words each: 4,096 positions
  (fixed before the run, for time; work.py used all 128).
PLATES: qkv, proj, fc, mproj of the 4 layers, and the output word table (head).
THE COUNT: engine.tipcount. The plate side is written in 17 places for all three plates (finished, blank, pushes):
  the pushes reach 16 places of size on the shared grid (largest 47,273), and the three are written alike so their
  diagonals line up. EXACT: the pile from the tips equals the direct pile at every position, every call.
  Also on the real scale (each meeting x 2^(neuron's power + position's power)), for the question of whether the
  paired-off mass grows with the pushes.
CALLS per plate and snapshot: fin, blank, push (the three plates on the finished light); fin_sh, blank_sh, push_sh (the
  same, the light shuffled across its lines: each position its own random order, one draw shared by the three).
ORDER-FREE CHECK (every snapshot, qkv.0 and head): the finished count with the lines in one random order on both sides.
    python3 tips_blank/tiny_tips.py recipe=B [width=128 layers=4]     -> tips_blank/tiny_w128_L4_<recipe>.json
"""
import os, sys, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tips_blank"; BG = f"{HERE}/blankgo"
sys.path.insert(0, OUT); sys.path.insert(0, BG); os.chdir(BG)
import engine
from model import data, held_windows, tag
from bmachine import Machine, machine_file
A = dict(a.split("=", 1) for a in sys.argv[1:])
RECIPE, d, L = A.get("recipe", "B"), int(A.get("width", 128)), int(A.get("layers", 4))
STEPS = [int(s) for s in A.get("steps", "0,10,30,100,300,1000,2000,3000").split(",")]
NWIN = int(A.get("testwin", 32))                                                 # 32 is the run; testwin only for a test
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()
_, held = data(); WIN = held_windows(held)[:NWIN]
PLATES = [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
name = f"{tag(d, L)}_{RECIPE}"; outf = f"{OUT}/tiny_{name}{'_test' if ('steps' in A or 'testwin' in A) else ''}.json"
res = dict(build=name, windows=NWIN, positions=NWIN * 128, snaps={})
print(f"TINY {name}: {NWIN} windows x 128 positions, {len(PLATES)} plates, steps {STEPS}", flush=True)
for t in STEPS:
    ts = time.time(); M = machine_file(d, L, RECIPE, t); m = Machine(d, L, RECIPE, t, "finished", capture=True, M=M)
    caps = {}
    for win in WIN:
        m.cap = {}; m.forward(win[:-1], 128)
        for k, v in m.cap.items(): caps.setdefault(k, []).append(v[0])
    snap = {}
    for i, pl in enumerate(PLATES):
        x = np.concatenate([a for a, _ in caps[pl]]).astype(np.int64); ex = np.concatenate([e for _, e in caps[pl]]).astype(np.int64)
        W = M[pl + ".w"].astype(np.int64); S = M[pl + ".s"].astype(np.int64); Pp = W - S; E = M[pl + ".e"].astype(np.int64)
        rw = np.left_shift(np.int64(1), ex - ex.min()); colw = np.left_shift(np.int64(1), E - E.min())
        rng = np.random.default_rng([2026, 1006, ord(RECIPE), t, i]); xs = engine.shuffled_light(x, rng)
        r = dict(N=W.shape[0], M=W.shape[1], coin=engine.coin(W.shape[0]), ex_min=int(ex.min()), e_min=int(E.min()),
                 push_max=int(np.abs(Pp).max()), blank_max=int(np.abs(S).max()), fin_max=int(np.abs(W).max()))
        for nm, xx, w in (("fin", x, W), ("blank", x, S), ("push", x, Pp), ("fin_sh", xs, W), ("blank_sh", xs, S), ("push_sh", xs, Pp)):
            r[nm] = engine.tojson(engine.tipcount(xx, w, nb=17, rw=rw, colw=colw))
        if pl in ("qkv.0", "head"):
            perm = np.random.default_rng([2026, 7, ord(RECIPE), t, i]).permutation(W.shape[0])
            R2 = engine.tipcount(x[:, perm], W[perm], nb=17)
            r["orderfree_same"] = bool(all(np.array_equal(np.asarray(r["fin"][k]), R2[k]) for k in ("FR", "SG", "DS")) and R2["bad"] == 0)
        bad = sum(r[k]["bad"] for k in ("fin", "blank", "push", "fin_sh", "blank_sh", "push_sh"))
        assert bad == 0, f"a pile from the tips differs from the direct pile at {pl}, step {t}"
        snap[pl] = r
    res["snaps"][t] = snap
    of = all(snap[p]["orderfree_same"] for p in ("qkv.0", "head"))
    line = []
    for pl in ("qkv.0", "fc.3", "mproj.3", "head"):
        dd = {k: engine.diag(snap[pl][k]) for k in ("fin", "blank", "push", "fin_sh")}; top = int(np.flatnonzero(dd["fin"]["used"])[-1])
        line.append(f"{pl} top: " + "/".join(f"{100 * dd[k]['paired'][top]:.1f}" for k in dd))
    print(f"  step {t:5d}: piles from the tips exact at every plate, order-free {'same' if of else 'DIFFERENT'}; paired off at the "
          f"top diagonal (fin/blank/push/fin_sh) " + "  ".join(line) + f"  ({time.time() - ts:.0f}s, {rss()} MB)", flush=True)
    json.dump(res, open(outf, "w"))
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
