"""PART D (AND C) ON QWEN3-0.6B: the paired-off share by diagonal, the finished plates against a fresh bell draw of each
neuron's own spread, and against the same plate met by the light shuffled across its lines. No blank was saved for
Qwen, so the fresh draw stands where the blank stood on Pythia.

HIS WORDS (2026-10-06), the only words quoted as his: "I really feel like the tips are the only thing the model learned
and needs and it's because all the non tips are the blank slate somehowz"
The analysis and the predictions (the task's and the builder's) are in PREDICTION.txt, written before this ran.

THE MACHINE AND THE LIGHT, as they are: qwen/qwen_base2.npz (16 places, one power of two per neuron, the water level's
gain folded into q, k, v, gate, up) and the machine's own light, qwen06b/light/ (capture.py, 613 positions of the 30
refs texts in Qwen's words). Plates as banks.py cuts them: q, k, v (from qkv), o, gate, up (from gu), down, at all 28
layers, and the word table (head, 1024 x 151,936, with the light at the top).
CHECK OF THE LIGHT (here): every slab's direct piles against capture.py's per-text digests (meta.json), as banks.py.
  The top light is capture.py's, checked there; here only its piles from the tips against its direct piles.
THE COUNT: engine.tipcount (16 places). EXACT: the pile from the tips equals the direct pile at every position, every call.
CALLS per plate: fin; fresh (a bell draw of each neuron's own spread in grains, rounded to the whole number on the
  neuron's power: it keeps neither the stored 8 significant places nor the folded gain's per-line scale; the light
  shuffle below keeps both); fin_sh (the light shuffled across its lines, each position its own random order).
ORDER-FREE CHECK at q.0, gate.14 and down.27.
banks.py already counted fin and fin_sh at every plate (ud_fringe_ab, ctl_ud_fringe_ab, with its own shuffle draws);
  fin is compared with its numbers here as a second count by other code.
    python3 tips_blank/qwen_tips.py          -> tips_blank/qwen.json
"""
import os, sys, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tips_blank"
sys.path.insert(0, OUT); sys.path.insert(0, f"{HERE}/qwen06b")
import engine
from capture import member
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
A = dict(a.split("=", 1) for a in sys.argv[1:])
t0 = time.time()
Z = np.load(f"{HERE}/qwen/qwen_base2.npz"); META = json.load(open(f"{HERE}/qwen06b/light/meta.json")); ROWS = META["rows"]
LIGHT = {s: np.load(f"{HERE}/qwen06b/light/{s}_light.npy", mmap_mode="r") for s in ("qkv", "o", "gu", "down", "top")}
KINDS = [("q", "qkv", 0, 2048), ("k", "qkv", 2048, 3072), ("v", "qkv", 3072, 4096), ("o", "o", 0, 1024),
         ("gate", "gu", 0, 3072), ("up", "gu", 3072, 6144), ("down", "down", 0, 1024)]
BANKS = {}
for f in os.listdir(f"{HERE}/qwen06b"):
    if f.startswith("banks_") and f.endswith(".json"): BANKS.update(json.load(open(f"{HERE}/qwen06b/{f}")))

def fresh(w, rng, cb=8192):
    """a bell draw of each neuron's (column's) own spread in grains, rounded half-even to whole numbers (int16)"""
    n, m = w.shape; out = np.empty((n, m), np.int16)
    for c0 in range(0, m, cb):
        wc = np.asarray(w[:, c0:c0 + cb]).astype(np.int64); sig = np.sqrt(np.maximum((wc * wc).mean(0) - wc.mean(0) ** 2, 0))
        out[:, c0:c0 + cb] = np.clip(np.rint(rng.normal(0.0, 1.0, wc.shape) * sig[None, :]), -32767, 32767)
    return out

PLATES = [(f"{k}.{l}", s, c0, c1) for l in range(28) for k, s, c0, c1 in KINDS] + [("head", "top", 0, None)]
if "only" in A: PLATES = [p for p in PLATES if p[0] in A["only"].split(",")]
ORDERFREE = {"q.0", "gate.14", "down.27"}
res = dict(model="qwen3-0.6b", positions=int(META["npos"]), plates={}); slab_bad = 0; slab_checked = set()
for i, (pl, slab, c0, c1) in enumerate(PLATES):
    tp = time.time()
    if pl == "head":
        x = np.asarray(LIGHT["top"]).astype(np.int64); W = np.ascontiguousarray(member("head.w"))
    else:
        l = int(pl.split(".")[1]); x = np.asarray(LIGHT[slab][l]).astype(np.int64)
        if (slab, l) not in slab_checked:                              # the slab's direct piles against the capture's digests
            ws = Z[f"{slab}.{l}.w"]; dp = (x.astype(np.float64) @ ws.astype(np.float64)).astype(np.int64)
            slab_bad += sum(digest(dp[a:b]) != META["dig"][f"{slab}.{l}"][t] for t, (a, b) in enumerate(ROWS)); slab_checked.add((slab, l)); del dp, ws
        W = np.asarray(Z[f"{slab}.{l}.w"][:, c0:c1])
    rng = np.random.default_rng([2026, 1006, 3, i]); F = fresh(W, rng); xs = engine.shuffled_light(x, rng)
    r = dict(N=W.shape[0], M=W.shape[1], coin=engine.coin(W.shape[0]))
    for nm, xx, w in (("fin", x, W), ("fresh", x, F), ("fin_sh", xs, W)):
        r[nm] = engine.tojson(engine.tipcount(xx, w))
    if pl in BANKS: r["banks_fin_same"] = bool(np.array_equal(np.asarray(r["fin"]["FR"]), np.asarray(BANKS[pl]["ud_fringe_ab"], dtype=np.int64)))
    if pl in ORDERFREE:
        perm = np.random.default_rng([2026, 7, 3, i]).permutation(W.shape[0]); R2 = engine.tipcount(x[:, perm], W[perm])
        r["orderfree_same"] = bool(all(np.array_equal(np.asarray(r["fin"][k]), R2[k]) for k in ("FR", "SG", "DS")) and R2["bad"] == 0)
    bad = sum(r[k]["bad"] for k in ("fin", "fresh", "fin_sh")); assert bad == 0, f"a pile from the tips differs at {pl}"
    res["plates"][pl] = r; del F, W
    dd = {k: engine.diag(r[k]) for k in ("fin", "fresh", "fin_sh")}; top = int(np.flatnonzero(dd["fin"]["used"])[-1])
    print(f"  {pl:8s} N={r['N']:5d} M={r['M']:6d}  exact" + (f"  banks.py's count {'same' if r['banks_fin_same'] else 'DIFFERENT'}" if "banks_fin_same" in r else "")
          + (f"  order-free {'same' if r['orderfree_same'] else 'DIFFERENT'}" if "orderfree_same" in r else "")
          + "  top diagonal paired off: " + " ".join(f"{k} {100 * dd[k]['paired'][top]:.2f}" for k in dd)
          + f"  (coin {100 * (1 - r['coin']):.2f}; {time.time() - tp:.0f}s, {rss()} MB)", flush=True)
    json.dump(res | dict(slab_digests_differing=slab_bad), open(f"{OUT}/qwen{'_test' if 'only' in A else ''}.json", "w"))
print(f"slab piles against the capture's digests: {slab_bad} texts differing")
assert slab_bad == 0
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
