"""PART D (AND C) ON GPT-2: the paired-off share by diagonal, the finished plates against a fresh bell draw of each
neuron's own spread, and against the same plate met by the light shuffled across its lines. No blank was saved for
GPT-2, so the fresh draw stands where the blank stood on Pythia.

HIS WORDS (2026-10-06), the only words quoted as his: "I really feel like the tips are the only thing the model learned
and needs and it's because all the non tips are the blank slate somehowz"
The analysis and the predictions (the task's and the builder's) are in PREDICTION.txt, written before this ran.

THE MACHINE AND THE LIGHT, as they are: base2_gpt2.npz's 48 banks (16 places, one power of two per neuron) and the word
table (head); the machine's own light at every bank and every position of the 30 refs texts (648 positions):
groups/light_gpt2/{qkv,proj,fc,mproj}.npy (capture_gpt2.py: kept from base2.py's own run, checked against
recording/C_ref/plain_refs.npz), and the light at the top from plain_refs.npz (x16).
CHECKS OF THE LIGHT (here): every bank's direct pile equals plain_refs.npz's pile at every position; the top light
  times the word table, through base2.py's own power and bias, gives every position's scores with plain_refs.npz's
  sha256.
THE COUNT: engine.tipcount (16 places, the machine's own form). EXACT: the pile from the tips equals the direct pile at
  every position of every call.
CALLS per plate: fin (finished, its light); fresh (a bell draw of each neuron's own spread in grains, rounded to the
  whole number on the neuron's power, as the plates were written from fp32); fin_sh (the light shuffled across its
  lines, each position its own random order).
ORDER-FREE CHECK at qkv.0, fc.6 and mproj.11.
    python3 tips_blank/gpt2_tips.py         -> tips_blank/gpt2.json
"""
import os, sys, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tips_blank"
sys.path.insert(0, OUT)
import engine
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
A = dict(a.split("=", 1) for a in sys.argv[1:])
t0 = time.time()
Z = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
LIGHT = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}

# the top light, checked through base2.py's own steps (power, bias) against the scores' sha256
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; B = {}; exec(_src, B); sys.argv = _argv; os.chdir(_cwd)
XTOP = REF["x16"].astype(np.int64); w, we, b = B["W"]["head"]; same = 0
for p0 in range(0, len(XTOP), 64):
    S = B["power"](B["product"](XTOP[p0:p0 + 64], w), (we[None, :] + REF["ex"][p0:p0 + 64, None] + B["ES"]) * 65536 - REF["LS"][p0:p0 + 64, None]) + b
    same += sum(bytes(np.frombuffer(hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).digest(), np.uint8)) == bytes(REF["score_sha"][p0 + i]) for i, s in enumerate(S))
del B, w, S
print(f"GPT-2: the top light gives plain_refs.npz's scores at {same} of {len(XTOP)} positions ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
assert same == len(XTOP)

def fresh(w, rng):
    """a bell draw of each neuron's (column's) own spread in grains, rounded half-even to whole numbers (int16)"""
    w = np.asarray(w); n, m = w.shape; s1 = np.zeros(m); s2 = np.zeros(m)
    for i0 in range(0, n, 256): bb = w[i0:i0 + 256].astype(np.int64); s1 += bb.sum(0); s2 += (bb * bb).sum(0)
    sig = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)); out = np.empty(w.shape, np.int16)
    for i0 in range(0, n, 64):
        k = min(64, n - i0); out[i0:i0 + k] = np.clip(np.rint(rng.normal(0.0, 1.0, (k, m)) * sig[None, :]), -32767, 32767)
    return out

PLATES = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
if "only" in A: PLATES = A["only"].split(",")
ORDERFREE = {"qkv.0", "fc.6", "mproj.11"}
res = dict(model="gpt2", positions=len(XTOP), plates={}); refbad = 0
for i, pl in enumerate(PLATES):
    tp = time.time()
    if pl == "head": x = XTOP; W = Z["head.w"]
    else:
        k, l = pl.split("."); x = np.asarray(LIGHT[k][int(l)]).astype(np.int64); W = Z[pl + ".w"]
        dp = x.astype(np.float64) @ W.astype(np.float64); refbad += int((dp.astype(np.int64) != REF[f"{k}_pile_{l}"]).any(1).sum()); del dp
    rng = np.random.default_rng([2026, 1006, 2, i]); F = fresh(W, rng); xs = engine.shuffled_light(x, rng)
    r = dict(N=W.shape[0], M=W.shape[1], coin=engine.coin(W.shape[0]))
    for nm, xx, w in (("fin", x, W), ("fresh", x, F), ("fin_sh", xs, W)):
        r[nm] = engine.tojson(engine.tipcount(xx, w))
    if pl in ORDERFREE:
        perm = np.random.default_rng([2026, 7, 2, i]).permutation(W.shape[0]); R2 = engine.tipcount(x[:, perm], np.asarray(W)[perm])
        r["orderfree_same"] = bool(all(np.array_equal(np.asarray(r["fin"][k]), R2[k]) for k in ("FR", "SG", "DS")) and R2["bad"] == 0)
    bad = sum(r[k]["bad"] for k in ("fin", "fresh", "fin_sh")); assert bad == 0, f"a pile from the tips differs at {pl}"
    res["plates"][pl] = r; del F
    dd = {k: engine.diag(r[k]) for k in ("fin", "fresh", "fin_sh")}; top = int(np.flatnonzero(dd["fin"]["used"])[-1])
    print(f"  {pl:8s} N={r['N']:5d} M={r['M']:6d}  exact" + (f"  order-free {'same' if r['orderfree_same'] else 'DIFFERENT'}" if "orderfree_same" in r else "")
          + "  top diagonal paired off: " + " ".join(f"{k} {100 * dd[k]['paired'][top]:.2f}" for k in dd)
          + f"  (coin {100 * (1 - r['coin']):.2f}; {time.time() - tp:.0f}s, {rss()} MB)", flush=True)
    json.dump(res | dict(bank_piles_differ_from_plain_refs=refbad), open(f"{OUT}/gpt2{'_test' if 'only' in A else ''}.json", "w"))
print(f"bank piles from the light against plain_refs.npz: positions differing {refbad}")
assert refbad == 0
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
