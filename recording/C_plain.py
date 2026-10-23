"""PHASE C, THE PLAIN REFERENCE (a separate process; the recording machine never runs in it).
base2.py's own steps (its gauged_bank, plain_bank, attention, edge, product, read from base2_gpt2.npz) on the 30 refs
texts (ids + ref, every position, 648 in all). Kept for the check, per text and concatenated over the texts:
  - every pile of all 48 banks (the whole-number product before the gauge or shift),
  - the stream after each half layer ("stream a l": after the arrival's bank, "stream l": after the neurons-out bank),
  - the light at the top (x16, ex, LS),
  - the top read at every position: each position's scores' sha256, its top 5 (judge5.top5) and the race winner.
    python3 recording/C_plain.py      -> recording/C_ref/plain_refs.npz and recording/C_plain.log
"""
import os, sys, json, time, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUTD = f"{HERE}/recording/C_ref"
LOG = open(f"{HERE}/recording/C_plain.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; B = {}; exec(_src, B); sys.argv = _argv; os.chdir(_cwd)
M, W = B["M"], B["W"]
_product = B["product"]; REC = []
def _rec(x, w): r = _product(x, w); REC.append(r); return r
B["product"] = _rec                                          # every product base2's steps make is kept, in order

def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
KEYS = [f"{k} {l}" for l in range(12) for k in ("qkv pile", "proj pile", "fc pile", "mproj pile", "stream a", "stream")]
out = {k: [] for k in KEYS}; X16, EX, LS_, DIG, T5, WIN, NPOS = [], [], [], [], [], [], []
for t, ids in enumerate(TEXTS):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        REC.clear(); QKV = B["gauged_bank"](R, f"qkv.{l}"); out[f"qkv pile {l}"].append(REC[0])
        REC.clear(); att = B["plain_bank"](B["attention"](QKV), f"proj.{l}"); out[f"proj pile {l}"].append(REC[-1])
        R = R + att; out[f"stream a {l}"].append(R)
        REC.clear(); H = B["gauged_bank"](R, f"fc.{l}"); out[f"fc pile {l}"].append(REC[0])
        REC.clear(); mo = B["plain_bank"](B["edge"](H), f"mproj.{l}"); out[f"mproj pile {l}"].append(REC[0])
        R = R + mo; out[f"stream {l}"].append(R)
    REC.clear()
    w, we, b = W["head"]; x16, ex, LS = B["gauge_log"](R)
    S = B["power"](_product(x16, w), (we[None, :] + ex[:, None] + B["ES"]) * 65536 - LS[:, None]) + b
    X16.append(x16); EX.append(ex); LS_.append(LS); NPOS.append(P)
    DIG.append(np.array([np.frombuffer(hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).digest(), np.uint8) for s in S]))
    T5.append(top5(S)); WIN.append(np.array([B["race"](s) for s in S])); del S
    log(f"  text {t + 1:2d}/30: {P} positions ({time.time() - t0:.0f}s, {rss()} MB)")
os.makedirs(OUTD, exist_ok=True)
del W, B["W"]                                                # the plates are no longer needed here
save = {k: np.concatenate(out.pop(k)) for k in KEYS}
save.update(x16=np.concatenate(X16), ex=np.concatenate(EX), LS=np.concatenate(LS_), score_sha=np.concatenate(DIG),
            top5=np.concatenate(T5), win=np.concatenate(WIN), npos=np.array(NPOS))
np.savez(f"{OUTD}/plain_refs.npz", **{k.replace(" ", "_"): v for k, v in save.items()})
log(f"saved recording/C_ref/plain_refs.npz: {sum(NPOS)} positions, {len(save)} arrays, "
    f"{os.path.getsize(f'{OUTD}/plain_refs.npz'):,} bytes ({time.time() - t0:.0f}s, max memory {rss()} MB)")
