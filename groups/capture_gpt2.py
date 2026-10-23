"""CAPTURE: THE LIGHT AT EVERY BANK OF THE ALL-BASE-2 GPT-2, AS IT MEETS THE PLATE, ON THE 30 REFS TEXTS, EVERY POSITION.

The machine is base2.py, run from its own source (as recording/C_plain.py does): its gauged_bank, plain_bank, attention,
edge, product, read from base2_gpt2.npz. One thing is replaced, and it changes no number: `product` is wrapped so that
when it meets one of the 48 bank plates, the light it is handed (x16: 16 places a value, one power of two a position,
as to16 writes it) is kept, with the pile it returns.
  qkv.l, fc.l   the light is the stream after the gauge (gauge_log's to16 of the stream)
  proj.l        the light is the arrival's output written by to16
  mproj.l       the light is the bend's output written by to16
CHECK: every kept pile equals recording/C_ref/plain_refs.npz's pile (base2's own run in a separate process), every bank,
every position; and the kept light times the plate (whole numbers, int64) gives that same pile again.
    python3 groups/capture_gpt2.py  -> groups/light_gpt2/{qkv,proj,fc,mproj}.npy  (12 layers x 648 positions x lines, int32)
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups/light_gpt2"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; B = {}; exec(_src, B); sys.argv = _argv; os.chdir(_cwd)
M, W = B["M"], B["W"]
BANKS = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")]
WID = {id(W[n][0]): n for n in BANKS}
_product = B["product"]; KEEP = {}
def _keep(x, w):
    r = _product(x, w); n = WID.get(id(w))
    if n is not None:
        x = np.asarray(x); assert x.dtype.kind == "i" and np.abs(x).max(initial=0) <= 1 << 15
        KEEP.setdefault(n, []).append((x.astype(np.int32), r))
    return r
B["product"] = _keep

REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
assert [len(t) for t in TEXTS] == REF["npos"].tolist()
for t, ids in enumerate(TEXTS):                               # base2.forward's own lines, every position kept
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + B["plain_bank"](B["attention"](B["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
        R = R + B["plain_bank"](B["edge"](B["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
    if t % 10 == 9: print(f"  text {t + 1}/30 ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
NP = int(REF["npos"].sum()); bad = 0; bad2 = 0
for kind in ("qkv", "proj", "fc", "mproj"):
    LT = []
    for l in range(12):
        n = f"{kind}.{l}"; assert len(KEEP[n]) == 30, (n, len(KEEP[n]))
        x = np.concatenate([a for a, _ in KEEP[n]]); pile = np.concatenate([b for _, b in KEEP[n]])
        ref = REF[f"{kind}_pile_{l}"]; assert pile.shape == ref.shape
        bad += int((pile != ref).any(1).sum())
        w = np.asarray(M[n + ".w"]).astype(np.int64); again = x.astype(np.int64) @ w        # whole numbers, int64
        bad2 += int((again != ref).any(1).sum()); LT.append(x); del KEEP[n]
    LT = np.stack(LT); assert LT.shape[:2] == (12, NP)
    np.save(f"{OUT}/{kind}.npy", LT)
    print(f"  {kind:5s} light {LT.shape} int32, largest |x16| {int(np.abs(LT).max())}, values at exactly 2^15: "
          f"{int((np.abs(LT) == 1 << 15).sum())}, dark values {100 * float((LT == 0).mean()):.2f}%", flush=True)
print(f"CHECK: kept piles against plain_refs.npz, positions differing over 48 banks: {bad}; the kept light times the plate "
      f"(int64) against plain_refs.npz: {bad2}")
assert bad == 0 and bad2 == 0
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
