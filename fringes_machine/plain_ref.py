"""THE PLAIN REFERENCE FOR THE FRINGES MACHINE: base2.py's own steps (its gauged_bank, plain_bank, attention, edge,
product, from base2_gpt2.npz) on every text, in a separate process; the fringes machine never runs here.
Kept per text: the sha256 of every array the fringes machine is compared on, in its row conventions:
  qkv pile l, proj pile l, fc pile l, fc out16 l (the bend's output after mproj's water level), fc sh l (its shift),
  mproj pile l, stream a l, stream l, for l = 0..11; at layer 0, proj pile / stream a / fc pile / fc out16 / fc sh /
  mproj pile over positions 1 on only (position 0 reads its stream after layer 0 whole); and the light at the top.
For the windows also the light at the top of the 20 judged positions (positions 3..22), checked against
collapse_top/light_top.npz (the plain machine's light saved earlier, src = 1).
    python3 fringes_machine/plain_ref.py set=windows|refs  -> fringes_machine/plain_digests_<set>.json (+ windows light)
"""
import sys, os, json, time, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
ARG = dict(a.split("=") for a in sys.argv[1:]); SET = ARG.get("set", "windows")
OUT = f"{HERE}/fringes_machine"; t0 = time.time()
LOG = open(f"{OUT}/plain_ref_{SET}.log", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; B = {}; exec(_src, B); sys.argv = _argv; os.chdir(_cwd)
M = B["M"]
class _Plates(dict):                                         # base2's W, each bank's float64 copy made when read
    def __missing__(self, k): return (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"])
B["W"] = _Plates()
_product = B["product"]; REC = []
def _rec(x, w): r = _product(x, w); REC.append(r); return r
B["product"] = _rec
def dg(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
if SET == "windows": TEXTS = [w["ids"] for w in json.load(open(f"{HERE}/judge5_texts.json"))]
else: TEXTS = [r["ids"] + r["ref"] for r in json.load(open(f"{HERE}/refs_many.json"))]
DIG = []; X16, EXS, LSS = [], [], []
for t, ids in enumerate(TEXTS):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; d = {}
    for l in range(12):
        sub = slice(1, None) if l == 0 else slice(None)
        REC.clear(); QKV = B["gauged_bank"](R, f"qkv.{l}"); d[f"qkv pile {l}"] = dg(REC[0])
        REC.clear(); att = B["plain_bank"](B["attention"](QKV), f"proj.{l}"); d[f"proj pile {l}"] = dg(REC[-1][sub])
        R = R + att; d[f"stream a {l}"] = dg(R[sub])
        REC.clear(); H = B["gauged_bank"](R, f"fc.{l}"); d[f"fc pile {l}"] = dg(REC[0][sub])
        G = B["edge"](H); g16, eg = B["to16"](G, -B["ES"]); d[f"fc out16 {l}"] = dg(g16[sub]); d[f"fc sh {l}"] = dg((eg + B["ES"])[sub])
        REC.clear(); mo = B["plain_bank"](G, f"mproj.{l}"); d[f"mproj pile {l}"] = dg(REC[0][sub])
        R = R + mo; d[f"stream {l}"] = dg(R)
    x16, ex, LS = B["gauge_log"](R); d["top light"] = dg(np.concatenate([x16.ravel(), ex, LS]))
    DIG.append(d)
    if SET == "windows": X16.append(x16[3:23]); EXS.append(ex[3:23]); LSS.append(LS[3:23])
    if t % 50 == 49 or t == len(TEXTS) - 1: log(f"  {t + 1}/{len(TEXTS)} texts ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
json.dump(DIG, open(f"{OUT}/plain_digests_{SET}.json", "w"))
if SET == "windows":
    X, EX, LS = np.concatenate(X16), np.concatenate(EXS), np.concatenate(LSS)
    np.savez(f"{OUT}/plain_windows_top_light.npz", x16=X, ex=EX, LS=LS)
    L = np.load(f"{HERE}/collapse_top/light_top.npz"); same = 0
    for i in range(len(TEXTS)):
        gi = np.flatnonzero((L["src"] == 1) & (L["win"] == i)); r = slice(20 * i, 20 * i + 20)
        same += int(np.array_equal(X[r], L["x16"][gi]) and np.array_equal(EX[r], L["ex"][gi]) and np.array_equal(LS[r], L["LS"][gi]))
    log(f"the light at the top of the 6,000 judged positions = collapse_top/light_top.npz (saved earlier): {same}/{len(TEXTS)} windows")
log(f"saved fringes_machine/plain_digests_{SET}.json: {len(DIG)} texts x {len(DIG[0])} arrays ({time.time() - t0:.0f}s)")
