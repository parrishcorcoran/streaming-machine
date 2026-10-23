"""CHECKER: is the light settle.py read the machine's own?  base2.py's own forward (every bank, unchanged) on all 30
refs texts (ids + ref, 648 positions); the stream entering each fc.l, each fc pile and the stream after each layer
must equal recording/C_ref/plain_refs.npz, the run settle.py read its light from.  Whole numbers only.
    python3 knife/xchk_forward.py   -> knife/xchk_forward.log
"""
import sys, os, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
LOG = open(f"{HERE}/knife/xchk_forward.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W = J["M"], J["W"]
_prod = J["product"]; REC = []
def _rec(x, w): r = _prod(x, w); REC.append(r); return r
J["product"] = _rec
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
TEXTS = [r["ids"] + r["ref"] for r in json.load(open(f"{HERE}/refs_many.json"))]
npos = REF["npos"]; assert [len(t) for t in TEXTS] == list(npos)
off = np.concatenate([[0], np.cumsum(npos)])
bad = {"stream_a": 0, "fc_pile": 0, "stream": 0}; vals = 0
for t, ids in enumerate(TEXTS):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; r0, r1 = off[t], off[t + 1]
    for l in range(12):
        R = R + J["plain_bank"](J["attention"](J["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
        bad["stream_a"] += int((R != REF[f"stream_a_{l}"][r0:r1]).sum()); vals += R.size
        REC.clear(); H = J["gauged_bank"](R, f"fc.{l}")
        bad["fc_pile"] += int((REC[0] != REF[f"fc_pile_{l}"][r0:r1]).sum())
        R = R + J["plain_bank"](J["edge"](H), f"mproj.{l}")
        bad["stream"] += int((R != REF[f"stream_{l}"][r0:r1]).sum())
    if t % 5 == 4: log(f"  text {t + 1}/30 ({int(time.time() - t0)}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
log(f"base2's own forward, 30 texts, {int(off[-1])} positions, 12 layers: values differing from the saved run: "
    f"stream entering fc {bad['stream_a']} (of {vals:,}), fc pile {bad['fc_pile']}, stream after the layer {bad['stream']}")
log(f"ALL IDENTICAL: {sum(bad.values()) == 0}  ({int(time.time() - t0)}s, max memory "
    f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
