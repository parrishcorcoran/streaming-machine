"""WHERE THE EDGE'S BOTTOM PLACES GO AFTER LENS 2 (a read of the machine as it is; nothing changes).
For the road block of knife/settle.py: the edge's output (out16) is exact only in the last 2-4 diagonals. But lens 2's
pile is itself shifted down before it joins the stream (plain_bank: rshift(pile, -(we + ex + 16))), and the next
layer's light keeps only the stream's top 16 places (gauge_log's to16). This reads, at every position of the 30 refs
texts and every layer, from base2's own steps on the saved run's stream:
  s_m   the places lens 2's pile drops when it joins the stream (per position and output line)
  s_n   the places of the stream the next light does not keep (per position)
  and that the stream rebuilt here equals the saved run's stream after the layer.
    python3 knife/levels.py  -> knife/levels.log
"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
LOG = open(f"{HERE}/knife/levels.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
_src = _src.replace(_banks, '[f"{b}.{l}" for l in range(12) for b in ("fc", "mproj")]')
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES = J["M"], J["W"], J["ES"]
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
def q(a): v = np.sort(np.asarray(a).ravel()); return [int(v[(len(v) - 1) * k // 4]) for k in range(5)]   # min, quartiles, max
log("layer | lens 2 drops s_m places (min q1 med q3 max) | next light drops s_n places | edge-output sh (min med max) | stream identical")
for l in range(12):
    R = REF[f"stream_a_{l}"].astype(np.int64)
    out = J["edge"](J["gauged_bank"](R, f"fc.{l}")); o16, eo = J["to16"](out, -ES)
    w, we, b = W[f"mproj.{l}"]
    s_m = -(we[None, :] + eo[:, None] + ES)
    mo = J["plain_bank"](out, f"mproj.{l}")
    Rn = R + mo; same = bool(np.array_equal(Rn, REF[f"stream_{l}"]))
    _, en = J["to16"](Rn, -ES); s_n = en + ES
    log(f"{l:5d} | {q(s_m)} | {q(s_n)} | {q(eo + ES)[0]} {q(eo + ES)[2]} {q(eo + ES)[4]} | {same}")
