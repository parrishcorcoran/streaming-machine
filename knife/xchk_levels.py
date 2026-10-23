"""CHECKER: knife/levels.py's read at layers 0, 5, 11, rebuilt: places lens 2's pile drops (s_m), places the next light
drops (s_n), the edge output's water-level shift sh, and out16's last place as a power of two of x (= sh - 16).
    python3 knife/xchk_levels.py -> knife/xchk_levels.log"""
import sys, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
LOG = open(f"{HERE}/knife/xchk_levels.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_src = _src.replace('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[f"{b}.{l}" for l in (0, 5, 11) for b in ("fc", "mproj")]')
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES = J["M"], J["W"], J["ES"]; REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
def q(a): v = np.sort(np.asarray(a).ravel()); return [int(v[(len(v) - 1) * k // 4]) for k in range(5)]
for l in (0, 5, 11):
    R = REF[f"stream_a_{l}"].astype(np.int64); out = J["edge"](J["gauged_bank"](R, f"fc.{l}"))
    o16, eo = J["to16"](out, -ES); sh = eo + ES
    x16, ex = J["to16"](out, -ES); w, we, b = W[f"mproj.{l}"]
    s_m = -(we[None, :] + ex[:, None] + ES)                        # plain_bank's shift of lens 2's pile, per position and line
    Rn = R + J["plain_bank"](out, f"mproj.{l}"); same = bool(np.array_equal(Rn, REF[f"stream_{l}"]))
    _, en = J["to16"](Rn, -ES)
    log(f"layer {l}: s_m (min q1 med q3 max) {q(s_m)}; s_n {q(en + ES)}; edge-output sh {q(sh)}; out16 last place 2^(sh-16): "
        f"median 2^{q(sh)[2] - 16}, max 2^{q(sh)[4] - 16}; stream identical {same}")
