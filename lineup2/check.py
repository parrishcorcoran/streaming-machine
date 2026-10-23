"""SELF-CHECK OF WAY 2 WITH SEPARATE CODE (numpy only; it never imports lineup2/way2.py or the walk in C).
Not the independent checker rule 10 asks for (same session); it rebuilds what it can a different way:
  - the light at five tables from base2.py's own saved run (recording/C_ref/plain_refs.npz) with base2's own steps:
    qkv.3 (the stream after layer 2), fc.6 (the stream after layer 6's arrival), mproj.9 (fc.9's saved pile through the
    water level and the bend), proj.0 (qkv.0's saved pile through the arrival), the top table (the saved light);
  - its own line-up: np.lexsort by (weight, neuron), its own gaps; its own walk: on each line the running product as a
    cumulative sum of x times the gaps, laid back on the neurons by the order;
  - its own count: signed digits found by the step-by-step rule (an odd value takes the digit 2 - (v mod 4)), ones of
    each running product, plain 1s;
  and compares the piles with base2's saved piles (the top table: base2's product), and the counts with what
  lineup2/run_refs.py stored for the same positions (refs_main_counts.npz).
    python3 lineup2/check.py [texts=3,17]   -> lineup2/check.log
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"
ARG = dict(a.split("=") for a in sys.argv[1:]); SEL = [int(v) for v in ARG.get("texts", "3,17").split(",")]
LOG = open(f"{OUT}/check.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _src.count(_a) == 1; _src = _src.replace(_a, _b)
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv; M = B["M"]; ES = B["ES"]
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL])
CN = np.load(f"{OUT}/refs_main_counts.npz"); CROWS = CN["rows"]; pos_in = {int(r): k for k, r in enumerate(CROWS)}
CR = np.array([pos_in[int(r)] for r in ROWS])
def sd_count(v):                                             # lit places in signed digits, the step-by-step rule
    a = np.abs(np.asarray(v, np.int64)).copy(); n = np.zeros(a.shape, np.int64)
    while a.any():
        odd = (a & 1) == 1; d = np.where(odd, 2 - (a % 4), 0); n += odd; a = (a - d) >> 1
    return n
def plain1(v): return np.bitwise_count(np.abs(np.asarray(v, np.int64))).astype(np.int64)
def gl(R): return B["gauge_log"](R)
def light(name):
    if name == "qkv.3": return gl(REF["stream_2"][ROWS])[0], REF["qkv_pile_3"][ROWS]
    if name == "fc.6": return gl(REF["stream_a_6"][ROWS])[0], REF["fc_pile_6"][ROWS]
    if name == "mproj.9":
        x16, ex, LS = gl(REF["stream_a_9"][ROWS])
        H = B["power"](REF["fc_pile_9"][ROWS], (M["fc.9.e"][None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + M["fc.9.b"]
        return B["to16"](B["edge"](H), -ES)[0], REF["mproj_pile_9"][ROWS]
    if name == "proj.0":
        outs = []
        for t in SEL:
            r = np.arange(offs[t], offs[t + 1]); ids = (json.load(open(f"{HERE}/refs_many.json"))[t]); ids = ids["ids"] + ids["ref"]
            x16, ex, LS = gl(M["wte"][ids] + M["wpe"][:len(ids)])
            QKV = B["power"](REF["qkv_pile_0"][r], (M["qkv.0.e"][None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + M["qkv.0.b"]
            outs.append(B["attention"](QKV))
        return B["to16"](np.concatenate(outs), -ES)[0], REF["proj_pile_0"][ROWS]
    if name == "head":
        x = REF["x16"][ROWS]; return x, B["product"](x, np.asarray(M["head.w"]).astype(np.float64))
for name in ("qkv.3", "fc.6", "mproj.9", "proj.0", "head"):
    x, ref = light(name); w = np.asarray(M[name + ".w"]).astype(np.int64); L, N = w.shape; P = len(x)
    pile = np.zeros((P, N), np.int64); pours = np.zeros(P, np.int64); gapsd = np.zeros(L, np.int64)
    for i in range(L):
        o = np.lexsort((np.arange(N), w[i])); g = np.diff(w[i][o], prepend=0); gapsd[i] = sd_count(g).sum()
        r = np.cumsum(x[:, i:i + 1] * g[None, :], axis=1)        # the running product on line i, every rank
        pile[:, o] += r; pours += plain1(r).sum(1)
    same = int((pile == ref).all(1).sum())
    gm = sd_count(x) @ gapsd; pl = plain1(x) @ plain1(w).sum(1)
    c = lambda k: CN[f"{name}|{k}"][CR]
    log(f"  {name:8s}: piles = base2's {same}/{P}; gap meetings {int(gm.sum()):,} (machine {int(c('gaps').sum()):,}, equal per position "
        f"{int((gm == c('gaps')).sum())}/{P}); pours {int(pours.sum()):,} (machine {int(c('pours').sum()):,}, equal {int((pours == c('pours')).sum())}/{P}); "
        f"plain {int(pl.sum()):,} (machine {int(c('plain').sum()):,}); gaps {100 * gm.sum() / pl.sum():.2f}% pours {100 * pours.sum() / pl.sum():.2f}% of plain "
        f"({time.time() - t0:.0f}s)")
log(f"texts {SEL}, {len(ROWS)} positions ({time.time() - t0:.0f}s)")
