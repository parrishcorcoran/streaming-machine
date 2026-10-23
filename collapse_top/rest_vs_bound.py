"""HOW MUCH OF THE ROOM THE UNREAD PART ACTUALLY USES. staged_top.py sets a word aside only when even the worst case
of its unread pairs (every one of them lined up the same way) cannot lift it to the top 5. Here, for the stamp
schedule after 8 and after 12 stamp places, on the 480 refs positions, every word's actual unread part (exact pile
minus the counted part) against that worst case (the bound). Also: the share of the unread pairs that pair off (up
against down) and the share left over.
CONTROL: the same with each position's light shuffled across its lines (the same kind of shuffle as staged_top.py).
PREDICTION (written before running): the unread part uses 2-5% of the room the bound allows (median over words), since
almost all pairs pair off.
    python3 collapse_top/rest_vs_bound.py
"""
import sys, os, types, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '["head"]')
_argv = sys.argv; sys.argv = ["x"]; _g = {}; exec(_src, _g); sys.argv = _argv; J = types.SimpleNamespace(**_g)
PL, NW, LINES = 15, 50257, 768
L = np.load(f"{HERE}/collapse_top/light_top.npz"); g = np.flatnonzero(L["src"] == 0)
X0 = L["x16"][g].astype(np.int64)
TT = np.ascontiguousarray(J.M["head.w"].T); TA = np.abs(TT.astype(np.int32)); TS = np.sign(TT).astype(np.int64)
NT = np.stack([((TA >> b) & 1).sum(1) for b in range(PL)], 1)
UP = {}; DN = {}
for k in (8, 12):
    cut = PL - k; low = (1 << cut) - 1
    Tlo = (TS * (TA & low)).astype(np.float64)                      # the unread stamp places, with direction
    absT = np.abs(Tlo)
    for light in ("real", "shuffled"):
        X = X0 if light == "real" else np.stack([x[np.random.default_rng(2026 + i).permutation(LINES)] for i, x in enumerate(X0)])
        rest = (X.astype(np.float64) @ Tlo.T).astype(np.int64)                                # actual unread part, every word
        TOT = (np.abs(X).astype(np.float64) @ absT.T)                                          # sum of |pair values|, every word
        ratios, left = [], []
        for p in range(len(X)):
            ax = np.abs(X[p]); nx = ((ax[None, :] >> np.arange(PL)[:, None]) & 1).sum(1)
            U = np.zeros((PL, PL), bool); U[:, :cut] = True
            G = (U.T * (1 << np.arange(PL))[None, :]) @ np.minimum(nx[:, None], np.arange(LINES + 1)[None, :])
            R = (G[np.arange(PL)[None, :], NT] << np.arange(PL)[None, :]).sum(1)
            ratios.append(np.median(np.abs(rest[p]) / R))
            # pairs: up and down (worth 2^(a+b) each), how much of their total is left after they pair off
            left.append(np.median(np.abs(rest[p]) / np.maximum(TOT[p], 1)))
        print(f"after {k} stamp places, light {light:8s}: the unread part is {int(round(100 * np.median(ratios)))}% of the bound "
              f"(median over words, then positions); of the unread pairs' total size, {int(round(100 * np.median(left)))}% is left after up and down pair off", flush=True)
