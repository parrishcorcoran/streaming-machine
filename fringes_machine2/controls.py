"""SHUFFLE CONTROLS FOR BASE2_FRINGES2's ADDITIONS 3 AND 4 (counts only; the machine itself always runs on the real light
and stays exact). On 6 refs texts (0, 5, 10, 15, 20, 25), at the level ADD=4:
  3. the arrival in signed digits: at every head, q and k each shuffled across their 64 lines, each position on its
     own (every value kept), and the scores' lit-digit meetings against the plain meetings of the same shuffled q, k;
     the shares shuffled among the positions each row reads (every share kept), and the shares x values meetings
     against the plain meetings of the same shuffled shares;
  4. the shared line order: at every plate the light that meets it shuffled across its lines, each position on its
     own, and the shared form's work (gaps x running sum, pours, rewriting) against signed digits on the same light.
The top read's control (additions 1 and 2) is fringes_machine2/top_read.py light=shuffled on all 648 refs positions.
PREDICTIONS (written before the run): 3: signed digits' share of the plain meetings within 1 point of the real light's
(the cut is the values' sizes); 4: the shared form's share of signed digits about the same on shuffled light as on real
light (the order is the plate's own; the light's lines carry no order the plate's line sums could follow).
    python3 fringes_machine2/controls.py   -> fringes_machine2/controls.log, controls.json
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
os.environ["FR2_ADD"] = "4"
OUT = f"{HERE}/fringes_machine2"; LOG = open(f"{OUT}/controls.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes2 as FM
SEL = [0, 5, 10, 15, 20, 25]; TEXTS = [FM.TEXTS[t] for t in SEL]
rng = np.random.default_rng(2026)
K = FM.Count()
def shuffle(x): return np.stack([r[rng.permutation(x.shape[1])] for r in np.asarray(x, np.int64)])
def ctl(kind, lay, pl, x, F=None, bias=None, c0=None):          # ADDITION 4 on the real and the shuffled light
    xr = np.asarray(x, np.int64); xs = shuffle(xr)
    for name, xx in (("real", xr), ("shuffled", xs)):
        _, W = FM.shared_work(pl, xx)
        K.add(f"{kind}|{name}|shared", sum(W.values()))
        xp = xx if pl.perm is None else xx[:, pl.perm]
        K.add(f"{kind}|{name}|signed", FM.naf(xp) @ pl.nrow + FM.ones(xp).sum())
def actl(lay, q16, k16, v, A, first_counted):                   # ADDITION 3 on the real and the shuffled light
    P = len(q16); low = np.tril(np.ones((P, P), bool))
    qs, ks = shuffle(q16), shuffle(k16)
    As = A.copy()
    for p in range(P): As[p, :p + 1] = A[p, rng.permutation(p + 1)]
    for name, qq, kk, AA in (("real", q16, k16, A), ("shuffled", qs, ks, As)):
        K.add(f"scores|{name}|signed", (FM.naf(qq) @ FM.naf(kk).T)[low].sum()); K.add(f"scores|{name}|plain", (FM.ones(qq) @ FM.ones(kk).T)[low].sum())
        K.add(f"mix|{name}|signed", (FM.naf(AA) @ FM.naf(v).sum(1)).sum()); K.add(f"mix|{name}|plain", (FM.ones(AA) @ FM.ones(v).sum(1)).sum())
FM.ARRIVAL_CTL = actl
(x16, ex, LS), STOPS, GATES, MEAS, LPLS = FM.forward(TEXTS, None, None, None, log=log, ctl=ctl)
n = sum(len(t) for t in TEXTS)
log(f"\nCONTROLS on refs texts {SEL} ({n} positions); the machine ran on the real light")
for part in ("scores", "mix"):
    log(f"  3. the arrival's {part:6s} in signed digits: real {FM.pct2(K[f'{part}|real|signed'], K[f'{part}|real|plain'])} of its plain "
        f"meetings; shuffled {FM.pct2(K[f'{part}|shuffled|signed'], K[f'{part}|shuffled|plain'])}")
for kind in ("qkv", "proj", "fc", "mproj"):
    log(f"  4. the shared order at {kind:5s}: real {FM.pct2(K[f'{kind}|real|shared'], K[f'{kind}|real|signed'])} of signed digits' work "
        f"on the same light; shuffled {FM.pct2(K[f'{kind}|shuffled|shared'], K[f'{kind}|shuffled|signed'])}")
json.dump(dict(K=dict(K), n=int(n), texts=SEL), open(f"{OUT}/controls.json", "w"), indent=0)
log(f"({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
