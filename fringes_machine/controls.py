"""SHUFFLE CONTROLS FOR THE FRINGES MACHINE'S CUTS (counts only; the machine itself always runs on the real light and
stays exact). On 6 refs texts (0, 5, 10, 15, 20, 25), at every plate the light that meets it is also shuffled across its
lines, each position on its own (every value kept, only which line it meets changes), and counted again:
  - signed digits: the lit-pixel meetings of the shuffled light, against the plain meetings of the same shuffled light;
  - knife stops: the whole knife (shared bound, gate, carrying to the edge's scale, stops) run on the piles the shuffled
    light makes, with the same stored part, gains and biases: the meetings it needs and its own carrying work;
  - layer 0 from storage: fc.0's split made with another word's stored rows (the same tables, the pairing broken):
    what would then still meet the plate, in signed digits.
The top read's control is fringes_machine/top_read.py light=shuffled (all 648 refs positions).
PREDICTIONS (written before the run): signed digits within 0.5 points of the real share (recording/C: 58.71 against
58.65%); knife stops within 2 points over all layers (settle.py: 91.1 against 91.8%), possibly further apart at layer
0 (settle.py's checker: 73.3 against 93.2% at layer 0); the layer-0 splits with broken pairing meet more than plain
(layer0/measure, plain photons: 64.6M against 61.3M plain at fc.0).
    python3 fringes_machine/controls.py   -> fringes_machine/controls.log, controls.json
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
OUT = f"{HERE}/fringes_machine"; LOG = open(f"{OUT}/controls.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes as FM
SEL = [0, 5, 10, 15, 20, 25]; TEXTS = [FM.TEXTS[t] for t in SEL]
rng = np.random.default_rng(2026)
C = FM.Count(); K = FM.Count()
def shuffle(x): return np.stack([r[rng.permutation(x.shape[1])] for r in np.asarray(x, np.int64)])
def ctl(kind, lay, pl, x, F=None, bias=None, c0=None):
    xs = shuffle(x); xr = np.asarray(x, np.int64)
    if pl.perm is not None: xs_, xr_ = xs[:, pl.perm], xr[:, pl.perm]
    else: xs_, xr_ = xs, xr
    K.add(f"{kind}|real|signed", FM.naf(xr_) @ pl.nrow); K.add(f"{kind}|real|plain", FM.ones(xr_) @ pl.wrow)
    K.add(f"{kind}|shuffled|signed", FM.naf(xs_) @ pl.nrow); K.add(f"{kind}|shuffled|plain", FM.ones(xs_) @ pl.wrow)
    if kind == "fc":                                         # the whole knife on the shuffled light's piles
        Ck = FM.Count()
        FM.knife(pl, xs, c0, F, bias, Ck, lay)
        for k_, v in Ck.items():
            who, l_, part, item = k_.split("|")
            if who in ("fr", "side"): K.add(f"knife|{lay}|shuffled|{item}", v)
# layer 0's broken pairings: read the light at fc.0 and proj.0 from the check callback
WTE, WPE = FM.M["wte"], FM.M["wpe"]; BPROJ = FM.BIAS["proj.0"]
ids = np.concatenate([np.asarray(t, np.int64) for t in TEXTS]); pos = np.concatenate([np.arange(len(t)) for t in TEXTS])
r1 = np.flatnonzero(pos > 0)
PV = {}
def check(key, value, rows=None):
    if key == "stream a 0":
        x1, e1, LS1 = FM.gauge_log(value); s1 = e1 + FM.ES
        other = rng.integers(0, 50257, len(r1))
        if "fc.0" not in PV: PV["fc.0"] = FM.table_values("fc.0")[0]
        w = PV["fc.0"]; nrow = FM.naf(w).sum(1)
        own = x1 - FM.rshift(WTE[ids[r1]], s1[:, None]) - FM.rshift(WPE[pos[r1]] + BPROJ[None, :], s1[:, None])
        oth = x1 - FM.rshift(WTE[other], s1[:, None]) - FM.rshift(WPE[pos[r1]] + BPROJ[None, :], s1[:, None])
        K.add("L0 fc.0|own rows|signed", FM.naf(own) @ nrow); K.add("L0 fc.0|another word's rows|signed", FM.naf(oth) @ nrow)
        K.add("L0 fc.0|whole light|signed", FM.naf(x1) @ nrow); K.add("L0 fc.0|whole light|plain", FM.ones(x1) @ FM.ones(w).sum(1))
C2 = FM.Count(); OTHER_KEY = "L0 fc.0|another word's rows|signed"
(x16, ex, LS), STOPS, GATES, MEAS, LPLS = FM.forward(TEXTS, C2, check, None, log=log, ctl=ctl)
n = len(ids)
log(f"\nCONTROLS on refs texts {SEL} ({n} positions); the machine ran on the real light")
for kind in ("qkv", "proj", "fc", "mproj"):
    rs, rp, ss, sp = (K.get(f"{kind}|{a}|{b}", 0) for a, b in (("real", "signed"), ("real", "plain"), ("shuffled", "signed"), ("shuffled", "plain")))
    log(f"  signed digits at {kind:5s}: real {FM.pct2(rs, rp)} of its plain meetings; light shuffled across its lines {FM.pct2(ss, sp)}")
tr = sum(C2.get(f"fr|L{l}|fc|meetings, signed digits, edges stopped", 0) for l in range(12))
tf = sum(C2.get(f"side|L{l}|fc|meetings, signed digits, no stops", 0) for l in range(12))
sr = sum(K.get(f"knife|L{l}|shuffled|meetings, signed digits, edges stopped", 0) for l in range(12))
sf = sum(K.get(f"knife|L{l}|shuffled|meetings, signed digits, no stops", 0) for l in range(12))
log(f"  knife stops, all layers: real {FM.pct2(tr, tf)} of fc's signed-digit meetings; shuffled {FM.pct2(sr, sf)}")
for l in range(12):
    a, b = C2.get(f"fr|L{l}|fc|meetings, signed digits, edges stopped", 0), C2.get(f"side|L{l}|fc|meetings, signed digits, no stops", 0)
    c, d = K.get(f"knife|L{l}|shuffled|meetings, signed digits, edges stopped", 0), K.get(f"knife|L{l}|shuffled|meetings, signed digits, no stops", 0)
    def carry(src, pre):
        return sum(v for k, v in src.items() if k.startswith(pre) and "knife:" in k)
    log(f"    L{l:2d}: real {FM.pct2(a, b)}, shuffled {FM.pct2(c, d)}; the knife's own carrying work real "
        f"{carry(C2, f'fr|L{l}|fc|') / n:,.0f}, shuffled {carry(K, f'knife|L{l}|shuffled|') / n:,.0f} a token")
log(f"  layer 0, fc.0 (positions 1 on, per token there): whole light plain {K.get('L0 fc.0|whole light|plain', 0) / len(r1):,.0f}, "
    f"signed {K.get('L0 fc.0|whole light|signed', 0) / len(r1):,.0f}; the split with its own rows {K.get('L0 fc.0|own rows|signed', 0) / len(r1):,.0f}; "
    f"with another word's rows {K.get(OTHER_KEY, 0) / len(r1):,.0f}")
json.dump(dict(K=dict(K), C=dict(C2), n=int(n), texts=SEL), open(f"{OUT}/controls.json", "w"), indent=0)
log(f"({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
