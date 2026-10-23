"""BASE2_FRINGES2 ON THE 30 REFS TEXTS: EVERY ARRAY AGAINST base2.py'S OWN RUN, AND THE COUNT (fringes_machine/run_refs.py's
checks, unchanged, with the level of additions ADD).

The reference is base2.py's own run kept by recording/C_plain.py (recording/C_ref/plain_refs.npz). Compared, at every
position the machine forms: qkv, proj and mproj piles; fc's whole pile (CHECK ONLY); fc's out16 and sh; the stream after
each half layer; the light at the top. The check path adds, at ADD >= 3, the arrival's products formed from signed digits
against base2's own products, and at ADD >= 4 the shared line order's sum of gaps x running sums against every pile.
The count: every part of the plain machine and of this machine on the same light; plain must come back 3,666,596,259.
    python3 fringes_machine2/run_refs.py add=N [texts=0,1,2] [order=SEED] [measure=0|1]
      -> fringes_machine2/refs_a<N>[_tag].log, refs_counts_a<N>[_tag].json, refs_top_light_a<N>[_tag].npz, refs_knife_a<N>[_tag].npz
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
os.environ["FR2_ADD"] = ARG["add"]; ADD = int(ARG["add"])
SEL = [int(v) for v in ARG["texts"].split(",")] if "texts" in ARG else list(range(30))
ORDER = int(ARG["order"]) if "order" in ARG else None; MEASURE = int(ARG.get("measure", 0))
TAG = f"_a{ADD}" + ("" if "texts" not in ARG else "_t" + "-".join(map(str, SEL))) + ("" if ORDER is None else f"_order{ORDER}")
OUT = f"{HERE}/fringes_machine2"
LOG = open(f"{OUT}/refs{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes2 as FM
assert FM.ADD == ADD
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL])
TEXTS = [FM.TEXTS[t] for t in SEL]
assert all(len(FM.TEXTS[t]) == npos[t] for t in range(30))
log(f"base2_fringes2 (ADD={ADD}) on refs texts {SEL if len(SEL) < 30 else 'all 30'} ({len(ROWS)} positions); order seed {ORDER}; "
    f"measure {MEASURE}")

_fcref = {}
def fc_ref(l):                                               # base2's own fc output, from its own fc pile
    if l not in _fcref:
        R = REF[f"stream_a_{l}"].astype(np.int64)[ROWS]; pile = REF[f"fc_pile_{l}"].astype(np.int64)[ROWS]
        x16, ex, LS = FM.gauge_log(R)
        H = FM.power(pile, (FM.M[f"fc.{l}.e"][None, :] + ex[:, None] + FM.ES) * 65536 - LS[:, None]) + FM.BIAS[f"fc.{l}"]
        o16, eo = FM.to16(FM.edge(H), -FM.ES); _fcref.clear(); _fcref[l] = (o16, eo + FM.ES)
    return _fcref[l]
DIFF = {}; CHECKED = {}
def check(key, value, rows=None):
    k, l = key.rsplit(" ", 1); l = int(l)
    if k == "fc out16": ref = fc_ref(l)[0]
    elif k == "fc sh": ref = fc_ref(l)[1]
    else: ref = REF[f"{k.replace(' ', '_')}_{l}"].astype(np.int64)[ROWS]
    if rows is not None: ref = ref[rows]
    v = np.asarray(value)
    assert v.shape == ref.shape, (key, v.shape, ref.shape)
    bad = (v != ref) if v.ndim == 1 else (v != ref).any(1)
    DIFF[key] = int(bad.sum()); CHECKED[key] = len(ref)
BAD = {}
def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)

C = FM.Count()
(x16, ex, LS), STOPS, GATES, MEAS, LPLS = FM.forward(TEXTS, C, check, chk, log=log,
                                               measure_layers=(tuple(range(12)) if MEASURE and ORDER is None else ()),
                                               order_seed=ORDER)
same_top = all(np.array_equal(a, REF[k][ROWS]) for a, k in ((x16, "x16"), (ex, "ex"), (LS, "LS")))
ndiff = sum(DIFF.values())
log(f"\nEXACT: {len(DIFF)} arrays checked; positions differing in any: {ndiff}; the light at the top identical: {same_top}")
for k in sorted(DIFF, key=lambda s: (int(s.rsplit(' ', 1)[1]), s)):
    if DIFF[k]: log(f"    DIFFERS: {k}: {DIFF[k]} of {CHECKED[k]}")
log(f"  the check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())) + f"; fallbacks {FM.FALLBACK}")
np.savez(f"{OUT}/refs_top_light{TAG}.npz", x16=x16, ex=ex, LS=LS, rows=ROWS)
np.savez(f"{OUT}/refs_knife{TAG}.npz", **{f"stop_{l}": STOPS[l] for l in range(12)}, **{f"gate_{l}": GATES[l] for l in range(12)})
dig = {k: FM.digest(v) for k, v in (("x16", x16), ("ex", ex), ("LS", LS))}
for l in range(12): dig[f"stop {l}"] = FM.digest(STOPS[l]); dig[f"gate {l}"] = FM.digest(GATES[l])
meas = {}
for l, Mz in MEAS.items():
    meas[str(l)] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in Mz.items()}
json.dump(dict(counts=dict(C), npos=int(len(ROWS)), texts=SEL, lpl=[int(v) for v in LPLS], diff=DIFF, bad=BAD, same_top=bool(same_top),
               digests=dig, measure=meas, fallback=FM.FALLBACK, add=ADD, rule4={k: list(v) for k, v in FM.RULE4.items()}),
          open(f"{OUT}/refs_counts{TAG}.json", "w"), indent=0)

n = len(ROWS)
def tot(who, part=None):
    return sum(v for k, v in C.items() if k.split("|")[0] == who and (part is None or k.split("|")[2] == part))
pl_, fr_ = tot("plain"), tot("fr")
log(f"\nWORK (stream part; the top read is fringes_machine2/top_read.py's): plain {FM.per_token(pl_, n):,} a token, "
    f"this machine {FM.per_token(fr_, n):,} a token")
for part in ("qkv", "arrival", "proj", "fc", "mproj", "rest"):
    log(f"  {part:8s} plain {FM.per_token(tot('plain', part), n):>14,}   this machine {FM.per_token(tot('fr', part), n):>14,}   "
        f"{FM.pct2(tot('fr', part), tot('plain', part))}")
if ADD >= 4:
    log("  ADDITION 4, the shared line order, MEASURED (count only; the rule kept every plate in signed digits):")
    for part in ("qkv", "proj", "fc", "mproj"):
        sh = sum(v for k, v in C.items() if k.startswith("side|") and k.split("|")[2] == part and "shared order: " in k and "against" not in k)
        sd = sum(v for k, v in C.items() if k.startswith("side|") and k.split("|")[2] == part and "shared order: against" in k)
        log(f"    {part:6s}: shared order (gaps x running sum, pours, rewriting) {FM.per_token(sh, n):>14,}; signed digits as the "
            f"machine meets this light {FM.per_token(sd, n):>14,}; {FM.pct2(sh, sd)}")
log(f"({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
