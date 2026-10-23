"""THE FRINGES MACHINE (base2_fringes.py) ON THE 30 REFS TEXTS: EVERY ARRAY AGAINST base2.py'S OWN RUN, AND THE COUNT.

The reference is base2.py's own run kept by recording/C_plain.py (recording/C_ref/plain_refs.npz: every pile of the
48 banks, the stream after every half layer, the light at the top). fc's output that mproj meets (out16 and the shift
sh) is made from that run's own fc pile with base2's power, edge and to16, here in the check.
Compared, at every position the machine forms: qkv, proj and mproj piles; fc's whole pile (CHECK ONLY: the machine
stops each edge early; the pieces past each stop are simulated only to be checked); fc's out16 and sh; the stream after
each half layer; the light at the top. Layer 0's position 0 reads its stream whole (Z0): its proj/fc/mproj values are
not formed, and base2's own layer 0 there is run in the check against Z0.
The count: every part of the plain machine and of the fringes machine on the same light (base2_fringes.py's rules).
The plain total must come back as 3,666,596,259 a token.
Also measured (knife_measure, every layer): settle.py's rule recounted in signed digits (per-edge bound, every stage)
and the shared bound at every stage with no gate, against the machine's (shared bound, gated).
    python3 fringes_machine/run_refs.py [texts=0,1,2] [order=SEED] [measure=0|1]
      -> fringes_machine/refs[_tag].log, refs_counts[_tag].json, refs_top_light[_tag].npz, refs_knife[_tag].npz
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
SEL = [int(v) for v in ARG["texts"].split(",")] if "texts" in ARG else list(range(30))
ORDER = int(ARG["order"]) if "order" in ARG else None; MEASURE = int(ARG.get("measure", 1))
TAG = ("" if "texts" not in ARG else "_t" + "-".join(map(str, SEL))) + ("" if ORDER is None else f"_order{ORDER}")
OUT = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/refs{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes as FM
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL])
TEXTS = [FM.TEXTS[t] for t in SEL]
assert all(len(FM.TEXTS[t]) == npos[t] for t in range(30))
log(f"fringes machine on refs texts {SEL if len(SEL) < 30 else 'all 30'} ({len(ROWS)} positions); order seed {ORDER}; "
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
               digests=dig, measure=meas, fallback=FM.FALLBACK), open(f"{OUT}/refs_counts{TAG}.json", "w"), indent=0)

# ---- a first look at the count (the full table is fringes_machine/report.py's) ----
n = len(ROWS)
def tot(who, part=None):
    return sum(v for k, v in C.items() if k.split("|")[0] == who and (part is None or k.split("|")[2] == part))
pl_, fr_ = tot("plain"), tot("fr")
log(f"\nWORK (stream part; the top read is fringes_machine/top_read.py's): plain {FM.per_token(pl_, n):,} a token, "
    f"fringes {FM.per_token(fr_, n):,} a token")
for part in ("qkv", "arrival", "proj", "fc", "mproj", "rest"):
    log(f"  {part:8s} plain {FM.per_token(tot('plain', part), n):>14,}   fringes {FM.per_token(tot('fr', part), n):>14,}   "
        f"{FM.pct2(tot('fr', part), tot('plain', part))}")
for l, Mz in sorted(MEAS.items()):
    log(f"  knife measurement L{l}: settle.py's rule (per-edge bound, every stage) {FM.pct2(Mz['per-edge work'], Mz['full'])} "
        f"of fc's signed-digit meetings; shared bound, every stage {FM.pct2(Mz['shared work'], Mz['full'])}")
for l in range(12):
    lay = f"L{l}"; full = C.get(f"side|{lay}|fc|meetings, signed digits, no stops", 0)
    stopped = C.get(f"fr|{lay}|fc|meetings, signed digits, edges stopped", 0)
    KDl = FM.grid(LPLS[l])[1]
    log(f"  knife L{l}: meetings with the stops {FM.pct2(stopped, full)} of fc's signed-digit meetings; gate opens at "
        f"top {KDl[int(np.median(GATES[l]))]} (median); light places {LPLS[l]}; stop stage histogram "
        + " ".join(str(int(v)) for v in np.bincount(STOPS[l].ravel(), minlength=len(KDl))))
log(f"({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
