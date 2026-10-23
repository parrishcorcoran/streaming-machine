"""WAY 2 ON THE 30 REFS TEXTS (648 positions): EVERY PILE AGAINST base2.py'S OWN RUN, AND THE COUNT.

The reference is base2.py's own run in a separate process (recording/C_ref/plain_refs.npz, recording/C_plain.py):
every pile of the 48 banks at every position, the stream after each half layer, the light at the top, and at the top
every position's score row (sha256), top 5 and race winner. The top table's piles are also checked against base2's
own product of the same light (the CHECK path here, never counted, never used by the machine).
    python3 lineup2/run_refs.py                      the machine, all 30 texts, every count
    python3 lineup2/run_refs.py mode=order           ORDER-FREE: texts 0, 14, 29; at every bank (and the top table)
                                                     the lines walked in a random order and every walk made DOWN from
                                                     the top weight
    python3 lineup2/run_refs.py mode=shuffle         SHUFFLE CONTROL: texts 0, 5, 10, 15, 20, 25; every line's gaps
                                                     taken in a random neuron order (the same weights, the line-up
                                                     broken); the walk still made and checked
  -> lineup2/refs_<mode>.log, refs_<mode>_counts.json, refs_<mode>_counts.npz
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"; sys.path.insert(0, OUT)
ARG = dict(a.split("=") for a in sys.argv[1:]); MODE = ARG.get("mode", "main")
SEL = {"main": list(range(30)), "order": [0, 14, 29], "shuffle": [0, 5, 10, 15, 20, 25]}[MODE]
TAG = MODE
if "texts" in ARG: SEL = [int(v) for v in ARG["texts"].split(",")]; TAG = MODE + "_t" + "-".join(map(str, SEL))
LOG = open(f"{OUT}/refs_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import way2 as W2
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL])
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [REFS[t]["ids"] + REFS[t]["ref"] for t in SEL]
assert [len(t) for t in TEXTS] == [int(npos[t]) for t in SEL]
log(f"WAY 2 on refs texts {SEL if len(SEL) < 30 else 'all 30'} ({len(ROWS)} positions), mode {MODE}")

if MODE == "shuffle": rec = W2.Record(order="random", seed=500, log=log)
else: rec = W2.Record(order="sorted", log=log, keep_shuffled_counts=(MODE == "main"))
mach = W2.Machine(rec, count=True, line_rng=(np.random.default_rng(7) if MODE == "order" else None), down=(MODE == "order"))

DIFF, CHECKED = {}, {}
def check(key, value):
    ref = REF[key.replace(" ", "_")][ROWS]; v = np.asarray(value); assert v.shape == ref.shape, (key, v.shape, ref.shape)
    DIFF[key] = int((v != ref).any(1).sum()); CHECKED[key] = len(ref)
x16, ex, LS = mach.forward(TEXTS, check=check, log=log)
same_top = all(np.array_equal(a, REF[k][ROWS]) for a, k in ((x16, "x16"), (ex, "ex"), (LS, "LS")))
log(f"EXACT (stream): {len(DIFF)} arrays checked at {len(ROWS)} positions; positions differing in any: {sum(DIFF.values())}; "
    f"the light at the top identical: {same_top}")
for k, v in DIFF.items():
    if v: log(f"    DIFFERS: {k}: {v} of {CHECKED[k]}")

# ---- the top table by the walk, every position; its piles checked against base2's own product (CHECK path) ----
HW = np.asarray(W2.M["head.w"]).astype(np.float64); TOPBAD = [0, 0]
def top_check(pile, xr):
    ref = W2.product(xr, HW); TOPBAD[0] += int((pile != ref).any(1).sum()); TOPBAD[1] += len(pile)
mach.top_check = top_check
SHA, T5, WIN = mach.top(x16, ex, LS); del HW
sha_ok = sum(bytes(s) == bytes(r) for s, r in zip(SHA, REF["score_sha"][ROWS]))
t5_ok = int((T5 == REF["top5"][ROWS]).all(1).sum()); win_ok = int((WIN == REF["win"][ROWS]).sum())
log(f"EXACT (top table): piles = base2's product of the same light {TOPBAD[1] - TOPBAD[0]}/{TOPBAD[1]}; score rows (sha256) "
    f"{sha_ok}/{len(ROWS)}; top 5 (set and order) {t5_ok}/{len(ROWS)}; race winner {win_ok}/{len(ROWS)}")

# ---- the count ----
C = mach.counts(); n = len(ROWS)
np.savez(f"{OUT}/refs_{TAG}_counts.npz", rows=ROWS, **{f"{t}|{k}": v for t, d in C.items() for k, v in d.items()})
tot = {t: {k: int(v.sum()) for k, v in d.items()} for t, d in C.items()}
json.dump(dict(mode=MODE, texts=SEL, npos=int(n), totals=tot, diff=DIFF, same_top=bool(same_top),
               top=dict(piles_bad=TOPBAD[0], sha=sha_ok, top5=t5_ok, win=win_ok)), open(f"{OUT}/refs_{TAG}_counts.json", "w"), indent=0)
pt = lambda v: W2.per_token(v, n)
def kind(t): return t.split(".")[0]
log(f"\nWORK A TOKEN (mean over the {n} positions; photons). Gap meetings and the light: signed digits. Pours: ones of each running product.")
log(f"  {'table':7s} {'plain':>14s} {'signed digits':>14s} {'gap meetings':>14s} {'pours':>14s} {'writing':>10s} {'way 2':>14s}   "
    f"{'gaps':>7s} {'pours':>7s} {'way 2':>7s} {'signed':>7s} {'gaps, shuffled order':>20s}")
agg = {}
for t in W2.ALL:
    d = tot[t]; a = agg.setdefault(kind(t), dict.fromkeys(d, 0))
    for k, v in d.items(): a[k] += v
for k in ("qkv", "proj", "fc", "mproj", "head"):
    d = agg[k]; w2 = d["gaps"] + d["pours"] + d["writing"]
    log(f"  {k:7s} {pt(d['plain']):>14,} {pt(d['signed']):>14,} {pt(d['gaps']):>14,} {pt(d['pours']):>14,} {pt(d['writing']):>10,} {pt(w2):>14,}   "
        f"{W2.pct2(d['gaps'], d['plain']):>7s} {W2.pct2(d['pours'], d['plain']):>7s} {W2.pct2(w2, d['plain']):>7s} {W2.pct2(d['signed'], d['plain']):>7s} "
        f"{W2.pct2(d['gaps_shuffled'], d['plain']):>20s}")
bk = {k: sum(agg[b][k] for b in ("qkv", "proj", "fc", "mproj")) for k in agg["qkv"]}
w2b = bk["gaps"] + bk["pours"] + bk["writing"]; hd = agg["head"]; w2h = hd["gaps"] + hd["pours"] + hd["writing"]
log(f"  48 banks: plain {pt(bk['plain']):,}; signed digits {pt(bk['signed']):,} ({W2.pct2(bk['signed'], bk['plain'])}); way 2 {pt(w2b):,} "
    f"({W2.pct2(w2b, bk['plain'])}) = gap meetings {W2.pct2(bk['gaps'], bk['plain'])} + pours {W2.pct2(bk['pours'], bk['plain'])} + writing "
    f"{W2.pct2(bk['writing'], bk['plain'])}")
log(f"  per pair (line x neuron, light lit): plain {bk['plain'] / bk['pairs']:.3f}, signed {bk['signed'] / bk['pairs']:.3f}, gap meetings "
    f"{bk['gaps'] / bk['pairs']:.3f}, pours {bk['pours'] / bk['pairs']:.3f} (in single waves {bk['pours_single_wave'] / bk['pairs']:.3f}); "
    f"top table: plain {hd['plain'] / hd['pairs']:.3f}, gaps {hd['gaps'] / hd['pairs']:.3f}, pours {hd['pours'] / hd['pairs']:.3f}")
log(f"  side: pours if each running product were poured in single waves (its writing not counted): banks "
    f"{W2.pct2(bk['pours_single_wave'], bk['plain'])}, top table {W2.pct2(hd['pours_single_wave'], hd['plain'])}")
if MODE == "main":
    log(f"  the plain count rebuilt: 48 banks {pt(bk['plain']):,} (recording/C: {W2.PLAIN_BANKS_C:,}); top table {pt(hd['plain']):,} "
        f"(recording/C: {W2.PLAIN_HEAD_C:,}); signed digits {pt(bk['signed']):,} (recording/C: {W2.SIGNED_BANKS_C:,})")
    whole = pt(w2b + w2h) + W2.REST_UNCHANGED; whole_banks_only = pt(w2b) + W2.PLAIN_HEAD_C + W2.REST_UNCHANGED
    log(f"\nTHE WHOLE MACHINE (way 2 at the 48 banks and the top table; arrival, bends, water levels, divides, biases, stream adds "
        f"unchanged: {W2.REST_UNCHANGED:,} a token)")
    log(f"  plain {W2.PLAIN_TOTAL:,}; way 2 everywhere {whole:,} = {W2.pct2(whole, W2.PLAIN_TOTAL)} of plain; way 2 at the banks only "
        f"(top read plain) {whole_banks_only:,} = {W2.pct2(whole_banks_only, W2.PLAIN_TOTAL)}")
    F = W2.FRINGES; fb = F["qkv"] + F["proj"] + F["fc"] + F["mproj"]
    log(f"  base2_fringes.py (fringes_machine/report.log): {F['total']:,} = {W2.pct2(F['total'], W2.PLAIN_TOTAL)}; its 48 banks "
        f"{fb:,} ({W2.pct2(fb, W2.PLAIN_BANKS_C)} of the plain bank meetings, the knife's own work and layer 0's "
        f"stored reads in it); its top read {F['top']:,} ({W2.pct2(F['top'], W2.PLAIN_HEAD_C)} of the plain read's meetings)")
    log(f"  way 2's 48 banks against the fringes machine's: {pt(w2b):,} against {fb:,} ({W2.pct2(pt(w2b), fb)}); top table "
        f"{pt(w2h):,} against {F['top']:,} ({W2.pct2(pt(w2h), F['top'])})")
log(f"\nBY LAYER (way 2 as a share of plain): " + "  ".join(
    f"L{l} " + "/".join(W2.pct2(tot[f'{b}.{l}']['gaps'] + tot[f'{b}.{l}']['pours'] + tot[f'{b}.{l}']['writing'], tot[f'{b}.{l}']['plain'])
                        for b in ("qkv", "proj", "fc", "mproj")) for l in range(12)))
log(f"  (each layer: qkv/proj/fc/mproj)")
log(f"({time.time() - t0:.0f}s, max memory {W2.rss()} MB)")
