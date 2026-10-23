"""ADDITION 4's RULE ON EVERY PLATE (no light): the shared line order (each line's sum over the plate's neurons, largest
first, ties by line number), the gaps in that order, and the rule fixed before any run: the plate takes the shared order
only if its gaps hold fewer lit digits (signed digits) than its weights, sum naf(g) < sum naf(w).
Also, for the record of the plate's own numbers: the gaps' plain 1s against the weights' plain 1s, and each neuron's OWN
order's gaps (the line-up of section 18) in lit digits, so the shared order can be seen against both.
PREDICTION (base2_fringes2.py's docstring): no plate passes; sum naf(g) is 1.1-1.3 x sum naf(w) on every plate.
    python3 fringes_machine2/shared_rule.py   -> fringes_machine2/shared_rule.log, shared_rule.json
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
os.environ["FR2_ADD"] = "4"
OUT = f"{HERE}/fringes_machine2"; LOG = open(f"{OUT}/shared_rule.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import base2_fringes2 as FM
FM.SHARED_GUARD = False                                     # here the rule is only read, never used
res = {}; TG = TW = 0
log("ADDITION 4: the shared line order's rule, plate by plate (lit digits in signed digits; plain 1s for the record)")
log(f"  {'plate':9s} {'gaps, lit':>12s} {'weights, lit':>13s} {'ratio':>7s} {'takes it':>9s} {'gaps, 1s':>11s} {'weights, 1s':>12s} {'own order gaps, lit':>20s}")
for l in range(12):
    for b in ("qkv", "proj", "fc", "mproj"):
        name = f"{b}.{l}"; pl = FM.Plate(name)
        w = pl.w.astype(np.int64); g = pl.sh_g.astype(np.int64)
        ws = -np.sort(-w, axis=0); og = ws.copy(); og[:-1] -= ws[1:]                        # each neuron's own order
        r = dict(gaps_lit=pl.sh_lit_g, w_lit=pl.sh_lit_w, takes=bool(pl.shared), gaps_ones=int(FM.ones(g).sum()),
                 w_ones=int(FM.ones(w).sum()), own_gaps_lit=int(FM.naf(og).sum()))
        res[name] = r; TG += r["gaps_lit"]; TW += r["w_lit"]
        log(f"  {name:9s} {r['gaps_lit']:>12,} {r['w_lit']:>13,} {r['gaps_lit'] / r['w_lit']:>7.3f} {str(r['takes']):>9s} "
            f"{r['gaps_ones']:>11,} {r['w_ones']:>12,} {r['own_gaps_lit']:>20,}")
        del pl
log(f"  all 48 plates: gaps {TG:,} against weights {TW:,} lit digits = {TG / TW:.3f}; plates the rule gives the shared order: "
    f"{sum(r['takes'] for r in res.values())} of 48 ({time.time() - t0:.0f}s, max memory {FM.rss()} MB)")
json.dump(res, open(f"{OUT}/shared_rule.json", "w"), indent=0)
