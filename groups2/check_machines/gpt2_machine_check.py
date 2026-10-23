"""CHECKER: base2_groups.py AS DELIVERED, run on the checker's 5 drawn refs texts [5, 12, 19, 20, 28] (two forward calls:
texts 5, 12 with the work counted; texts 19, 20, 28 with the work counted), every kept array compared with base2.py's own
steps run fresh (gpt2_ref.npz); the light at the top compared; then the top read (gtop.run_top, topg (ii) per place,
g = 16, as base2_groups' phase_top runs it) on the machine's light, its top 5 and race winner against base2's fresh top
read. Nothing of base2_groups' own output files is written; everything goes to check_machines/.
    python3 gpt2_machine_check.py -> gpt2_machine_check.json / .log
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/groups2")
argv = sys.argv; sys.argv = ["base2_groups.py"]
import base2_groups as BG
sys.argv = argv
FM = BG.FM
LOGF = open(f"{OUT}/gpt2_machine_check.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
REF = np.load(f"{OUT}/gpt2_ref.npz"); TS = [int(t) for t in REF["texts"]]
SAVED = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = SAVED["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
t0 = time.time(); out = dict(texts=TS, runs={})
for sel in ([5, 12], [19, 20, 28]):
    assert set(sel) <= set(TS)
    texts = [FM.TEXTS[t] for t in sel]
    # base2.py's fresh arrays, texts concatenated in the same order
    def ref(key):
        return np.concatenate([REF[f"t{t}_{key}"] for t in sel])
    # the saved record too (what the builder checked against), only as a second look
    rows_saved = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in sel])
    DIFF, N, SAVED_DIFF = {}, {}, {}
    def check(key, value, rows=None):
        k, l = key.rsplit(" ", 1); r = ref(f"{k.replace(' ', '_')}_{l}")
        if rows is not None: r = r[rows]
        v = np.asarray(value); assert v.shape == r.shape, (key, v.shape, r.shape)
        bad = (v != r) if v.ndim == 1 else (v != r).any(1); DIFF[key] = int(bad.sum()); N[key] = len(r)
        sk = f"{k.replace(' ', '_')}_{l}"
        if sk in SAVED.files:
            s = SAVED[sk].astype(np.int64)[rows_saved]
            if rows is not None: s = s[rows]
            SAVED_DIFF[key] = int(((s != r) if r.ndim == 1 else (s != r).any(1)).sum())
    BAD = {}
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    C = FM.Count(); MZ = {}
    (x16, ex, LS), STOPS = BG.forward(texts, C, check, chk, log=lambda s: None, MZ=MZ)
    top_same = bool(np.array_equal(x16, ref("top_x16")) and np.array_equal(ex, ref("top_ex")) and np.array_equal(LS, ref("top_LS")))
    log(f"texts {sel}: {len(DIFF)} arrays compared with base2.py's fresh run, {sum(N.values())} rows; rows differing {sum(DIFF.values())}"
        f" {[k for k in DIFF if DIFF[k]]}; light at the top identical {top_same}")
    log(f"   kinds checked: {sorted({k.rsplit(' ', 1)[0] for k in DIFF})}")
    log(f"   base2's fresh run against the saved record recording/C_ref/plain_refs.npz: rows differing {sum(SAVED_DIFF.values())} of {len(SAVED_DIFF)} arrays")
    log(f"   the machine's own check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())) + f"; fallbacks {FM.FALLBACK}")
    out["runs"]["-".join(map(str, sel))] = dict(diff=DIFF, bad=BAD, top_same=top_same, saved_diff=SAVED_DIFF, counts=dict(C),
                                               npos=int(sum(len(t) for t in texts)), x16=None)
    np.savez(f"{OUT}/gpt2_machine_top_light_{'-'.join(map(str, sel))}.npz", x16=x16, ex=ex, LS=LS)
    log(f"   ({time.time() - t0:.0f}s, {BG.rss()} MB)")
# ---- the top read on the machine's light, all 5 texts ----
Z = [np.load(f"{OUT}/gpt2_machine_top_light_{s}.npz") for s in ("5-12", "19-20-28")]
X = np.concatenate([z["x16"] for z in Z]).astype(np.int64); EX = np.concatenate([z["ex"] for z in Z]); LSa = np.concatenate([z["LS"] for z in Z])
order = [5, 12, 19, 20, 28]
R5 = np.concatenate([REF[f"t{t}_top5"] for t in order]); RW = np.concatenate([REF[f"t{t}_win"] for t in order]); S6 = np.concatenate([REF[f"t{t}_s5"] for t in order])
TG = BG.top_module(False)
rec, bad, pwrite, lwrite, ppmask, lab = BG.run_top(TG, X, EX, LSa, False, lambda s: None)
r = rec["pp"]; n = len(X)
log(f"top read on the machine's light ({n} positions): checks " + ", ".join(f"{k} {v}" for k, v in sorted(bad["pp"].items())))
same5 = (r["ii_top5"] == R5).all(1); log(f"   (ii) top 5 = base2's fresh top 5 (set and order) {int(same5.sum())}/{n}; race winner = base2's fresh race winner {int((r['ii_win'] == RW).sum())}/{n}")
log(f"   topg's own plain read top 5 = base2's fresh {int((rec['ptop5'] == R5).all(1).sum())}/{n}, race winner {int((rec['pwin'] == RW).sum())}/{n}")
for q in np.flatnonzero(~same5):
    log(f"   position {q}: (ii) {r['ii_top5'][q].tolist()}, base2 {R5[q].tolist()}, base2's six best scores {S6[q].tolist()} (tie for 5th: {S6[q][4] == S6[q][5]})")
pl = rec["plain_meet"] + rec["plain_div"] + rec["plain_stand"]
out["top"] = dict(n=n, plain=pl.tolist(), ii_total=r["ii_total"].tolist(), pwrite=pwrite.tolist(), lwrite=lwrite.tolist(), table=r["table"].tolist(),
                  ii_reads=r["ii_reads"].tolist(), ii_light=r["ii_light"].tolist(), ii_bound=r["ii_bound"].tolist(), ii_end=r["ii_end"].tolist(),
                  ii_stand=r["ii_stand"].tolist(), ii_surv=r["ii_surv"].tolist(), bad=bad, ppmask=int(ppmask),
                  same5=int(same5.sum()), samewin=int((r['ii_win'] == RW).sum()))
log(f"   plain read {pl.sum() / n:,.0f} a position; (ii) {r['ii_total'].sum() / n:,.0f}; + ready piles written {pwrite.sum() / n:,.0f} + light {lwrite.sum() / n:,.0f}"
    f" = {100 * (r['ii_total'] + pwrite + lwrite).sum() / pl.sum():.2f}% of the plain read")
json.dump(out, open(f"{OUT}/gpt2_machine_check.json", "w"))
log(f"done ({time.time() - t0:.0f}s, max memory {BG.rss()} MB)")
