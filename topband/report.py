"""THE BAND-BY-BAND TOP READ: THE REPORT, FROM THE SAVED RUNS ONLY (no machine run here).
  topband/top_refs.npz                     the 648 refs positions (the plain read's basis: 904,017,313 a position)
  topband/top_windows_p1of2, p2of2.npz     the 6,000 judged positions
  topband/top_refs_shuffled.npz            CONTROL: the light shuffled across its lines (fringes' own draw)
  topband/top_refs_order7.npz              ORDER-FREE: lines in a random order, walks down, words relabelled
  topband/judge.log, check.log             judge5's own judge; the session's recount
Against: the collapsing read (fringes_machine/top_refs.json, top_windows_p*of2.json, top_refs_shuffled.json) and the
whole fringes machine (fringes_machine/report.log: 1,745,178,133 of 3,666,596,259, its top read 191,031,613).
    python3 topband/report.py   -> topband/report.log
"""
import json, glob, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/topband"; FM = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/report.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def pct2(a, b): return f"{(10000 * a + b // 2) // b / 100:.2f}%" if b else "-"
def per(v, n): return (int(v) + n // 2) // n
NB = 8; NT = 7; BANDS = ["29-26", "25-22", "21-18", "17-14", "13-10", "9-6", "5-2", "1-0"]
PLAIN_TOTAL, FR_TOTAL, FR_TOP = 3666596259, 1745178133, 191031613
def load(paths):
    Z = [np.load(p) for p in paths]; keys = [k for k in Z[0].files if k not in ("bad_keys", "bad")]
    d = {k: np.concatenate([z[k] for z in Z]) for k in keys}
    bad = {}
    for z in Z:
        for k, v in zip(z["bad_keys"], z["bad"]): bad[str(k)] = bad.get(str(k), 0) + int(v)
    d["bad"] = bad; return d
def tot(d): return (d["walk"] + d["writing"] + d["standing_written"] + d["standing_poured"] + d["pours"].sum(1)
                    + d["bound"].sum(1) + d["end"])
def plain(d): return d["plain_meet"] + d["plain_div"] + d["plain_stand"]
def fr(jsons):
    f = p = n = 0
    for j in jsons:
        z = json.load(open(j)); c = z["counts"]; n += z["n"]
        f += sum(v for k, v in c.items() if k.startswith("fr|")); p += sum(v for k, v in c.items() if k.startswith("plain|"))
    return f, p, n

R = load([f"{OUT}/top_refs.npz"]); W = load(sorted(glob.glob(f"{OUT}/top_windows_p*of2.npz")))
A = {k: np.concatenate([R[k], W[k]]) for k in R if k != "bad"}
SH = load([f"{OUT}/top_refs_shuffled.npz"]) if os.path.exists(f"{OUT}/top_refs_shuffled.npz") else None
OR = load([f"{OUT}/top_refs_order7.npz"]) if os.path.exists(f"{OUT}/top_refs_order7.npz") else None
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); ROWS = np.load(f"{FM}/refs_top_light.npz")["rows"]
J5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5)

log("THE TOP TABLE READ BAND BY BAND (way 2's walk once; each exposure poured from its top places down; words set aside")
log("after every band by the collapsing read's rule at the pile's own scale, one shared bound: lit lines x (2^p - 1))")
log("\nEXACT")
log(f"  refs (648): top 5 (set and order) = the plain read {int((R['top5'] == R['ptop5']).all(1).sum())}/648; = base2.py's saved run "
    f"{int((R['top5'] == REF['top5'][ROWS]).all(1).sum())}/648; race winner = saved {int((R['win'] == REF['win'][ROWS]).sum())}/648")
log(f"  judged (6,000): top 5 (set and order) = the plain read {int((W['top5'] == W['ptop5']).all(1).sum())}/6000; = judge5_ref "
    f"{int((W['top5'] == J5).all(1).sum())}/6000; race winner = the plain read's {int((W['win'] == W['pwin']).sum())}/6000")
for f in ("judge.log", "check.log"):
    if os.path.exists(f"{OUT}/{f}"): log(f"  {f}: " + open(f"{OUT}/{f}").read().strip().splitlines()[-1])
for name, d in (("refs", R), ("judged", W)) + ((("shuffled light", SH),) if SH else ()) + ((("order-free", OR),) if OR else ()):
    log(f"  checks, {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(d["bad"].items())))

def table(d, label):
    n = len(d["walk"]); pl = int(plain(d).sum()); t = int(tot(d).sum())
    log(f"\nBY BAND, {label} ({n} positions; photons a position; shares of the plain read a position, {per(pl, n):,})")
    log(f"  {'band':6s} {'places':>6s}  {'words still in after its test: median   mean  least   most':>58s}  {'pour photons':>13s} {'share':>7s}"
        f"  {'bound photons':>13s} {'compares':>9s}  {'its photons over every word':>27s}")
    log(f"  {'walk':6s} {'':6s}  {'(once a line, before band 0; every word)':>58s}  {per(d['walk'].sum(), n):>13,} {pct2(int(d['walk'].sum()), pl):>7s}")
    for b in range(NB):
        if b < NT:
            sv = d["surv"][:, b]; s = f"{int(np.median(sv)):>6d} {sv.mean():>9.1f} {sv.min():>6d} {sv.max():>6d}"
            bd = f"{per(d['bound'][:, b].sum(), n):>13,} {per(d['cmp'][:, b].sum(), n):>9,}"
        else: s = f"{'(read to the end)':>31s}"; bd = f"{'':>13s} {'':>9s}"
        log(f"  {b:<6d} {BANDS[b]:>6s}  {s:>58s}  {per(d['pours'][:, b].sum(), n):>13,} {pct2(int(d['pours'][:, b].sum()), pl):>7s}"
            f"  {bd}  {per(d['hall'][:, b].sum(), n):>13,} ({pct2(int(d['hall'][:, b].sum()), int(d['hall'].sum()))})")
    rest = {k: int(d[k].sum()) for k in ("writing", "standing_written", "standing_poured", "end")}
    log(f"  the light's writing {per(rest['writing'], n):,}; standing amounts written {per(rest['standing_written'], n):,} and poured "
        f"{per(rest['standing_poured'], n):,}; words read to the end {per(rest['end'], n):,}")
    log(f"  TOTAL {per(t, n):,} a position = {pct2(t, pl)} of the plain read (walk {pct2(int(d['walk'].sum()), pl)}, pours "
        f"{pct2(int(d['pours'].sum()), pl)}, standing amounts {pct2(rest['standing_written'] + rest['standing_poured'], pl)}, bound "
        f"{pct2(int(d['bound'].sum()), pl)}, the rest {pct2(rest['writing'] + rest['end'], pl)}); compares (a count) {per(d['cmp'].sum(), n):,}")
    log(f"  every band into every word (way 2's pours) {per(d['hall'].sum(), n):,} = {pct2(int(d['hall'].sum()), pl)} of the plain read, "
        f"{pct2(int(d['hall'].sum()), int(d['plain_meet'].sum()))} of its meetings; poured here {pct2(int(d['pours'].sum()), int(d['hall'].sum()))} of them")
    return t, pl, n

tR, pR, nR = table(R, "the 648 refs positions")
tW, pW, nW = table(W, "the 6,000 judged positions")
tA, pA, nA = table(A, "all 6,648 read positions")

log("\nAGAINST THE PLAIN READ AND THE COLLAPSING READ")
fR, fpR, _ = fr([f"{FM}/top_refs.json"]); fW, fpW, _ = fr(sorted(glob.glob(f"{FM}/top_windows_p*of2.json")))
log(f"  refs: plain read {per(pR, nR):,} (base2_fringes.py's basis 904,017,313: same {per(pR, nR) == 904017313}); "
    f"band read {per(tR, nR):,} = {pct2(tR, pR)}; collapsing read {per(fR, nR):,} = {pct2(fR, fpR)}; "
    f"the band read is {pct2(tR, fR)} of the collapsing read")
log(f"  judged: plain read {per(pW, nW):,}; band read {per(tW, nW):,} = {pct2(tW, pW)}; collapsing read {per(fW, nW):,} = "
    f"{pct2(fW, fpW)}; the band read is {pct2(tW, fW)} of the collapsing read")
log(f"  all 6,648: band read {pct2(tA, pA)} of the plain read")

log("\nTHE WHOLE MACHINE (base2_fringes.py, with this read in place of its top read; every other part unchanged)")
new = FR_TOTAL - FR_TOP + per(tR, nR)
log(f"  {FR_TOTAL:,} - {FR_TOP:,} + {per(tR, nR):,} = {new:,} a token = {pct2(new, PLAIN_TOTAL)} of plain ({PLAIN_TOTAL:,}); "
    f"it was {pct2(FR_TOTAL, PLAIN_TOTAL)}")
log(f"  the top read is now {pct2(per(tR, nR), new)} of the whole machine (it was {pct2(FR_TOP, FR_TOTAL)})")

if SH:
    tS, pS, nS = table(SH, "CONTROL, the light shuffled across its lines (refs)")
    fS, fpS, _ = fr([f"{FM}/top_refs_shuffled.json"])
    same_hall = pct2(int(SH['hall'][:, :4].sum()), int(SH['hall'].sum()))
    log(f"  shuffled {pct2(tS, pS)} against real {pct2(tR, pR)}; the collapsing read on the same shuffled light {pct2(fS, fpS)} "
        f"(real {pct2(fR, fpR)}); places 14 and up hold {same_hall} of the shuffled light's exposure photons "
        f"(real {pct2(int(R['hall'][:, :4].sum()), int(R['hall'].sum()))})")
    log(f"  words still in after band 3, median: shuffled {int(np.median(SH['surv'][:, 3]))}, real {int(np.median(R['surv'][:, 3]))}; "
        f"after band 2, positions with any word set aside: shuffled {int((SH['surv'][:, 2] < 50257).sum())}, real {int((R['surv'][:, 2] < 50257).sum())}")
if OR:
    keys = ["n", "mu", "walk", "writing", "standing_written", "standing_poured", "end", "plain_meet", "plain_div", "plain_stand",
            "hall", "pours", "surv", "bound", "cmp"]
    eq = {k: int((np.asarray(OR[k]) == np.asarray(R[k])).reshape(len(R[k]), -1).all(1).sum()) for k in keys}
    log(f"\nORDER-FREE (lines in a random order at every batch, every walk DOWN from the top weight, the words relabelled; refs)")
    log("  positions with every count equal to the main run's: " + ", ".join(f"{k} {v}/648" for k, v in eq.items()))
    log(f"  top 5 = the main run's: set and order {int((OR['top5'] == R['top5']).all(1).sum())}/648, as a set "
        f"{int((np.sort(OR['top5'], 1) == np.sort(R['top5'], 1)).all(1).sum())}/648; race winner {int((OR['win'] == R['win']).sum())}/648; "
        f"its own top 5 = its own plain read {int((OR['top5'] == OR['ptop5']).all(1).sum())}/648")
    dif = np.flatnonzero(~(OR['top5'] == R['top5']).all(1))
    for q in dif: log(f"    position {q}: main {R['top5'][q].tolist()}, order-free {OR['top5'][q].tolist()}")
