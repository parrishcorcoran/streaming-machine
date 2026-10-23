"""WHY A SMALL RELIEF MOVES A TALL LEVEL: the arrival row's scores at the qkv plate's own layer. 2026-10-06.

HIS SENTENCE (his exact words): "It should be 100% removable after training."
SESSION'S RESTATEMENT (not his): what survives of the random start in plate W is round(s x W0) (../foundation_share.json).

WHAT plate_look.py FOUND (plate_look_160m.log): removing the surviving foundation from qkv.l alone moves layer l's
level in 42, 54, 61% of rows at l = 9, 10, 11, and a scrambled copy does the same (37-45, 53-66, 57-67%), although
qkv.9-11's surviving start is only 3.3-4.2% of the plate's size (rho), and although in the finished machine the next
score stands far below the level there (median 6.1, 15.2, 16.7 e-folds; in the rows that moved, 4.0, 6.6, 12.9).
The other plates (proj, fc, mproj) move a level only where the next score stands within about 0.01-2 e-folds of it.

WHAT IS RUN: the same base-2 machine (plate_look.py's copy of pythia_base2.py, imported from nothing: copied below in
the one function that changes, the arrival, which here also hands back its scores; nothing else of the machine
changes), on the same 30 texts with the finished machine's words fed in, no greedy run. At layer l only (the plate's own
layer, where only qkv.l has changed, so the stream coming in is the finished machine's, exactly):
  - the row's span in the finished machine: the level minus the lowest live score (e-folds), the height of the row
  - the ripple: each position's score change (case minus finished), its spread over the row (median absolute size)
  - in the rows that moved: the old winner's score change and the new winner's score change (e-folds)
Cases: fin; b:qkv.l and cb:qkv.l@{20261006,2,3} for l = 9, 10, 11 at 160m (4, 5 at 70m).

PREDICTIONS, written before the first run:
  (1) the rows are tall at the late layers: median span at layer 11 (160m) 30 e-folds or more, and at least 1.5 times
      the span at layer 9.
  (0) a check, not a prediction: the share of rows moved here equals plate_look.py's for the same case exactly.
  (2) the ripple a 3-4% relief makes in those rows is several e-folds: median absolute score change 1-10 e-folds at
      layers 9-11, about 3-6% of the row's span (the relief's size, rho, carried through the score).
  (3) in the rows that moved, the old winner (mostly the first word) drops: its median change is below 0 and larger in
      size than the new winner's change. The scrambled copies show the same within their spread.
  (4) real against scramble: b's ripple within the three draws' spread at every layer read.

  python3 level_height.py size=160m
"""
import os, sys, json, time, zlib, resource, runpy
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "160m")
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE)
SEEDS = [20261006, 2, 3]
LAYERS = {"160m": [9, 10, 11], "70m": [4, 5]}[SIZE]
LOGF = open(os.path.join(LATE, f"level_height_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

# load plate_look.py's machine and plate loader without running its cases: run it with cases set to fin only and
# texts=0 would still open its log, so its functions are taken by executing the file up to its runs.
src = open(os.path.join(LATE, "plate_look.py")).read()
cut = src.index("# ---------------------------------------------------------------- the runs")
G = {"__file__": os.path.join(LATE, "plate_look.py"), "__name__": "plate_look_parts"}
sys_argv = sys.argv; sys.argv = ["plate_look.py", f"size={SIZE}", "tag=level_height_parts"]
exec(compile(src[:cut], "plate_look.py", "exec"), G)
sys.argv = sys_argv
os.remove(os.path.join(LATE, f"plate_look_{SIZE}-level_height_parts.log"))   # the parts' empty log, not a result
M, L, NH, HS, d = G["M"], G["L"], G["NH"], G["HS"], G["d"]
ARR, SC, HALF = G["ARR"], G["SC"], G["HALF"]
to16, product, power, turn, plain, gauge, rshift, log2fix = (G[k] for k in
    ("to16", "product", "power", "turn", "plain", "gauge", "rshift", "log2fix"))

def attention_scores(QKV):
    """the machine's arrival exactly as in plate_look.py, also handing back each head's scores (a look)"""
    P = len(QKV); pos = np.arange(P); out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
    S = np.zeros((NH, P, P), np.int64); win = np.zeros((NH, P), np.int64)
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        q16, eq = to16(q); k16, ek = to16(k)
        sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - HALF)
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
        win[h] = np.where(live & (sc == lev[:, None]), pos[None, :], P).min(1)
        S[h] = sc
        dd = lev[:, None] - sc
        a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
        out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
    return out, S, win

def scores_at(ids, lay):
    """run the machine to layer lay and hand back that layer's scores and winners"""
    ids = np.asarray(ids); EW, EE = G["PL"]["embed"]
    R = rshift(EW[ids].astype(np.int64), -(EE[ids][:, None] + G["ES"]))
    for l in range(lay + 1):
        At, S, win = attention_scores(plain(gauge(R, f"ln1.{l}"), f"qkv.{l}"))
        if l == lay: return S, win
        At = plain(At, f"proj.{l}")
        Mo = plain(G["edge"](plain(gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        R = R + At + Mo

refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:30]
seqs = [r["ids"] + r["ref"][:15] for r in refs]
log(f"level_height.py  size {SIZE}  layers {LAYERS}  (every score at 2^-10 of an e-fold; shown in e-folds)")
G["set_case"]("fin")
FIN = {lay: [scores_at(s, lay) for s in seqs] for lay in LAYERS}
RES = {}
for lay in LAYERS:
    span = []
    for (S, w) in FIN[lay]:
        P = S.shape[1]
        for h in range(NH):
            for i in range(1, P): span.append((S[h, i, :i + 1].max() - S[h, i, :i + 1].min()) / 1024)
    RES[f"fin.L{lay}"] = dict(span_median=float(np.median(span)), span_q90=float(np.quantile(span, 0.9)))
    log(f"  layer {lay}: finished row span (level minus lowest live score), median {np.median(span):.2f} e-folds, "
        f"90th {np.quantile(span, 0.9):.2f}")
    for case in [f"b:qkv.{lay}"] + [f"cb:qkv.{lay}@{s}" for s in SEEDS]:
        G["set_case"](case)
        rip, rel, dold, dnew, nmv, nall, old0 = [], [], [], [], 0, 0, 0
        for (Sf, wf), s in zip(FIN[lay], seqs):
            Sc, wc = scores_at(s, lay); P = Sf.shape[1]
            for h in range(NH):
                for i in range(1, P):
                    dlt = (Sc[h, i, :i + 1] - Sf[h, i, :i + 1]) / 1024; sp = (Sf[h, i, :i + 1].max() - Sf[h, i, :i + 1].min()) / 1024
                    rip.append(np.median(np.abs(dlt))); rel.append(np.median(np.abs(dlt)) / sp if sp > 0 else 0.0)
                    nall += 1
                    if wc[h, i] != wf[h, i]:
                        nmv += 1; dold.append(dlt[wf[h, i]]); dnew.append(dlt[wc[h, i]]); old0 += wf[h, i] == 0
        o = dict(moved=nmv / nall, ripple_median=float(np.median(rip)), ripple_over_span=float(np.median(rel)),
                 old_winner_change_median=float(np.median(dold)) if dold else None,
                 new_winner_change_median=float(np.median(dnew)) if dnew else None,
                 old_winner_dropped=float(np.mean(np.array(dold) < 0)) if dold else None,
                 old_was_first_word=old0 / max(nmv, 1),
                 old_abs_median=float(np.median(np.abs(dold))) if dold else None,
                 new_abs_median=float(np.median(np.abs(dnew))) if dnew else None)
        RES[f"{case}"] = o
        log(f"    {case:18s} moved {100*o['moved']:5.1f}%  ripple median {o['ripple_median']:6.2f} e-folds "
            f"({100*o['ripple_over_span']:4.1f}% of the row's span)  | moved rows: old winner change median "
            f"{o['old_winner_change_median']:+7.2f} (|.| {o['old_abs_median']:6.2f}, dropped in {100*o['old_winner_dropped']:4.1f}%), "
            f"new winner {o['new_winner_change_median']:+7.2f} (|.| {o['new_abs_median']:6.2f}); old winner was the first word in "
            f"{100*o['old_was_first_word']:4.1f}%")
json.dump(RES, open(os.path.join(LATE, f"level_height_{SIZE}.json"), "w"))
log(f"done  peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
