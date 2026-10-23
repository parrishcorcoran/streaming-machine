"""WHERE IT BLOCKS: a box of +-1 (one unit of the machine's own whole numbers) placed at one point of layer 11 (or the
stream into it), everything else exact, carried up step by step; the box's width is read after every step.

HIS WORDS (2026-10-08): "Start backward I think that's your problem. Start on the next token and work backwards".

THE SESSION'S RESTATEMENT (the session's words). req.py says which points can be known loosely and which must be exact.
This shows, for the points that must be exact, which step of the machine turns one unit of doubt into too much: the
water level (to16: the stream's places dropped, so a unit moves a line's light only near a rounding edge), the gauge
(LS, one number from every line; a range in it scales every pile of the bank), the plate (its piles gather every line's
doubt with no cancelling allowed in an exact bound: the doubt times the sum of |weights|), the bend (the share table),
the arrival (the race's start, each distance, the share read from the arrival table, the total), and the top read.
Widths are medians over positions and lines (each a whole number of that point's units); the gauge's range in units of
1/65536 of a place; branches = how many water levels the box allows.

    python3 backward/diag.py      -> backward/out/diag_refs.log
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward")
import core
from core import *
LOG = open(f"{OUT}/diag_refs.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
TRACE = []
def wrap(name, f, kind):
    def g(*a, **k):
        out = f(*a, **k)
        if kind == "box": TRACE.append((name, out[1] - out[0]))
        elif kind == "ls": TRACE.append((name, out[1] - out[0]))
        return out
    return g
core.ls_box = wrap("gauge LS range", core.ls_box, "ls")
for nm in ("i_gauged", "i_attention", "i_plain", "i_edge"):
    setattr(core, nm, wrap(nm, getattr(core, nm), "box"))
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
CASES = [(10, "Rout", None), (11, "QKV", "q"), (11, "QKV", "k"), (11, "QKV", "v"), (11, "ATT", None), (11, "A", None),
         (11, "H", None), (11, "E", None), (11, "Mw", None)]
agg = {c: {} for c in CASES}; proved = {c: [0, 0] for c in CASES}
for t in range(0, 30, 3):                                                  # every third text (10 texts, ~216 positions)
    rec = record(TEXTS[t], keep_from=10); rows = np.arange(rec["P"])
    for c in CASES:
        l, pt, part = c; v = rec["L"][l][pt][rows]; w = np.ones(v.shape, np.int64)
        if part: m = np.zeros(2304, bool); m[{"q": 0, "k": 768, "v": 1536}[part]:{"q": 768, "k": 1536, "v": 2304}[part]] = True; w = w * m
        TRACE.clear(); lo, hi, bl = push(rec, rows, l, pt, v - w, v + w)
        names = {"i_gauged": "gauged bank out", "i_attention": "arrival out", "i_plain": "plain bank out", "i_edge": "bend out"}
        seen = {}
        for nm, wd in TRACE:
            lab = names.get(nm, nm); seen[lab] = seen.get(lab, 0) + 1
            if nm == "gauge LS range" and wd.ndim == 1:
                val = float(np.median(wd))
            else:
                val = float(np.median(wd[~bl])) if (~bl).any() else np.nan
            agg[c].setdefault((lab, seen[lab]), []).append(val)
        br = top_branches(lo[~bl], hi[~bl]) if (~bl).any() else None
        if br is not None:
            cr = certify(br); proved[c][0] += int(cr["cert5"].sum())
            agg[c].setdefault(("final light (x16) width", 1), []).append(float(np.median(br[0]["xhi"] - br[0]["xlo"])))
            agg[c].setdefault(("final gauge LS range", 1), []).append(float(np.median(br[0]["LShi"] - br[0]["LSlo"])))
            agg[c].setdefault(("blown (positions)", 1), []).append(float(bl.sum()))
        proved[c][1] += len(rows)
    log(f"  text {t} ({time.time() - t0:.0f}s)")
for c in CASES:
    l, pt, part = c
    log(f"\n+-1 at layer {l} {pt}{':' + part if part else ''}: top 5 proved {proved[c][0]}/{proved[c][1]}")
    for (lab, n), vals in agg[c].items():
        log(f"   {lab:26s} #{n}: median width {np.nanmedian(vals):16,.1f}")
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
