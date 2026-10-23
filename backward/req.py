"""STEPS 2-3, WORKING BACKWARD: how loosely each value of each layer may be known and the top 5 still be proved.

HIS WORDS (2026-10-08): "Start backward I think that's your problem. Start on the next token and work backwards".

THE SESSION'S RESTATEMENT (the session's words). Step 1 (top.py) says what the top read needs of the final light. One
stage further back, the value that feeds it is known only as a box: every line within +-2^k of its true value (k = -1:
known exactly). Everything else of that position is exact, and every earlier position's keys and values are exact.
The box is carried up through the rest of the machine by exact whole-number interval steps (core.push): every water
level the box allows as its own branch, the gauge's range, the plates (2 x pile_lo = (lo + hi) . w - (hi - lo) . |w|),
the bend's exact range, the arrival's race, distances, shares and total. At the top, core.certify asks whether the top
5 (set and order) is the same for every value the boxes allow. The requirement at a point is the coarsest k proved
(proved at k means proved at every smaller k: the boxes are nested and every step keeps nesting).

POINTS, per layer l (the machine's own values at each point, in its own whole numbers):
  Rout  the stream after layer l (for l = 11 the final stream; for l < 11 the stream into layer l+1)
  Mw    the gate write (mproj's output joining the stream)      E    the bend's outputs (mproj's light, 3,072)
  H     the gauged fc value at the knife edge (3,072)            Rmid the stream after the arrival write
  A     the arrival write (proj's output joining the stream)     ATT  the arrival's output (proj's light)
  QKV   q|k|v of this position (2,304); QKV:q, QKV:k, QKV:v the box on that part alone
  Rin   the stream into layer l (= Rout of layer l-1)

PREDICTION (the session's, written before this ran): R_final +-2^4 to 2^9 at most positions (the final water level
drops about 9 places, and the top read needs the light to place 1); Mw the same as R_final (it adds to an exact
stream); E and H need under one unit at most positions (3,072 lines each carry their box through mproj's plate, which
adds them all with no cancelling allowed); Rmid, A and Rin of layer 11 need exact at nearly every position (the fc
gauge reads every line, a range in LS scales all 3,072 piles); QKV needs exact through the arrival; nothing below
layer 11 can be loose (blocked at layer 11's water level and gauge).

    python3 backward/req.py set=refs|windows layers=11,10,... [points=...] [win=a:b]
      -> backward/out/req_<set>[_win a-b].npz / .log
"""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward")
from core import *
ARG = dict(a.split("=") for a in sys.argv[1:]); SET = ARG.get("set", "refs")
LAYERS = [int(v) for v in ARG.get("layers", "11").split(",")]
PTS = ARG.get("points", "Rout,Mw,E,H,Rmid,A,ATT,QKV,QKV:q,QKV:k,QKV:v").split(",")
KMAX = 31; CB = int(ARG.get("batch", 64))
WIN = ARG.get("win", None)
TAG = f"req_{SET}" + ("" if WIN is None else f"_win{WIN.replace(':', '-')}") + f"_L{'-'.join(map(str, LAYERS))}" + \
      ("" if "points" not in ARG else "_" + ARG["points"].replace(":", "").replace(",", "-"))
LOG = open(f"{OUT}/{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if SET == "refs":
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    JUDGE = [np.arange(len(t)) for t in TEXTS]
    T5 = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")["top5"]
else:
    WINS = json.load(open(f"{HERE}/judge5_texts.json")); a, b = (int(v) for v in (WIN or "0:300").split(":"))
    TEXTS = [w["ids"] for w in WINS[a:b]]; JUDGE = [np.arange(3, 23) for _ in TEXTS]
    T5 = np.load(f"{HERE}/judge5_ref.npz")["top5"][a:b].reshape(-1, 5)
NPOS = sum(len(j) for j in JUDGE)
log(f"WORKING BACKWARD: {SET} {'' if WIN is None else WIN} ({len(TEXTS)} texts, {NPOS} positions); layers {LAYERS}; points {PTS}")

def box_at(rec, l, pt, rows, k):
    """the true value at (l, pt) for these rows, +-2^k (k = -1: exact); QKV:x boxes that part only"""
    key = pt.split(":")[0]; v = rec["L"][l][key][rows]; w = np.where(k >= 0, np.left_shift(np.int64(1), np.maximum(k, 0)), 0)[:, None]
    if ":" in pt:
        part = {"q": slice(0, 768), "k": slice(768, 1536), "v": slice(1536, 2304)}[pt.split(":")[1]]
        m = np.zeros(v.shape[1], bool); m[part] = True; w = np.where(m[None, :], w, 0)
    return key, v - w, v + w

def proved_rows(recs, items, l, pt):
    """items: list of (text index, position, k). Returns (top 5 proved, winner proved, the five words) per item"""
    n = len(items); Rlo = np.zeros((n, 768), np.int64); Rhi = np.zeros((n, 768), np.int64); blown = np.zeros(n, bool)
    it = np.array(items, np.int64)
    for t in np.unique(it[:, 0]):
        sel = np.flatnonzero(it[:, 0] == t); rows = it[sel, 1]; k = it[sel, 2]
        key, lo, hi = box_at(recs[t], l, pt, rows, k)
        Rlo[sel], Rhi[sel], blown[sel] = push(recs[t], rows, l, key, lo, hi)
    p5 = np.zeros(n, bool); pw = np.zeros(n, bool); cand = np.zeros((n, 5), np.int64)
    ok = ~blown
    idx = np.flatnonzero(ok)
    for b0 in range(0, len(idx), CB):
        ii = idx[b0:b0 + CB]; br = top_branches(Rlo[ii], Rhi[ii]); c = certify(br)
        p5[ii] = c["cert5"]; pw[ii] = c["certwin"]; cand[ii] = c["cand"]
    return p5, pw, cand, blown

def requirement(recs, l, pt, gidx):
    """per judged position: the coarsest k proved (-1 = exact needed), by halving each position's range"""
    n = len(gidx); lo = np.full(n, -1, np.int64); hi = np.full(n, KMAX + 1, np.int64); wrong = 0; nblown = 0
    # first k = 0: a position not proved at +-1 needs the exact value
    while (hi - lo > 1).any():
        act = np.flatnonzero(hi - lo > 1)
        mid = np.where(lo[act] < 0, 0, (lo[act] + hi[act]) // 2)
        items = [(gidx[i][0], gidx[i][1], int(m)) for i, m in zip(act, mid)]
        p5, pw, cand, bl = proved_rows(recs, items, l, pt); nblown += int(bl.sum())
        tt = np.array([gidx[i][2] for i in act]); wrong += int((p5 & ~(cand == T5[tt]).all(1)).sum())
        lo[act[p5]] = mid[p5]; hi[act[~p5]] = mid[~p5]
    return lo, wrong, nblown

res = {}
CHUNK = 10
for c0 in range(0, len(TEXTS), CHUNK):
    tl = time.time()
    recs = [record(TEXTS[t], keep_from=min(LAYERS), keys=KEEP_SMALL) for t in range(c0, min(len(TEXTS), c0 + CHUNK))]
    for t, r in enumerate(recs):                                               # the record must be the machine's own
        assert np.array_equal(r["top"]["top5"][JUDGE[c0 + t]], T5[sum(len(j) for j in JUDGE[:c0 + t]):sum(len(j) for j in JUDGE[:c0 + t + 1])])
    gidx = []; g0 = sum(len(j) for j in JUDGE[:c0])
    for t in range(len(recs)):
        for p in JUDGE[c0 + t]: gidx.append((t, int(p), g0)); g0 += 1
    for l in LAYERS:
        for pt in PTS:
            k, wrong, nb = requirement(recs, l, pt, gidx)
            res.setdefault(f"{l}:{pt}", []).append(k)
            res.setdefault(f"{l}:{pt}:wrong", []).append(wrong); res.setdefault(f"{l}:{pt}:blown", []).append(nb)
    del recs
    log(f"  texts {c0}-{min(len(TEXTS), c0 + CHUNK) - 1} done ({time.time() - tl:.0f}s; total {time.time() - t0:.0f}s, {rss()} MB)")

log(f"\nTHE REQUIREMENT AT EACH POINT: the coarsest box +-2^k on every line that still proves the top 5 (set and order).")
log(f"  k = -1: the exact value is needed. Shares of the {NPOS} positions; median k over the positions not needing exact.")
out = {}
for l in LAYERS:
    for pt in PTS:
        k = np.concatenate(res[f"{l}:{pt}"]); out[f"{l}:{pt}"] = k
        h = np.bincount(k + 1, minlength=KMAX + 2)
        ne = k >= 0
        log(f"  layer {l:2d} {pt:6s}: exact needed at {int((k < 0).sum()):5d} ({(k < 0).mean():6.1%}); "
            f"median k {int(np.median(k[ne])) if ne.any() else '-':>3}; k by count " + " ".join(f"{c - 1}:{h[c]}" for c in range(KMAX + 2) if h[c])
            + f"   [proofs naming other words: {sum(res[f'{l}:{pt}:wrong'])}]")
np.savez(f"{OUT}/{TAG}.npz", **{k.replace(":", "__"): v for k, v in out.items()})
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
