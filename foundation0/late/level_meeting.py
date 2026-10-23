"""HOW A 3-4% RELIEF MAKES RIPPLES OF TENS OF E-FOLDS: the light leaving qkv.l, with and without its foundation. 2026-10-06.

HIS SENTENCE (his exact words): "It should be 100% removable after training."
SESSION'S RESTATEMENT (not his): what survives of the random start in plate W is round(s x W0).

WHAT level_height.py FOUND (level_height_160m.log, _70m.log): removing round(s x W0) from qkv.l alone (rho = 3.3-4.2% of
the plate at 160m qkv.9-11) changes the scores at layer l by a median 12-57 e-folds per position (39-74% of the row's
height, which is 28-88 e-folds), and the level moves because another position's score RISES by 41-137 e-folds; the
scrambled copies do the same. Prediction (2) there (ripple 1-10 e-folds, 3-6% of the row) was wrong by a factor 4-15.

WHAT IS LOOKED AT HERE (the same machine and plates as plate_look.py, no new part; layer l only, where the stream coming
in is the finished machine's exactly): for each position, the question q and key k leaving qkv.l after their turns,
  - the relief's share of the light leaving the plate: |q_case - q_fin| / |q_fin|, the same for k (medians)
  - the meeting's full size against what it gives: |q_i||k_j|/sqrt(hs) in e-folds against the score q_i.k_j/sqrt(hs),
    and the cosine between q_i and k_j (median |cos| over live pairs), in the finished machine
  - how much of the finished q and k is the plate's bias (|bias part| / |q|), since the bias is never touched
Cases: fin, b:qkv.l, cb:qkv.l@{20261006,2,3}; 160m l = 9, 10, 11; 70m l = 4, 5.

PREDICTIONS, written before the first run:
  (1) the relief's share of the light leaving the plate is far above its share of the plate: |dq|/|q| and |dk|/|k| at
      160m qkv.9-11 between 0.10 and 0.40 (rho 0.033-0.042), the scrambled copies the same within 20% of b's value.
  (2) the meetings are long and nearly at right angles: in the finished machine at 160m layers 10-11 the median
      |q_i||k_j|/sqrt(hs) is 5 times the row's height (88 e-folds at layer 11) or more, and the median |cos| is 0.2 or less.
  (3) the bias is not what makes the meeting long: its part is under half of |q| and of |k| at those layers.

  python3 level_meeting.py size=160m
"""
import os, sys, json, resource
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "160m")
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE)
SEEDS = [20261006, 2, 3]
LAYERS = {"160m": [9, 10, 11], "70m": [4, 5]}[SIZE]
LOGF = open(os.path.join(LATE, f"level_meeting_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

src = open(os.path.join(LATE, "plate_look.py")).read()
cut = src.index("# ---------------------------------------------------------------- the runs")
G = {"__file__": os.path.join(LATE, "plate_look.py"), "__name__": "plate_look_parts"}
sys_argv = sys.argv; sys.argv = ["plate_look.py", f"size={SIZE}", "tag=level_meeting_parts"]
exec(compile(src[:cut], "plate_look.py", "exec"), G)
sys.argv = sys_argv
os.remove(os.path.join(LATE, f"plate_look_{SIZE}-level_meeting_parts.log"))
L, NH, HS, d, ES = G["L"], G["NH"], G["HS"], G["d"], G["ES"]
plain, gauge, rshift, turn, to16, product, power = (G[k] for k in ("plain", "gauge", "rshift", "turn", "to16", "product", "power"))
HALFE = 0.5 * np.log(HS)                                                 # 1/sqrt(hs), as the natural log of the divide

def qk_at(ids, lay):
    """the stream into layer lay (finished machine up to there in every case here), then qkv.lay's q and k after turns,
    as real numbers for the look (value = whole number x 2^-ES); also the bias part of q and k"""
    ids = np.asarray(ids); EW, EE = G["PL"]["embed"]
    R = rshift(EW[ids].astype(np.int64), -(EE[ids][:, None] + ES))
    for l in range(lay + 1):
        X = gauge(R, f"ln1.{l}"); QKV = plain(X, f"qkv.{l}")
        if l == lay:
            P = len(ids); pos = np.arange(P); bias = G["PL"][f"qkv.{l}"][2]
            out = []
            for h in range(NH):
                o = h * 3 * HS
                q, k = turn(QKV[:, o:o + HS], pos), turn(QKV[:, o + HS:o + 2 * HS], pos)
                bq, bk = bias[o:o + HS], bias[o + HS:o + 2 * HS]
                out.append((q / 2.0 ** ES, k / 2.0 ** ES, np.linalg.norm(bq) / 2.0 ** ES, np.linalg.norm(bk) / 2.0 ** ES))
            return out
        At = plain(G["attention"](QKV), f"proj.{l}")
        Mo = plain(G["edge"](plain(gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        R = R + At + Mo

refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:30]
seqs = [r["ids"] + r["ref"][:15] for r in refs]
log(f"level_meeting.py  size {SIZE}  layers {LAYERS}  (a look in real numbers at the machine's own whole-number q and k)")
G["set_case"]("fin")
FIN = {lay: [qk_at(s, lay) for s in seqs] for lay in LAYERS}
RES = {}
for lay in LAYERS:
    full, sc, cosv, height, bqs, bks = [], [], [], [], [], []
    for heads in FIN[lay]:
        for (q, k, bq, bk) in heads:
            P = len(q); live = np.tril(np.ones((P, P), bool)); live[0, 0] = False
            nq, nk = np.linalg.norm(q, axis=1), np.linalg.norm(k, axis=1)
            S = (q @ k.T) / np.sqrt(HS); F = np.outer(nq, nk) / np.sqrt(HS)
            full += F[live].tolist(); cosv += np.abs(S[live] / np.maximum(F[live], 1e-30)).tolist()
            for i in range(1, P): height.append(S[i, :i + 1].max() - S[i, :i + 1].min())
            bqs += (bq / np.maximum(nq, 1e-30)).tolist(); bks += (bk / np.maximum(nk, 1e-30)).tolist()
    o = dict(full_median=float(np.median(full)), height_median=float(np.median(height)), cos_median=float(np.median(cosv)),
             bias_q=float(np.median(bqs)), bias_k=float(np.median(bks)))
    RES[f"fin.L{lay}"] = o
    log(f"  layer {lay}: finished meeting |q||k|/sqrt(hs) median {o['full_median']:.1f} e-folds; row height median "
        f"{o['height_median']:.1f} (x{o['full_median']/o['height_median']:.1f}); median |cos(q,k)| {o['cos_median']:.3f}; "
        f"bias part of |q| {o['bias_q']:.3f}, of |k| {o['bias_k']:.3f}")
    for case in [f"b:qkv.{lay}"] + [f"cb:qkv.{lay}@{s}" for s in SEEDS]:
        G["set_case"](case)
        rq, rk = [], []
        for heads_f, s in zip(FIN[lay], seqs):
            heads_c = qk_at(s, lay)
            for (qf, kf, _, _), (qc, kc, _, _) in zip(heads_f, heads_c):
                rq += (np.linalg.norm(qc - qf, axis=1) / np.maximum(np.linalg.norm(qf, axis=1), 1e-30)).tolist()
                rk += (np.linalg.norm(kc - kf, axis=1) / np.maximum(np.linalg.norm(kf, axis=1), 1e-30)).tolist()
        o = dict(dq=float(np.median(rq)), dk=float(np.median(rk)))
        RES[case] = o
        log(f"    {case:18s} the relief's share of the light leaving the plate: |dq|/|q| {o['dq']:.3f}  |dk|/|k| {o['dk']:.3f}  "
            f"(plate share rho {G['SHARE'][f'qkv.{lay}']['rho']:.3f})")
json.dump(RES, open(os.path.join(LATE, f"level_meeting_{SIZE}.json"), "w"))
log(f"done  peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
