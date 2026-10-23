"""CHECK 4 (independent, small): step 1's whole-set numbers recomputed from the session's stop places and the saved
lights with the checker's own formulas; the last water level's drop; refs position 494 (the exact tie)."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward/check")
from common import *
LOG = open(f"{CK}/check_extra.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
pph = np.bitwise_count(np.abs(HW.astype(np.int64))).sum(1).astype(np.int64)            # 1s of the top table per line
for st, lf, tf in (("refs", "refs_top_light", "top_refs"), ("windows", "windows_top_light", "top_windows")):
    Z = np.load(f"{HERE}/fringes_machine/{lf}.npz"); c = np.load(f"{HERE}/backward/out/{tf}.npz")["c5"]
    X = np.abs(Z["x16"]); ex = Z["ex"]; N = len(c)
    lines = (X >= (np.int64(1) << c[:, None])).sum(1)
    pl_read = np.maximum(nbits(X) - c[:, None], 0).sum(); pl_all = nbits(X).sum()
    ph_read = np.bitwise_count(X >> c[:, None]).sum(); ph_all = np.bitwise_count(X).sum()
    meet_read = (np.bitwise_count(X >> c[:, None]).astype(np.int64) @ pph).sum(); meet_all = (np.bitwise_count(X).astype(np.int64) @ pph).sum()
    h = np.bincount(c, minlength=8)
    log(f"{st} ({N}): c5 median {int(np.median(c))}; every place at {(c == 0).mean():.1%}; by place " + " ".join(f"{k}:{h[k]}" for k in range(8) if h[k])
        + f"; lines read median {int(np.median(lines))}; places read {pl_read / pl_all:.2%}; photons read {ph_read / ph_all:.2%}; "
        f"meetings of the read part with every word {meet_read / meet_all:.2%}")
    sh = ex + ES
    log(f"   the last water level's drop: median {int(np.median(sh))}, range {int(sh.min())}..{int(sh.max())}, 5-95% {np.percentile(sh, [5, 95]).astype(int).tolist()}")
Q = np.load(f"{HERE}/backward/out/req_refs_L11.npz")
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
off = np.concatenate([[0], np.cumsum([len(t) for t in TEXTS])]); t = int(np.searchsorted(off, 494, side="right") - 1); p = 494 - off[t]
rec = forward_keep(TEXTS[t], keep=-1); S = rec["S"][p]; o = np.argsort(-S, kind="stable")[:7]
log(f"refs 494 (text {t}, position {p}): the seven largest scores {[(int(j), int(S[j])) for j in o]}; req k at Mw {Q['11__Mw'][494]}, H {Q['11__H'][494]}, A {Q['11__A'][494]}")
k = Q["11__Rout"]; log(f"refs final stream box: median k {int(np.median(k[k >= 0]))}, by count {np.bincount(k + 1).tolist()} (index 0 = exact)")
