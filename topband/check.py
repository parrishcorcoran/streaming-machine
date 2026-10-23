"""THE SESSION'S RECOUNT OF THE BAND-BY-BAND TOP READ (not an independent checker: the same session wrote it).
Its own code throughout: no walk and no C. Each exposure is formed directly as x_i * w_iw (whole numbers, numpy int64),
split into the 8 bands by place masks on |r|, the gaps for the walk's count taken from np.sort of each line, gamma and
delta worked out again with Python fractions, and the set-aside made by its own loop. At every position checked it
compares, number by number, with topband/top_<set>.npz: the words still in after every test, the pours of every band,
the photons of every band over every word, the walk's photons, the light's writing, the standing amounts, the bound, the
words read to the end, mu, the top 5 and the race winner; and the plain read's three parts.
Positions: every 27th of the 648 refs (24) and every 250th of the 6,000 judged positions (24).
PREDICTION: every number equal at all 48 positions.
    python3 topband/check.py   -> topband/check.log
"""
import sys, glob, math, time, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/topband"; sys.path.insert(0, OUT)
LOG = open(f"{OUT}/check.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import topband as TB                                           # only base2.py's steps (power, race) and file paths
power, race, EXP2, ES, HEAD_E = TB.power, TB.race, TB.EXP2, TB.ES, TB.HEAD_E
T = np.asarray(TB.M["head.w"]).astype(np.int64); bias = np.asarray(TB.M["head.b"]).astype(np.int64); L, NW = T.shape
pc = lambda a: np.bitwise_count(np.abs(a)).astype(np.int64)
def sd(a): a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
# the walk's photons per line: signed-digit lit places of each sorted line's gaps (the lowest weight itself first)
srt = np.sort(T, axis=1); gaps = np.diff(srt, axis=1, prepend=0); GN_LINE = sd(gaps).sum(1); del srt, gaps
PLAIN_LINE = pc(T).sum(1); BONES = pc(bias)
BANDS = [(29, 26), (25, 22), (21, 18), (17, 14), (13, 10), (9, 6), (5, 2), (1, 0)]
MASKS = [sum(1 << k for k in range(lo, hi + 1)) for hi, lo in BANDS]

def recount(x, ex, LSv):
    x = x.astype(np.int64); n = int((x != 0).sum())
    Fp = int((int(HEAD_E[0]) + int(ex) + ES) * 65536 - int(LSv)); I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
    e2 = int(EXP2[(-Fp) & 0xFFFF]); s2 = 16 - ((-Fp) >> 16)
    gam = Fraction(1, 2) + int(np.abs(bias).max()) * Fraction(2) ** s * abs(e * e2 - Fraction(2) ** (s + s2)) / (e * Fraction(2) ** (s + s2))
    gamma = math.ceil(gam); delta = max(1, math.ceil(Fraction(2) ** s / e)); mu = 2 * gamma + delta
    beta = power(bias, -Fp)
    part = np.zeros((8, NW), np.int64); phot = np.zeros((8, NW), np.int64)
    for i0 in range(0, L, 96):                                 # exposures directly, 96 lines at a time
        r = x[i0:i0 + 96, None] * T[i0:i0 + 96]; a = np.abs(r); sg = np.sign(r)
        for b, m in enumerate(MASKS):
            am = a & m; part[b] += (sg * am).sum(0); phot[b] += pc(am).sum(0)
        del r, a, sg
    pile = part.sum(0)
    o = dict(n=n, mu=mu, walk=int(sd(x) @ GN_LINE), writing=int(pc(x).sum()), hall=phot.sum(1),
             standing_written=int(BONES.sum()) * bin(e2).count("1"), standing_poured=int(pc(beta).sum()))
    inn = np.ones(NW, bool); pours = []; surv = []; bound = []; cmp_ = []
    have = beta.copy()
    for b, (hi, lo) in enumerate(BANDS):
        pours.append(int(phot[b][inn].sum())); have = have + np.where(inn, part[b], 0)
        if b == 7: break
        R = n * ((1 << lo) - 1); vals = np.sort(have[inn])[::-1]; v5 = int(vals[4])
        cut = v5 - 2 * R - mu
        bound.append(bin(n).count("1") + bin(mu).count("1") + bin(2 * R + mu).count("1")); cmp_.append(2 * int(inn.sum()))
        inn = inn & (have > cut); surv.append(int(inn.sum()))
    w = np.flatnonzero(inn); pw = have[w] - beta[w]; sc = power(pw, Fp) + bias[w]
    o["end"] = int(pc(beta[w]).sum() + pc(pw).sum() * bin(e).count("1") + BONES[w].sum())
    full = np.full(NW, -(1 << 62), np.int64); full[w] = sc
    o.update(pours=np.array(pours), surv=np.array(surv), bound=np.array(bound), cmp=np.array(cmp_),
             top5=TB.top5(full[None])[0], win=race(full), plain_meet=int(pc(x) @ PLAIN_LINE),
             plain_div=int(pc(pile).sum()) * bin(e).count("1"), plain_stand=int(BONES.sum()))
    return o

KEYS = ["n", "mu", "walk", "writing", "standing_written", "standing_poured", "end", "plain_meet", "plain_div", "plain_stand",
        "hall", "pours", "surv", "bound", "cmp", "top5", "win"]
allbad = 0; done = 0
for SET, every in (("refs", 27), ("windows", 250)):
    if SET == "refs":
        Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); RUN = [np.load(f"{OUT}/top_refs.npz")]
    else:
        Z = np.load(f"{HERE}/fringes_machine/windows_top_light.npz"); RUN = [np.load(f) for f in sorted(glob.glob(f"{OUT}/top_windows_p*of2.npz"))]
    X, EX, LSa = Z["x16"], Z["ex"], Z["LS"]
    run = {k: np.concatenate([z[k] for z in RUN]) for k in KEYS}; sel = np.concatenate([z["sel"] for z in RUN])
    assert np.array_equal(sel, np.arange(len(X)))
    for q in range(0, len(X), every):
        o = recount(X[q], EX[q], LSa[q]); diff = [k for k in KEYS if not np.array_equal(np.asarray(o[k]), np.asarray(run[k][q]))]
        allbad += len(diff); done += 1
        if diff: log(f"  {SET} position {q}: DIFFERS in {diff}")
    log(f"  {SET}: {len(range(0, len(X), every))} positions recounted ({time.time() - t0:.0f}s, {TB.rss()} MB)")
log(f"RECOUNT: {done} positions, {len(KEYS)} numbers or arrays each: differing {allbad}  ({time.time() - t0:.0f}s)")
