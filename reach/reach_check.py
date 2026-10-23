"""reach/reach_check.py: (1), AN INDEPENDENT CHECK BY REAL REMOVAL OF SINGLE DIGITS.

reach.py and reach_alone.py found, by the bucket and the largest possible move, that no digit can go (at most 274 of
548.7M could go even by themselves). This checks it the other way, with no bucket and no bound: single digits taken
at random from every plate (2,000 a plate: a random weight that is not 0, then one of its lit places at random) are
each removed by themselves for real, the pile moved by exactly -(the digit) x (the light on its line) at every
position, and the bank's own rounding (base2's power or rshift, its own F or shift) run on the moved pile. Counted:
the digits whose removal changes at least one kept value at some position (7,848 positions for the 48 plates, the
6,480 read positions for the top table).
Also written down for (2): each plate's place-0 digits and the photon meetings they take part in (on each line, the
light's lit places in signed digits x the plate's place-0 digits), over the same positions.
PREDICTION (written before the run): every sampled digit's real removal changes some kept value, 2,000/2,000 on every
  plate (the closest pile of nearly every neuron sits on its bucket's edge, and the real move has a sign that crosses
  it at roughly half of those). At most a handful on mproj.0, where 274 digits pass even the bound.
    python3 reach/reach_check.py      -> reach/reach_check.log, reach/reach_check.json
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastm as F
F.ROWS = 256
HERE = F.HERE; OUT = f"{HERE}/reach"; LOG = open(f"{OUT}/reach_check.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
RNG = np.random.default_rng(7); K = 2000; RES = {}

def sample(w):
    nz = np.flatnonzero(w.reshape(-1)); pick = RNG.choice(nz, K, replace=False)
    i, j = np.divmod(pick, w.shape[1]); v = w[i, j].astype(np.int64); up, dn = F.signed_digits(v); lit = up | dn
    p = np.array([RNG.choice(np.flatnonzero((int(L) >> np.arange(17)) & 1)) for L in lit])
    d = np.where(((up >> p) & 1) == 1, 1, -1)                 # the digit's sign
    return i, j, p, d

def check(name, x16, P, kept_fn):
    """P: (T, K) the sampled neurons' piles; kept_fn(X, cols): the bank's rounding for those columns."""
    w = F.plate(name); i, j, p, d = sample(w)
    changed = np.zeros(K, bool)
    for s0 in range(0, K, 200):                                # 200 digits at a time
        ss = slice(s0, s0 + 200)
        move = (d[ss] << p[ss])[None, :] * x16[:, i[ss]]       # removing the digit takes it off the pile
        Pj = P[:, j[ss]]
        changed[ss] = (kept_fn(Pj, j[ss]) != kept_fn(Pj - move, j[ss])).any(0)
    p0 = np.zeros(w.shape[0], np.int64); nrow = np.zeros(w.shape[0], np.int64)
    for c0 in range(0, w.shape[1], 4096):                      # a piece of neurons at a time
        up, dn = F.signed_digits(w[:, c0:c0 + 4096].astype(np.int64)); lit = up | dn
        p0 += (lit & 1).sum(1, dtype=np.int64); nrow += np.bitwise_count(lit).sum(1, dtype=np.int64)
    nx = F.naf(x16).sum(0)
    r = dict(sampled=K, changed=int(changed.sum()), place_of_unchanged=np.bincount(p[~changed], minlength=17).tolist(),
             place0_digits=int(p0.sum()), digits=int(nrow.sum()), place0_meetings=int(nx @ p0), meetings=int(nx @ nrow))
    RES[name] = r
    log(f"  {name:8s}: single digits whose real removal changes a kept value: {r['changed']}/{K}; place-0 digits {r['place0_digits']:,} of "
        f"{r['digits']:,} ({100 * r['place0_digits'] / r['digits']:.2f}%), their meetings {100 * r['place0_meetings'] / r['meetings']:.2f}% "
        f"of the plate's ({time.time() - t0:.0f}s, {F.rss()} MB)")

def hook(name, l, d):
    if name in ("stream in", "stream a", "stream"): return
    x16, P, ex, LS, we = d["x16"], d["pile"], d["ex"], d["LS"], d["we"]
    if LS is not None:
        fn = lambda X, js: F.power(X, (we[None, js] + ex[:, None] + F.ES) * 65536 - LS[:, None])
    else:
        fn = lambda X, js: F.rshift(X, -(we[None, js] + ex[:, None] + F.ES))
    check(name, x16, P, fn)
log("48 plates at all 7,848 positions; 2,000 single digits a plate")
x16, ex, LS = F.run(hook=hook, log=None)
RR = F.READ_ROWS; xt, et, lt = x16[RR], ex[RR], LS[RR]; del x16
WH = F.plate("head"); HE = F.E_["head"]
def head_fn(X, js):
    return F.power(X, (HE[None, js] + et[:, None] + F.ES) * 65536 - lt[:, None])
st = RNG.bit_generator.state
i, j, p, d = sample(WH)                                         # the top table: piles of the sampled words only
RNG.bit_generator.state = st                                    # the same draw again inside check()
Ph = F.product_lean(xt, WH[:, np.unique(j)].astype(np.float64)); col = {c: k for k, c in enumerate(np.unique(j))}
class Cols:                                                     # the sampled words' piles, indexed by word
    def __getitem__(self, idx): return Ph[:, [col[c] for c in idx[1]]]
check("head", xt, Cols(), head_fn)
tot = {k: sum(r[k] for r in RES.values()) for k in ("sampled", "changed")}
log(f"\nALL 49 PLATES: single digits whose real removal changes a kept value {tot['changed']:,}/{tot['sampled']:,}")
json.dump(RES, open(f"{OUT}/reach_check.json", "w"), indent=1)
log(f"saved reach/reach_check.json ({time.time() - t0:.0f}s, max memory {F.rss()} MB)")
