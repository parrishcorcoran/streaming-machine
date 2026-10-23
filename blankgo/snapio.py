"""Reading the builds back, exactly, and the scrambled blanks.

  snapshots(d, L, recipe) yields (t, Bt, P, W) in step order, all whole numbers at 2^-30:
    Bt  the blank as it stands at step t (A, B: the draw itself; C: the draw with every shave replayed exactly)
    P   the pushes as they stand (stored by build.py)
    W   = Bt + P, the weight
  and checks Bt and W against the hashes build.py logged at that step (any difference stops the run).

  scramble(name, a, k) permutes a drawn part's values WITHIN EACH NEURON (draw k, fixed seed): a plate's or the output
  word table's column (one neuron, one power of two in the machine); an input word table's or position table's row.
  Same values, same neuron, same power of two; only which input line each value meets is broken.
"""
import os, json, hashlib, numpy as np
from model import HERE, layout, drawn, tag, blank, step_size

def _h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def hall(D, names): return hashlib.sha256(b"".join(_h(D[n]).encode() for n in names)).hexdigest()[:16]

def build_log(d, L, recipe): return json.load(open(os.path.join(HERE, f"build_{tag(d, L)}_{recipe}.json")))

def snapshots(d, L, recipe, only=None):
    log = build_log(d, L, recipe); names = [n for n, _, _ in layout(d, L)]; dr = set(drawn(d, L))
    b0 = blank(d, L); Bt = {n: b0[n].copy() for n in names}; done = 0
    for t in sorted(int(s) for s in log["snaps"]):
        if log["wd"]:
            for s in range(done + 1, t + 1):
                f = step_size(s, log["peak"]) * log["wd"]
                for n in dr: Bt[n] -= np.rint(Bt[n] * f).astype(np.int64)
            done = t
        if only is not None and t not in only: continue
        z = np.load(os.path.join(HERE, "snaps", f"{tag(d, L)}_{recipe}_step{t}.npz"))
        P = {n: z["P." + n] for n in names}
        W = {n: Bt[n] + P[n] for n in names}
        rec = log["snaps"][str(t)]
        assert hall(Bt, names) == rec["hash_Bt"], f"replayed blank differs at step {t}"
        assert hall(W, names) == rec["hash_W"], f"weights differ at step {t}"
        yield t, {n: a.copy() for n, a in Bt.items()}, P, W

AXIS = lambda name: 1 if name in ("wte", "wpe") else 0       # which axis runs along one neuron's values

def scramble(name, a, k):
    rng = np.random.default_rng(7000 + 101 * k + sum(map(ord, name)))
    ax = AXIS(name)
    idx = np.argsort(rng.random(a.shape), axis=ax)
    return np.take_along_axis(a, idx, axis=ax)
