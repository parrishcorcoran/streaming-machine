"""The camera's toolkit on the ranked plate (each neuron's weights sorted by size). All exact: same piles, same words.
Counted: photon meetings (1s x 1s) and light adds, against the plain plate. And: is every neuron's ranked plate one
shared curve, only the order of its lines different? (weights already scaled a neuron to its loudest by a power of two)
    python3 rank_camera.py
"""
import numpy as np
exec(open("rank_gaps.py").read().split("tot = {}")[0])          # base2 machine, pc(), lights per bank (the 30 pieces)
res = {}
for name, xs in lights.items():
    kind = name.split(".")[0]; r = res.setdefault(kind, {})
    w = M[name + ".w"].astype(np.int64); nin, nout = w.shape
    order = np.argsort(w, axis=0, kind="stable"); ws = np.take_along_axis(w, order, 0)
    pw = pc(w); gaps = np.diff(ws, axis=0, prepend=0); pg = pc(gaps)
    Q = np.median(ws, axis=1).round().astype(np.int64)[:, None]; dev = ws - Q   # the one shared curve, and each neuron off it
    add = lambda k, v: r.__setitem__(k, r.get(k, 0) + int(v))
    add("photons in weights", pw.sum()); add("photons off the one curve", pc(dev).sum())
    bins = {}
    for B in (16, 64, 256):
        idx = (np.arange(nin) * B) // nin                                        # rank -> bin
        lo = np.array([ws[idx == b][0] for b in range(B)]); hi = np.array([ws[idx == b][-1] for b in range(B)])
        rep = (lo + hi) // 2; resid = ws - rep[idx]; bins[B] = (idx, rep, pc(rep), pc(resid))
    for x in xs[:6]:
        X = x.astype(np.int64)[order]; px = pc(X)
        add("plain", (pw * pc(x)[:, None]).sum())
        S = np.cumsum(X[::-1], axis=0)[::-1]; add("gaps", (pg * pc(S)).sum())
        for B, (idx, rep, prep, presid) in bins.items():
            T = np.zeros((B, nout), np.int64); np.add.at(T, idx, X)               # each bin's light, added once
            if name == "fc.0" and x is xs[0] and B == 64: assert np.array_equal((rep * T).sum(0) + ((ws - rep[idx]) * X).sum(0), (w * x[:, None]).sum(0))
            add(f"bins {B}", (prep * pc(T)).sum() + (presid * px).sum())
print(f"  {'bank':6s} | photon meetings as a share of plain:   gaps    16 bins   64 bins   256 bins | photons off the one curve")
for kind, r in res.items():
    p = r["plain"]; print(f"  {kind:6s} |{'':38s}{100 * r['gaps'] / p:5.0f}%  {100 * r['bins 16'] / p:7.0f}%  {100 * r['bins 64'] / p:7.0f}%  {100 * r['bins 256'] / p:8.0f}%  | {100 * r['photons off the one curve'] / r['photons in weights']:5.0f}% of the weights' photons")
