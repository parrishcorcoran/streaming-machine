"""REFERENCE BEAM family, version 5: the lens turned the other way -- after the line reference, an integer Haar lens
ALONG THE WORDS of each line (pairs of neighbouring words, then pairs of pairs ...). A slowly changing reference (the
line's light drifting from word to word) would come out in the low band. Word order: the stored id order (a fixed rule
that never looks at the values), and a random order (seed 0) as the control. 8 levels on the first 50,176 words
(196 x 256); the last 81 words go as they are. Odds per (line, band).
Prediction (analysis only): both words of a pair sit on the same line, so they have the same spread; the change per
value for one level is 1/4 log2(1 - r^2) <= 0 with r the correlation of neighbouring words -- only a gain, and only as
big as r^2 allows.
    python3 refbeam_v5_haar_words.py
"""
import time, numpy as np
from refbeam_lib import *
from refbeam_lift import haar, ihaar

NW, LV = 50176, 8

def band_of_index(nw, lv):                           # which band each position along the words falls in
    b = np.zeros(nw, np.int64); n = nw
    for k in range(lv): b[n // 2:n] = k + 1; n //= 2
    return b                                          # 0 = lowest band

for name in ("E", "T"):
    A = load(name); n = A.size; t = time.time()
    ref = np.sort(A, axis=0)[A.shape[0] // 2]; side = small(ref); R = A - ref[None, :]
    for order in ("id", "random"):
        X = R if order == "id" else R[np.random.default_rng(0).permutation(R.shape[0])]
        a, b = X[0:NW:2].astype(np.float64), X[1:NW:2].astype(np.float64)
        a -= a.mean(0); b -= b.mean(0); r = (a * b).mean(0) / np.sqrt((a * a).mean(0) * (b * b).mean(0))
        print(f"   {name} {order} order: neighbouring words r^2 mean {float((r * r).mean()):.5f}, largest |r| {float(np.abs(r).max()):.3f};"
              f" level-1 prediction {float((0.25 * np.log2(1 - r * r)).mean()):+.5f} b/v")
        del a, b
        H = X.copy(); H[:NW] = haar(X[:NW].T, LV).T
        back = X.copy(); back[:NW] = ihaar(H[:NW].T, LV).T; ok_t = bool(np.array_equal(back, X)); del back
        band = np.concatenate([band_of_index(NW, LV), np.full(X.shape[0] - NW, LV + 1)])
        ctx = (np.arange(768)[None, :] * (LV + 2) + band[:, None]).astype(np.uint32)
        nb, ok = rc(H, ctx, False)
        report("v5", name, f"line ref + Haar along words ({order} order), 8 lvl, rc line x band", nb,
               side + (8 if order == "random" else 0), n, ok and ok_t)
        if order == "id":
            nb, ok = rc(R, (np.arange(768)[None, :] * (LV + 2) + band[:, None]).astype(np.uint32), False)
            report("v5", name, "same contexts, NO lens (to separate the lens from the contexts)", nb, side, n, ok)
            print(f"     photons/value: residual {ones(R).mean():.3f} -> through the lens {ones(H).mean():.3f}")
        del H, ctx
    del R, A
    print(f"   ({time.time() - t:.0f} s)")
