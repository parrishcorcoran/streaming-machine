"""PLATES v8 -- fewer photons per weight. Start from v7's development (line + neuron metering, place floor s_ij),
then let no weight carry more than k photons in the single-wave writing: each photon is the signed power of two
nearest what is left; the last one chooses between the two powers around the remainder by chance (fixed seed), or
nearest. Weights already within k photons are untouched; only the brightest weights lose their faint photons.
Stored: as v7 (whole numbers coded as photons, so a capped weight is cheaper to store).
Prediction before running: at v7 a=4 (1.70 photons/weight) a cap of 2 touches only the big weights: saves ~0.2-0.4
bit/weight, costs 1-3 words (a cap is coarser exactly where weights are biggest; equal-scatter says that is the
wrong place to save). Also prints how close the photon coder sits to each plate's own one-value-at-a-time floor."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
def v7(q, a):
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    tp = bl(qa.sum(0)) - bl(np.array(R)) + 1
    tr = bl(qa.sum(1)) - bl(np.array(C)) + 1; off = tr - mode(tr)
    return np.maximum(0, (tp - a)[None, :] + off[:, None])
# how close is the coder to the floor (v7 a=4, no words call)
fl = 0.0; cb = 0
for n in NAMES:
    q = plate(n); s = v7(q, 4); v = cut(q, s); fl += entropy_bits(v); cb += store(v, check=False)[0]
print(f"v7 a=4: coder {8*cb/NVAL:.4f} bits/value, per-plate order-0 floor of the stored numbers {fl/NVAL:.4f} (model not counted)", flush=True)
log({"version": "v7 a=4 coder vs floor", "coder_bits": round(8*cb/NVAL, 4), "order0_floor_bits": round(fl/NVAL, 4)})
runs = [(4, 2, True), (4, 3, True), (4, 2, False)]
for a, k, ch in runs:
    rng = np.random.default_rng(800 + 10 * a + k)
    def dev(n, q, e, a=a, k=k, ch=ch, rng=rng):
        s = v7(q, a); v = cut(q, s)
        return topk(v, k, chance=ch, rng=rng), s
    run_version(f"v8 v7 a={a} + at most {k} photons ({'chance' if ch else 'nearest'})", dev, side_per_col=2, extra_side=extra,
                note=f"v7 a={a} then cap at {k} single-wave photons per weight, last photon {'by chance' if ch else 'nearest'}")
