"""PLATES: look before coding. (1) how lit each place is (the chance a place holds a 1) over all 48 plates -- a place
that is lit half the time with no pattern is one full bit that no coder can remove; (2) the spread of the neurons'
powers of two in each plate; (3) the shared part: how much of a plate's light is a per-line or per-neuron constant
(the mean across the other direction, whole-number sums and shifts), against a random-shuffle control;
(4) coder test on one plate: meters on / off / values shuffled at random (fixed seed)."""
from plates_lib import *
import time
rng = np.random.default_rng(1)
tot_on = np.zeros(16); tot_naf_on = np.zeros(17); n_all = 0
for n in NAMES:
    q = plate(n).astype(np.int64); a = np.abs(q); n_all += q.size
    for b in range(16): tot_on[b] += ((a >> b) & 1).sum()
print("chance a place is lit (plain base 2), place 0 = lowest:", " ".join(f"{x/n_all:.3f}" for x in tot_on))
print("values:", n_all, " order-0 floor over all plates (per-plate histograms):")
h = 0.0
for n in NAMES: h += entropy_bits(plate(n))
print(f"  {h/n_all:.4f} bits/value -> {h/8/1e6:.2f} MB")
for n in ["qkv.0", "proj.0", "fc.0", "mproj.0", "qkv.6", "fc.11", "mproj.11"]:
    e = colexp(n); vals, cnt = np.unique(e, return_counts=True)
    print(f"{n:9s} shape {plate(n).shape}  neuron powers of two: " + " ".join(f"{v}:{c}" for v, c in zip(vals, cnt)))
# shared part: per-line mean across neurons, per-neuron mean across lines (sums, then the variance of those means
# against the plate's variance; for i.i.d. values the line means would hold ~1/n_out of it)
print("shared part (variance share of per-line / per-neuron constants) vs what i.i.d. values would give:")
for n in ["qkv.0", "proj.0", "fc.0", "mproj.0", "qkv.6", "proj.6", "fc.6", "mproj.6"]:
    q = plate(n).astype(np.float64); R, C = q.shape; vt = q.var()
    lm = q.mean(1).var() / vt; nm = q.mean(0).var() / vt
    qs = rng.permutation(q.ravel()).reshape(R, C)
    print(f"  {n:8s} per-line {lm:.5f} (iid {1/C:.5f}, shuffled {qs.mean(1).var()/vt:.5f})   per-neuron {nm:.5f} (iid {1/R:.5f})")
for n in ["fc.0", "qkv.6"]:
    q = plate(n).astype(np.int32)
    for ctx, lab, A in [(1, "meters on", q), (0, "meters off", q), (1, "meters on, values shuffled", rng.permutation(q.ravel()).reshape(q.shape))]:
        t = time.time(); nb, ok = store(A, ctx=ctx)
        print(f"  {n} {lab:28s} {8*nb/q.size:.4f} bits/value  roundtrip {ok}  {time.time()-t:.1f}s   (order-0 floor {entropy_bits(q)/q.size:.4f})")
