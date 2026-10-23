"""Is step0 the random start? Each plate read one at a time (safetensors mmap), fp16 values widened to float64 only
to measure them; nothing here is the machine.

PREDICTION (written before the run, 2026-10-06):
  Spreads from Pythia's own build settings (settings/pythia-*.yml: "init_method": "small_init",
  "output_layer_init_method": "wang_init"):
      small_init  sigma = sqrt(2 / (5 d))     query_key_value, dense_h_to_4h, embed_in, embed_out
      wang_init   sigma = 2 / (L sqrt(d))     attention.dense, dense_4h_to_h
      14m (d 128, L 6):   small 0.05590  wang 0.02946
      70m (d 512, L 6):   small 0.02795  wang 0.01473
      160m (d 768, L 12): small 0.02282  wang 0.006014
  1. Every step0 plate: centred on 0 (|mean| under 4 sigma/sqrt(N)); measured spread within 1% of the settings'
     sigma; a bell curve: 68.27 / 95.45 / 99.73% inside 1 / 2 / 3 sigma, each within 0.3 points on the big plates;
     fourth moment / sigma^4 = 3.00 +- 0.05 (bigger allowance on the 14m's small plates).
  2. Every norm gain exactly 1 and every bias and norm offset exactly 0 at step0.
  3. Shuffle control (the pairing of value to place broken, every value kept): the spread across neurons of each
     neuron's sum of squares, divided by the same for the shuffled plate. Step0: 1.00 within a few % (no structure
     beyond the draw). step143000: well above 1 on most plates (structure built on top).
    python3 check_step0.py
"""
import os, json, math, numpy as np
from safetensors import safe_open

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
rng = np.random.default_rng(0)

def bell(x, sig):                        # in slabs of rows, so no full-size copies (peak under 3 GB)
    n = [0, 0, 0]; m2 = m4 = 0.0; zmax = 0.0
    for i in range(0, len(x), 4096):
        z = np.abs(x[i:i + 4096] / sig); n = [n[j] + int((z < k).sum()) for j, k in enumerate((1, 2, 3))]
        z2 = z * z; m2 += float(z2.sum()); m4 += float((z2 * z2).sum()); zmax = max(zmax, float(z.max()))
    return [v * 100 / x.size for v in n], (m4 / x.size) / (m2 / x.size) ** 2, zmax

def cv_ratio(W):                         # spread across neurons of each neuron's sum of squares: plate / shuffled plate
    def cv(M, ax):
        s = (M * M).sum(ax); return s.std() / s.mean()
    W = W.astype(np.float32); a = cv(W, 1), cv(W, 0)        # float32 copies keep the peak under 3 GB
    S = rng.permutation(W.ravel()).reshape(W.shape); del W
    return float(a[0] / cv(S, 1)), float(a[1] / cv(S, 0))

out = {}
for name, d, L in [("pythia-14m", 128, 6), ("pythia-70m", 512, 6), ("pythia-160m", 768, 12)]:
    small, wang = math.sqrt(2 / (5 * d)), 2 / (L * math.sqrt(d))
    print(f"\n{name}: settings' spread small_init {small:.5f}, wang_init {wang:.5f}")
    print(f"  {'plate':42s} {'kind':5s} {'mean/(s/rtN)':>12s} {'spread/set':>10s} {'<1s':>6s} {'<2s':>6s} {'<3s':>6s} {'m4':>5s} {'max|z|':>6s}"
          f" {'shuf rows':>9s} {'shuf cols':>9s} | finished: {'rows':>6s} {'cols':>6s}")
    f0 = safe_open(os.path.join(D, f"{name}-step0", "model.safetensors"), "np")
    f1 = safe_open(os.path.join(D, f"{name}-step143000", "model.safetensors"), "np")
    gains = biases = 0; bad = []
    for k in sorted(f0.keys()):
        x = f0.get_tensor(k).astype(np.float64)
        if k.endswith(".bias") or "layernorm" in k or "layer_norm" in k:
            want = 1.0 if ("norm" in k and k.endswith(".weight")) else 0.0
            ok = bool((x == want).all()); gains += want == 1.0; biases += want == 0.0
            if not ok: bad.append((k, float(np.abs(x - want).max())))
            continue
        kind = "wang" if (k.endswith("attention.dense.weight") or k.endswith("dense_4h_to_h.weight")) else "small"
        sig = wang if kind == "wang" else small
        N = x.size; m = x.mean(); s = x.std()
        (p1, p2, p3), m4, zmax = bell(x - 0, sig)
        r0 = cv_ratio(x); del x
        r1 = cv_ratio(f1.get_tensor(k))
        short = k.replace("gpt_neox.", "").replace("layers.", "L")
        print(f"  {short:42s} {kind:5s} {m / (s / math.sqrt(N)):12.2f} {s / sig:10.4f} {p1:6.2f} {p2:6.2f} {p3:6.2f} {m4:5.2f} {zmax:6.2f}"
              f" {r0[0]:9.3f} {r0[1]:9.3f} |           {r1[0]:6.2f} {r1[1]:6.2f}")
        out.setdefault(name, {})[k] = dict(kind=kind, settings=sig, mean_over_se=m / (s / math.sqrt(N)), spread_ratio=s / sig,
                                           within=[p1, p2, p3], m4=m4, zmax=zmax, shuf_step0=r0, shuf_finished=r1)
    print(f"  norm gains checked {gains} vectors, biases/offsets checked {biases} vectors; not exactly 1/0: {bad if bad else 'none'}")
    out.setdefault(name, {})["_norms_biases"] = dict(gain_vectors=gains, zero_vectors=biases, bad=bad)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_step0.json"), "w"), indent=1)
