"""Part A of PREDICTIONS_STONE.md, done right: the neurons that went dark during the making. They are not exactly zero:
their largest weight is below 1e-6 (the plates' rms is 0.016-0.018). For each: its gate, its up and its down line."""
import glob, numpy as np
from safetensors import safe_open
SNAP = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/")[0]
F = {}
for f in glob.glob(SNAP + "*.safetensors"):
    h = safe_open(f, "pt")
    for k in h.keys(): F[k] = h
def get(k): return F[k].get_tensor(k).float().numpy().astype(np.float64)
n = 0
for l in range(24):
    GU = get(f"model.layers.{l}.block_sparse_moe.input_linear.weight"); DN = get(f"model.layers.{l}.block_sparse_moe.output_linear.weight")
    mx = np.abs(GU).max(2)
    for e, r in zip(*np.nonzero(mx < 1e-6)):
        if r >= 512: continue
        j = r; g, u, d = GU[e, j], GU[e, 512 + j], DN[e][:, j]; n += 1
        def rms(a): return np.sqrt((a ** 2).mean())
        z = g / g.std()
        print(f"layer {l:2d} expert {e:2d} neuron {j:3d}: rms gate {rms(g):.1e} up {rms(u):.1e} down line {rms(d):.1e} | plate {rms(GU[e]):.4f} | "
              f"gate's shape: within one spread {100 * (np.abs(z) < 1).mean():.1f}% (bell 68.3%), kurtosis {(z ** 4).mean():.1f} (bell 3)")
print(f"{n} dark neurons in all 24 layers")
