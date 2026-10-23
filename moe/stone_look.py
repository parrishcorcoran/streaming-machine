"""The stone across the 32 sibling experts (PREDICTIONS_STONE.md). Read-only on the plates; no decomposing.
Each plate is looked at from its stream-place side (the side the siblings share):
  gate|up: the 1,024 stream places it reads, each met by k = 1,024 hidden entries
  down:    the 1,024 stream places it writes, each fed by k = 512 hidden entries
A. the never-lit neurons (gate and up all zero): what their down lines hold
B. each expert's mean square; the floor across siblings; brightness of each stream place (spread, and siblings'
   profiles related), against the stone's sqrt(2/k)
C. relations between stream places: R = k x spread of their cosines (a stone gives 1); siblings' patterns related
Controls: every weight of the plate shuffled (R); stream places shuffled in each expert (siblings).
Reference: 32 fresh bell plates of the same shapes (seed 2026), the stone's own look.
    python3 stone_look.py
"""
import glob, json, numpy as np, torch
from safetensors import safe_open
SNAP = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/")[0]
F = {}
for f in glob.glob(SNAP + "*.safetensors"):
    h = safe_open(f, "pt")
    for k in h.keys(): F[k] = h
def get(k): return F[k].get_tensor(k).float().numpy()          # bf16 -> float32 is exact
rng = np.random.default_rng(2026)
IU = None
def look(P, k_side):
    """P: [32][k][1024]: each expert's plate with its hidden side first and its 1,024 stream places second."""
    global IU
    n = P.shape[0]; P64 = P.astype(np.float64)
    ms = (P64 ** 2).mean((1, 2))
    b = (P64 ** 2).sum(1)                                            # brightness of each stream place [32][1024]
    cv = b.std(1) / b.mean(1)
    prof = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    sib_b = (prof @ prof.T / prof.shape[1])[np.triu_indices(n, 1)]
    perm = np.stack([rng.permutation(1024) for _ in range(n)]); profp = np.take_along_axis(prof, perm, 1)
    sib_bc = (profp @ profp.T / 1024)[np.triu_indices(n, 1)]
    if IU is None: IU = np.triu_indices(1024, 1)
    off, offc, R, Rs = [], [], [], []
    for e in range(n):
        W = torch.from_numpy(P[e]); Wn = W / W.norm(dim=0, keepdim=True)
        C = (Wn.T @ Wn).numpy(); o = C[IU]; R.append(k_side * o.var()); off.append(o)
        pe = perm[e]; offc.append(C[np.ix_(pe, pe)][IU])
        S = torch.from_numpy(rng.permutation(P[e].ravel()).reshape(P[e].shape)); Sn = S / S.norm(dim=0, keepdim=True)
        Rs.append(k_side * (Sn.T @ Sn).numpy()[IU].var())
    def sib(X):
        X = np.stack(X).astype(np.float64); X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        return (X @ X.T / X.shape[1])[np.triu_indices(n, 1)]
    return dict(ms_ratio=float(ms.max() / ms.min()), bound_median=float(np.median(ms.min() / ms)), rms_median=float(np.sqrt(np.median(ms))),
                rms_min=float(np.sqrt(ms.min())), cv_median=float(np.median(cv)), sib_b=float(sib_b.mean()), sib_b_ctl=float(sib_bc.mean()),
                R_median=float(np.median(R)), R_min=float(np.min(R)), R_shuf=float(np.mean(Rs)), sib_rel=float(sib(off).mean()), sib_rel_ctl=float(sib(offc).mean()))

out = {"A": [], "layers": {}, "reference": {}}
print("A. THE NEVER-LIT NEURONS (gate and up columns all zero)")
for l in range(24):
    GU = get(f"model.layers.{l}.block_sparse_moe.input_linear.weight"); DN = get(f"model.layers.{l}.block_sparse_moe.output_linear.weight")
    for e in range(32):
        zc = np.flatnonzero(~GU[e].any(1))                           # rows of [1024 out][1024 in] = gate 0..511, up 512..1023
        if len(zc) == 0: continue
        neurons = sorted(set(int(c % 512) for c in zc))
        for j in neurons:
            g0, u0 = not GU[e][j].any(), not GU[e][512 + j].any(); d = DN[e][:, j]
            rec = dict(layer=l, expert=e, neuron=j, gate_zero=g0, up_zero=u0, down_line_zero=not d.any(),
                       down_line_rms=float(np.sqrt((d.astype(np.float64) ** 2).mean())), down_plate_rms=float(np.sqrt((DN[e].astype(np.float64) ** 2).mean())))
            out["A"].append(rec)
            print(f"  layer {l:2d} expert {e:2d} neuron {j:3d}: gate all zero {g0}, up all zero {u0} | its down line all zero {rec['down_line_zero']} "
                  f"(rms {rec['down_line_rms']:.2e}; the plate's rms {rec['down_plate_rms']:.2e})", flush=True)

print("\nB, C. EACH LAYER (median over the 32 experts unless said)")
hdr = f"  {'plate':10s} {'rms':>8s} {'loud/quiet':>10s} {'stone<=':>8s} {'bright spread':>13s} {'sib bright':>10s} {'(ctl)':>6s} {'R':>7s} {'R min':>6s} {'(R shuf)':>8s} {'sib rel':>8s} {'(ctl)':>7s}"
print(hdr)
def row(name, r, law):
    print(f"  {name:10s} {r['rms_median']:8.5f} {r['ms_ratio']:10.2f} {100 * r['bound_median']:7.1f}% {100 * r['cv_median']:6.1f}% (st {law:.2f}%) "
          f"{r['sib_b']:7.3f} {r['sib_b_ctl']:+6.3f} {r['R_median']:7.2f} {r['R_min']:6.2f} {r['R_shuf']:8.3f} {r['sib_rel']:8.4f} {r['sib_rel_ctl']:+7.4f}", flush=True)
ref = {"gate|up": rng.standard_normal((32, 1024, 1024)).astype(np.float32) * 0.02, "down": rng.standard_normal((32, 512, 1024)).astype(np.float32) * 0.02}
for name, k in (("gate|up", 1024), ("down", 512)):
    out["reference"][name] = r = look(ref[name], k); row(f"STONE {name}", r, 100 * np.sqrt(2 / k))
del ref
for l in range(24):
    GU = get(f"model.layers.{l}.block_sparse_moe.input_linear.weight")          # [32][1024 hidden out][1024 stream in]: hidden side first
    DN = get(f"model.layers.{l}.block_sparse_moe.output_linear.weight")         # [32][1024 stream out][512 hidden in]
    print(f" layer {l}")
    out["layers"][l] = {}
    for name, P, k in (("gate|up", GU, 1024), ("down", np.ascontiguousarray(DN.transpose(0, 2, 1)), 512)):
        r = look(P, k); out["layers"][l][name] = r; row(name, r, 100 * np.sqrt(2 / k))
json.dump(out, open("stone_look.json", "w"), indent=1)
