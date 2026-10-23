"""COMBINED v11 -- SHUFFLE CONTROLS for the two structural claims (5 frames each).
Claim 1 (v3, W1): the middle layers 4-7 take a coarser exposure for less loss than other layers.
  Control: the same number of layers (4) at A=5, chosen at random (fixed seeds), the rest A=8, tables c=35 -- the
  same A histogram and nearly the same bytes, the depth placement lost. If 'middle' is real, random picks that miss
  the middle land lower than W1's level 27.6.
Claim 2 (R1): within proj and fc, the end layers need the finer exposure (A=8) and the middle can be A=5.
  Control: proj's and fc's A values permuted across the 12 layers at random (fixed seed); qkv and mproj (all A=5)
  unchanged; tables c=35. Same per-kind bytes to within the layers' small size differences.
Predictions before running: random-4-layers levels 26-27 (below W1's 27.6, if any pick includes an end pair);
R1 permuted ~25.5-26 (below R1's 26.6)."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])
runs = []
for seed in (11, 12):
    pick = sorted(np.random.default_rng(seed).permutation(12)[:4].tolist())
    runs.append((f"v11 CONTROL W1 with 4 random layers {pick} at A=5, rest A=8, c=35", cfg_by(lambda n, p=pick: ("v12", 5 if layer(n) in p else 8)), 35))
r1 = alloc(5, 5, qkv_end=5); perm = np.random.default_rng(13).permutation(12)
ctl = {n: (r1[n] if kind(n) in ("qkv", "mproj") else r1[f"{kind(n)}.{perm[layer(n)]}"]) for n in NAMES}
runs.append((f"v11 CONTROL R1 with proj/fc A permuted across layers (perm {perm.tolist()}), c=35", ctl, 35))
for name, cfg, c in runs:
    r = judge(name, cfg, c, note="shuffle control, 5 frames", predicted="see docstring")
    w = judge_phase(name, cfg, c)
    print(f"{name}: nearest {r['words']}, frames {w}, level {(r['words'] + sum(w)) / 5:.1f}, total {r['total_MB']} MB", flush=True)
