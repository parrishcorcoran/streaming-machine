"""Compares check_fwd.py's forward numbers with halves.py's reverse numbers (turns/halves_part_*.npz), and recomputes the
report's step-2 and step-3 figures from the forward numbers alone.   python3 turns/check_fwd_compare.py"""
import numpy as np, re
D = "/Users/abundancemachine/streaming-machine/turns"
A = [np.load(f"{D}/check_fwd_00_10.npz"), np.load(f"{D}/check_fwd_10_30.npz")]
s2 = np.concatenate([A[0]["s2"][:10], A[1]["s2"][10:]]); s3 = np.concatenate([A[0]["s3"][:10], A[1]["s3"][10:]])
lead = np.concatenate([A[0]["lead"][:10], A[1]["lead"][10:]]).astype(float); words = np.concatenate([A[0]["words"][:10], A[1]["words"][10:]])
H = [np.load(f"{D}/halves_part_00_15.npz"), np.load(f"{D}/halves_part_15_30.npz")]
h2 = np.concatenate([H[0]["s2"][:15], H[1]["s2"][15:]]); h3 = np.concatenate([H[0]["s3"][:15], H[1]["s3"][15:]])
ht = np.concatenate([H[0]["target"][:15], H[1]["target"][15:]]); hpos = np.concatenate([H[0]["s3_pos"][:15], H[1]["s3_pos"][15:]])
hw = np.stack([np.concatenate([H[0][k][:15], H[1][k][15:]]) for k in ("ans", "second", "rnd")], 1)
print("same words (answer, 2nd, random) 30/30:", bool((hw == words).all()), "; same targets:", bool((ht == lead).all()))
def rel(a, b): return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
r2 = rel(s2[:, :, :2, :], h2[:, :, :, :3]); print(f"STEP 2 forward vs reverse, straight/sym/part, 30 texts x 12 layers x 2 rows: max relative gap {r2.max():.2e}")
mp = {0: 0, 1: 1, 2: 2, 3: 6, 4: 7, 5: 8, 6: 4, 7: 3}           # mine: entry symN symO fcB fcR arr mp top  -> theirs
for w in range(3):
    g = max(rel(s3[:, w, k], h3[:, w, j]).max() for k, j in mp.items() if j != 3)
    gt = np.abs(s3[:, w, 7] - h3[:, w, 3]).max()
    gp = rel(s3[:, w, 8:13], hpos[:, w, :5]).max()
    print(f"STEP 3 row {w}: max relative gap over the buckets {g:.2e}; top constant max abs gap {gt:.2e} units; per-token entry max rel gap {gp:.2e}")
# exact text 0 fraction from the two logs
a = open(f"{D}/check_fwd_00_10.log").read(); b = open(f"{D}/halves_00_15.log").read()
fa = re.search(r"text 0 all layers: fixed entry part = (-?\d+)/(\d+)", a); fb = re.search(r"in whole numbers over 0 places: fixed (-?\d+)/(\d+)", b)
print("text 0 fixed entry part, exact fraction, forward == reverse log:", fa.groups() == fb.groups(), f"({len(fa.group(1))} digits)")
print("\nSTEP 2 from the forward numbers (lead = answer - 2nd), median [10th, 90th] over 30 texts")
for l in range(12):
    x = s2[:, l, 0] / lead[:, 0:1]
    o = int((np.sign(x[:, 0]) != np.sign(x[:, 1])).sum())
    print(f"  layer {l:2d}: straight/lead {np.median(x[:,0]):7.3f} [{np.percentile(x[:,0],10):7.3f},{np.percentile(x[:,0],90):7.3f}]  sym/lead {np.median(x[:,1]):7.3f}"
          f"  part/lead {np.median(x[:,2]):7.3f}  |str|/(|str|+|sym|) {np.median(np.abs(x[:,0])/(np.abs(x[:,0])+np.abs(x[:,1]))):.3f}  opposite {o}/30")
x = s2[:, :, 0] / lead[:, None, 0:1]
print(f"  all 360: |sym|>|str| {int((np.abs(x[...,1])>np.abs(x[...,0])).sum())}; opposite {int((np.sign(x[...,0])!=np.sign(x[...,1])).sum())};"
      f" both > |part| {int(((np.abs(x[...,0])>np.abs(x[...,2]))&(np.abs(x[...,1])>np.abs(x[...,2]))).sum())}; median |str| {np.median(np.abs(x[...,0])):.3f}"
      f" |sym| {np.median(np.abs(x[...,1])):.3f} |part| {np.median(np.abs(x[...,2])):.3f}")
x = s2[:, :, 1] / lead[:, None, 1:2]
print(f"  random lead, all 360: |sym|>|str| {int((np.abs(x[...,1])>np.abs(x[...,0])).sum())}; opposite {int((np.sign(x[...,0])!=np.sign(x[...,1])).sum())};"
      f" median |str| {np.median(np.abs(x[...,0])):.3f} |sym| {np.median(np.abs(x[...,1])):.3f} |part| {np.median(np.abs(x[...,2])):.3f}")
print("\nSTEP 3 from the forward numbers: each part / target, median [10th, 90th], |.| median")
nm = ["entry (fixed machine)", "sym newest", "sym older", "fc bias", "fc roundings", "arrival consts", "mproj consts", "top const"]
for w, wn in ((0, "LEAD"), (1, "RANDOM LEAD"), (2, "SCORE")):
    ld = lead[:, w]; print(f"  {wn}")
    for k, n in enumerate(nm):
        y = s3[:, w, k] / ld; print(f"    {n:22s} {np.median(y):11.4g} [{np.percentile(y,10):11.4g},{np.percentile(y,90):11.4g}]  |.| {np.median(np.abs(y)):11.4g}")
    e = s3[:, w, 0] / ld; sy = (s3[:, w, 1] + s3[:, w, 2]) / ld
    print(f"    sym all {np.median(sy):.4g}; opposite sign entry vs sym {int((np.sign(e)!=np.sign(sy)).sum())}/30; entry < 1/3 in {int((e<1/3).sum())}/30;"
          f" in [0,1/3) {int(((e>=0)&(e<1/3)).sum())}/30; |entry| > 2 in {int((np.abs(e)>2).sum())}/30; log10|entry| median {np.median(np.log10(np.abs(e))):.2f}")
    print("    per token position: " + ", ".join(f"{np.median(s3[:, w, 8+p]/ld):.4g}" for p in range(5)))
    tot = s3[:, w, :8].sum(1) / ld; print(f"    sum of all parts / target: min {tot.min():.12f} max {tot.max():.12f} (floats of exact parts)")
