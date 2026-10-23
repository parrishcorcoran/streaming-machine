"""Every row of layer 0 up_proj (3,072 rows): how much 8-bit and 4-bit change each row's length and turn its direction,
and how many weights changed side (sign flipped) or became 0."""
import numpy as np
exec(open("circles.py").read().split("q8, c8 = q8_0(row)")[0].replace("t[0:1]", "t[:]").replace("[0].astype", ".astype"))
W = row.reshape(-1, 1024); print(W.shape)
for name, fn in (("8-bit", q8_0), ("4-bit", q4_0)):
    Q = np.stack([fn(r)[0] for r in W]).astype(np.float64); X = W.astype(np.float64)
    Lx, Lq = np.linalg.norm(X, axis=1), np.linalg.norm(Q, axis=1)
    dl = 100 * np.abs(Lq / Lx - 1)
    ang = np.degrees(np.arccos(np.clip(np.sum(X * Q, 1) / (Lx * Lq), -1, 1)))
    flip = int(np.sum(np.sign(X) * np.sign(Q) < 0)); zero = int(np.sum((Q == 0) & (X != 0)))
    print(f"{name}: length changed mean {dl.mean():.3f}% (largest {dl.max():.3f}%)   direction turned mean {ang.mean():.3f} deg "
          f"(largest {ang.max():.3f})   weights that changed side {flip} of {X.size}; became 0 {zero} ({100*zero/X.size:.1f}%)")
