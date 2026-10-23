"""How the crude grid and the tips treat the word table and the plates, counted on the cut itself: the share of numbers
put dark (to 0), and each number's error as a share of its own size (median). Plus the row's largest number against
the row's middle-sized number (median of |x|)."""
import glob, numpy as np
from safetensors.numpy import load_file
S = load_file(glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors")[0])
def grid(R, b):
    L = 2 ** (b - 1) - 1; sc = (np.abs(R).max(1) / L).astype(np.float16).astype(np.float32)
    return np.clip(np.rint(R / sc[:, None]), -L, L) * sc[:, None]
def tips1(x, k):
    m, e = np.frexp(np.abs(x.astype(np.float64))); m = np.rint(m * 2 ** 24).astype(np.int64)
    ah = m >> 1; a3 = m + ah; c = ah ^ a3; up, ho = a3 & c, ah & c; lit = up | ho
    cnt = np.bitwise_count(lit)
    while (cnt > k).any(): s = cnt > k; lit = np.where(s, lit & (lit - 1), lit); cnt = cnt - s
    return np.sign(x) * np.ldexp(((up & lit) - (ho & lit)).astype(np.float64), e - 24)
for name, key, rowsT in [("word table", "wte.weight", False), ("plates h.5 mlp.c_fc", "h.5.mlp.c_fc.weight", True),
                         ("plates h.5 attn.c_attn", "h.5.attn.c_attn.weight", True)]:
    A = S[key]; R = (A.T if rowsT else A)[:8192].astype(np.float32)
    ratio = np.abs(R).max(1) / np.median(np.abs(R), 1)
    print(f"{name}: row's largest / row's middle-sized number, median over rows {np.median(ratio):.1f} (10%-90%: {np.percentile(ratio, 10):.1f}-{np.percentile(ratio, 90):.1f})")
    for b in (4, 5, 6):
        V = grid(R, b); nz = R != 0
        rel = np.abs(V - R)[nz] / np.abs(R)[nz]
        print(f"   grid b{b}: dark {100 * (V == 0).mean():.1f}%, error as a share of its own number: median {np.median(rel):.3f}")
    for k in (1, 2):
        V = tips1(R, k); nz = R != 0; rel = np.abs(V - R)[nz] / np.abs(R)[nz]
        print(f"   tips k{k}: dark {100 * (V == 0).mean():.1f}%, error as a share of its own number: median {np.median(rel):.3f}")
