"""Candidates for the all-seeds sign scan: the first 2,048 values of every Qwen3-0.6B plate (word table + 196 body
plates), as stored and as transposed. Plus 6 planted checks (a known seed's draw mixed into a real plate at a set cos,
one per draw kind) and 16 shuffled controls (a real candidate's signs in shuffled order).
Writes cands.bin: per candidate 32 u64 sign bits (1 = negative) then 32 u64 mask bits (1 = value not zero); cands.txt names."""
import numpy as np, torch, glob
from safetensors import safe_open
N = 2048
path = glob.glob(f"{__import__('os').path.expanduser('~')}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
f = safe_open(path, framework="pt")
names = [n for n in f.keys() if n != "lm_head.weight" and len(f.get_slice(n).get_shape()) == 2]
names.sort(key=lambda n: (0, 0, n) if "embed" in n else (1, int(n.split(".")[2]), n))
vals, labels = [], []
for n in names:
    s = f.get_slice(n); R, C = s.get_shape()
    st = s[0:-(-N // C), :].float().reshape(-1)[:N]                 # stored layout, row after row
    tr = s[:, 0:-(-N // R)].float().T.reshape(-1)[:N]               # transposed: column after column
    vals += [st.numpy(), tr.numpy()]; labels += [f"{n}|stored", f"{n}|transposed"]
print(len(names), "plates ->", len(vals), "real candidates")

def draw(seed, conv):
    torch.manual_seed(seed)
    if conv == "E": x = torch.empty(2 * N, dtype=torch.float64)[::2]; x.normal_(); return x.numpy().astype(np.float64)
    if conv == "H": x = torch.empty(N, dtype=torch.float32); torch.nn.init.trunc_normal_(x, std=0.02); return x.double().numpy()
    if conv == "I": return torch.empty(N, dtype=torch.float64).uniform_(-1, 1).numpy()
    dt = {"A": torch.bfloat16, "B": torch.float16, "C": torch.float32, "D": torch.float64}[conv]
    return torch.empty(N, dtype=dt).normal_().double().numpy()
PLANT = [("A", 123456789, 0.46, "model.layers.0.self_attn.q_proj.weight|stored"),
         ("B", 987654321, 0.46, "model.layers.3.self_attn.k_proj.weight|stored"),
         ("C", 2718281828, 0.46, "model.embed_tokens.weight|stored"),
         ("C", 1618033988, 0.30, "model.layers.10.mlp.down_proj.weight|stored"),
         ("D", 3141592653, 0.46, "model.layers.20.self_attn.o_proj.weight|transposed"),
         ("E", 4000000007, 0.46, "model.layers.27.mlp.up_proj.weight|stored"),
         ("H", 2000000011, 0.46, "model.layers.14.mlp.gate_proj.weight|stored"),
         ("I", 3000000019, 0.46, "model.layers.7.self_attn.v_proj.weight|transposed")]
for conv, seed, rho, base in PLANT:
    w = vals[labels.index(base)].astype(np.float64); d = draw(seed, conv)
    m = 1248 if conv in "DEI" else N                                  # D and E are scanned on their first 1,248 values
    d[m:] = 0; d = d - (d @ w) / (w @ w) * w * (np.arange(N) < m)
    p = rho * d / np.linalg.norm(d[:m]) * np.sqrt(m) + np.sqrt(1 - rho**2) * w / np.linalg.norm(w[:m]) * np.sqrt(m)
    assert np.all(np.isfinite(p)), "planted values not finite"
    c_d = float(p[:m] @ d[:m] / np.linalg.norm(p[:m]) / np.linalg.norm(d[:m])); c_w = float(p[:m] @ w[:m] / np.linalg.norm(p[:m]) / np.linalg.norm(w[:m]))
    print(f"plant {conv} {seed}: cos with the draw {c_d:.3f}, with the real plate {c_w:.3f}, sign agreement with the draw {np.mean((p[:m]<0)==(d[:m]<0)):.3f}")
    vals.append(p); labels.append(f"PLANT|{conv}|{seed}|cos{rho}|{base}")
rng = np.random.default_rng(20261009)
for i in rng.choice(len(names) * 2, 16, replace=False):
    vals.append(rng.permutation(vals[i])); labels.append(f"SHUFFLED|{labels[i]}")
out = np.zeros((len(vals), 64), np.uint64)
for k, v in enumerate(vals):
    neg = np.packbits((v < 0).astype(np.uint8), bitorder="little").view(np.uint64)
    nz = np.packbits((v != 0).astype(np.uint8), bitorder="little").view(np.uint64)
    out[k, :32], out[k, 32:] = neg, nz
out.tofile("cands.bin"); open("cands.txt", "w").write("\n".join(labels) + "\n")
print(len(vals), "candidates written; values exactly 0 in the real ones:", int(sum((v == 0).sum() for v in vals[:len(names)*2])))
