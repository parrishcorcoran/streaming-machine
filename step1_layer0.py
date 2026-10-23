"""Baby step 1: ONE layer. Layer 0 collapsed into one layer (5 positions x 768 in -> 5 x 768 out, offsets folded in as
the always-on channel); layers 1-11 and the read left exactly as they are. Layer 0 replaced by: its own collapsed
layer (control), the base (weights only, gates open), the base + one turn per circle (output, input, both; read off
from the input's own). Scored by the right next word only.      python3 step1_layer0.py
"""
import json, numpy as np, torch
exec(open("base_layer_turns.py").read().split("def collapsed(")[0])          # sd, gelu, run() machinery
def layer(x, l, G=None):
    S = x.shape[0]; g = lambda k: sd[f"h.{l}.{k}"]
    c = x - x.mean(-1, keepdim=True); s1 = (c.var(-1, unbiased=False) + 1e-5).rsqrt() if G is None else G["s1"]
    h = c * s1[:, None] * g("ln_1.weight") + g("ln_1.bias")
    q, k, v = (h @ g("attn.c_attn.weight") + g("attn.c_attn.bias")).split(768, -1)
    q, k, v = [t.view(S, 12, 64).transpose(0, 1) for t in (q, k, v)]
    A = torch.softmax((q @ k.transpose(1, 2) / 8).masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), -1e300), -1) if G is None else G["A"]
    x = x + (A @ v).transpose(0, 1).reshape(S, 768) @ g("attn.c_proj.weight") + g("attn.c_proj.bias")
    c = x - x.mean(-1, keepdim=True); s2 = (c.var(-1, unbiased=False) + 1e-5).rsqrt() if G is None else G["s2"]
    f = (c * s2[:, None] * g("ln_2.weight") + g("ln_2.bias")) @ g("mlp.c_fc.weight") + g("mlp.c_fc.bias")
    share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f)) if G is None else G["share"]
    return x + (f * share) @ g("mlp.c_proj.weight") + g("mlp.c_proj.bias"), dict(s1=s1, A=A, s2=s2, share=share)
def rest(x):                                            # layers 1-11 and the read, exactly as they are
    for l in range(1, 12): x, _ = layer(x, l)
    c = x - x.mean(-1, keepdim=True); top = (c * (c.var(-1, unbiased=False) + 1e-5).rsqrt()[:, None] * sd["ln_f.weight"] + sd["ln_f.bias"])[-1]
    return int((top @ sd["wte.weight"].T).argmax())
def collapsed0(x0, G):
    with torch.enable_grad(): J = torch.autograd.functional.jacobian(lambda x: layer(x, 0, G)[0], x0, vectorize=True).reshape(S * 768, S * 768)
    return torch.cat([J, layer(torch.zeros_like(x0), 0, G)[0].reshape(-1, 1)], 1)
S = 5; ones = lambda *s: torch.ones(*s, dtype=torch.float64)
OPEN0 = dict(s1=ones(S), A=(torch.tril(ones(S, S)) / torch.arange(1, S + 1, dtype=torch.float64)[:, None])[None].expand(12, S, S), s2=ones(S), share=ones(S, 3072))
base = collapsed0(torch.zeros(S, 768, dtype=torch.float64), OPEN0)
right = {k: 0 for k in ("own (control)", "base", "base + turns at output", "base + turns at input", "base + turns at both")}; off = {k: [] for k in right}
for r in refs:
    ids = (r["ids"] + r["ref"])[:S]; x0 = sd["wte.weight"][ids] + sd["wpe.weight"][:S]
    true_word = rest(layer(x0, 0)[0]); _, G = layer(x0, 0); own = collapsed0(x0, G)
    xin = torch.cat([x0.reshape(-1), ones(1)]); y_own = own @ xin
    V = {"own (control)": own, "base": base}
    for mode in ("base + turns at output", "base + turns at input", "base + turns at both"):
        out, inn = ones(S * 768), ones(S * 768 + 1)
        for _ in range(60):
            if mode != "base + turns at input": Bc = base * inn[None]; out = (own * Bc).sum(1) / (Bc * Bc).sum(1).clamp(min=1e-300)
            if mode != "base + turns at output": Br = base * out[:, None]; inn = (own * Br).sum(0) / (Br * Br).sum(0).clamp(min=1e-300)
        V[mode] = out[:, None] * base * inn[None]
    for k, J in V.items():
        y = J @ xin; right[k] += rest(y.reshape(S, 768)) == true_word; off[k].append(float((y - y_own).norm() / y_own.norm()))
print(f"layer 0 alone collapsed; layers 1-11 as they are; {len(refs)} inputs of {S} pieces")
for k in right: print(f"  {k:24s}: right next word {right[k]:2d} / {len(refs)}   layer 0's output off by {100 * np.median(off[k]):8.3f}% (median)")
