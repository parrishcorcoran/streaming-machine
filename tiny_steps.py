"""The tiny model in steps, for the new piece (last position) with the earlier pieces standing. Step k: layers 0..k-1
collapsed into ONE block (the new piece's 768 in -> 768 out, its own gates held, the standing pieces' keys/values
folded in as offsets), applied in whole numbers; layers k..11 run as they are. Size and right next word at every step.
    python3 tiny_steps.py
"""
import json, math, numpy as np, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").double().eval(); sd = {k.replace("transformer.", ""): v for k, v in m.state_dict().items()}
refs = json.load(open("refs_many.json")); S = 5
def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
g = lambda l, k: sd[f"h.{l}.{k}"]
def standing(X, l):                                      # the earlier pieces' keys and values at layer l (they never change)
    c = X - X.mean(-1, keepdim=True); h = c * (c.var(-1, unbiased=False) + 1e-5).rsqrt()[:, None] * g(l, "ln_1.weight") + g(l, "ln_1.bias")
    _, k, v = (h @ g(l, "attn.c_attn.weight") + g(l, "attn.c_attn.bias")).split(768, -1); return k.view(-1, 12, 64), v.view(-1, 12, 64)
def layer_new(x, l, K, V, G=None):                       # the new piece through layer l; G holds its gates, else they act live
    c = x - x.mean(); s1 = (c.var(unbiased=False) + 1e-5).rsqrt() if G is None else G["s1"]
    q, k, v = ((c * s1 * g(l, "ln_1.weight") + g(l, "ln_1.bias")) @ g(l, "attn.c_attn.weight") + g(l, "attn.c_attn.bias")).split(768, -1)
    Kall, Vall = torch.cat([K, k.view(1, 12, 64)]), torch.cat([V, v.view(1, 12, 64)])
    A = torch.softmax(torch.einsum("hd,phd->hp", q.view(12, 64), Kall) / 8, -1) if G is None else G["A"]
    x = x + torch.einsum("hp,phd->hd", A, Vall).reshape(768) @ g(l, "attn.c_proj.weight") + g(l, "attn.c_proj.bias")
    c = x - x.mean(); s2 = (c.var(unbiased=False) + 1e-5).rsqrt() if G is None else G["s2"]
    f = (c * s2 * g(l, "ln_2.weight") + g(l, "ln_2.bias")) @ g(l, "mlp.c_fc.weight") + g(l, "mlp.c_fc.bias")
    share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f)) if G is None else G["share"]
    return x + (f * share) @ g(l, "mlp.c_proj.weight") + g(l, "mlp.c_proj.bias"), dict(s1=s1, A=A, s2=s2, share=share)
def read(x):
    c = x - x.mean(); top = c * (c.var(unbiased=False) + 1e-5).rsqrt() * sd["ln_f.weight"] + sd["ln_f.bias"]; return int((top @ sd["wte.weight"].T).argmax())
def whole(J):                                            # the block in whole numbers: 16 bits per number, one power of two per row
    e = torch.ceil(torch.log2(J.abs().max(1).values.clamp(min=1e-300) / 32767)); return torch.round(J / 2.0 ** e[:, None]).long(), e
MB = lambda n: n * 2 / 1e6
plate_layer = sum(sd[f"h.0.{k}"].numel() for k in ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"))
setups = []
for r in refs:
    ids = (r["ids"] + r["ref"])[:S]; X = sd["wte.weight"][ids] + sd["wpe.weight"][:S]
    st = []; xs = X.clone()
    for l in range(12):                                   # the earlier pieces, standing: each layer's keys/values, from their own run
        K, V = standing(xs, l); st.append((K[:-1], V[:-1]))
        xs = torch.stack([layer_new(xs[p], l, K[:p], V[:p])[0] for p in range(S)])
    x, G = X[-1], []
    for l in range(12): x, gl = layer_new(x, l, *st[l]); G.append(gl)
    setups.append((X[-1], st, G, read(x)))
print(f"one layer's plate: {plate_layer:,} weights = {MB(plate_layer):.1f} MB;  a collapsed block: 768 x 769 = {768 * 769:,} = {MB(768 * 769):.2f} MB")
for k in range(0, 13):
    right = 0
    for x0, st, G, true_word in setups:
        x = x0
        if k > 0:
            def block(z):
                for l in range(k): z = layer_new(z, l, *st[l], G[l])[0]
                return z
            with torch.enable_grad(): J = torch.autograd.functional.jacobian(block, x0)
            off = block(torch.zeros_like(x0)); Ji, e = whole(J)
            x = ((Ji @ torch.round(x0 * 2 ** 16).long()).double() * 2.0 ** e / 2 ** 16) + off      # the collapsed block, whole numbers
        for l in range(k, 12): x = layer_new(x, l, *st[l])[0]
        right += read(x) == true_word
    size = (MB(768 * 769) if k else 0) + (12 - k) * MB(plate_layer)
    print(f"  step {k:2d}: layers 0-{k - 1 if k else '-'} collapsed into one block, {12 - k:2d} layers as they are:  {size:7.1f} MB of layers   right next word {right} / 30", flush=True)
