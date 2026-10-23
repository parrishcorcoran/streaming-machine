"""Is each input's collapsed layer = the base collapsed layer (weights only, every bend, gauge and arrival open) with
one turn per circle at the top (768) and one per input channel (3,840 + the always-on channel)?
Collapse: GPT-2's 12 layers into one layer from the input chords (5 positions x 768, positions included) to the top
(768, after the final gauge), the offsets folded in as one always-on input channel. Turns read off by bringing the base
onto the input's own collapsed layer. Run in whole numbers; scored by accuracy (the right next word) only.
    python3 base_layer_turns.py
"""
import json, math, numpy as np, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").double().eval(); sd = {k.replace("transformer.", ""): v for k, v in m.state_dict().items()}
refs = json.load(open("refs_many.json")); B2 = np.load("base2_gpt2.npz"); WTE = B2["wte"].astype(np.int64)      # the table, whole numbers
def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def run(x, G=None):
    """x: (S,768) input chords. G=None: the real run (returns its gates); else every gate frozen to G."""
    S = x.shape[0]; rec = []
    for l in range(12):
        g = lambda k: sd[f"h.{l}.{k}"]
        c = x - x.mean(-1, keepdim=True); s1 = (c.var(-1, unbiased=False) + 1e-5).rsqrt() if G is None else G[l]["s1"]
        h = c * s1[:, None] * g("ln_1.weight") + g("ln_1.bias")
        q, k, v = (h @ g("attn.c_attn.weight") + g("attn.c_attn.bias")).split(768, -1)
        q, k, v = [t.view(S, 12, 64).transpose(0, 1) for t in (q, k, v)]
        A = torch.softmax((q @ k.transpose(1, 2) / 8).masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), -1e300), -1) if G is None else G[l]["A"]
        x = x + (A @ v).transpose(0, 1).reshape(S, 768) @ g("attn.c_proj.weight") + g("attn.c_proj.bias")
        c = x - x.mean(-1, keepdim=True); s2 = (c.var(-1, unbiased=False) + 1e-5).rsqrt() if G is None else G[l]["s2"]
        f = (c * s2[:, None] * g("ln_2.weight") + g("ln_2.bias")) @ g("mlp.c_fc.weight") + g("mlp.c_fc.bias")
        share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f)) if G is None else G[l]["share"]
        x = x + (f * share) @ g("mlp.c_proj.weight") + g("mlp.c_proj.bias")
        rec.append(dict(s1=s1, A=A, s2=s2, share=share))
    c = x - x.mean(-1, keepdim=True); sf = (c.var(-1, unbiased=False) + 1e-5).rsqrt() if G is None else G[12]
    top = (c * sf[:, None] * sd["ln_f.weight"] + sd["ln_f.bias"])[-1]
    return top, rec + [sf]
def collapsed(x0, G):                                   # one layer, offsets folded in as the always-on channel: 768 x (S*768 + 1)
    with torch.enable_grad(): J = torch.autograd.functional.jacobian(lambda x: run(x, G)[0], x0, vectorize=True).reshape(768, -1)
    return torch.cat([J, run(torch.zeros_like(x0), G)[0][:, None]], 1)
S = 5
OPEN = [dict(s1=torch.ones(S, dtype=torch.float64), A=torch.tril(torch.ones(S, S, dtype=torch.float64)) / torch.arange(1, S + 1, dtype=torch.float64)[:, None],
             s2=torch.ones(S, dtype=torch.float64), share=torch.ones(S, 3072, dtype=torch.float64)) for _ in range(12)] + [torch.ones(S, dtype=torch.float64)]
for l in range(12): OPEN[l]["A"] = OPEN[l]["A"][None].expand(12, S, S)
base = collapsed(torch.zeros(S, 768, dtype=torch.float64), OPEN)                   # from the weights alone, every gate open
def whole(J):                                            # each circle at the top as 16-bit whole numbers with its own power of two
    e = torch.ceil(torch.log2(J.abs().max(1).values.clamp(min=1e-300) / 32767)); return torch.round(J / 2.0 ** e[:, None]).long(), e
def word(J, xin):                                        # the input through a collapsed layer, then the table: whole numbers
    Ji, e = whole(J); xi = torch.round(xin * 2 ** 16).long()
    top = ((Ji @ xi).double() * 2.0 ** e).round().long()                             # the top at the stream's scale (2^-16)
    return int((top.numpy() @ WTE.T).argmax()), top
right = {k: 0 for k in ("own", "base", "turns at the top only", "turns at the input only", "turns at both")}; close = {k: [] for k in right}
for r in refs:
    ids = (r["ids"] + r["ref"])[:S]; x0 = sd["wte.weight"][ids] + sd["wpe.weight"][:S]
    true_top, G = run(x0); true_word = int((true_top @ sd["wte.weight"].T).argmax())
    own = collapsed(x0, G); xin = torch.cat([x0.reshape(-1), torch.ones(1, dtype=torch.float64)])
    out = torch.ones(768, dtype=torch.float64); inn = torch.ones(S * 768 + 1, dtype=torch.float64)
    variants = {"own": own, "base": base}
    for mode in ("turns at the top only", "turns at the input only", "turns at both"):
        out[:] = 1; inn[:] = 1
        for _ in range(60):                               # read the turns off: bring the base onto this input's own collapsed layer
            if mode != "turns at the input only":
                Bc = base * inn[None, :]; out = (own * Bc).sum(1) / (Bc * Bc).sum(1).clamp(min=1e-300)
            if mode != "turns at the top only":
                Br = base * out[:, None]; inn = (own * Br).sum(0) / (Br * Br).sum(0).clamp(min=1e-300)
        variants[mode] = out[:, None] * base * inn[None, :]
    _, own_top = word(own, xin)
    for k, J in variants.items():
        w, t = word(J, xin); right[k] += (w == true_word)
        close[k].append(float((t - own_top).double().norm() / own_top.double().norm()))
print(f"{len(refs)} inputs of {S} pieces; right next word, and how far the top lands from the input's own collapsed layer:")
for k in right: print(f"  {k:24s}: {right[k]:2d} / {len(refs)}   top off by {100 * np.median(close[k]):7.2f}% (median)")
