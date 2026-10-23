"""THE TINY MODEL FOR QWEN3-0.6B: all 28 depths collapsed into ONE layer (as basics' tiny_waveform.py did for GPT-2).
Settled once from the prompt: every gauge, every gated bend's share and every arrival frozen; what is left is pure
superposition, input chords -> the top: one layer of notes. Then the table reads it. Same word as fp32?
    python3 tiny_qwen.py
"""
import json, torch, transformers.models.qwen3.modeling_qwen3 as q3
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float64, attn_implementation="eager").eval()
refs = json.load(open("refs_qwen.json")); E = m.model.embed_tokens.weight
orig_attn = q3.eager_attention_forward
def tiny(ids):
    S = len(ids); x0 = E[ids]
    share, landed, scale = {}, [], {}
    hooks = []
    for k, L in enumerate(m.model.layers):
        hooks.append(L.mlp.act_fn.register_forward_hook(lambda mod, i, o, k=k: share.__setitem__(k, o.clone())))      # the whole gate value, frozen
        for nm in ("input_layernorm", "post_attention_layernorm"):
            hooks.append(getattr(L, nm).register_forward_hook(lambda mod, i, o, key=(nm, k): scale.__setitem__(key, torch.rsqrt(i[0].pow(2).mean(-1, keepdim=True) + mod.variance_epsilon).clone())))
    hooks.append(m.model.norm.register_forward_hook(lambda mod, i, o: scale.__setitem__(("norm", 0), torch.rsqrt(i[0].pow(2).mean(-1, keepdim=True) + mod.variance_epsilon).clone())))
    out = m(inputs_embeds=x0[None], output_attentions=True); landed = [a.clone() for a in out.attentions]; real = int(out.logits[0, -1].argmax())
    for h in hooks: h.remove()
    # frozen: each gauge a fixed scale, each gated bend a fixed share, each arrival fixed weights -> pure superposition
    saved = []
    for k, L in enumerate(m.model.layers):
        saved.append((L.mlp.act_fn.forward, L.input_layernorm.forward, L.post_attention_layernorm.forward))
        L.mlp.act_fn.forward = (lambda k: (lambda x: share[k] + 0 * x))(k)          # gate x up -> fixed gain x up: one superposition
        L.input_layernorm.forward = (lambda key, w: (lambda x: x * scale[key] * w))(("input_layernorm", k), L.input_layernorm.weight)
        L.post_attention_layernorm.forward = (lambda key, w: (lambda x: x * scale[key] * w))(("post_attention_layernorm", k), L.post_attention_layernorm.weight)
    nf = m.model.norm.forward; m.model.norm.forward = lambda x: x * scale[("norm", 0)] * m.model.norm.weight
    layer = [0]
    def fixed_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        w = landed[layer[0] % len(landed)].to(v.dtype); layer[0] += 1
        vv = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        return torch.matmul(w, vv).transpose(1, 2).contiguous(), w
    q3.eager_attention_forward = fixed_attn
    def F(x):
        layer[0] = 0; return m.model(inputs_embeds=x[None]).last_hidden_state[0, -1]
    with torch.enable_grad(): notes = torch.autograd.functional.jacobian(F, x0, vectorize=True).reshape(1024, S * 1024)
    offs = F(torch.zeros_like(x0))
    q3.eager_attention_forward = orig_attn; m.model.norm.forward = nf
    for L, (a, f1, f2) in zip(m.model.layers, saved): L.mlp.act_fn.forward, L.input_layernorm.forward, L.post_attention_layernorm.forward = a, f1, f2
    return notes, offs, x0.reshape(-1), real
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
for r in refs[:5]:
    notes, offs, strike, real = tiny(r["ids"])
    top = notes @ strike + offs; word = int(m.lm_head(top).argmax())
    print(f"  {r['prompt']!r:28s} tiny model: {tok.decode([word])!r:12s} fp32: {tok.decode([real])!r:12s} {'same' if word == real else 'DIFFERS'}   (one layer, {notes.numel():,} notes)")

# ---- the tiny model in base 2: every note a 16-bit whole number (one power of two per row), the strike in whole
# numbers, the top read by the whole-number race (QuantumQwen3's head plate)
import numpy as np
B2 = np.load("qwen_base2.npz"); EMB = B2["embed"].astype(np.int64)
def race(v): return int(np.argmax(v))
print("\nbase 2:")
for r in refs[:5]:
    notes, offs, strike, real = tiny(r["ids"])
    N = notes.numpy(); mx = np.abs(N).max(1); ne = np.ceil(np.log2(np.maximum(mx, 1e-30) / 32767)).astype(np.int64)
    n16 = np.round(N / 2.0 ** ne[:, None]).astype(np.int64)                         # the notes: 16-bit whole numbers
    s16 = B2["embed"][r["ids"]].reshape(-1).astype(np.int64)                         # the strike: whole numbers at 2^-16
    top = (n16 @ s16).astype(np.float64)                                             # one layer fires: whole-number piles
    top_int = np.round(top * 2.0 ** ne).astype(np.int64) + np.round(offs.numpy() * 2 ** 16).astype(np.int64)   # at the stream's scale
    t16 = top_int >> max(int(np.abs(top_int).max()).bit_length() - 15, 0)            # onto 16 bits for the table
    lg = t16 @ EMB.T                                                                 # the table reads it: the same table as the bottom, whole numbers
    word = race(lg)
    print(f"  {r['prompt']!r:28s} tiny model, base 2: {tok.decode([word])!r:12s} fp32: {tok.decode([real])!r:12s} {'same' if word == real else 'DIFFERS'}")
