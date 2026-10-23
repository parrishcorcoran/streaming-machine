"""Blank mini model vs GPT-2 (built), the same text, the streaming machine's whole numbers.
THE STONE: the first token's stream at each layer -- share on its 4 largest lanes, its size against later tokens'.
WHAT CANCELLED, two places:
  in each pile (gate): everything poured (|light| x |weight| over all lines) against what is left (|the pile|);
  across the layers:   each layer's push added up by size, against what the pushes add up to together.
Control: GPT-2 with each plate's weights shuffled (same values, structure gone)."""
import torch, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
HALF = 4096 * 4096 // 2 - 1
tok = GPT2Tokenizer.from_pretrained("gpt2")
text = ("The capital of France is Paris. It is known for the Eiffel Tower, the Louvre and its cafes. "
        "Every year millions of people visit the city to see its museums and walk along the river.")
ids = tok(text, return_tensors="pt").input_ids
def whole(v, axis):  # positions on the circles: one gauge per token (light) or per column (weights)
    g = v.abs().amax(axis, keepdim=True).clamp_min(1e-30) / HALF
    return torch.round(v / g)
def measure(name, m):
    m = m.double().eval(); tr = m.transformer; caps = {}
    hooks = [mod.register_forward_hook(lambda mod, i, o, k=k: caps.__setitem__(k, i[0][0]))
             for blk_i, blk in enumerate(tr.h) for k, mod in [(f"qkv.{blk_i}", blk.attn.c_attn), (f"fc.{blk_i}", blk.mlp.c_fc),
                                                              (f"mproj.{blk_i}", blk.mlp.c_proj)]]
    blocks = []
    hooks += [blk.register_forward_hook(lambda mod, i, o: blocks.append(o[0][0])) for blk in tr.h]
    out = m(ids, output_hidden_states=True); [h.remove() for h in hooks]
    H = [out.hidden_states[0][0]] + blocks                    # the stream before each layer, and at the top (before the top water level)
    L = len(tr.h)
    print(f"\n{name}  ({L} layers, {tr.wte.weight.shape[1]} wide)  next word: {tok.decode([int(out.logits[0, -1].argmax())])!r}")
    print("  THE STONE (first token's stream):")
    for l in sorted(set([1, L // 2, L])):
        s = H[l]; first = s[0].abs(); rest = s[1:].abs()
        top4 = first.topk(4).values.sum() / first.sum(); later4 = (rest.topk(4, dim=1).values.sum(1) / rest.sum(1)).mean()
        print(f"    after layer {l:2d}: {top4*100:5.1f}% on its 4 largest lanes (later tokens {later4*100:4.1f}%), "
              f"size {s[0].norm() / s[1:].norm(dim=1).mean():5.1f}x the later tokens'  (biggest lane {int(first.argmax())})")
    print("  WHAT CANCELLED IN EACH PILE (left / poured, median over gates and tokens after the first):")
    for kind in ["qkv", "fc", "mproj"]:
        row = []
        for l in sorted(set([0, L // 2, L - 1])):
            mod = dict(qkv=tr.h[l].attn.c_attn, fc=tr.h[l].mlp.c_fc, mproj=tr.h[l].mlp.c_proj)[kind]
            x = whole(caps[f"{kind}.{l}"][1:], 1); w = whole(mod.weight, 0)
            left = (x @ w).abs(); poured = x.abs() @ w.abs()
            row.append(f"layer {l}: {torch.median(left / poured.clamp_min(1))*100:4.1f}%")
        print(f"    {kind:5s} " + " | ".join(row))
    push = [H[l + 1][1:] - H[l][1:] for l in range(L)]
    each = sum(p.norm(dim=1) for p in push); together = sum(push).norm(dim=1)
    print(f"  WHAT CANCELLED ACROSS LAYERS: the {L} pushes add to {(together / each).median()*100:4.1f}% of their sizes added up; "
          f"pushes together are {(together / H[0][1:].norm(dim=1)).median():.2f}x the token as it came in")
measure("BLANK mini model, never built on", GPT2LMHeadModel.from_pretrained("blank_mini/model"))
g2 = GPT2LMHeadModel.from_pretrained("gpt2"); measure("GPT-2, built", g2)
gs = GPT2LMHeadModel.from_pretrained("gpt2"); gen = torch.Generator().manual_seed(0)
for blk in gs.transformer.h:
    for mod in [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]:
        w = mod.weight.data; mod.weight.data = w.reshape(-1)[torch.randperm(w.numel(), generator=gen)].reshape(w.shape)
measure("CONTROL: GPT-2 with every plate shuffled", gs)
