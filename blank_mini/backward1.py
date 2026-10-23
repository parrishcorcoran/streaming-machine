"""Backward from the next token, step 1. At the top read, with this position's water level fixed (one number per
position), the next token's pile is an exact sum of everything that wrote into the stream: the table row, the position
row, every gate of every layer (through its out-weights), every arrival head, the biases. Each piece's part of the
pile is exact; the parts add to the pile (checked).
Measured: everything poured into the next token's pile (sum of sizes of the parts) against what is left (the pile);
which parts carry it. Control: the same for a random other word at the same position.
PREDICTION (written first): blank -- the table row carries the pile; the gates and heads pour a lot and cancel to about
0 (over 95% cancels). GPT-2 -- 85-95% of what is poured into the next token's pile cancels; the pile comes from gates and
heads, not the table row; a random word's pile cancels even more (left near 0)."""
import torch, sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
tok = GPT2Tokenizer.from_pretrained("gpt2")
text = ("The capital of France is Paris. It is known for the Eiffel Tower, the Louvre and its cafes. "
        "Every year millions of people visit the city to see its museums and walk along the river.")
ids = tok(text, return_tensors="pt").input_ids; P = ids.shape[1]; g0 = torch.Generator().manual_seed(0)
def backward(name, m):
    m = m.double().eval(); tr = m.transformer; L = len(tr.h); H = m.config.n_head; D = m.config.n_embd
    zin, ain, tops = {}, {}, []
    hs = []
    for l, b in enumerate(tr.h):
        hs.append(b.attn.c_proj.register_forward_hook(lambda mod, i, o, l=l: zin.__setitem__(l, i[0][0])))
        hs.append(b.mlp.c_proj.register_forward_hook(lambda mod, i, o, l=l: ain.__setitem__(l, i[0][0])))
    hs.append(tr.h[-1].register_forward_hook(lambda mod, i, o: tops.append(o[0][0])))
    out = m(ids); [h.remove() for h in hs]
    x = tops[0]; mu = x.mean(1, keepdim=True); sig = ((x - mu) ** 2).mean(1, keepdim=True).add(tr.ln_f.eps).sqrt()
    nxt = out.logits[0].argmax(1); rnd = torch.randint(0, tr.wte.weight.shape[0], (P,), generator=g0)
    print(f"\n{name}: {L} layers, {D} wide, {P} positions")
    for label, words in [("the next token", nxt), ("CONTROL: a random word", rnd)]:
        Wbar = tr.wte.weight.mean(0)                                 # the average word: what every word's pile shares
        E = (tr.wte.weight[words] - Wbar) * tr.ln_f.weight           # (P, D): this word's lead over the average word
        v = (E - E.mean(1, keepdim=True)) / sig                      # what one unit on each lane is worth to this word's pile
        parts = {"table row": (tr.wte(ids)[0] * v).sum(1, keepdim=True),
                 "position row": (tr.wpe(torch.arange(P)) * v).sum(1, keepdim=True)}
        gates, heads, biases = [], [], []
        for l, b in enumerate(tr.h):
            Wm = b.mlp.c_proj.weight                                 # (gates, D)
            gates.append(ain[l] * (v @ Wm.T))                        # each gate's part: its output x its out-weights . v
            Wo = b.attn.c_proj.weight; z = zin[l]; hd = D // H
            heads.append(torch.stack([(z[:, h*hd:(h+1)*hd] @ Wo[h*hd:(h+1)*hd]) .mul(v).sum(1) for h in range(H)], 1))
            biases.append(((b.mlp.c_proj.bias + b.attn.c_proj.bias) * v).sum(1, keepdim=True))
        parts["gates"] = torch.cat(gates, 1); parts["arrival heads"] = torch.cat(heads, 1); parts["biases"] = torch.cat(biases, 1)
        fixed = (tr.ln_f.bias * (tr.wte.weight[words] - Wbar)).sum(1)
        pile = sum(p.sum(1) for p in parts.values()) + fixed
        ref = out.logits[0].gather(1, words[:, None])[:, 0] - out.logits[0].mean(1)
        if label == "the next token":
            print(f"  the last row's shared part (same for every word): {out.logits[0].mean(1).abs().median():.2f}; "
                  f"the next token's lead over it: {ref.median():.2f}")
        assert torch.allclose(pile, ref, atol=1e-8), (pile - ref).abs().max()
        allp = torch.cat(list(parts.values()), 1)
        poured = allp.abs().sum(1); left = (allp.sum(1)).abs()
        print(f"  {label} (its LEAD over the average word): {allp.shape[1]:,} parts feed it (sum checked exact). Left of all poured: "
              f"{(left/poured).median()*100:5.2f}%  -> {100-(left/poured).median()*100:5.2f}% cancels")
        for k, p in parts.items():
            print(f"      {k:14s} poured {(p.abs().sum(1)/poured).median()*100:5.1f}% of all | its net {(p.sum(1)/allp.sum(1)).median()*100:+7.1f}% of the pile"
                  f" | inside it {100 - (p.sum(1).abs()/p.abs().sum(1).clamp_min(1e-30)).median()*100:5.1f}% cancels")
backward("BLANK mini model", GPT2LMHeadModel.from_pretrained("blank_mini/model"))
backward("GPT-2", GPT2LMHeadModel.from_pretrained("gpt2"))
