"""Turn the base into zero, exactly. Every share stays where the full model put it at this token (each water level,
each gate's share s = a/H, each head's arrival weights). With the shares fixed, every word's pile is an exact sum of
what each source delivers: table rows, position rows, biases (checked: the sum of all sources' deliveries = the real
piles). The base (the average table row, in every token) is carried up alone through the same shares; what it delivers
to every word is taken out. Control: a random vector of the same size carried up the same way.
PREDICTION (written first): the base's delivery taken out keeps the top word at 97-99.5% of positions."""
import json, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").double().eval(); tr = m.transformer
L, D, H = len(tr.h), m.config.n_embd, m.config.n_head; hd = D // H
W = tr.wte.weight; base = W.mean(0)
g = torch.Generator().manual_seed(0); rnd = torch.randn(D, generator=g); rnd = rnd / rnd.norm() * base.norm()
def carry(ids):
    """the full run, its shares, and a linear carrier that pushes any input through those same shares"""
    cap = {k: {} for k in ["pre", "mid", "a", "H"]}; tops = []; hs = []
    for l, b in enumerate(tr.h):
        hs += [b.ln_1.register_forward_hook(lambda mod, i, o, l=l: cap["pre"].__setitem__(l, i[0][0])),
               b.ln_2.register_forward_hook(lambda mod, i, o, l=l: cap["mid"].__setitem__(l, i[0][0])),
               b.mlp.c_fc.register_forward_hook(lambda mod, i, o, l=l: cap["H"].__setitem__(l, o[0])),
               b.mlp.c_proj.register_forward_hook(lambda mod, i, o, l=l: cap["a"].__setitem__(l, i[0][0]))]
    hs.append(tr.h[-1].register_forward_hook(lambda mod, i, o: tops.append(o[0][0])))
    out = m(torch.tensor([ids]), output_attentions=True); [h.remove() for h in hs]
    sd = lambda x, eps: ((x - x.mean(1, keepdim=True)) ** 2).mean(1, keepdim=True).add(eps).sqrt()
    S1 = [sd(cap["pre"][l], tr.h[l].ln_1.eps) for l in range(L)]; S2 = [sd(cap["mid"][l], tr.h[l].ln_2.eps) for l in range(L)]
    SF = sd(tops[0], tr.ln_f.eps); A = [out.attentions[l][0] for l in range(L)]
    s = [torch.where(cap["H"][l].abs() > 1e-12, cap["a"][l] / cap["H"][l], torch.zeros(1)) for l in range(L)]
    def push(x, with_biases):
        for l, b in enumerate(tr.h):
            v = ((x - x.mean(1, keepdim=True)) / S1[l] * b.ln_1.weight + (b.ln_1.bias if with_biases else 0)) @ b.attn.c_attn.weight[:, 2*D:]
            if with_biases: v = v + b.attn.c_attn.bias[2*D:]
            z = torch.cat([A[l][h] @ v[:, h*hd:(h+1)*hd] for h in range(H)], 1)
            x = x + z @ b.attn.c_proj.weight + (b.attn.c_proj.bias if with_biases else 0)
            Hl = ((x - x.mean(1, keepdim=True)) / S2[l] * b.ln_2.weight + (b.ln_2.bias if with_biases else 0)) @ b.mlp.c_fc.weight
            if with_biases: Hl = Hl + b.mlp.c_fc.bias
            x = x + (s[l] * Hl) @ b.mlp.c_proj.weight + (b.mlp.c_proj.bias if with_biases else 0)
        return ((x - x.mean(1, keepdim=True)) / SF * tr.ln_f.weight + (tr.ln_f.bias if with_biases else 0)) @ W.T
    full = out.logits[0]
    rebuilt = push(W[ids] + tr.wpe.weight[:len(ids)], True)
    assert torch.allclose(rebuilt, full, atol=1e-6), (rebuilt - full).abs().max()       # every source's delivery adds to the real piles
    return full, push
wins = [w["ids"] for w in json.load(open("judge5_texts.json"))]; refs = json.load(open("refs_many.json"))
res = {k: [0, 0, 0, 0] for k in ["BASE TO ZERO (exact)", "CONTROL: random of the same size"]}
for ids in wins + [r["ids"] + r["ref"] for r in refs]:
    full, push = carry(ids); P = len(ids)
    for k, vec in [("BASE TO ZERO (exact)", base), ("CONTROL: random of the same size", rnd)]:
        cut = full - push(vec[None].expand(P, D).clone(), False)
        r = res[k]; r[0] += int((cut.argmax(1) == full.argmax(1)).sum()); r[1] += int((cut.topk(5, 1).indices == full.argmax(1)[:, None]).any(1).sum()); r[2] += P
        if len(ids) != 24 or ids in [rr["ids"] + rr["ref"] for rr in refs]:
            rr = [x for x in refs if x["ids"] + x["ref"] == ids]
            if rr: n0 = len(rr[0]["ids"]); r[3] += int((cut.argmax(1)[n0 - 1:-1] == torch.tensor(rr[0]["ref"])).sum())
print("every piece's delivery checked: all sources together rebuild the real piles exactly, at every position")
for k, r in res.items():
    print(f"  {k:34s} same top word {r[0]/r[2]*100:6.2f}% | full word in its top 5 {r[1]/r[2]*100:6.2f}% | 30 texts: {r[3]}/480 words")
