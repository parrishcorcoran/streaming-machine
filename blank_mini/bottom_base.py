"""All the way down to the table, signed, one pass. At this token every share is fixed (each gate's s = a/H, each head's
arrival weights, each water level), so the next token's lead (over the average word) is an exact sum over the sources:
every table number of every token in the text, every position row number, every bias. One pass back gives each
source's worth; table number x its worth = what that table number delivers to the next token after all cancelling.
Checked: the sources' parts add to the lead exactly. Control: a random word.
BASE SPLIT: every table row = the base (the average row, in every token) + the token's own part. The base, turned by
this context and carried up, is always there; it is taken out. What is left is what each token's own part delivers.
PREDICTION (written first): the base's part is large where the next word is common and small where it is specific;
the own parts carry the specific words; inside the own parts, number by number, about 80% cancels."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok("The capital of France is Paris. It is known for the Eiffel", return_tensors="pt").input_ids; P = ids.shape[1]
def bottom(name, path):
    m = GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager").double().eval(); tr = m.transformer
    L, D, H = len(tr.h), m.config.n_embd, m.config.n_head; hd = D // H
    cap = {k: {} for k in ["pre", "mid", "a", "H"]}; tops = []; hs = []
    for l, b in enumerate(tr.h):
        hs += [b.ln_1.register_forward_hook(lambda mod, i, o, l=l: cap["pre"].__setitem__(l, i[0][0])),
               b.ln_2.register_forward_hook(lambda mod, i, o, l=l: cap["mid"].__setitem__(l, i[0][0])),
               b.mlp.c_fc.register_forward_hook(lambda mod, i, o, l=l: cap["H"].__setitem__(l, o[0])),
               b.mlp.c_proj.register_forward_hook(lambda mod, i, o, l=l: cap["a"].__setitem__(l, i[0][0]))]
    hs.append(tr.h[-1].register_forward_hook(lambda mod, i, o: tops.append(o[0][0])))
    out = m(ids, output_attentions=True); [h.remove() for h in hs]
    sd = lambda x, eps: ((x - x.mean(1, keepdim=True)) ** 2).mean(1).add(eps).sqrt()
    C = lambda W: W - W.mean(0, keepdim=True)
    S1 = [sd(cap["pre"][l], tr.h[l].ln_1.eps) for l in range(L)]; S2 = [sd(cap["mid"][l], tr.h[l].ln_2.eps) for l in range(L)]
    SF = sd(tops[0], tr.ln_f.eps); A = [out.attentions[l][0] for l in range(L)]
    R1 = [C(tr.h[l].ln_1.weight[:, None] * tr.h[l].attn.c_attn.weight[:, 2*D:]) for l in range(L)]
    R2 = [C(tr.h[l].ln_2.weight[:, None] * tr.h[l].mlp.c_fc.weight) for l in range(L)]
    FX1 = [tr.h[l].ln_1.bias @ tr.h[l].attn.c_attn.weight[:, 2*D:] + tr.h[l].attn.c_attn.bias[2*D:] for l in range(L)]
    FX2 = [tr.h[l].ln_2.bias @ tr.h[l].mlp.c_fc.weight + tr.h[l].mlp.c_fc.bias for l in range(L)]
    s = [torch.where(cap["H"][l].abs() > 1e-12, cap["a"][l] / cap["H"][l], torch.zeros(1)) for l in range(L)]
    Wbar = tr.wte.weight.mean(0); hof = torch.arange(D) // hd
    print(f"\n{name}: {L} layers, {P} tokens: {[tok.decode([i]) for i in ids[0].tolist()]}")
    rows = []
    for T in range(P):
        for label, w in [("next", int(out.logits[0, T].argmax())),
                         ("random", int(torch.randint(0, tr.wte.weight.shape[0], (1,), generator=torch.Generator().manual_seed(T))))]:
            e = (tr.wte.weight[w] - Wbar) * tr.ln_f.weight
            G = torch.zeros(P, D); G[T] = (e - e.mean()) / SF[T]           # worth of each lane of the top stream
            lead = float(out.logits[0, T, w] - out.logits[0, T].mean())
            other = float(tr.ln_f.bias @ (tr.wte.weight[w] - Wbar))         # fixed parts and biases, gathered as we go
            for l in reversed(range(L)):
                b = tr.h[l]
                u = G @ b.mlp.c_proj.weight.T                                  # worth of one unit of each gate's output
                other += float((G @ b.mlp.c_proj.bias).sum() + (u * s[l] * FX2[l]).sum())
                Gm = G + (u * s[l]) @ R2[l].T / S2[l][:, None]                 # worth of each lane after the arrivals
                wz = Gm @ b.attn.c_proj.weight.T                               # worth of one unit of each head channel, per reader t
                other += float((Gm @ b.attn.c_proj.bias).sum() + (wz * FX1[l]).sum())
                wv = torch.einsum("ctp,tc->pc", A[l][hof], wz)                 # worth of each value channel at each position p
                G = Gm + wv @ R1[l].T / S1[l][:, None]                         # worth of each lane at the bottom of layer l
            tab = tr.wte.weight[ids[0]] * G; pos = tr.wpe.weight[:P] * G      # every table number x its worth
            base = float((Wbar[None] * G).sum()); own = (tr.wte.weight[ids[0]] - Wbar) * G   # the base turned by context; own parts
            total = float(tab.sum() + pos.sum()) + other
            assert abs(total - lead) < 1e-7 * max(1, abs(lead)), (total, lead)
            rows.append((T, label, tok.decode([w]), lead, float(tab.sum()), float(pos.sum()), other,
                         float(tab[T].sum()), float(tab.abs().sum()), tab, base, float(own.sum()), float(own.abs().sum()), float(own[T].sum())))
    nb = float(Wbar.norm()); nr = float(tr.wte.weight[ids[0]].norm(dim=1).median())
    print(f"  the base (average row) is {nb/nr*100:.0f}% of a table row's size here")
    for label in ["next", "random"]:
        R = [r for r in rows if r[1] == label]
        g = lambda i: sorted(r[i] / r[3] for r in R)[len(R) // 2] * 100
        print(f"  {label:6s} word, BASE SPLIT: base (turned by context) {g(10):+6.1f}% | own parts {g(11):+6.1f}% "
              f"(own row {g(13):+6.1f}%) | position rows {g(5):+6.1f}% | biases {g(6):+6.1f}% | "
              f"inside the own parts {sorted(100 - abs(r[11]) / r[12] * 100 for r in R)[len(R)//2]:5.1f}% cancels")
        f = lambda i: sorted(r[i] / r[3] for r in R)[len(R) // 2] * 100
        canc = sorted(100 - abs(r[4]) / r[8] * 100 for r in R)[len(R) // 2]
        print(f"  {label:6s} word: the lead comes from -- table rows {f(4):+6.1f}% | position rows {f(5):+6.1f}% | biases {f(6):+6.1f}%"
              f"   (medians over {len(R)} positions; sums checked exact)")
        print(f"         the token's own table row {f(7):+6.1f}% of the lead | inside the table rows, number by number: {canc:5.1f}% cancels")
    print("  per position (next token), % of the lead: base turned by context | own part of the token's row | own parts of earlier rows | position rows | biases")
    for r in [r for r in rows if r[1] == "next"]:
        print(f"    {r[0]:2d} {tok.decode([ids[0, r[0]].item()])!r:10s} -> {r[2]!r:10s} lead {r[3]:+6.2f}: base {r[10]/r[3]*100:+7.1f} | own {r[13]/r[3]*100:+7.1f}"
              f" | earlier own {(r[11]-r[13])/r[3]*100:+7.1f} | pos {r[5]/r[3]*100:+7.1f} | biases {r[6]/r[3]*100:+7.1f}")
    return rows
bottom("BLANK mini model", "blank_mini/model")
rows = bottom("GPT-2", "gpt2")

