"""Backward from the next token, step 2: one layer down through the gates.
Step 1 split the next token's lead (over the average word) exactly into parts; gate g's part is c_g = a_g * u_g
(a_g its output, u_g what one unit of it is worth to the lead). Here each gate's own pile H_g is split exactly into
everything that wrote into the stream before it (table row, position row, every earlier gate, every arrival head up to
this layer, biases), through this position's water level (fixed, one number per position). The bend passes a share
s_g = a_g / H_g of the pile at this token, so every two-step path (part k -> gate g -> lead) is exact:
path(k,g) = s_g * u_g * (part k's push into H_g). Sums checked exact.
Measured: (a) inside each gate's pile, poured vs left; (b) end to end through two steps, everything poured along all
paths vs what is left; which parts carry the survivors. Control: a random word. Arrival heads are not gone through yet.
PREDICTION (written first): GPT-2 -- inside each gate's pile 90-95% cancels; through two steps over 99% of everything
poured cancels for the next token and more for a random word. Blank -- the gates' small part is fed from the table row."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
tok = GPT2Tokenizer.from_pretrained("gpt2")
text = ("The capital of France is Paris. It is known for the Eiffel Tower, the Louvre and its cafes. "
        "Every year millions of people visit the city to see its museums and walk along the river.")
ids = tok(text, return_tensors="pt").input_ids; P = ids.shape[1]
POS = torch.linspace(1, P - 1, 10).round().long().tolist()
def step2(name, m):
    m = m.double().eval(); tr = m.transformer; L = len(tr.h); H = m.config.n_head; D = m.config.n_embd; hd = D // H
    zin, ain, hin, mid, tops = {}, {}, {}, {}, []
    hs = []
    for l, b in enumerate(tr.h):
        hs.append(b.attn.c_proj.register_forward_hook(lambda mod, i, o, l=l: zin.__setitem__(l, i[0][0])))
        hs.append(b.mlp.c_proj.register_forward_hook(lambda mod, i, o, l=l: ain.__setitem__(l, i[0][0])))
        hs.append(b.mlp.c_fc.register_forward_hook(lambda mod, i, o, l=l: hin.__setitem__(l, o[0])))
        hs.append(b.ln_2.register_forward_hook(lambda mod, i, o, l=l: mid.__setitem__(l, i[0][0])))
    hs.append(tr.h[-1].register_forward_hook(lambda mod, i, o: tops.append(o[0][0])))
    out = m(ids); [h.remove() for h in hs]
    x = tops[0]; sig = ((x - x.mean(1, keepdim=True)) ** 2).mean(1, keepdim=True).add(tr.ln_f.eps).sqrt()
    Wbar = tr.wte.weight.mean(0)
    nxt = out.logits[0].argmax(1); rnd = torch.randint(0, tr.wte.weight.shape[0], (P,), generator=torch.Generator().manual_seed(0))
    print(f"\n{name}: {L} layers x {tr.h[0].mlp.c_fc.weight.shape[1]} gates, {len(POS)} positions")
    res = {}
    for label, words in [("next token", nxt), ("random word", rnd)]:
        res[label] = dict(poured1=0., net1=0., poured2=0., left2=0., inside=[], carry={})
    for t in POS:
        for l in range(L):
            b = tr.h[l]
            # every part that wrote into the stream before this layer's gates, as its vector at position t
            vecs, kinds = [tr.wte.weight[ids[0, t]][None], tr.wpe.weight[t][None]], ["table row", "position row"]
            for l2 in range(l + 1):
                z = zin[l2][t]; Wo = tr.h[l2].attn.c_proj.weight
                vecs.append(torch.stack([z[h*hd:(h+1)*hd] @ Wo[h*hd:(h+1)*hd] for h in range(H)])); kinds += ["arrival heads"] * H
                vecs.append(tr.h[l2].attn.c_proj.bias[None]); kinds += ["biases"]
                if l2 < l:
                    vecs.append(ain[l2][t][:, None] * tr.h[l2].mlp.c_proj.weight); kinds += ["earlier gates"] * ain[l2].shape[1]
                    vecs.append(tr.h[l2].mlp.c_proj.bias[None]); kinds += ["biases"]
            V = torch.cat(vecs); assert torch.allclose(V.sum(0), mid[l][t], atol=1e-8)
            xm = mid[l][t]; s2 = ((xm - xm.mean()) ** 2).mean().add(b.ln_2.eps).sqrt()
            Wp = b.ln_2.weight[:, None] * b.mlp.c_fc.weight; Wp = (Wp - Wp.mean(0, keepdim=True)) / s2      # route into each gate's pile
            fixed = b.ln_2.bias @ b.mlp.c_fc.weight + b.mlp.c_fc.bias
            R = V @ Wp                                                            # (parts, gates): each part's push into each pile
            Hg = hin[l][t]; assert torch.allclose(R.sum(0) + fixed, Hg, atol=1e-6)
            a = ain[l][t]; s = torch.where(Hg.abs() > 1e-12, a / Hg, torch.zeros_like(a))
            inside = (Hg.abs() / (R.abs().sum(0) + fixed.abs()))
            for label, words in [("next token", nxt), ("random word", rnd)]:
                e = (tr.wte.weight[words[t]] - Wbar) * tr.ln_f.weight; v = (e - e.mean()) / sig[t, 0]
                u = b.mlp.c_proj.weight @ v                                       # each gate's worth to the lead
                c = a * u; r = res[label]
                r["poured1"] += c.abs().sum(); r["net1"] += c.sum()
                path = R * (s * u)[None]                                          # every two-step path
                fx = fixed * s * u
                r["poured2"] += path.abs().sum() + fx.abs().sum(); r["left2"] += path.sum() + fx.sum()
                assert torch.allclose(path.sum() + fx.sum(), c.sum(), atol=1e-6)
                w = c.abs() / c.abs().sum(); r["inside"].append(float((inside * w).sum()))
                net_by = path.sum(1)
                for k in set(kinds):
                    idx = torch.tensor([i for i, kk in enumerate(kinds) if kk == k])
                    r["carry"][k] = r["carry"].get(k, 0.) + float(net_by[idx].sum())
                r["carry"]["biases"] = r["carry"].get("biases", 0.) + float(fx.sum())
    for label in res:
        r = res[label]; inside = sum(r["inside"]) / len(r["inside"])
        print(f"  {label}: step 1, the gates' parts: {100 - abs(r['net1'])/r['poured1']*100:5.2f}% cancels | "
              f"inside each gate's own pile: {100 - inside*100:5.2f}% cancels | "
              f"two steps end to end: {100 - abs(r['left2'])/r['poured2']*100:6.3f}% cancels "
              f"(poured {r['poured2']/r['poured1']:.1f}x step 1's)")
        tot = sum(r["carry"].values())
        print("      what the gates' survivors come from: " + " | ".join(f"{k} {v/tot*100:+.1f}%" for k, v in sorted(r["carry"].items(), key=lambda kv: -abs(kv[1]))))
step2("BLANK mini model", GPT2LMHeadModel.from_pretrained("blank_mini/model"))
step2("GPT-2", GPT2LMHeadModel.from_pretrained("gpt2"))
