"""Backward from the next token, all the way down. At this token every share is a fixed number: each gate passes
s = a/H of its pile, each arrival head reads earlier positions with fixed weights, each water level is one number per
position. Then the next token's lead (over the average word) is an exact sum over paths: source -> part -> part -> ...
-> lead. Parts are gates (3072 a layer) and arrival-head channels (768 a layer), at every position. Sources are the
table rows, the position rows and the biases (nothing else enters).
Level k: every part reached in k steps back is broken into what fed it; sources stop. Poured at level k = the sum of
the sizes of all paths; left = the lead (checked exact at every level, signed). Control: a random word.
PREDICTION (written first): GPT-2 next token -- level 1 about 93% cancels, level 2 about 99.9%, level 3 about 99.99%;
a random word cancels more at every level. Blank -- all the way down over 99.9% cancels and the next token and a random
word differ little; its survivors are the table row going straight through."""
import torch, sys, time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok("The capital of France is Paris. It is known for the Eiffel", return_tensors="pt").input_ids; P = ids.shape[1]
def run(name, path, KMAX, TARGETS):
    m = GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager").double().eval(); tr = m.transformer
    L, D, H = len(tr.h), m.config.n_embd, m.config.n_head; hd = D // H; F = tr.h[0].mlp.c_fc.weight.shape[1]
    cap = {k: {} for k in ["pre", "mid", "z", "a", "H"]}; tops = []
    hs = []
    for l, b in enumerate(tr.h):
        hs += [b.ln_1.register_forward_hook(lambda mod, i, o, l=l: cap["pre"].__setitem__(l, i[0][0])),
               b.ln_2.register_forward_hook(lambda mod, i, o, l=l: cap["mid"].__setitem__(l, i[0][0])),
               b.attn.c_proj.register_forward_hook(lambda mod, i, o, l=l: cap["z"].__setitem__(l, i[0][0])),
               b.mlp.c_fc.register_forward_hook(lambda mod, i, o, l=l: cap["H"].__setitem__(l, o[0])),
               b.mlp.c_proj.register_forward_hook(lambda mod, i, o, l=l: cap["a"].__setitem__(l, i[0][0]))]
    hs.append(tr.h[-1].register_forward_hook(lambda mod, i, o: tops.append(o[0][0])))
    out = m(ids, output_attentions=True); [h.remove() for h in hs]
    sd = lambda x, eps: ((x - x.mean(1, keepdim=True)) ** 2).mean(1).add(eps).sqrt()
    S1 = [sd(cap["pre"][l], tr.h[l].ln_1.eps) for l in range(L)]; S2 = [sd(cap["mid"][l], tr.h[l].ln_2.eps) for l in range(L)]
    SF = sd(tops[0], tr.ln_f.eps)
    A = [out.attentions[l][0] for l in range(L)]                          # (H, P, P) fixed arrival weights
    C = lambda W: W - W.mean(0, keepdim=True)
    R1 = [C(tr.h[l].ln_1.weight[:, None] * tr.h[l].attn.c_attn.weight[:, 2*D:]) for l in range(L)]     # into value channels
    R2 = [C(tr.h[l].ln_2.weight[:, None] * tr.h[l].mlp.c_fc.weight) for l in range(L)]                  # into gate piles
    FX1 = [tr.h[l].ln_1.bias @ tr.h[l].attn.c_attn.weight[:, 2*D:] + tr.h[l].attn.c_attn.bias[2*D:] for l in range(L)]
    FX2 = [tr.h[l].ln_2.bias @ tr.h[l].mlp.c_fc.weight + tr.h[l].mlp.c_fc.bias for l in range(L)]
    Hp = [cap["H"][l] for l in range(L)]; a = [cap["a"][l] for l in range(L)]; z = [cap["z"][l] for l in range(L)]
    s = [torch.where(Hp[l].abs() > 1e-12, a[l] / Hp[l], torch.zeros_like(a[l])) for l in range(L)]
    for l in range(L):   # the identities everything rests on
        assert torch.allclose(cap["mid"][l] @ R2[l] / S2[l][:, None] + FX2[l], Hp[l], atol=1e-7)
        v_in = cap["pre"][l] @ R1[l] / S1[l][:, None] + FX1[l]
        zz = torch.cat([A[l][h] @ v_in[:, h*hd:(h+1)*hd] for h in range(H)], 1); assert torch.allclose(zz, z[l], atol=1e-7)
    Wo = [tr.h[l].attn.c_proj.weight for l in range(L)]; Wm = [tr.h[l].mlp.c_proj.weight for l in range(L)]
    bo = [tr.h[l].attn.c_proj.bias for l in range(L)]; bm = [tr.h[l].mlp.c_proj.bias for l in range(L)]
    Wbar = tr.wte.weight.mean(0)
    hof = torch.arange(D) // hd                                            # which head each channel belongs to
    # writers at each position: (kind, layer, out-vectors, value, slot). slot: 0 rows; 2l+1 heads/attn bias; 2l+2 gates/mlp bias
    def writers(p):
        W = [("table row", -1, tr.wte.weight[ids[0, p]][None], torch.ones(1), 0), ("position row", -1, tr.wpe.weight[p][None], torch.ones(1), 0)]
        for l in range(L):
            W += [("heads", l, Wo[l], z[l][p], 2*l+1), ("biases", l, bo[l][None], torch.ones(1), 2*l+1),
                  ("gates", l, Wm[l], a[l][p], 2*l+2), ("biases", l, bm[l][None], torch.ones(1), 2*l+2)]
        return W
    WR = [writers(p) for p in range(P)]
    print(f"\n{name}: {L} layers x {F} gates + {D} head channels a layer, {P} positions; identities checked exact", flush=True)
    for T in TARGETS:
        nxt = int(out.logits[0, T].argmax()); rnd = int(torch.randint(0, tr.wte.weight.shape[0], (1,), generator=torch.Generator().manual_seed(T)))
        for label, w in [("next token " + repr(tok.decode([nxt])), nxt), ("random word", rnd)]:
            e = (tr.wte.weight[w] - Wbar) * tr.ln_f.weight; v = (e - e.mean()) / SF[T]
            lead = float(out.logits[0, T, w] - out.logits[0, T].mean()); fxl = float(tr.ln_f.bias @ (tr.wte.weight[w] - Wbar))
            # B[k]: per writer (p, index in WR[p]) -> (signed, abs) worth of one unit of its value to the lead via exactly k steps
            Bg = {"s": [torch.zeros(P, F) for _ in range(L)], "a": [torch.zeros(P, F) for _ in range(L)]}   # readers: gates
            Bh = {"s": [torch.zeros(P, D) for _ in range(L)], "a": [torch.zeros(P, D) for _ in range(L)]}   # readers: head channels
            lead_k = {"s": 1.0, "a": 1.0}                                                                   # level 0: the lead itself
            src_s = src_a = 0.0
            src_s += fxl; src_a += abs(fxl)                                                                  # the top's own fixed part
            line = []
            for k in range(1, KMAX + 1):
                nBg = {"s": [torch.zeros(P, F) for _ in range(L)], "a": [torch.zeros(P, F) for _ in range(L)]}
                nBh = {"s": [torch.zeros(P, D) for _ in range(L)], "a": [torch.zeros(P, D) for _ in range(L)]}
                part_s = part_a = 0.0
                for p in range(P):
                    for kind, lw, O, val, slot in WR[p]:
                        bs = torch.zeros(O.shape[0]); ba = torch.zeros(O.shape[0])
                        for l2 in range(L):
                            if slot <= 2*l2 + 1 and Bg["a"][l2][p].any():
                                M = O @ R2[l2] / S2[l2][p]
                                bs += M @ (s[l2][p] * Bg["s"][l2][p]); ba += M.abs() @ (s[l2][p].abs() * Bg["a"][l2][p])
                            if slot <= 2*l2:
                                al = A[l2][:, :, p][hof]                              # (D channels, P readers t): weight from p
                                gs = (al * Bh["s"][l2].T).sum(1); ga = (al * Bh["a"][l2].T).sum(1)
                                if ga.any():
                                    M = O @ R1[l2] / S1[l2][p]; bs += M @ gs; ba += M.abs() @ ga
                        if p == T and lead_k["a"]:
                            ov = O @ v; bs += ov * lead_k["s"]; ba += ov.abs() * lead_k["a"]
                        if kind in ("gates", "heads"):
                            (nBg if kind == "gates" else nBh)["s"][lw][p] = bs; (nBg if kind == "gates" else nBh)["a"][lw][p] = ba
                            part_s += float(val @ bs); part_a += float(val.abs() @ ba)
                        else:
                            src_s += float(val @ bs); src_a += float(val.abs() @ ba)
                # fixed parts that enter readers directly (sources), reached at this level through the readers' worth
                for l2 in range(L):
                    src_s += float((Bg["s"][l2] * s[l2] * FX2[l2]).sum()); src_a += float((Bg["a"][l2] * (s[l2] * FX2[l2]).abs()).sum())
                    src_s += float((Bh["s"][l2] * FX1[l2]).sum()); src_a += float((Bh["a"][l2] * FX1[l2].abs()).sum())
                left = src_s + part_s; poured = src_a + part_a
                assert abs(left - lead) < 1e-6 * max(1, abs(lead)), (k, left, lead)
                line.append(f"level {k}: {100 - abs(lead)/poured*100:8.4f}% cancels")
                Bg, Bh, lead_k = nBg, nBh, {"s": 0.0, "a": 0.0}
            print(f"  position {T:2d}, {label:22s} lead {lead:+7.2f} | " + " | ".join(line), flush=True)
t0 = time.time()
run("BLANK mini model", "blank_mini/model", 10, [P - 1, 7])
print(f"  ({time.time()-t0:.0f}s)", flush=True)
run("GPT-2", "gpt2", int(sys.argv[1]) if len(sys.argv) > 1 else 3, [P - 1, 7, 4])
print(f"  ({time.time()-t0:.0f}s)")
