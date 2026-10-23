"""n neurons of layer 0's first row collapsed TOGETHER into one block (768 in -> 768 out), their bends LIVE: each
neuron's share computed from this piece's own light as it arrives, nothing held. Every other neuron and layer as they
are. Whole model, right next word out of 30, and sizes. (float64 like the tiny model; words counted.)
    python3 two_together.py
"""
import json, math, numpy as np, torch
exec(open("tiny_steps.py").read().split("print(f\"one layer")[0])          # sd, g, gelu, standing, layer_new, read, the 30 pieces
def layer0_with_block(x, K, V, J):
    l = 0; c = x - x.mean(); s1 = (c.var(unbiased=False) + 1e-5).rsqrt()
    q, k, v = ((c * s1 * g(l, "ln_1.weight") + g(l, "ln_1.bias")) @ g(l, "attn.c_attn.weight") + g(l, "attn.c_attn.bias")).split(768, -1)
    Kall, Vall = torch.cat([K, k.view(1, 12, 64)]), torch.cat([V, v.view(1, 12, 64)])
    A = torch.softmax(torch.einsum("hd,phd->hp", q.view(12, 64), Kall) / 8, -1)
    x = x + torch.einsum("hp,phd->hd", A, Vall).reshape(768) @ g(l, "attn.c_proj.weight") + g(l, "attn.c_proj.bias")
    c = x - x.mean(); h2 = c * (c.var(unbiased=False) + 1e-5).rsqrt() * g(l, "ln_2.weight") + g(l, "ln_2.bias")
    Wf, bf, Wm = g(l, "mlp.c_fc.weight"), g(l, "mlp.c_fc.bias"), g(l, "mlp.c_proj.weight")
    f = h2 @ Wf + bf; share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f))                  # every bend live, from this light
    rest = [j for j in range(3072) if j not in set(J)]
    out = (f[rest] * share[rest]) @ Wm[rest]                                                         # the other neurons as they are
    B = torch.einsum("j,ij,jk->ik", share[J], Wf[:, J], Wm[J])                                        # the n neurons collapsed together: one 768 x 768 block
    out = out + h2 @ B + (share[J] * bf[J]) @ Wm[J]
    return x + out + g(l, "mlp.c_proj.bias")
print(f"{'neurons together':>16} | {'their plate (numbers)':>21} | {'their collapsed block':>21} | right next word")
for n in (2, 4, 10, 100, 1000, 3072):
    J = list(range(n)); right = 0
    for x0, st, G, true_word in setups:
        x = layer0_with_block(x0, *st[0], J)
        for l in range(1, 12): x = layer_new(x, l, *st[l])[0]
        right += read(x) == true_word
    print(f"{n:>16} | {2 * 768 * n:>21,} | {768 * 768:>21,} | {right} / 30", flush=True)
