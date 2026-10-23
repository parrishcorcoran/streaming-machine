"""One rotation for the whole row (layer 0's 3,072 bends), in the plane (arrival at the piece before, arrival now):
f(s+1) = a f(s) + b f(s-1) + c, the same a, b, c for every neuron -- each neuron moves according to where it sits.
Measured on the pieces before (t-2, t-1, t), applied forward to get piece t+1's arrivals without running its slab;
shares from the fixed edge; layer 0 run with them, everything else live; right next word out of 30.
    python3 row_rotation.py
"""
import json, math, numpy as np, torch
exec(open("tiny_steps.py").read().split("setups = []")[0])
def layer_f(x, l, K, V, G=None):                         # layer_new, also handing back the arrivals at the bends (f)
    c = x - x.mean(); s1 = (c.var(unbiased=False) + 1e-5).rsqrt()
    q, k, v = ((c * s1 * g(l, "ln_1.weight") + g(l, "ln_1.bias")) @ g(l, "attn.c_attn.weight") + g(l, "attn.c_attn.bias")).split(768, -1)
    Kall, Vall = torch.cat([K, k.view(1, 12, 64)]), torch.cat([V, v.view(1, 12, 64)])
    A = torch.softmax(torch.einsum("hd,phd->hp", q.view(12, 64), Kall) / 8, -1)
    x = x + torch.einsum("hp,phd->hd", A, Vall).reshape(768) @ g(l, "attn.c_proj.weight") + g(l, "attn.c_proj.bias")
    c = x - x.mean(); f = (c * (c.var(unbiased=False) + 1e-5).rsqrt() * g(l, "ln_2.weight") + g(l, "ln_2.bias")) @ g(l, "mlp.c_fc.weight") + g(l, "mlp.c_fc.bias")
    share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f)) if G is None else G
    return x + (f * share) @ g(l, "mlp.c_proj.weight") + g(l, "mlp.c_proj.bias"), f
def run_piece(ids):                                       # the last piece, earlier ones standing: its x0, standing pages, layer-0 arrivals, true word
    X = sd["wte.weight"][ids] + sd["wpe.weight"][:len(ids)]; st = []; xs = X.clone()
    for l in range(12):
        K, V = standing(xs, l); st.append((K[:-1], V[:-1])); xs = torch.stack([layer_new(xs[p], l, K[:p], V[:p])[0] for p in range(len(ids))])
    x, f0 = layer_f(X[-1], 0, *st[0])
    for l in range(1, 12): x = layer_new(x, l, *st[l])[0]
    return X[-1], st, f0, read(x)
def word_with(piece, share0):
    x0, st, _, _ = piece; x = layer_f(x0, 0, *st[0], share0)[0]
    for l in range(1, 12): x = layer_new(x, l, *st[l])[0]
    return read(x)
def fit(prev2, prev1, now):                               # one rotation of the plane: now = a*prev1 + b*prev2 + c for every neuron
    Amat = torch.stack([prev1, prev2, torch.ones_like(prev1)], 1); return torch.linalg.lstsq(Amat, now[:, None]).solution[:, 0]
edge = lambda f: gelu(f) / torch.where(f != 0, f, torch.ones_like(f))
res = {"no rotation (shares from t)": 0, "one rotation, measured on the pieces before": 0, "one rotation, best possible (measured with t+1)": 0}; errs = {k: [] for k in res}
for r in refs:
    full = r["ids"] + r["ref"]; P = [run_piece(full[:n]) for n in (4, 5, 6, 7)]    # pieces t-2, t-1, t, t+1
    f = [p[2] for p in P]; target = P[3]
    preds = {"no rotation (shares from t)": f[2]}
    a, b, c = fit(f[0], f[1], f[2]); preds["one rotation, measured on the pieces before"] = a * f[2] + b * f[1] + c
    a, b, c = fit(f[1], f[2], f[3]); preds["one rotation, best possible (measured with t+1)"] = a * f[2] + b * f[1] + c
    for k, fp in preds.items():
        res[k] += word_with(target, edge(fp)) == target[3]; errs[k].append(float((fp - f[3]).norm() / f[3].norm()))
print("piece t+1's layer-0 bends from one rotation of the row; everything else live:")
for k in res: print(f"  {k:48s}: right next word {res[k]:2d} / 30   arrivals off by {100 * np.median(errs[k]):5.1f}% (median)")
