"""Piece t and the next piece t+1 of the same text, layer 0's row of 3,072 bends: each share as a delay on the time
circle (log2, in doublings). Just a rotation = every neuron's delay moves by the same amount. Then: piece t+1 run with
layer 0's shares taken from piece t, turned by that one common amount; whole model; right next word out of 30.
    python3 shares_rotation.py
"""
import json, math, numpy as np, torch
exec(open("tiny_steps.py").read().split("setups = []")[0])
def pieces(ids):                                         # piece at each position, earlier ones standing; returns its gates and true word
    X = sd["wte.weight"][ids] + sd["wpe.weight"][:len(ids)]; st = []; xs = X.clone()
    for l in range(12):
        K, V = standing(xs, l); st.append((K[:-1], V[:-1])); xs = torch.stack([layer_new(xs[p], l, K[:p], V[:p])[0] for p in range(len(ids))])
    x, G = X[-1], []
    for l in range(12): x, gl = layer_new(x, l, *st[l]); G.append(gl)
    return X[-1], st, G, read(x)
moves, spread, right_same, right_turned = [], [], 0, 0
for n, r in enumerate(refs):
    full = r["ids"] + r["ref"]; a = pieces(full[:5]); b = pieces(full[:6])
    da = torch.log2(a[2][0]["share"].clamp(min=1e-30)); db = torch.log2(b[2][0]["share"].clamp(min=1e-30))
    move = db - da; common = float(move.median())
    moves.append(common); spread.append(float((move - common).abs().median()))
    if n < 2:
        print(f"text {n}: first 8 neurons' delay (doublings) piece t -> t+1:")
        for j in range(8): print(f"    neuron {j}: {float(da[j]):7.2f} -> {float(db[j]):7.2f}   moved {float(move[j]):+7.2f}")
    for turned in (False, True):                          # piece t+1 with layer 0's shares from piece t (turned by the one common amount, or not)
        G0 = dict(b[2][0]); G0["share"] = a[2][0]["share"] * (2.0 ** common if turned else 1.0)
        x = layer_new(b[0], 0, *b[1][0], {**b[2][0], "share": G0["share"]})[0]
        for l in range(1, 12): x = layer_new(x, l, *b[1][l])[0]
        if turned: right_turned += read(x) == b[3]
        else: right_same += read(x) == b[3]
print(f"\nall 30 texts: how far each neuron's delay moves from the common move (median): {np.median(spread):.2f} doublings")
print(f"piece t+1 with layer 0's shares from piece t, unturned: right next word {right_same} / 30")
print(f"piece t+1 with layer 0's shares from piece t, turned by the one common amount: right next word {right_turned} / 30")
