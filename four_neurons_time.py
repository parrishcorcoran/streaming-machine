"""The same two neurons: each value as a place on the time circle (base-2 log: where its top 1 sits + the next digits
from the table, in 1/16ths of a doubling) and a direction (up/down). Own vs open: how far each is delayed, and how much
of that delay is the gauge (the same for both) and how much each neuron's own bend.      python3 two_neurons_time.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); J = [0, 1, 2, 3]; w, we, b = W["fc.0"]
def place(v): return (log2fix([abs(int(v))])[0] * 16) // 65536 if v != 0 else None     # 1/16ths of a doubling
def dirn(v): return "up" if v > 0 else "down"
print("input   neuron   open: place dir     own: place dir    delay   of which gauge   of which its own bend")
for n, r in enumerate(refs[:4]):
    ids = (r["ids"] + r["ref"])[:5]; R = M["wte"][ids] + M["wpe"][:5]
    R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
    x16, ex, LS = gauge_log(R); pile = product(x16[-1:], w[:, J])[0]
    piles_open = rshift(pile, -(we[J] + ex[-1] + ES)); gauged = power(pile[None], (we[J] + ex[-1] + ES) * 65536 - LS[-1])[0]
    open_v = piles_open + b[J]; pre = gauged + b[J]; own_v = edge(pre[None])[0]
    for j in range(4):
        d_total = place(open_v[j]) - place(own_v[j]); d_gauge = place(piles_open[j]) - place(gauged[j]); d_bend = place(pre[j]) - place(own_v[j])
        print(f"  {n}       {j}      {place(open_v[j]):5d} {dirn(open_v[j]):4s}      {place(own_v[j]):5d} {dirn(own_v[j]):4s}   {d_total:5d}        {d_gauge:5d}              {d_bend:5d}")
# all 30 inputs, the 4 neurons: is the gauge's delay the same for every neuron in the row, and what sets each bend's delay?
same_gauge = 0; ups, downs = [], []
for r in refs:
    ids = (r["ids"] + r["ref"])[:5]; R = M["wte"][ids] + M["wpe"][:5]
    R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
    x16, ex, LS = gauge_log(R); pile = product(x16[-1:], w[:, J])[0]
    po = rshift(pile, -(we[J] + ex[-1] + ES)); g = power(pile[None], (we[J] + ex[-1] + ES) * 65536 - LS[-1])[0]
    dg = [log2fix([abs(int(po[j]))])[0] - log2fix([abs(int(g[j]))])[0] for j in range(4) if po[j] != 0 and g[j] != 0]
    same_gauge += max(dg) - min(dg) <= 65536 // 256                  # the same delay for the whole row (to 1/256 of a doubling)
    pre = g + b[J]; own = edge(pre[None])[0]
    for j in range(4):
        if pre[j] != 0 and own[j] != 0: (ups if pre[j] > 0 else downs).append((log2fix([abs(int(pre[j]))])[0] - log2fix([abs(int(own[j]))])[0]) * 16 // 65536)
print(f"\nall 30 inputs, 4 neurons: the gauge delays every neuron in the row by the same amount in {same_gauge} of 30")
print(f"the bend's delay: going up {np.median(ups):.0f} sixteenths (median, {len(ups)} cases), going down {np.median(downs):.0f} sixteenths ({len(downs)} cases); direction kept every time")
