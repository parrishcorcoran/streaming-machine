"""10 neurons of layer 0 (last position, base-2 GPT-2), each as an angle on the time circle (its base-2 place; one lap
= one doubling, 16 steps of 1/16) and a direction (down = the opposite side). Added as circles into the row's one
circle (whole-number cos/sin tables at 2^14). Is the input's own row circle the open row circle turned by one angle?
    python3 ten_circles.py
"""
import sys, json, math, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); J = list(range(10)); w, we, b = W["fc.0"]
STEPS = 16; COS = [round(math.cos(2 * math.pi * k / STEPS) * 2 ** 14) for k in range(STEPS)]; SIN = [round(math.sin(2 * math.pi * k / STEPS) * 2 ** 14) for k in range(STEPS)]
def step(v): return (int(log2fix([abs(int(v))])[0]) * STEPS // 65536 + (STEPS // 2 if v < 0 else 0)) % STEPS   # angle on the circle; down = opposite side
def row(vals): s = [step(v) for v in vals if v != 0]; return sum(COS[k] for k in s), sum(SIN[k] for k in s), s
best_turns, offs = [], []
for n, r in enumerate(refs):
    ids = (r["ids"] + r["ref"])[:5]; R = M["wte"][ids] + M["wpe"][:5]
    R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
    x16, ex, LS = gauge_log(R); pile = product(x16[-1:], w[:, J])[0]
    open_v = rshift(pile, -(we[J] + ex[-1] + ES)) + b[J]; own_v = edge((power(pile[None], (we[J] + ex[-1] + ES) * 65536 - LS[-1]) + b[J]))[0]
    oc, os_, ost = row(open_v); wc, ws, wst = row(own_v)
    # turn the open row circle by k steps; how far is it from the own row circle (whole numbers)
    def miss(k): c, s = COS[k], SIN[k]; tc, ts = (oc * c - os_ * s) >> 14, (oc * s + os_ * c) >> 14; return abs(tc - wc) + abs(ts - ws)
    k = min(range(STEPS), key=miss); best_turns.append(k); offs.append(miss(k) / max(1, abs(wc) + abs(ws)))
    if n < 3: print(f"input {n}: open angles {ost}  own angles {wst}\n         open row circle ({oc}, {os_})  own row circle ({wc}, {ws})  best turn {k} steps -> off by {100 * offs[-1]:.0f}%")
print(f"\n30 inputs: the own row circle = the open row circle turned by one angle? off by {100 * np.median(offs):.0f}% (median); turns used: {sorted(set(best_turns))}")
