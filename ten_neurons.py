"""10 neurons of layer 0, last position, base-2 GPT-2, whole numbers. Looking down the row: the gauge is one delay for
the whole row; the bend splits the row into an up wave and a down wave. Each wave given ONE shared delay (its group's
middle delay for that input). How close each neuron's value comes to its own.      python3 ten_neurons.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); J = list(range(10)); w, we, b = W["fc.0"]
F16 = 16 * 65536 // 16                                   # one sixteenth of a doubling, in the log table's units (65536 per doubling)
def L(v): return int(log2fix([abs(int(v))])[0]) if v != 0 else None
def delay_by(v, d):                                      # v delayed by d (log units): v * 2^(-d/65536), whole numbers
    return int(power(np.array([[int(v)]]), np.array([[-d]]))[0, 0])
errs, spread_up, spread_dn = [], [], []
for n, r in enumerate(refs):
    ids = (r["ids"] + r["ref"])[:5]; R = M["wte"][ids] + M["wpe"][:5]
    R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
    x16, ex, LS = gauge_log(R); pile = product(x16[-1:], w[:, J])[0]
    pre = power(pile[None], (we[J] + ex[-1] + ES) * 65536 - LS[-1])[0] + b[J]          # after the row's one shared delay (the gauge), + standing offsets
    own = edge(pre[None])[0]
    d = {j: L(pre[j]) - L(own[j]) for j in J if pre[j] != 0 and own[j] != 0}
    up = [d[j] for j in d if pre[j] > 0]; dn = [d[j] for j in d if pre[j] < 0]
    du, dd = (int(np.median(up)) if up else 0), (int(np.median(dn)) if dn else 0)    # one delay for the up wave, one for the down wave
    two = np.array([delay_by(pre[j], du if pre[j] > 0 else dd) for j in J])
    errs.append(float(np.abs(two - own).sum() / np.abs(own).sum()))
    if up: spread_up.append((max(up) - min(up)) * 16 / 65536)
    if dn: spread_dn.append((max(dn) - min(dn)) * 16 / 65536)
    if n < 3: print(f"input {n}: up wave {len(up)} neurons, delays {[round(x * 16 / 65536) for x in up]} -> one delay {round(du * 16 / 65536)};   down wave {len(dn)} neurons, delays {[round(x * 16 / 65536) for x in dn]} -> one delay {round(dd * 16 / 65536)}")
print(f"\n30 inputs, 10 neurons, each wave given one shared delay:")
print(f"  spread of delays inside the up wave: median {np.median(spread_up):.0f} sixteenths;  inside the down wave: median {np.median(spread_dn):.0f} sixteenths")
print(f"  the 10 values off from their own by {100 * np.median(errs):.1f}% (median over inputs)")
