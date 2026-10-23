"""Two neurons, one layer (layer 0, buckets 0 and 1), last position, base-2 GPT-2, whole numbers.
open: gauge and bend open (the weights alone); own: the input's own gauge and bend. Row first: the two values struck
into one line (harmonics 1 and 2 on a circle of 5 ticks; whole-number cos table at 2^14). Then: the open line delayed
by 0..4 ticks, and which delay (if any) lands it on the input's own line.      python3 two_neurons.py
"""
import sys, json, math, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); J = [0, 1]; T = 5
C = np.array([[round(math.cos(2 * math.pi * (k + 1) * t / T) * 2 ** 14) for t in range(T)] for k in range(len(J))], dtype=np.int64)
def line(v): return v @ C                                          # two notes -> one line of 5 ticks
w, we, b = W["fc.0"]
exact = 0
for n, r in enumerate(refs):
    ids = (r["ids"] + r["ref"])[:5]; R = M["wte"][ids] + M["wpe"][:5]
    R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
    x16, ex, LS = gauge_log(R); pile = product(x16[-1:], w[:, J])[0]
    open_v = rshift(pile, -(we[J] + ex[-1] + ES)) + b[J]                               # gauge open, bend open
    own_v = edge((power(pile[None], (we[J] + ex[-1] + ES) * 65536 - LS[-1]) + b[J]))[0]   # the input's own gauge and bend
    Lo, Lw = line(open_v), line(own_v)
    best = min(range(T), key=lambda d: int(np.abs(np.roll(Lo, d) - Lw).sum()))
    hit = bool(np.array_equal(np.roll(Lo, best), Lw)); exact += hit
    if n < 4:
        print(f"input {n}: open values {open_v.tolist()}   own values {own_v.tolist()}")
        print(f"         open line {Lo.tolist()}\n         own line  {Lw.tolist()}")
        print(f"         best delay {best} ticks: open line delayed {np.roll(Lo, best).tolist()}  -> {'lands exactly on own' if hit else 'does not land on own'}")
print(f"\n30 inputs: a delay of the open line lands exactly on the input's own line in {exact} of 30")
