"""Every layer's light as ONE dial position (the chord's notes superposed, 16,384 positions, whole numbers), every layer
a table position -> position, the read a table position -> word. Single words (the arrival sees only itself: out = v).
Made from all 50,257 words through the base-2 machine; then every word through the 12 tables only, against the machine.
    python3 thirteen_tables.py
"""
import sys, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
V = 50257; ids = np.arange(V); R = M["wte"][ids] + M["wpe"][0][None, :]
def position(R):                                  # the whole chord superposed into one place on a 16,384 dial (whole numbers)
    s = R.sum(1); top = int(np.abs(s).max()).bit_length(); return rshift(s, max(top - 13, 0)), top
pos = [position(R)[0]]
for l in range(12):
    Q = gauged_bank(R, f"qkv.{l}"); A = Q[:, 1536:]                       # a single word's arrival is its own value
    R = R + plain_bank(A, f"proj.{l}"); R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    pos.append(position(R)[0])
w, we, b = W["head"]; x16, ex, LS = gauge_log(R); words = np.array([race(r) for r in (power(x16[i:i + 1] @ w, (we[None, :] + ex[i] + ES) * 65536 - LS[i]) + b for i in range(V))]).reshape(-1) if False else None
lg_words = []
for i in range(0, V, 2048):
    lg = power(product(x16[i:i + 2048], w), (we[None, :] + ex[i:i + 2048, None] + ES) * 65536 - LS[i:i + 2048, None]) + b
    lg_words += [int(np.argmax(r)) for r in lg]
words = np.array(lg_words)
def table(a, b):                                  # position in -> the most common position out
    t = {}
    for x, y in zip(a.tolist(), b.tolist()): t.setdefault(x, {}); t[x][y] = t[x].get(y, 0) + 1
    return {x: max(c, key=c.get) for x, c in t.items()}
T = [table(pos[l], pos[l + 1]) for l in range(12)]; READ = table(pos[12], words)
print("distinct positions used per depth:", [len(np.unique(p)) for p in pos])
p = pos[0].tolist(); 
for l in range(12): p = [T[l].get(x, x) for x in p]
got = np.array([READ.get(x, 0) for x in p])
print(f"next word through the 12 tables only == the machine's next word: {int((got == words).sum()):,} of {V:,} ({100 * (got == words).mean():.1f}%)")
print(f"for comparison, always saying the most common word: {100 * (words == np.bincount(words).argmax()).mean():.1f}%")
