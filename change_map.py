"""The change one new piece makes, marked: the whole machine before and after, laid digit over digit (XOR).
Every 1 of the XOR is a digit that changed. Mapped by depth, by part, and by binary digit place.
    python3 change_map.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json"))
def state(ids):                                   # everything the text puts on the machine, by (depth, part)
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; S = {(0, "stream"): R}
    for l in range(12):
        Q = gauged_bank(R, f"qkv.{l}"); A = attention(Q); R = R + plain_bank(A, f"proj.{l}")
        S[(l, "query")], S[(l, "key")], S[(l, "value")], S[(l, "arrival")] = Q[:, :768], Q[:, 768:1536], Q[:, 1536:], A
        H = edge(gauged_bank(R, f"fc.{l}")); R = R + plain_bank(H, f"mproj.{l}")
        S[(l, "edge buckets")], S[(l + 1, "stream")] = H, R
    return S
def marks(a, b):                                   # XOR of sign-and-size digits: (count, count per digit place)
    sa, sb = (a < 0), (b < 0); ma, mb = np.abs(a).astype(np.uint64), np.abs(b).astype(np.uint64)
    x = ma ^ mb; places = np.array([int(np.count_nonzero((x >> np.uint64(k)) & np.uint64(1))) for k in range(48)])
    return int(places.sum()) + int(np.count_nonzero(sa != sb)), places
r = refs[0]; ids = r["ids"] + r["ref"][:4]; nxt = r["ref"][4]
before, after = state(ids), state(ids + [nxt]); P = len(ids)
old = sum(marks(before[k], after[k][:P])[0] for k in before)
total, byd, byp, places = 0, {}, {}, np.zeros(48, int)
for k in after:
    pad = np.zeros_like(after[k]); pad[:P] = before[k]
    n, pl = marks(pad, after[k]); total += n; places += pl
    byd[k[0]] = byd.get(k[0], 0) + n; byp[k[1]] = byp.get(k[1], 0) + n
print(f"{r['prompt']!r} + 4 pieces, then the next piece {GPT2Tokenizer.from_pretrained('gpt2').decode([nxt]) if False else nxt}")
print(f"marked digits (XOR 1s): {total:,}    of them in any earlier piece's spots: {old}")
bar = lambda n, m: "#" * int(round(40 * n / m))
print("\nby depth"); m = max(byd.values())
for d in sorted(byd): print(f"  {d:2d}  {byd[d]:8,}  {bar(byd[d], m)}")
print("\nby part"); m = max(byp.values())
for p, n in sorted(byp.items(), key=lambda t: -t[1]): print(f"  {p:13s} {n:8,}  {bar(n, m)}")
print("\nby binary digit place (0 = smallest)"); m = places.max()
for k in range(47, -1, -1):
    if places[k]: print(f"  {k:2d}  {places[k]:8,}  {bar(places[k], m)}")
