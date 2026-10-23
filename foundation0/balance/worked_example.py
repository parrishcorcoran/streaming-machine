"""A small worked example: one neuron with 8 lines, its weights as whole numbers in base 2, ups and downs exactly
equal at every place. What an even light and an uneven light meet there. Whole numbers only.

    python3 worked_example.py
"""
W = [+5, -3, +2, -5, +6, -4, +1, -2]          # the neuron's 8 weights, one per line
P = 3                                          # places 0, 1, 2 (sizes up to 7)

def ups_downs(w):
    return [(sum(1 for x in w if x > 0 and (x >> p) & 1), sum(1 for x in w if x < 0 and (-x >> p) & 1)) for p in range(P)]

print("weights, sign and size in base 2 (place 2, 1, 0):")
for j, x in enumerate(W):
    print(f"  line {j}: {x:+d} = {'+' if x > 0 else '-'}{abs(x):03b}")
print("ups and downs per place:", [(p, u, d) for p, (u, d) in enumerate(ups_downs(W))])

def meet(light):
    """each 1 at place p on line j meets brightness light[j]; ups add, downs take away"""
    rows = []; pile = 0
    for p in range(P):
        up = sum(light[j] for j, x in enumerate(W) if x > 0 and (x >> p) & 1)
        dn = sum(light[j] for j, x in enumerate(W) if x < 0 and (-x >> p) & 1)
        rows.append((p, up, dn, (up - dn) << p)); pile += (up - dn) << p
    assert pile == sum(b * x for b, x in zip(light, W))
    met = sum(b * abs(x) for b, x in zip(light, W))
    width2 = sum((b * x) ** 2 for b, x in zip(light, W))          # coin toss: each line's part could have been + or -
    return rows, pile, met, width2

for name, light in (("even light, every line 1", [1] * 8), ("even light, every line 3", [3] * 8),
                    ("uneven light", [2, 1, 3, 1, 2, 3, 1, 2]), ("another uneven light", [0, 3, 1, 0, 2, 1, 3, 1])):
    rows, pile, met, w2 = meet(light)
    print(f"\n{name} {light}:")
    for p, up, dn, left in rows:
        print(f"  place {p}: ups meet {up:2d}, downs meet {dn:2d}, leftover {up - dn:+d} x 2^{p} = {left:+d}")
    print(f"  pile {pile:+d}; light met in all {met}; coin-toss width sqrt({w2}) = {w2 ** 0.5:.1f}")
