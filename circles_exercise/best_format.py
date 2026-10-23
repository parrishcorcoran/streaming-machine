"""One row: the same-step grid for every lane (circles cut at one place across the row, rounded to nearest),
packed to its least size (each weight's kept value written as one symbol), against the least any format can reach."""
import math, numpy as np
exec(open("circles.py").read().split("q8, c8 = q8_0(row)")[0])
x = row.astype(np.float64); L = np.linalg.norm(x); sd = math.sqrt(np.mean(x * x))
def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def turn(q): return math.degrees(math.acos(min(1, np.dot(x, q) / (L * np.linalg.norm(q)))))
def least(deg):   # the least bits a weight any format can reach for this turn, for a bell-shaped row of this size
    s = math.sin(math.radians(deg)); return 0.5 * math.log2(1 / s ** 2)
q4 = q4_0(row)[0].astype(np.float64); c4 = q4_0(row)[1]
print(f"4-bit (Q4_0): turned {turn(q4):.2f} deg, written 4.5, packed {H(c4) + 0.5:.2f}; least any format: {least(turn(q4)):.2f}")
for F in range(-6, -11, -1):
    step = 2.0 ** F; q = np.round(x / step) * step
    print(f"same step 2^{F} for every lane: turned {turn(q):5.2f} deg, packed {H(q) + 8/1024:.2f} bits a weight; "
          f"least any format at that turn: {least(turn(q)):.2f}")
