"""Each weight's angle written as bits of pi (one shared half turn; bit 1 = pi/2, bit 2 = pi/4, ...), the length once.
How many leading bits every weight shares, and the turn and packed size at each number of bits of pi."""
import math, numpy as np
exec(open("circles.py").read().split("q8, c8 = q8_0(row)")[0])
x = row.astype(np.float64); L = np.linalg.norm(x)
t = np.arccos(x / L) / math.pi                                       # each angle as a fraction of pi
print(f"angles run from {180*t.min():.2f} to {180*t.max():.2f} degrees (all within {180*np.max(np.abs(t-0.5)):.2f} of 90)")
def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def turn(q): return math.degrees(math.acos(min(1, np.dot(x, q) / (L * np.linalg.norm(q)))))
for b in range(7, 13):
    code = np.round(t * 2 ** b).astype(np.int64)
    q = L * np.cos(math.pi * code / 2 ** b)
    bits = [(code >> (b - 1 - k)) & 1 for k in range(b)]               # bit k of every weight's angle
    shared = sum(1 for k in range(b) if np.all(bits[k] == bits[k][0]))
    print(f"{b:2d} bits of pi: turned {turn(q):5.2f} deg; bits the same in every weight {shared}; "
          f"written {b} bits, packed {H(code):.2f} bits a weight")
