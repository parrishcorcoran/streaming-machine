"""FAX family, look before coding: how lit is each place's page, plain base 2 and single waves (up/hole), and how much
the top lit place of a value (where its light starts) depends on its line and on its neighbours. Reference numbers
only (Shannon floors from histograms); the real stored sizes come from the coders in fax_v*.py.
    python3 fax_look.py E|T|fc.0
"""
import sys, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

def H(*cols):                                   # empirical joint entropy (bits per sample) of whole-number columns
    key = np.zeros(len(cols[0]), np.int64)
    for c in cols: key = key * (int(c.max()) + 1) + c
    _, n = np.unique(key, return_counts=True); p = n / n.sum(); return float(-(p * np.log2(p)).sum())
def hb(p): return 0.0 if p <= 0 or p >= 1 else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))

name = sys.argv[1] if len(sys.argv) > 1 else "E"
A = load(name); R, C = A.shape; N = A.size; a = np.abs(A); nb = nplanes(A)
print(f"{name}: {R} x {C}, {N:,} values, {nb} places + sign; order-0 floor {H((A - A.min()).ravel().astype(np.int64)):.3f} bits/value")
P, Q = naf_masks(A)
print("  place  plain lit share  h(share)  | single-wave lit share  up share  h(lit)+lit share (direction)")
s1 = s2 = 0.0
for k in range(nb + 1):
    p = float(((a >> k) & 1).mean()); l = float((((P | Q) >> k) & 1).mean())
    s1 += hb(p); s2 += hb(l) + l
    print(f"  {k:5d}  {p:15.4f}  {hb(p):8.4f}  | {l:21.4f}  {float(((P >> k) & 1).mean()):8.4f}  {hb(l) + l:8.4f}")
nz = float((A != 0).mean())
print(f"  every page on its own (no page knows another): plain {s1:.3f} + sign {nz:.3f} = {s1 + nz:.3f} bits/value;"
      f" single waves {s2:.3f} bits/value")
lead = np.zeros(A.shape, np.int64)              # top lit place + 1 (0 for a dark value): where the value's light starts
for k in range(nb): lead[(a >> k) != 0] = k + 1
line = np.broadcast_to(np.arange(C), A.shape)
left = np.zeros_like(lead); left[:, 1:] = lead[:, :-1]
up = np.zeros_like(lead); up[1:] = lead[:-1]
L, Li, Lf, Lu = lead.ravel(), line.ravel(), left.ravel(), up.ravel()
h0 = H(L); hl = H(L, Li) - H(Li); hlf = H(L, Li, Lf) - H(Li, Lf); hlu = H(L, Li, Lu) - H(Li, Lu)
print(f"  top lit place: H = {h0:.3f};  given its line {hl:.3f};  given line + the place of the line to its left "
      f"{hlf:.3f};  given line + the same line of the word before {hlu:.3f}")
below = np.maximum(lead - 1, 0).ravel()          # places below the top lit one: their pages are what is left
print(f"  places below the top lit one (every one of them is a coin flip if the light is grain): {below.mean():.3f} per value;"
      f"  + top place {h0:.3f} + sign {nz:.3f} = {below.mean() + h0 + nz:.3f} bits/value")
