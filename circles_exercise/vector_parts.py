"""The row is one vector. Each weight = the row's length x the cosine of the angle between the row and that lane.
For bf16, Q8_0, Q4_0: the row's length, and each weight's cosine and angle."""
import math, numpy as np
exec(open("circles.py").read().split("# --- base-2 circles")[0])   # loads row, q8, q4
forms = {"bf16": row.astype(np.float64), "8-bit": q8.astype(np.float64), "4-bit": q4.astype(np.float64)}
P = {}
for k, x in forms.items():
    L = float(np.linalg.norm(x)); c = x / L; a = np.degrees(np.arccos(c))
    P[k] = (L, c, a)
    print(f"{k:6s} length {L:.7f}   cosines squared add to {np.sum(c*c):.12f}")
a16 = P["bf16"][2]
for k in ("8-bit", "4-bit"):
    d = np.abs(P[k][2] - a16)
    print(f"{k:6s} vs bf16: length {100*(P[k][0]/P['bf16'][0]-1):+.3f}%   each weight's angle moved: mean {d.mean():.4f} deg, "
          f"largest {d.max():.4f} deg; moved to exactly 90 deg (became 0): {int(np.sum((P[k][1]==0)&(P['bf16'][1]!=0)))}")
with open("vector_parts.tsv", "w") as f:
    f.write("i\t" + "\t".join(f"{k}_weight\t{k}_length\t{k}_cosine\t{k}_degrees" for k in forms) + "\n")
    for i in range(row.size):
        f.write(f"{i}\t" + "\t".join(f"{forms[k][i]:.9g}\t{P[k][0]:.7f}\t{P[k][1][i]:.7f}\t{P[k][2][i]:.4f}" for k in forms) + "\n")
print()
print("i  | bf16: cosine  degrees | 8-bit: cosine  degrees | 4-bit: cosine  degrees")
for i in range(8):
    print(f"{i}  | {P['bf16'][1][i]:+.5f}  {P['bf16'][2][i]:8.4f} | {P['8-bit'][1][i]:+.5f}  {P['8-bit'][2][i]:8.4f} | "
          f"{P['4-bit'][1][i]:+.5f}  {P['4-bit'][2][i]:8.4f}")
