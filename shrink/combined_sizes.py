"""COMBINED: fill the size caches -- every plate under rule v12 and v14 at A = 3..10, the tables at c = 32..40 (cap 12)
-- each coded for real, decoded and checked; then print bytes by kind and by layer band. Sizes only, no words."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
t0 = time.time()
for rule in ("v12", "v14"):
    for A in range(3, 11):
        tot = {k: 0 for k in KINDS}; band = [0, 0, 0]
        for n in NAMES:
            b = plate_bytes(n, rule, A); tot[kind(n)] += b; band[layer(n) // 4] += b
        print(f"{rule} A={A:2d}  " + "  ".join(f"{k} {tot[k]/1e6:6.2f}" for k in KINDS) +
              f"   layers 0-3 {band[0]/1e6:6.2f}  4-7 {band[1]/1e6:6.2f}  8-11 {band[2]/1e6:6.2f}   all {sum(tot.values())/1e6:6.2f} MB"
              f"   ({time.time()-t0:.0f}s)", flush=True)
for c in range(32, 41):
    print(f"tables c={c} cap 12: {table_bytes(c)/1e6:.3f} MB", flush=True)
print("rest:", rest_bytes())
