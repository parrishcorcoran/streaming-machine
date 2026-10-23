"""FAX family, version 6 (JBIG contexts) on every plate of the machine, the line context being the plate's lines in
(each plate turned so its lines in run across the page). Lossless, decoded and checked equal, real bytes.
    python3 fax_v6_plates.py
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *
import fax_v6_jbig as v6

tot_bits = tot_vals = 0; allok = True; rows = []
for L in range(12):
    for b in ("qkv", "proj", "fc", "mproj"):
        nm = f"{b}.{L}"; A = np.ascontiguousarray(load(nm).T)
        bufs, nb = v6.code(A); bits = 8 * sum(len(x) for x in bufs) + 2 * 32 + 5 + 64 * len(bufs)
        ok = bool(np.array_equal(v6.decode(bufs, nb, A.shape), A)); allok &= ok
        tot_bits += bits; tot_vals += A.size; rows.append((nm, bits / A.size, ok))
    print(f"  layer {L}: " + "  ".join(f"{n} {x:.2f}" for n, x, _ in rows[-4:]), flush=True)
print(f"  all 48 plates: {tot_vals:,} values, {tot_bits / tot_vals:.3f} bits/value, {tot_bits / 8e6:.1f} MB "
      f"(stored as 16-bit: {tot_vals * 2 / 1e6:.1f} MB), every plate decoded equal: {allok}")
json.dump(dict(bits_per_value=tot_bits / tot_vals, MB=tot_bits / 8e6, ok=allok, plates=rows),
          open("/Users/abundancemachine/streaming-machine/shrink/fax_out_v6_plates.json", "w"))
