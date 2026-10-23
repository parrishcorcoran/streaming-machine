"""Phase 1a / Phase 3: the plates' sheets as stored. Each weight in up/hole form, one sheet per place, each place up,
hole or dark. AS IS: every place of every sheet kept (log2 3 bits a place). FLOOR: each sheet written at its own
content (the share of up, hole and dark at that place, per plate). Nothing dropped either way. Against ternary 1.58."""
import numpy as np, sys
sys.argv = ["x", "size"]; exec(open("plate_machine.py").read().split("C = {}")[0])
tot_w = raw = floor = 0; rows = []
for name in [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    w = M[name + ".w"] if name != "head" else M["head.w"]
    Ws = updown(w.astype(np.int32)); n = w.size
    used = [p for p in range(NPL) if Ws[p].any()]
    fl = 0.0
    for p in used:
        f = np.array([(Ws[p] == 1).sum(), (Ws[p] == -1).sum(), (Ws[p] == 0).sum()], float) / n
        f = f[f > 0]; fl += float(-(f * np.log2(f)).sum())
    raw += len(used) * np.log2(3) * n; floor += fl * n; tot_w += n
    if name in ("qkv.0", "fc.6", "mproj.11", "head"): rows.append(f"  {name:9s} places used {len(used)}: as is {len(used)*np.log2(3):5.2f} bits a weight, floor {fl:5.2f}")
    del Ws
print("\n".join(rows))
print(f"ALL PLATES ({tot_w:,} weights, the word plate included): as is {raw/tot_w:.2f} bits a weight ({raw/8e6:.0f} MB) | "
      f"floor {floor/tot_w:.2f} bits a weight ({floor/8e6:.0f} MB) | ternary 1.58 ({1.585*tot_w/8e6:.0f} MB) | 16-place whole numbers 16 ({16*tot_w/8e6:.0f} MB)")
