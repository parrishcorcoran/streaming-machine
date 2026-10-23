"""PLANES family, version 9: two exposure meters and no identities. Version 8b gave every neuron its own chances; a
neuron of a 768-line plate has only 768 values to learn ~150 chances from, and the learning cost (about 1/2 log2 n
bits for every chance a context has to learn) ate more than the neuron's own histogram was worth (proj plates got
worse by 0.06-0.16 bits; only mproj, 3,072 values a neuron, and fc.0 gained). So keep only what is shared by all:
  context of a still-dark pixel = place x (bit length of how many values of its ROW are lit above this place)
                                        x (bit length of how many values of its COLUMN are lit above this place)
  -- 17 x 12 x 18 chances for a whole plate, learned from all its values. A row of a plate is a line in (its layer-norm
  gain is folded in, so some rows are brighter), a column is a neuron. Lit values: version 1 context. Sign: one chance.
Run on all 48 plates, and on E and T (where the column meter replaces the line number).
Prediction: plates gain on version 8a (the folded gain makes rows of different brightness, and the meters read it
without paying for 768 x 17 identities); on E and T the meters lose a little to the line number (E has 50,257 words a
line, plenty to learn a line's own histogram, which is richer than one brightness count).
    python3 planes_v9_meters.py
"""
import sys, json; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from planes_lib import *
MODE = M_HIGH | M_WEXP | M_LEXP
for tb in ["E", "T"]:
    A = load(tb); run("v9 two exposure meters (row x column), no identities", tb, A, MODE); del A
names = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
v8 = {r["plate"]: r for r in json.load(open("/Users/abundancemachine/streaming-machine/shrink/planes_out_v8_plates.json"))["plates"]}
tot, tot8a, n, rows = 0.0, 0.0, 0, []
for nm in names:
    A = load(nm)
    r, _ = run("v9 plate, two exposure meters", nm, A, MODE)
    tot += r["bits_per_value"] * A.size; tot8a += v8[nm]["v8a"] * A.size; n += A.size
    rows.append(dict(plate=nm, v9=r["bits_per_value"], v8a=v8[nm]["v8a"], ok=r["roundtrip_ok"]))
    del A
json.dump(dict(values=n, v9_bits=round(tot / n, 4), v9_MB=round(tot / 8e6, 3), v8a_bits=round(tot8a / n, 4),
               all_roundtrip_ok=all(r["ok"] for r in rows), plates=rows),
          open("/Users/abundancemachine/streaming-machine/shrink/planes_out_v9_plates.json", "w"), indent=1)
print(f"ALL 48 PLATES v9: {tot / n:.4f} bits/value = {tot / 8e6:.2f} MB (v8a {tot8a / n:.4f}); all roundtrip {all(r['ok'] for r in rows)}")
