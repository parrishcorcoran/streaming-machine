"""ONLY THE EXTRA -- the table, read from onlyextra/results.json (nothing re-run).
    python3 onlyextra/summary.py [tag=]
"""
import sys, json, numpy as np
ARG = dict(a.split("=", 1) for a in sys.argv[1:]); TAG = ARG.get("tag", "")
R = json.load(open(f"/Users/abundancemachine/streaming-machine/onlyextra/results{TAG}.json"))
NV_PLATES, NV_E = 84_934_656, 50257 * 768; NV = NV_PLATES + NV_E
LENS_BITS = 2892 * 8; CELL = 40
TERNARY = 1.584962500721156 * NV / 8e6; FOUR = 4 * NV / 8e6; RECORD = 80.711
full = R.get("full")
PLAIN_ONES = full["work"]["plates_ones"] + full["work"]["top_ones"] if full else None
PLAIN_SW = full["work"]["plates"] + full["work"]["top"] if full else None

def size(row, which="chain"):
    p, t = row["size"]["plates"], row["size"]["table"]
    if f"{which}_bits" not in t or not p.get(f"{which}_bits"): return None
    pb = p[f"{which}_bits"] + CELL * p[f"{which}_cells"]; tb = t[f"{which}_bits"] + CELL * t[f"{which}_cells"] + LENS_BITS
    return dict(plates_bpw=pb / NV_PLATES, table_bpv=tb / NV_E, bpw=(pb + tb) / NV, MB=(pb + tb) / 8e6, plates_MB=pb / 8e6, table_MB=tb / 8e6)

print(f"Reference sizes: ternary {TERNARY:.1f} MB (1.58 bits), 4-bit {FOUR:.1f} MB, the record {RECORD} MB (about "
      f"{RECORD * 8e6 / NV:.2f} bits a value), as written 16-bit plates + 18-bit word table 256.7 MB")
if full: print(f"Plain work a token (full machine, judge5's windows): plates {full['work']['plates_ones']:,.0f} + top read "
               f"{full['work']['top_ones']:,.0f} = {PLAIN_ONES:,.0f} meetings in plain base-2 ones; in single waves {PLAIN_SW:,.0f}")
print()
hdr = f"{'case':22s} {'full word in case top5':>22s} {'case word in full top5':>22s} {'same top':>9s} {'greedy (full)':>14s} {'greedy (fp32)':>14s} " \
      f"{'lit/w':>6s} {'bits/w':>7s} {'MB':>7s} {'work vs plain':>14s} {'vs full sw':>10s}"
print(hdr); print("-" * len(hdr))
for name, row in R.items():
    J = row["judged"]; g = row["greedy_vs_full"]; gf = row["greedy_vs_fp32"]; s = size(row)
    w = row["work"]["plates"] + row["work"]["top"]
    print(f"{name:22s} {100 * row['full_word_in_case_top5'] / J:21.2f}% {100 * row['in_full_top5'] / J:21.2f}% {100 * row['same_top'] / J:8.2f}% "
          f"{(str(g['words']) + '/480 ' + str(g['texts']) + '/30') if g else '-':>14s} {str(gf['words']) + '/480 ' + str(gf['texts']) + '/30':>14s} "
          f"{row['size']['plates']['lit_kept']:6.3f} {('%.3f' % s['bpw']) if s else '-':>7s} {('%.1f' % s['MB']) if s else '-':>7s} "
          f"{(100 * w / PLAIN_ONES) if PLAIN_ONES else 0:13.2f}% {(100 * w / PLAIN_SW) if PLAIN_SW else 0:9.2f}%")
print()
print("Stored size at the floor (chain: where each kept photon sits and its sign, counts per plate; joint: the kept value as one symbol):")
for name, row in R.items():
    s, sj = size(row), size(row, "joint")
    if not s: continue
    print(f"  {name:20s} plates {s['plates_bpw']:.3f} bits a weight ({s['plates_MB']:.1f} MB), word table {s['table_bpv']:.3f} bits a value "
          f"({s['table_MB']:.1f} MB) -> {s['bpw']:.3f} bits, {s['MB']:.1f} MB  [joint {sj['bpw']:.3f} bits, {sj['MB']:.1f} MB]  "
          f"x{s['MB'] / TERNARY:.2f} ternary, x{s['MB'] / FOUR:.2f} 4-bit, x{s['MB'] / RECORD:.2f} the record")
print()
print("Work a token (single-wave meetings of kept light photons with kept plate photons):")
for name, row in R.items():
    wk = row["work"]
    print(f"  {name:20s} plates {wk['plates']:>14,.0f}  top read {wk['top']:>13,.0f}  (plain ones of the same: {wk['plates_ones']:,.0f} + {wk['top_ones']:,.0f})"
          f"  top table {row['size']['top_lit_per_value']:.3f} lit a value; word table lit {row['size']['table']['lit_kept']:.3f}")
for name, row in R.items():
    if "order_free" in row: print(f"\nORDER-FREE ({name}): {row['order_free']}")
