"""confirm_groups/orderfree_compare.py -- THE ORDER-FREE CHECK, SET SIDE BY SIDE (reads only this folder's records).

Two runs of rebuild_gpt2.py on refs texts 0, 1, 2 (62 positions): seed 1 in the draw's own order
(refs_s1_t0-1-2.json, _top.npz) and seed 1 with reorder=5 (refs_s1_t0-1-2_order5.json, _top.npz): at every bank the
groups in a random order, each group's lines in a random order and the up/hole places summed in a random order; at
the top the groups and their lines reordered and the words relabelled. Both runs check every array against
plain_refs.npz on their own (their logs).

PREDICTION (written before this file ran): every pile, stream and top 5 identical (both logs: 0 arrays differing,
62/62); every count identical item by item except the half tables (bank "table" items and the top "table"), which
move by under 0.01 point of plain; the top's survivors after every place identical (the read part after places >= b
is the same sum for any grouping).
    python3 confirm_groups/orderfree_compare.py
"""
import json, numpy as np
H = "/Users/abundancemachine/streaming-machine/confirm_groups"
a = json.load(open(f"{H}/refs_s1_t0-1-2.json")); b = json.load(open(f"{H}/refs_s1_t0-1-2_order5.json"))
ta = np.load(f"{H}/refs_s1_t0-1-2_top.npz"); tb = np.load(f"{H}/refs_s1_t0-1-2_order5_top.npz")
ca, cb = a["counts"], b["counts"]; NP = a["npos"]; PLAIN = 3666596259
same = [k for k in ca if ca[k] == cb.get(k)]; diff = [k for k in ca if ca[k] != cb.get(k)]
print(f"bank, arrival, rest and side counts: {len(same)} identical, {len(diff)} differ")
for k in diff: print(f"  {k:60s} {ca[k]:16,d} {cb[k]:16,d} ({(cb[k] - ca[k]) / NP:+,.0f} a token, {100 * (cb[k] - ca[k]) / NP / PLAIN:+.4f} points)")
items = sorted(set(ta.files) & set(tb.files) - {"rows"})
for k in items:
    eq = np.array_equal(ta[k], tb[k])
    print(f"  top {k:8s} identical {eq}" + ("" if eq else f"  ({(int(tb[k].sum()) - int(ta[k].sum())) / NP:+,.0f} a token, "
                                                         f"{100 * (int(tb[k].sum()) - int(ta[k].sum())) / NP / PLAIN:+.4f} points)"))
print(f"whole: {a['whole'] / NP:,.0f} against {b['whole'] / NP:,.0f} a token ({100 * (b['whole'] - a['whole']) / NP / PLAIN:+.4f} points)")
