"""Reads plate_look_<size>.json (no machine run) and prints the tables for the report.
  python3 plate_look_report.py size=160m
"""
import os, sys, json
import numpy as np
A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "160m")
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, f"plate_look_{SIZE}{('-' + A['tag']) if A.get('tag') else ''}.json")))
R, SEEDS, LATE = J["cases"], J["seeds"], J["late"]
L = len(R["fin"]["wl_moved"]); KINDS = ("qkv", "proj", "fc", "mproj")
P = lambda v: f"{100 * v:5.1f}"
def cbs(p): return [R[f"cb:{p}@{s}"] for s in SEEDS if f"cb:{p}@{s}" in R]
def rng(vals): return f"{100*min(vals):5.1f}..{100*max(vals):5.1f}" if vals else "      -     "
def side(b, vals, tol):
    if not vals: return "?"
    if b < min(vals) - tol: return "b BELOW"
    if b > max(vals) + tol: return "b above"
    return "inside" if min(vals) <= b <= max(vals) else "near"

print(f"size {SIZE}  layers {L}  late {LATE}  seeds {SEEDS}")
print(f"finished first word share: {' '.join(P(a) for a in R['fin']['first_word'])}")
print(f"pairing-broken level moved: {' '.join(P(a) for a in R['control_finished_vs_next_text']['wl_moved'])}")
print(f"finished gap (e-folds) by layer: {R['fin_gap_median_efolds']}")

print("\n1. words: top-1 of all positions, b against the three scrambled draws (tolerance 2 points)")
for k in KINDS:
    for l in range(L):
        p = f"{k}.{l}"; b = R.get(f"b:{p}")
        if not b: continue
        c = cbs(p); ct = [x["top1_all"] for x in c]
        print(f"  {p:8s} b {P(b['top1_all'])}  cb {rng(ct)}  {side(b['top1_all'], ct, 0.02):8s}  top-5 b {P(b['top5_all'])} "
              f"cb {rng([x['top5_all'] for x in c])}  greedy texts b {b['greedy_texts']:2d} cb {[x['greedy_texts'] for x in c]}  "
              f"greedy words b {b['greedy_words']} cb {[x['greedy_words'] for x in c]}")

print("\n2. the level: rows whose level moves at each layer, b (and cb min..max), by plate")
for k in KINDS:
    for l in range(L):
        p = f"{k}.{l}"; b = R.get(f"b:{p}")
        if not b: continue
        c = cbs(p)
        cells = []
        for m in range(L):
            if b["wl_moved"][m] == 0 and all(x["wl_moved"][m] == 0 for x in c): cells.append("    .     "); continue
            cv = [x["wl_moved"][m] for x in c]
            cells.append(f"{100*b['wl_moved'][m]:3.0f}({100*min(cv):2.0f}-{100*max(cv):2.0f})" if cv else f"{100*b['wl_moved'][m]:3.0f}")
        print(f"  {p:8s} " + " ".join(f"{s:10s}" for s in cells))

print("\n3. the first word's share of the levels at the late layers, b (cb min..max); finished: "
      + " ".join(P(R['fin']['first_word'][m]) for m in LATE))
for k in KINDS:
    for l in range(L):
        p = f"{k}.{l}"; b = R.get(f"b:{p}")
        if not b: continue
        c = cbs(p)
        print(f"  {p:8s} " + "  ".join(f"L{m} {P(b['first_word'][m])} ({rng([x['first_word'][m] for x in c])})" for m in LATE))

print("\n4. knife edges that change side, b (cb min..max), at the plate's first reachable layer and the mean of the later ones")
for k in KINDS:
    for l in range(L):
        p = f"{k}.{l}"; b = R.get(f"b:{p}")
        if not b: continue
        c = cbs(p); f0 = l if k == "fc" else l + 1
        if f0 >= L: print(f"  {p:8s} (reaches no knife edge)"); continue
        later = list(range(f0 + 1, L))
        mb = np.mean([b["ke_flipped"][m] for m in later]) if later else None
        mc = [np.mean([x["ke_flipped"][m] for m in later]) for x in c] if later else []
        print(f"  {p:8s} first L{f0}: b {P(b['ke_flipped'][f0])} cb {rng([x['ke_flipped'][f0] for x in c])}   "
              + (f"later mean: b {P(mb)} cb {rng(mc)}" if later else ""))

print("\n5. by kind, every layer at once")
for k in KINDS:
    p = f"K={k}"; b = R.get(f"b:{p}")
    if not b: continue
    c = cbs(p); ct = [x["top1_all"] for x in c]
    print(f"  {k:6s} top-1 b {P(b['top1_all'])} cb {rng(ct)} {side(b['top1_all'], ct, 0.02)}; top-5 b {P(b['top5_all'])} "
          f"cb {rng([x['top5_all'] for x in c])}; greedy texts b {b['greedy_texts']} cb {[x['greedy_texts'] for x in c]}; "
          f"greedy words b {b['greedy_words']} cb {[x['greedy_words'] for x in c]}")
    print(f"         level moved b  {' '.join(f'{100*a:3.0f}' for a in b['wl_moved'])}")
    for x, s in zip(c, SEEDS): print(f"         level moved cb {' '.join(f'{100*a:3.0f}' for a in x['wl_moved'])}   (draw {s})")
    print(f"         first word  b  {' '.join(f'{100*a:3.0f}' for a in b['first_word'])}")
    for x, s in zip(c, SEEDS): print(f"         first word  cb {' '.join(f'{100*a:3.0f}' for a in x['first_word'])}")
    print(f"         edges       b  {' '.join(f'{100*a:3.0f}' for a in b['ke_flipped'])}")
    for x, s in zip(c, SEEDS): print(f"         edges       cb {' '.join(f'{100*a:3.0f}' for a in x['ke_flipped'])}")
    print(f"         finished gap (e-folds) moved rows {b['gap_moved']}")
    print(f"         finished gap (e-folds) held rows  {b['gap_held']}")

print("\n6. the look: finished gap (e-folds, level minus the next score) of rows that moved / held, b, own reachable layer")
for k in KINDS:
    for l in range(L):
        p = f"{k}.{l}"; b = R.get(f"b:{p}")
        if not b: continue
        f0 = l if k == "qkv" else l + 1
        if f0 >= L: continue
        print(f"  {p:8s} L{f0}: moved {b['gap_moved'][f0]}  held {b['gap_held'][f0]}"
              + (f"   | L{L-1}: moved {b['gap_moved'][L-1]} held {b['gap_held'][L-1]}" if f0 != L - 1 else ""))

print("\n7. real against scramble, per kind over its single plates: b below all draws / inside / above all draws;"
      " sum over plates of top-1 points lost, b against each draw")
for k in KINDS + ("all",):
    ps = [f"{kk}.{l}" for kk in (KINDS if k == "all" else (k,)) for l in range(L) if f"b:{kk}.{l}" in R]
    below = inside = above = 0; lb = 0.0; lc = np.zeros(len(SEEDS))
    for p in ps:
        b = R[f"b:{p}"]["top1_all"]; ct = [R[f"cb:{p}@{s}"]["top1_all"] for s in SEEDS]
        below += b < min(ct); above += b > max(ct); inside += min(ct) <= b <= max(ct)
        lb += 1 - b; lc += 1 - np.array(ct)
    print(f"  {k:6s} plates {len(ps):2d}: below {below:2d} inside {inside:2d} above {above:2d}  (exchangeable: 1/4, 1/2, 1/4)  "
          f"points lost b {100*lb:6.1f}  cb {' '.join(f'{100*x:6.1f}' for x in lc)}")
print("\n8. the late level, per kind over its single plates: rows moved at each late layer, summed over the kind's plates, b / cb draws")
for k in KINDS:
    ps = [f"{k}.{l}" for l in range(L) if f"b:{k}.{l}" in R]
    for m in LATE:
        sb = sum(R[f"b:{p}"]["wl_moved"][m] for p in ps); sc = [sum(R[f"cb:{p}@{s}"]["wl_moved"][m] for p in ps) for s in SEEDS]
        own = R[f"b:{k}.{m}"]["wl_moved"][m] if k == "qkv" else None
        print(f"  {k:6s} L{m}: b {100*sb:5.1f}  cb {' '.join(f'{100*x:5.1f}' for x in sc)}" + (f"   (of which its own qkv.{m}: {100*own:5.1f})" if own is not None else ""))
