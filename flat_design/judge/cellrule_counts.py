"""JUDGE: the exact read counts of the per-(gate, group) sign-run rule (pick 2), from the plates alone, for the build
to match number for number. Whole numbers only. Reuses runcount.py's cell count (each cell's identity asserted there).

The rule, fixed from the plate alone (the per-place rule's own 1-to-1 pricing: one read against one lit digit):
  for each (gate j, group k):  form C  reads c16 + 1 places of w + 2^15 (places 0..c16-1, then the sign sheet once at
                                       place c16), no signed digits
                               form B  the confirmed per-place cell (h up/hole places of the centred w_lo) with the run
                                       inside: c_lo + 1 reads, plus the lit signed digits of w_hi met directly
  take form C when c16 + 1 <= (c_lo + 1) + (the cell's lit w_hi digits); otherwise form B.
g: 12 at qkv and fc, 11 at proj and mproj (pick 1); groups keyed [1, slab, layer, g].
Also formC_everywhere_uh_reads: every cell in form C (the build's recommended form: up = all 16 places, the run in
every cell, no signed digits at all).
Writes cellrule_counts.json: per bank, the confirmed form's up/hole reads and lit top digits a position, and the rule's.
These are reads a POSITION (they do not depend on the light): the build's read COUNTS (not photons) must equal them.
    python3 flat_design/judge/cellrule_counts.py
"""
import sys, json, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/flat_design/judge")
import runcount as RC
HERE = RC.HERE; OUT = f"{HERE}/flat_design/judge/cellrule_counts.json"
Z = np.load(f"{HERE}/base2_gpt2.npz"); out = {}
for l in range(12):
    for kind in ("qkv", "proj", "fc", "mproj"):
        w = np.asarray(Z[f"{kind}.{l}.w"]).astype(np.int64); N, M = w.shape; g = RC.GREL[kind]
        up, _ = RC.PP.rule(w, [(kind, 0, M)], g); u = int(up[0]); h = u.bit_length(); assert u == (1 << h) - 1
        glines, goff = RC.R.make_groups(N, g, [1, RC.SLABS.index(kind), l, g])
        rB, tB = RC.cells(w, glines, goff, h); rC, _ = RC.cells(w, glines, goff, 16)
        useC = rC <= rB + tB
        out[f"{kind}.{l}"] = dict(g=g, h=h, cells=int(rB.size),
                                  confirmed_uh_reads=int(h * rB.size), confirmed_top_digits=int(tB.sum()),
                                  rule_uh_reads=int(np.where(useC, rC, rB).sum()), rule_top_digits=int(np.where(useC, 0, tB).sum()),
                                  cells_form_C=int(useC.sum()), formC_everywhere_uh_reads=int(rC.sum()))
        r = out[f"{kind}.{l}"]
        print(f"{kind}.{l:<2d} g={g} h={h}: reads {r['confirmed_uh_reads']:,} -> {r['rule_uh_reads']:,}; top digits "
              f"{r['confirmed_top_digits']:,} -> {r['rule_top_digits']:,}; form C {100 * r['cells_form_C'] / r['cells']:.1f}% of cells; "
              f"form C everywhere: reads {r['formC_everywhere_uh_reads']:,}, no top digits", flush=True)
json.dump(out, open(OUT, "w"), indent=1)
