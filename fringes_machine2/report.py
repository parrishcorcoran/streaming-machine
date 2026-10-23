"""BASE2_FRINGES2's REPORT AT ONE LEVEL OF ADDITIONS: work per token by part against plain and against base2_fringes.py.
Reads only saved counts: the stream (fringes_machine2/refs_counts_<stream>.json) and the top read
(fringes_machine2/top_<top>_refs.json), and base2_fringes.py's own (fringes_machine/refs_counts.json, top_refs.json).
Every number is a whole-number total over the 648 refs positions, shown per token (rounded half up, per_token's rule).
    python3 fringes_machine2/report.py add=N stream=aX top=aY   -> fringes_machine2/report_a<N>.log
"""
import sys, os, json, re
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/fringes_machine2"; OLD = f"{HERE}/fringes_machine"
ARG = dict(a.split("=") for a in sys.argv[1:]); ADD = int(ARG["add"]); ST, TP = ARG["stream"], ARG["top"]
LOG = open(f"{OUT}/report_a{ADD}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def load(sp, tp):
    S = json.load(open(sp)); T = json.load(open(tp)); assert S["npos"] == 648 and T["n"] == 648
    C = dict(S["counts"])
    for k, v in T["counts"].items(): C[k] = C.get(k, 0) + v
    return C, S
NEW, SN = load(f"{OUT}/refs_counts_{ST}.json", f"{OUT}/top_{TP}_refs.json")
OLDC, _ = load(f"{OLD}/refs_counts.json", f"{OLD}/top_refs.json")
n = 648; PLAIN = 3666596259
def pt(x): return (x + n // 2) // n
def pc(a, b): return f"{100 * a / b:6.2f}%" if b else "     -"
def g(C, who=None, lay=None, part=None, item_re=None):
    tot = 0
    for k, v in C.items():
        w, l, p, i = k.split("|")
        if who is not None and w not in (who if isinstance(who, tuple) else (who,)): continue
        if lay is not None and l not in (lay if isinstance(lay, tuple) else (lay,)): continue
        if part is not None and p not in (part if isinstance(part, tuple) else (part,)): continue
        if item_re is not None and not re.search(item_re, i): continue
        tot += v
    return tot
PL = g(NEW, "plain"); assert PL == g(OLDC, "plain"), "the plain count differs between the two runs"
FR, FO = g(NEW, "fr"), g(OLDC, "fr")
log(f"BASE2_FRINGES2, ADD={ADD} (stream counts {ST}, top read {TP}), 648 refs positions; photon meetings and photons added, a token")
log(f"  plain, rebuilt in the same run: {pt(PL):,} (the unit of all earlier results {PLAIN:,}; same: {pt(PL) == PLAIN})")
log(f"  base2_fringes.py:               {pt(FO):,} = {pc(FO, PL)} of plain")
log(f"  base2_fringes2.py, ADD={ADD}:        {pt(FR):,} = {pc(FR, PL)} of plain; against base2_fringes.py {pc(FR, FO)} "
    f"(saves {pt(FO - FR):,} a token, {pc(FO - FR, PL).strip()} of plain)")
log(f"  compares (a count): {pt(g(NEW, 'cmp')):,} (base2_fringes.py {pt(g(OLDC, 'cmp')):,}); table reads (a count): {pt(g(NEW, 'reads')):,}")
log(f"\n  {'part':62s} {'plain':>14s} {'base2_fringes':>14s} {'this':>14s} {'of plain':>9s} {'of fringes':>10s}")
def row(name, a, b, c): log(f"  {name:62s} {pt(a):>14,} {pt(b):>14,} {pt(c):>14,} {pc(c, a) if a else '':>9s} {pc(c, b) if b else '':>10s}")
for part in ("qkv", "proj", "fc", "mproj"):
    row(f"{part}, all layers (meetings, layer-0 stored reads; fc: the knife)", g(NEW, "plain", None, part), g(OLDC, "fr", None, part), g(NEW, "fr", None, part))
row("the arrival, all layers", g(NEW, "plain", None, "arrival"), g(OLDC, "fr", None, "arrival"), g(NEW, "fr", None, "arrival"))
for it in ("^scores", "^shares x values", "^shares' total", "^divides", "written in signed digits"):
    row(f"    {it.strip('^')}", g(NEW, "plain", None, "arrival", it), g(OLDC, "fr", None, "arrival", it), g(NEW, "fr", None, "arrival", it))
ptop = g(NEW, "plain", "top", "top read") + g(NEW, "plain", "top", "rest", "^top (divide|standing)")
row("the top read, all of it", ptop, g(OLDC, "fr", "top", "top read"), g(NEW, "fr", "top", "top read"))
row("    its meetings", g(NEW, "plain", "top", "top read", "^meetings"), g(OLDC, "fr", "top", "top read", "^meetings"), g(NEW, "fr", "top", "top read", "^meetings"))
row("    standing amounts (divides, poured), words read to the end", g(NEW, "plain", "top", "rest", "^top (divide|standing)"),
    g(OLDC, "fr", "top", "top read", "^(standing|words read)"), g(NEW, "fr", "top", "top read", "^(standing|words read)"))
row("    the bound (tables, table values, carried, poured, lows)", 0, g(OLDC, "fr", "top", "top read", "^(bound|lows)"), g(NEW, "fr", "top", "top read", "^(bound|lows)"))
row("    the start rule (the light's largest unread part, the spread)", 0, g(OLDC, "fr", "top", "top read", "^start"), g(NEW, "fr", "top", "top read", "^start"))
row("    the light written in signed digits", 0, g(OLDC, "fr", "top", "top read", "^light written"), g(NEW, "fr", "top", "top read", "^light written"))
for it in sorted({k.split("|")[3] for k in NEW if k.startswith("fr|top|top read|")}):
    log(f"        {it:80s} {pt(g(NEW, 'fr', 'top', 'top read', '^' + re.escape(it) + '$')):>14,}")
row("the rest: water levels, bank divides, biases, stream adds", g(NEW, "plain", None, "rest") - g(NEW, "plain", "top", "rest", "^top (divide|standing)"),
    g(OLDC, "fr", None, "rest"), g(NEW, "fr", None, "rest"))
row("TOTAL", PL, FO, FR)
for C_, T_ in ((NEW, FR), (OLDC, FO)):
    s_ = sum(g(C_, "fr", None, p) for p in ("qkv", "proj", "fc", "mproj", "arrival", "rest")) + g(C_, "fr", "top", "top read")
    assert s_ == T_, ("the parts do not add up to the total", s_, T_)
log("  (the parts add up to each total exactly)")
if any("shared order" in k for k in NEW):
    log("\nADDITION 4, MEASURED ON THE REAL LIGHT (count only: the rule kept every plate in signed digits)")
    for part in ("qkv", "proj", "fc", "mproj"):
        parts = {i: g(NEW, "side", None, part, "^shared order: " + re.escape(i) + "$") for i in
                 ("gaps x running sum, signed digits", "pours into the one running sum", "running sum written in signed digits",
                  "against signed digits, meetings", "against signed digits, writing")}
        sh = sum(v for k, v in parts.items() if not k.startswith("against")); sd = sum(v for k, v in parts.items() if k.startswith("against"))
        log(f"  {part:6s}: shared order {pt(sh):>13,} (gaps x sum {pt(parts['gaps x running sum, signed digits']):,}, pours "
            f"{pt(parts['pours into the one running sum']):,}, rewriting {pt(parts['running sum written in signed digits']):,}); "
            f"signed digits {pt(sd):>13,}: {pc(sh, sd)}")
    r4 = SN.get("rule4", {})
    if r4: log(f"  the rule on the plates seen in this run: {sum(v[2] for v in r4.values())} of {len(r4)} take the shared order")
