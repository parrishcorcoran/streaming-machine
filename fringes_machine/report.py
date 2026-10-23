"""THE FRINGES MACHINE'S REPORT: work per token by part against plain, the total, and which cut gave what.
Reads only the saved counts (fringes_machine/refs_counts.json: the stream; top_refs.json: the top read; the controls)
and the run logs. Every number is a whole-number total over the 648 refs positions, shown per token (rounded half up,
as combined/machine.py's per_token does), and every share is computed from the totals.
THE CUTS, each against the machine before it (the four savings must add up exactly to plain - fringes):
  1. signed digits at all 48 banks, every light whole (as recording/C, but position 0 and layer 0 included), less the
     writing of every light in signed digits;
  2. layer 0 from storage, on top of 1 (fc.0 still met whole, in signed digits);
  3. the knife stops at every fc plate, on top of 1 and 2: the meetings past each stop, against the knife's own
     carrying to the edge's scale (which replaces the plain divide at fc), the shared bound, biases and bends;
  4. the top read carried down the fine diagonals with the pile-scale set-aside, against the plain read with its
     divide and standing amounts.
    python3 fringes_machine/report.py   -> fringes_machine/report.log
"""
import sys, os, json, re, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/report.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
S = json.load(open(f"{OUT}/refs_counts.json")); T = json.load(open(f"{OUT}/top_refs.json"))
n = S["npos"]; assert n == 648 and T["n"] == 648
CNT = dict(S["counts"]);
for k, v in T["counts"].items(): CNT[k] = CNT.get(k, 0) + v
PLAIN = 3666596259
def pt(x): return (x + n // 2) // n
def pc(a, b): return f"{100 * a / b:6.2f}%" if b else "     -"
def g(who=None, lay=None, part=None, item=None, item_re=None):
    tot = 0
    for k, v in CNT.items():
        w, l, p, i = k.split("|")
        if who is not None and w not in (who if isinstance(who, tuple) else (who,)): continue
        if lay is not None and l not in (lay if isinstance(lay, tuple) else (lay,)): continue
        if part is not None and p not in (part if isinstance(part, tuple) else (part,)): continue
        if item is not None and i != item: continue
        if item_re is not None and not re.search(item_re, i): continue
        tot += v
    return tot
PL_TOT, FR_TOT = g("plain"), g("fr")
log(f"THE FRINGES MACHINE ON THE 648 POSITIONS OF THE 30 REFS TEXTS (photon meetings and photons added, a token)")
log(f"  plain, rebuilt in the same run: {pt(PL_TOT):,} a token (the unit of all earlier results: {PLAIN:,}; same: {pt(PL_TOT) == PLAIN})")
log(f"  fringes machine, every add counted: {pt(FR_TOT):,} a token = {pc(FR_TOT, PL_TOT)} of plain")
log(f"  table reads (a count): {pt(g('reads')):,} a token; compares (a count): {pt(g('cmp')):,} a token")

# ---------------- by part ----------------
LAYS = [f"L{l}" for l in range(12)]
log("\nWORK PER TOKEN BY PART")
log(f"  {'part':58s} {'plain':>15s} {'fringes':>15s} {'share':>8s}")
def row(name, a, b): log(f"  {name:58s} {pt(a):>15,} {pt(b):>15,} {pc(b, a) if a else '':>8s}")
L0p = g("plain", "L0", ("qkv", "proj", "fc", "mproj")); L0f = g("fr", "L0", ("qkv", "proj", "fc", "mproj"))
row("layer 0, its four banks (meetings, stored reads, splits, knife)", L0p, L0f)
for part in ("qkv", "proj", "fc", "mproj"):
    row(f"    {part}.0", g("plain", "L0", part), g("fr", "L0", part))
row("    position 0's stream after layer 0 read (Z0)", 0, g("fr", "L0", "rest", "position 0: the stream after layer 0 read"))
for l in range(1, 12):
    lay = f"L{l}"
    log(f"  layer {l:2d}: " + "  ".join(f"{p} {pt(g('plain', lay, p)) / 1e6:6.1f}M -> {pt(g('fr', lay, p)) / 1e6:6.1f}M ({pc(g('fr', lay, p), g('plain', lay, p)).strip()})"
                                     for p in ("qkv", "proj", "fc", "mproj")))
for part in ("qkv", "proj", "fc", "mproj"):
    row(f"all layers: {part} (meetings; at fc also the knife's own work)", g("plain", None, part), g("fr", None, part))
row("the arrival (scores, shares, mix, divide), all layers", g("plain", None, "arrival"), g("fr", None, "arrival"))
row("the bends (fc's, all layers)", g("plain", None, "fc", item_re="^bends"), g("fr", None, "fc", item_re="^bends"))
row("the top read, all of it", g("plain", "top", ("top read",)) + g("plain", "top", "rest", item_re="^top (divide|standing)"),
    g("fr", "top", "top read"))
row("    its meetings", g("plain", "top", "top read", "meetings, plain"), g("fr", "top", "top read", "meetings, plain photons, staged"))
row("    its divide and standing amounts (plain: every word)", g("plain", "top", "rest", item_re="^top (divide|standing)"),
    g("fr", "top", "top read", item_re="^(standing amounts|words read to the end)"))
row("    its bound (tables, table values, carried down, poured, lows)", 0, g("fr", "top", "top read", item_re="^(bound|lows)"))
row("bounds: the knife's (shared bound, carrying to the edge's scale)", g("plain", None, "fc", item_re="^divides"),
    g("fr", None, "fc", item_re="^knife:"))
row("bounds: the top read's (as above, with its standing amounts)", 0, g("fr", "top", "top read", item_re="^(bound|lows|standing)"))
row("stored rows read and added, splits (layer 0)", 0, g("fr", "L0", None, item_re="^(stored|split|position 0)"))
row("the light's signed-digit writing (every bank)", 0, g("fr", None, None, item_re="^light written"))
row("the rest: water levels, bank divides (qkv), biases, stream adds",
    g("plain", None, "rest") - g("plain", "top", "rest", item_re="^top (divide|standing)") + g("plain", None, "fc", item_re="^biases"),
    g("fr", None, "rest") + g("fr", None, "fc", item_re="^biases"))
log(f"  {'TOTAL':58s} {pt(PL_TOT):>15,} {pt(FR_TOT):>15,} {pc(FR_TOT, PL_TOT):>8s}")

# ---------------- which cut gave what ----------------
log("\nWHICH CUT GAVE WHAT (each against the machine before it; photons a token, share of plain)")
pm = g("plain", None, ("qkv", "proj", "fc", "mproj"), item_re="^meetings")           # every plain bank meeting
sdw = (g("side", "L0", "qkv", "meetings, signed digits, whole light") + g("side", "L0", "proj", "meetings, signed digits, whole light")
       + g("side", "L0", "proj", "meetings, signed digits, position 0") + g("side", "L0", "fc", "meetings, signed digits, whole light")
       + g("side", "L0", "fc", "meetings, signed digits, position 0") + g("fr", "L0", "mproj", "meetings, signed digits")
       + g("side", "L0", "mproj", "meetings, signed digits, position 0")
       + sum(g("fr", f"L{l}", ("qkv", "proj", "mproj"), "meetings, signed digits") + g("side", f"L{l}", "fc", "meetings, signed digits, no stops") for l in range(1, 12)))
wrw = (g("side", "L0", None, "light written, whole light") + g("fr", "L0", "mproj", "light written in signed digits")
       + sum(g("fr", f"L{l}", None, "light written in signed digits") for l in range(1, 12)))
S_SD = pm - sdw - wrw
# layer 0: the signed-digits-only machine's layer 0 against the stored layer 0 with fc.0 still met whole
L0_items_plain_rest = g("plain", "L0", ("rest", "arrival")) + g("plain", "L0", "fc", item_re="^(divides|biases|bends)")
sd_only_L0 = (g("side", "L0", "qkv", "meetings, signed digits, whole light") + g("side", "L0", "proj", "meetings, signed digits, whole light")
              + g("side", "L0", "proj", "meetings, signed digits, position 0") + g("side", "L0", "fc", "meetings, signed digits, whole light")
              + g("side", "L0", "fc", "meetings, signed digits, position 0") + g("fr", "L0", "mproj", "meetings, signed digits")
              + g("side", "L0", "mproj", "meetings, signed digits, position 0")
              + g("side", "L0", None, "light written, whole light") + g("fr", "L0", "mproj", "light written in signed digits")
              + L0_items_plain_rest)
knife_fr = lambda lay: (g("fr", lay, "fc", "meetings, signed digits, edges stopped") + g("fr", lay, "fc", item_re="^knife:")
                        + g("fr", lay, "fc", item_re="^(biases|bends)"))
knife_plain = lambda lay: (g("side", lay, "fc", "meetings, signed digits, no stops")
                           + g("plain", lay, "fc", item_re="^(divides|biases|bends)$"))
stored_L0 = g("fr", "L0", ("qkv", "proj", "fc", "mproj", "rest", "arrival")) - knife_fr("L0") + knife_plain("L0")
S_L0 = sd_only_L0 - stored_L0
S_K = sum(knife_plain(f"L{l}") - knife_fr(f"L{l}") for l in range(12))
plain_top = g("plain", "top", "top read") + g("plain", "top", "rest", item_re="^top (divide|standing)")
S_T = plain_top - g("fr", "top", "top read")
log(f"  plain                                                   {pt(PL_TOT):>15,}  100.00%")
run = PL_TOT
for name, sv in (("1. signed digits at all 48 banks (less their writing)", S_SD), ("2. layer 0 from storage", S_L0),
                 ("3. knife stops at every fc plate (less their own work)", S_K), ("4. the top read down the fine diagonals", S_T)):
    run -= sv
    log(f"  {name:55s} saves {pt(sv):>13,} ({pc(sv, PL_TOT).strip()} of plain) -> {pt(run):>15,} {pc(run, PL_TOT)}")
log(f"  identity: plain - the four savings = the fringes total exactly: {run == FR_TOT} (remainder {run - FR_TOT})")
for l in range(12):
    lay = f"L{l}"
    log(f"    knife {lay:3s}: meetings past the stops {pt(g('side', lay, 'fc', 'meetings, signed digits, no stops') - g('fr', lay, 'fc', 'meetings, signed digits, edges stopped')):>11,}; "
        f"its own work {pt(g('fr', lay, 'fc', item_re='^knife:')):>10,} against the plain divide {pt(g('plain', lay, 'fc', item_re='^divides$')):>10,}; "
        f"net {pt(knife_plain(lay) - knife_fr(lay)):>11,}")
log("  the knife's own work, all layers: " + "; ".join(f"{i[7:]} {pt(v):,}" for i, v in sorted(
    {k.split('|')[3]: g('fr', None, 'fc', k.split('|')[3]) for k in CNT if k.split('|')[3].startswith('knife:')}.items())))

# ---------------- measurements and controls ----------------
log("\nKNIFE STOPS, MEASURED (fc's signed-digit meetings needed, all 12 layers)")
ms = S.get("measure", {})
if ms:
    full = sum(v["full"] for v in ms.values())
    log(f"  settle.py's rule in signed digits (per-edge bound, every stage): {pc(sum(v['per-edge work'] for v in ms.values()), full)}")
    log(f"  the shared bound, every stage, no gate:                          {pc(sum(v['shared work'] for v in ms.values()), full)}")
mach = g("fr", None, "fc", "meetings, signed digits, edges stopped"); fullm = g("side", None, "fc", "meetings, signed digits, no stops")
log(f"  the machine (shared bound, gate where the bound is narrower than the knife edge): {pc(mach, fullm)}")
K = np.load(f"{OUT}/refs_knife.npz")
hist = sum(np.bincount(K[f"stop_{l}"].ravel(), minlength=16) for l in range(12)); tot = hist.sum()
KD = list(range(2, 31, 2)) + [31]
log("  edges settled by each stage (cumulative): " + " ".join(f"top{k} {100 * c / tot:.1f}%" for k, c in zip(KD, np.cumsum(hist)) if c))
gates = np.concatenate([K[f"gate_{l}"] for l in range(12)]); gh = np.bincount(gates, minlength=16)
log("  gate opened at (positions x layers): " + " ".join(f"top{KD[i]} {100 * c / gh.sum():.1f}%" for i, c in enumerate(gh) if c))
if os.path.exists(f"{OUT}/controls.json"):
    Cj = json.load(open(f"{OUT}/controls.json")); Kc = Cj["K"]
    log("\nCONTROLS (6 refs texts; each light shuffled across its lines, every value kept)")
    for kind in ("qkv", "proj", "fc", "mproj"):
        log(f"  signed digits at {kind:5s}: real {pc(Kc[f'{kind}|real|signed'], Kc[f'{kind}|real|plain'])}, shuffled {pc(Kc[f'{kind}|shuffled|signed'], Kc[f'{kind}|shuffled|plain'])} of the plain meetings of the same light")
    cr = sum(Cj["C"].get(f"fr|L{l}|fc|meetings, signed digits, edges stopped", 0) for l in range(12)); cf = sum(Cj["C"].get(f"side|L{l}|fc|meetings, signed digits, no stops", 0) for l in range(12))
    sr = sum(Kc.get(f"knife|L{l}|shuffled|meetings, signed digits, edges stopped", 0) for l in range(12)); sf = sum(Kc.get(f"knife|L{l}|shuffled|meetings, signed digits, no stops", 0) for l in range(12))
    log(f"  knife stops: real {pc(cr, cf)}, shuffled {pc(sr, sf)}")
if os.path.exists(f"{OUT}/top_refs_shuffled.json"):
    Ts = json.load(open(f"{OUT}/top_refs_shuffled.json"))["counts"]
    a = sum(v for k, v in Ts.items() if k.startswith("fr|")); b = sum(v for k, v in Ts.items() if k.startswith("plain|"))
    a0 = g("fr", "top", "top read"); log(f"  top read: real {pc(a0, plain_top)}, light shuffled {pc(a, b)} of the plain read of the same light")
log("\nLOGS")
for f in ("refs.log", "windows_join.log", "judge.log", "top_refs.log", "refs_t0-14-29_order7.log", "top_refs_order7.log", "controls.log"):
    p = f"{OUT}/{f}"
    if os.path.exists(p):
        lines = [s for s in open(p).read().splitlines() if re.search(r"EXACT|identical|judge5|top 5 \(set|check path|checks:|joined|race winner", s)]
        for s in lines[:6]: log(f"  {f}: {s.strip()[:230]}")
