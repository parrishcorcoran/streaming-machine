"""DESIGNER 3: the summary of out/main.json, shuffle.json, seed2.json (sample: 8 plates, 16 positions), with the
whole-machine projection on the confirmed group machine's parts (docs/09: GPT-2 711,431,068 a token of 3,666,596,259).
    python3 flat_design/3/report.py -> out/report.log
"""
import json, sys, math, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/flat_design/3/out"
sys.argv = [sys.argv[0]]; sys.path.insert(0, f"{HERE}/flat_design/3"); import measure as MS
R = {r: json.load(open(f"{D}/{r}.json"))["plates"] for r in ("main", "shuffle", "seed2")}
LOG = open(f"{D}/report.log", "w")
def log(s=""): print(s); LOG.write(s + "\n")
pct = lambda a, b: 100.0 * a / b
P = list(R["main"].keys())

log("A. THE EXTRA IN THREE DIGIT FORMS (each pile rebuilt from its tips alone: 0 differ everywhere)")
log("   plate     plain/pos    uh extra/meet  (shuf)   uh extra/plain  uh tips' digits/plain | 1s extra/plain (shuf) | sd meet/plain  sd extra/plain  sd tips/plain")
for k in P:
    A, S = R["main"][k]["A"], R["shuffle"][k]["A"]; pl = A["plain"]
    log(f"   {k:8s} {pl / 16:13,.0f}   {pct(A['uh']['extra'], A['uh']['meet']):6.3f}%  ({pct(S['uh']['extra'], S['uh']['meet']):6.3f})"
        f"   {pct(A['uh']['extra'], pl):6.2f}%        {pct(A['uh']['tipdig'], pl):5.2f}%            | {pct(A['one']['extra'], pl):5.2f}%  ({pct(S['one']['extra'], S['plain']):5.2f})"
        f"        | {pct(A['sd']['meet'], pl):6.2f}%       {pct(A['sd']['extra'], pl):5.2f}%          {pct(A['sd']['tipdig'], pl):5.2f}%")
    assert A["uh"]["orderfree_same"]
tot = {f: sum(R["main"][k]["A"][f]["extra"] for k in P) for f in ("uh", "one", "sd")}; pl = sum(R["main"][k]["A"]["plain"] for k in P)
td = {f: sum(R["main"][k]["A"][f]["tipdig"] for k in P) for f in ("uh", "one", "sd")}
log(f"   all 8:    extra/plain  up/hole {pct(tot['uh'], pl):.2f}%   1s {pct(tot['one'], pl):.2f}%   signed digits {pct(tot['sd'], pl):.2f}%;"
    f"  tips' digits/plain  up/hole {pct(td['uh'], pl):.2f}%   1s {pct(td['one'], pl):.2f}%   signed digits {pct(td['sd'], pl):.2f}%")
log("   order-free: tips identical under a random line order at every plate (checked at position 0 of each)")

log("\nB-D. GROUP PILES, PER-PLACE FORM, AS A SHARE OF EACH PLATE'S PLAIN (every pile exact in every run)")
log("   plate     B12    reads  top  table+write | g*(real) R=2^   | Bg     reads  table+write | D12    Dg   | shuffle B12  Dg  | seed2 B12 | s_V(12) s_V(g*)")
for k in P:
    m, s, s2 = R["main"][k], R["shuffle"][k], R["seed2"][k]; pl = m["plain"]
    b, bg, rel = m["B12"], m["Bg"], m["relation"]
    log(f"   {k:8s} {pct(b['total'], pl):6.2f} {pct(b['reads'], pl):6.2f} {pct(b['top_plate'], pl):4.2f}  {pct(b['build'] + b['write'], pl):5.2f}      |"
        f" {rel['g_star']:2d} ({rel['g_real']:5.2f}) {rel['log2R']:5.2f} | {pct(bg['total'], pl):6.2f} {pct(bg['reads'], pl):6.2f}  {pct(bg['build'] + bg['write'], pl):5.2f}      |"
        f" {pct(m['D12']['total'], pl):6.2f} {pct(m['Dg']['total'], pl):6.2f} | {pct(s['B12']['total'], s['plain']):6.2f} {pct(s['Dg']['total'], s['plain']):6.2f} |"
        f" {pct(s2['B12']['total'], s2['plain']):6.2f}    | {b['reads'] / b['reads_n']:.3f}   {bg['reads'] / bg['reads_n']:.3f}")

# whole machine: the confirmed parts (docs/09) moved by the sample's change of share at each plate kind
PLAIN = dict(qkv=679_699_985, proj=248_461_379, fc=902_064_996, mproj=914_738_348); WHOLE, PALL = 711_431_068, 3_666_596_259
def proj(var):
    d = 0.0; per = {}
    for kind in PLAIN:
        ks = [k for k in P if k.startswith(kind + ".")]
        num = sum(R["main"][k][var]["total"] - R["main"][k]["B12"]["total"] for k in ks); den = sum(R["main"][k]["plain"] for k in ks)
        per[kind] = 100 * num / den; d += num / den * PLAIN[kind]
    return per, d
log("\nWHOLE GPT-2, the confirmed machine's parts moved by the sample's change at each plate kind (points of that kind's plain):")
for var, name in (("Bg", "readers relation only"), ("Dg", "readers relation + the light's top places met directly (as predicted)"), ("D12", "light's top places only, g = 12")):
    per, d = proj(var); new = WHOLE + d
    log(f"   {name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in per.items()) + f"  ->  {new:,.0f} a token = {pct(new, PALL):.2f}% of plain"
        f" (from 19.40%); strictest count {33.67 * new / WHOLE:.2f}% (from 33.67%)")

log("\nTHE READERS RELATION ON THE SAMPLE (work a weight at g, the up/hole reads and the table; the plate's top digits apart)")
for k in P:
    m = R["main"][k]; rel = m["relation"]; b = m["B12"]; M, N, h = b["M"], b["N"], b["h"]
    sV, oV, sx = rel["sV"], rel["oV"], rel["sx"]; plw = m["plain"] / (16 * N * M)
    W = lambda g: (h / g) * sV + 2 ** (g - 1) * (sx + oV) / (g * M)
    need = h * sV / (plw * 0.03)                                    # lines a ready pile must cover for 3% of plain
    log(f"   {k:8s} plain {plw:5.2f} a weight; at g* {rel['g_star']}: {W(rel['g_star']):.2f} a weight ({pct(W(rel['g_star']), plw):.1f}% of plain);"
        f" at g = log2(readers) = {rel['log2R']:.1f}: {W(rel['log2R']):.2f} ({pct(W(rel['log2R']), plw):.1f}%);"
        f" 3% of plain needs a ready pile of about {need:.0f} lines, a table of 2^{need:.0f}")

log("\nPLAIN ADDS AND WORD OPERATIONS (sample, g = 12 and g*)")
MZ = np.load(f"{HERE}/base2_gpt2.npz")
for k in P:
    kind, l = k.split("."); l = int(l); m = R["main"][k]
    x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l][MS.POS]).astype(np.int64)
    w = np.asarray(MZ[f"{kind}.{l}.w"]).astype(np.int64); N, M = w.shape
    adds_plain = int((x != 0).sum(0) @ MS.ones(w).sum(1)); out = []
    for var in ("B12", "Bg"):
        c = m[var]; g = 12 if var == "B12" else m["relation"]["g_star"]
        h, wlo, whi, L = MS.plate_form(w, g); top_adds = int((x != 0).sum(0) @ MS.sdg(whi).sum(1))
        adds = c["reads_n"] + top_adds + c["build_n"] + 16 * (N - 1 + M)
        out.append(f"{var} {pct(adds, adds_plain):5.2f}% of plain adds ({adds / (16 * N * M):.2f} a weight; + {c['write_n'] / (16 * N * M):.3f} entries written)")
    nx = m["A"]["uh"]["places"]; wops = 3 * nx * 16 * (N // 64) * M
    log(f"   {k:8s} plain adds {adds_plain / (16 * N * M):.2f} a weight; " + "; ".join(out) +
        f"; both sides as bit planes (XOR, popcount, add per 64 lines): {wops / (N * M):.1f} word ops a weight = {pct(wops * 16, adds_plain):.0f}% of plain adds")
LOG.close()
