"""THE GROUP MACHINE'S REPORT: every run of groupmachine.py put side by side with the cuts already measured, the controls,
the weight-only part, and a labelled estimate at 14B widths.
    python3 groups/report.py          -> groups/out/report.log, groups/out/report.json
Reads: groups/out/{gpt2,qwen}_g{8,12}_{main,seed2,shuffle,order}.json, qwen_g16_main_L*.json (the g = 16 sample, for the
estimate only), lineup2/refs_main_counts.npz (way 2, GPT-2), qwen06b/banks_*.json (way 2, Qwen, per plate),
qwen_fringes2/report.log (the slab line-up), and (weight-only, g = 16, every Qwen plate) qwen/qwen_base2.npz.
"""
import os, re, sys, json, glob, math, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUTD = f"{HERE}/groups/out"
sys.argv = [sys.argv[0]]; sys.path.insert(0, f"{HERE}/groups")
import groupmachine as GM
LOG = open(f"{OUTD}/report.log", "w"); OUT = {}
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def pc(a, b): return f"{100 * a / b:6.2f}%"
NPOS = {"gpt2": 648, "qwen": 613}
KINDS = {"gpt2": ["qkv", "proj", "fc", "mproj"], "qwen": ["q", "k", "v", "o", "gate", "up", "down"]}
NLINES = {"qkv": 768, "proj": 768, "fc": 768, "mproj": 3072, "q": 1024, "k": 1024, "v": 1024, "o": 2048, "gate": 1024, "up": 1024, "down": 3072}
NOUT = {"qkv": 2304, "proj": 768, "fc": 3072, "mproj": 768, "q": 2048, "k": 1024, "v": 1024, "o": 1024, "gate": 3072, "up": 3072, "down": 1024}
SLABOF = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down"}
SLABFIRST = {"qkv": "q", "o": "o", "gu": "gate", "down": "down"}            # the plate whose record carries the slab's table
LAYERS = {"gpt2": 12, "qwen": 28}
def load(model, g, run):
    f = f"{OUTD}/{model}_g{g}_{run}.json"
    return json.load(open(f))["plates"] if os.path.exists(f) else None
R = {(m, g, r): load(m, g, r) for m in ("gpt2", "qwen") for g in (8, 12) for r in ("main", "seed2", "shuffle", "order")}

def table_owner(model, kind):                       # True if this plate's record carries its light's table
    return model == "gpt2" or SLABFIRST[SLABOF[kind]] == kind

def sums(rec, model, kind, form):
    """summed over the layers: plain, signed, pairs, adds; the machine's counts; the weight-only counts. Qwen: the slab's
    one table (q, k, v; gate, up) is shared out to its plates in proportion to their plain work, layer by layer."""
    s = dict(plain=0, signed=0, pairs=0, adds_plain=0, adds_signed=0, light_sd=0, table=0, table_n=0, half=0, half_n=0,
             flipbuild=0, flipbuild_n=0, read=0, read_n=0, combine=0, combine_n=0, flip=0, flip_n=0,
             cells_all=0, cells_lit=0, both=0, up_lit=0, dn_lit=0, up_lines=0, dn_lines=0, piles_differ=0)
    for l in range(LAYERS[model]):
        r = rec[f"{kind}.{l}"]; b = r["base"]; c = r[form]["counts"]; w = r[form]["weight"]
        for k in ("plain", "signed", "pairs", "adds_plain", "adds_signed", "light_sd"): s[k] += b[k]
        if model == "gpt2": share = 1.0
        else:
            sib = [k for k in KINDS["qwen"] if SLABOF[k] == SLABOF[kind]]
            share = b["plain"] / sum(rec[f"{k}.{l}"]["base"]["plain"] for k in sib)
        for k, q in (("table", "build_full"), ("table_n", "build_full_n"), ("half", "build_half"), ("half_n", "build_half_n"),
                     ("flipbuild", "flipbuild"), ("flipbuild_n", "flipbuild_n")):
            s[k] += share * c[q]
        for k in ("read", "read_n", "combine", "combine_n", "flip", "flip_n"): s[k] += c[k]
        for k in ("cells_all", "cells_lit", "both", "up_lit", "dn_lit", "up_lines", "dn_lines"): s[k] += w[k]
        s["piles_differ"] += r[form]["piles_differ"] or 0
    return s
def add(a, b): return {k: a[k] + b[k] for k in a}

# ---------- the cuts already measured ----------
def way2_gpt2():
    z = np.load(f"{HERE}/lineup2/refs_main_counts.npz"); out = {}
    for kind in KINDS["gpt2"]:
        out[kind] = dict(way2=sum(int(z[f"{kind}.{l}|gaps"].sum() + z[f"{kind}.{l}|pours"].sum() + z[f"{kind}.{l}|writing"].sum()) for l in range(12)),
                         plain=sum(int(z[f"{kind}.{l}|plain"].sum()) for l in range(12)), signed=sum(int(z[f"{kind}.{l}|signed"].sum()) for l in range(12)))
    return out
def way2_qwen():
    J = {}
    for f in glob.glob(f"{HERE}/qwen06b/banks_*.json"): J.update(json.load(open(f)))
    out = {}
    for kind in KINDS["qwen"]:
        out[kind] = dict(way2=sum(J[f"{kind}.{l}"]["gaps"] + J[f"{kind}.{l}"]["pours"] + J[f"{kind}.{l}"]["writing"] for l in range(28)),
                         plain=sum(J[f"{kind}.{l}"]["plain"] for l in range(28)), signed=sum(J[f"{kind}.{l}"]["signed"] for l in range(28)))
    return out
def slab_lineup():
    txt = open(f"{HERE}/qwen_fringes2/report.log").read(); out = {}
    for slab in ("qkv", "o", "gu", "down"):
        wk = re.search(rf"^\s+{slab} walk \(of the slab's plain\)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", txt, re.M)
        po = re.search(rf"^\s+{slab} pours\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", txt, re.M)
        out[slab] = dict(plain=int(wk.group(1).replace(",", "")), work=int(wk.group(3).replace(",", "")) + int(po.group(3).replace(",", "")))
    wr = re.search(r"^\s+the light's writing\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", txt, re.M); out["writing"] = int(wr.group(3).replace(",", ""))
    tot = re.search(r"^\s+the 196 plates, \(a\) way 2\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", txt, re.M)
    out["plates"] = dict(plain=int(tot.group(1).replace(",", "")), work=int(tot.group(3).replace(",", "")))
    return out
W2 = {"gpt2": way2_gpt2(), "qwen": way2_qwen()}; SL = slab_lineup()

# ---------- 1. the machine, per plate kind ----------
log("THE GROUP MACHINE: WORK SHARED ACROSS LIGHT LINES BY A TABLE OF GROUP PILES (groups/groupmachine.py)")
log("photons a token (lit places added), every position of the 30 refs texts (GPT-2 648, Qwen3-0.6B 613); every pile exact")
log("SD = the plate in signed digits; SM = the plate as a sign and its 1s. 'table' = the full table by Gray steps (shown apart);")
log("'read' = each ready pile added into the neuron's pile at its place; 'comb' = the down pile taken from the up pile where both are lit.")
log("'machine' = table + reads + combines. 'half+flip' = the half table (the flip) with every lookup of the other half made by")
log("the flip at the read. Qwen: q, k, v share one table a token (one light); gate and up share one; the slab's one table is")
log("shared out to its plates in proportion to their plain work (counted once). 'adds/direct': the machine's adds against the")
log("direct machine's (each lit plate place met by a lit light value adds that value once), the plate in the same form.")
for model in ("gpt2", "qwen"):
    P = NPOS[model]
    for g in (8, 12):
        rec = R[(model, g, "main")]
        if rec is None: log(f"\n{model} g={g}: no record"); continue
        log(f"\n{model.upper()}  g = {g}   ({LAYERS[model]} layers; plain and the cuts per token)")
        log(f"  {'kind':5s} {'lines':>5s} {'neur':>5s} {'plain a token':>15s} {'signed':>7s} {'way 2':>7s} | form {'table':>6s} {'read':>7s} {'comb':>7s} "
            f"{'MACHINE':>8s} {'half+flip':>9s} | {'adds':>6s} {'/direct':>7s}")
        tot = {f: None for f in ("sd", "sm")}; OUT.setdefault(model, {})[g] = {}
        for kind in KINDS[model]:
            w2 = W2[model][kind]
            for form in ("sd", "sm"):
                s = sums(rec, model, kind, form); tot[form] = s if tot[form] is None else add(tot[form], s)
                mach = s["table"] + s["read"] + s["combine"]; half = s["half"] + s["read"] + s["combine"] + s["flip"]
                adds = s["table_n"] + s["read_n"] + s["combine_n"]; direct = s["adds_signed"] if form == "sd" else s["adds_plain"]
                assert s["piles_differ"] == 0
                if form == "sd":
                    assert w2["plain"] == s["plain"], (model, kind, w2["plain"], s["plain"])          # the plain count rebuilt
                    head = f"  {kind:5s} {NLINES[kind]:5d} {NOUT[kind]:5d} {s['plain'] // P:15,d} {pc(s['signed'], s['plain'])} {pc(w2['way2'], s['plain'])} |"
                else: head = " " * 52 + "|"
                log(f"{head}  {form}  {pc(s['table'], s['plain'])} {pc(s['read'], s['plain'])} {pc(s['combine'], s['plain'])} {pc(mach, s['plain'])}  "
                    f"{pc(half, s['plain'])} | {pc(adds, direct)}")
                OUT[model][g][f"{kind}.{form}"] = dict(s, way2=w2["way2"], machine=mach, half_flip=half)
        for form in ("sd", "sm"):
            s = tot[form]; mach = s["table"] + s["read"] + s["combine"]; half = s["half"] + s["read"] + s["combine"] + s["flip"]
            adds = s["table_n"] + s["read_n"] + s["combine_n"]
            w2all = sum(W2[model][k]["way2"] for k in KINDS[model])
            log(f"  ALL {('48 banks' if model == 'gpt2' else '196 plates')} {form}: plain {s['plain'] // P:,} a token; signed digits {pc(s['signed'], s['plain'])}; "
                f"way 2 {pc(w2all, s['plain'])}" + (f"; slab line-up {pc(SL['plates']['work'] * P, s['plain'])}" if model == "qwen" else "")
                + f"\n      the group machine {pc(mach, s['plain'])} = table {pc(s['table'], s['plain'])} + reads {pc(s['read'], s['plain'])} + combines "
                f"{pc(s['combine'], s['plain'])};  half table + flips {pc(half, s['plain'])} (flips {pc(s['flip'], s['plain'])}, half build "
                f"{pc(s['half'], s['plain'])});  other half made by the flip once a token {pc(s['half'] + s['flipbuild'] + s['read'] + s['combine'], s['plain'])}"
                f"\n      adds: {adds / P:,.0f} a token = {pc(adds, s['adds_signed'])} of the direct machine's adds with the plate in signed digits, "
                f"{pc(adds, s['adds_plain'])} with its 1s (table {s['table_n'] / P:,.0f}, reads {s['read_n'] / P:,.0f}, combines {s['combine_n'] / P:,.0f})"
                f"\n      per pair (line x neuron, lit light): plain {s['plain'] / s['pairs']:.3f}, signed {s['signed'] / s['pairs']:.3f}, the machine "
                f"{mach / s['pairs']:.3f}; adds per pair: the machine {adds / s['pairs']:.3f}, direct (signed digits) {s['adds_signed'] / s['pairs']:.3f}, direct (1s) {s['adds_plain'] / s['pairs']:.3f}")
            OUT[model][g][f"ALL.{form}"] = dict(s, machine=mach, half_flip=half, adds=adds, way2=w2all)
        if model == "qwen":
            log(f"  the slab line-up (qwen_fringes2), per slab, against the group machine (SD) on the same slab:")
            for slab, kinds in (("qkv", ["q", "k", "v"]), ("o", ["o"]), ("gu", ["gate", "up"]), ("down", ["down"])):
                ss = None
                for k in kinds:
                    s = sums(rec, model, k, "sd"); ss = s if ss is None else add(ss, s)
                mach = ss["table"] + ss["read"] + ss["combine"]
                log(f"    {slab:5s} plain {ss['plain'] // P:15,d}  slab line-up {pc(SL[slab]['work'] * P, ss['plain'])}  group machine {pc(mach, ss['plain'])} "
                    f"(table {pc(ss['table'], ss['plain'])})")
                assert abs(SL[slab]["plain"] - ss["plain"] / P) < 1, (slab, SL[slab]["plain"], ss["plain"] / P)

# ---------- 2. controls ----------
log("\nCONTROLS (the machine, table + reads + combines, as a share of the same run's own plain; SD / SM)")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        main = R[(model, g, "main")]
        line = f"  {model:4s} g={g:2d}"
        for run in ("seed2", "shuffle"):
            rr = R[(model, g, run)]
            if rr is None: line += f"  {run}: no record"; continue
            parts = []
            for form in ("sd", "sm"):
                a = b = None
                for kind in KINDS[model]:
                    sa, sb = sums(main, model, kind, form), sums(rr, model, kind, form)
                    a = sa if a is None else add(a, sa); b = sb if b is None else add(b, sb)
                ma = (a["table"] + a["read"] + a["combine"]) / a["plain"]; mb = (b["table"] + b["read"] + b["combine"]) / b["plain"]
                parts.append(f"{100 * mb:.2f}% ({100 * (mb - ma):+.2f} pt)")
                OUT.setdefault("controls", {})[f"{model}.g{g}.{run}.{form}"] = dict(main=ma, run=mb, plain_main=a["plain"], plain_run=b["plain"])
            line += f"  {run}: " + " / ".join(parts)
        log(line)
        rr = R[(model, g, "order")]
        if rr is None: log(f"    order: no record"); continue
        for form in ("sd", "sm"):
            same_pile = same_rc = n = 0; dt = dfl = 0; T0 = F0 = PL = 0
            for k in rr:
                n += 1; a, b = main[k][form], rr[k][form]
                same_pile += a["pile_digest"] == b["pile_digest"]
                same_rc += all(a["counts"][q] == b["counts"][q] for q in ("read", "read_n", "combine", "combine_n"))
                dt += b["counts"]["build_full"] - a["counts"]["build_full"]; T0 += a["counts"]["build_full"]
                dfl += b["counts"]["flip"] - a["counts"]["flip"]; F0 += a["counts"]["flip"]; PL += main[k]["base"]["plain"]
            log(f"    order-free ({form}): piles identical at {same_pile} of {n} plates; reads and combines identical at {same_rc} of {n}; "
                f"the table's photons changed {100 * dt / T0:+.3f}% of the table ({100 * dt / PL:+.4f} pt of plain); the flips {100 * dfl / F0:+.3f}% of the flips")
            OUT.setdefault("controls", {})[f"{model}.g{g}.order.{form}"] = dict(plates=n, piles_same=same_pile, rc_same=same_rc, table_change=dt / T0, flip_change=dfl / F0)

# ---------- 3. the weight-only part ----------
log("\nWEIGHT-ONLY (fixed by the plate's digits and g; no light): per (neuron, place, group) cell")
log(f"  {'model':5s} {'kind':5s} {'lines x neurons':>16s}  g form {'empty':>7s} {'both lit':>8s} {'lines/up':>8s} {'lines/down':>10s} {'lines/read':>10s}")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        rec = R[(model, g, "main")]
        for kind in KINDS[model]:
            for form in ("sd", "sm"):
                s = sums(rec, model, kind, form)
                log(f"  {model:5s} {kind:5s} {NLINES[kind]:6d} x {NOUT[kind]:6d}  {g:2d} {form}  {pc(s['cells_all'] - s['cells_lit'], s['cells_all'])} "
                    f"{pc(s['both'], s['cells_lit']):>8s} {s['up_lines'] / s['up_lit']:8.3f} {s['dn_lines'] / s['dn_lit']:10.3f} "
                    f"{(s['up_lines'] + s['dn_lines']) / s['cells_lit']:10.3f}")

# ---------- 4. the light's piles: signed digits of a pile of k lines' light ----------
def group_sizes(N, g): return [g] * (N // g) + ([N % g] if N % g else [])
def light_curve(rec, model, kinds, g, form="sd", layers=None):
    """f(k): the mean signed digits of a pile of k light values (every subset of every group, every position)"""
    num = np.zeros(17); den = np.zeros(17)
    for kind in kinds:
        for l in (layers if layers is not None else range(LAYERS[model])):
            key = f"{kind}.{l}"
            if key not in rec: continue
            num += np.array(rec[key][form]["naf_by_size"], float)
            for gk in group_sizes(NLINES[kind], g):
                for k in range(gk + 1): den[k] += math.comb(gk, k) * NPOS[model]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)
log("\nTHE LIGHT'S PILES: signed digits of a pile of k light values (k = 1..g), every subset, every group, every position")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        rec = R[(model, g, "main")]
        for kind in KINDS[model]:
            if not table_owner(model, kind): continue
            f = light_curve(rec, model, [kind], g)
            log(f"  {model:4s} g={g:2d} {SLABOF.get(kind, kind) if model == 'qwen' else kind:5s} " + " ".join(f"{v:5.2f}" for v in f[1:g + 1]))

# ---------- 5. the estimate relation, checked at 0.6B, then carried to 14B ----------
def est_rc(hist, f):
    """reads + combines a position from the weight-only cell counts (up lines a, down lines b) and the light curve f"""
    H = np.array(hist, float); a, b = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]), indexing="ij")
    f = np.nan_to_num(np.asarray(f, float))                           # sizes beyond g never hold a cell (H is 0 there)
    assert not H[(a + b) > np.flatnonzero(np.asarray(f) > 0).max(initial=0)].any(), "a cell's pile is larger than the curve"
    cost = f[np.minimum(a + b, len(f) - 1)] + np.where((a > 0) & (b > 0), f[np.minimum(b, len(f) - 1)], 0.0)
    return float((H * cost).sum())
log("\nTHE RELATION (session's, fixed before the 14B numbers): reads + combines a position = sum over cells of f(lines in the ready")
log("pile) + [both lit] f(lines in the down pile), with f the light's curve above and the cells from the plate alone.")
log("Checked at 0.6B against the machine's own counts (SD):")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        rec = R[(model, g, "main")]; parts = []
        for kind in KINDS[model]:
            est = meas = 0.0
            for l in range(LAYERS[model]):
                r = rec[f"{kind}.{l}"]["sd"]; f = light_curve(rec, model, [kind], g, layers=[l]); f[0] = 0.0
                est += est_rc(r["weight"]["hist_ud"], f) * NPOS[model]; meas += r["counts"]["read"] + r["counts"]["combine"]
            parts.append(f"{kind} {100 * (est / meas - 1):+.2f}%")
        log(f"  {model:4s} g={g:2d}: estimate against measured: " + ", ".join(parts))

g16 = sorted(glob.glob(f"{OUTD}/qwen_g16_main_L*.json"))
REC16 = {}
for f in g16: REC16.update(json.load(open(f))["plates"])
L16 = sorted({int(k.split(".")[1]) for k in REC16})
if REC16:
    parts = []
    for kind in KINDS["qwen"]:
        est = meas = 0.0
        for l in L16:
            r = REC16[f"{kind}.{l}"]["sd"]; f = light_curve(REC16, "qwen", [kind], 16, layers=[l]); f[0] = 0.0
            est += est_rc(r["weight"]["hist_ud"], f) * 613; meas += r["counts"]["read"] + r["counts"]["combine"]
        parts.append(f"{kind} {100 * (est / meas - 1):+.2f}%")
    log(f"  qwen g=16 (layers {L16}, the sample run for the estimate): estimate against measured: " + ", ".join(parts))

log("\nESTIMATE AT 14B WIDTHS (LABELLED: NOT A RUN; no 14B plates or light here). Qwen3-0.6B's plate digits and light stand in")
log("for the 14B's: the cells per (neuron, group) are fixed by the digits and g, not the width; the light's piles by the light.")
log("Only the table's share changes with width: (2^g - 1) x (the light's signed digits a value) a group a token, shared by the")
log("neurons that read the same light. 14B shapes: hidden 5,120; q 5,120 out, k and v 1,024 out (one table for q, k, v: 7,168")
log("neurons); o 5,120 -> 5,120; gate and up 5,120 -> 17,408 each (one table: 34,816 neurons); down 17,408 -> 5,120.")
log("f(13..16) from the g = 16 sample (layers 0, 14, 27), scaled to every layer by f(12) of every layer / f(12) at those layers.")
SH14 = {"q": (5120, 5120, 7168), "k": (5120, 1024, 7168), "v": (5120, 1024, 7168), "o": (5120, 5120, 5120),
        "gate": (5120, 17408, 34816), "up": (5120, 17408, 34816), "down": (17408, 5120, 5120)}
SH06 = {k: (NLINES[k], NOUT[k], {"qkv": 4096, "o": 1024, "gu": 6144, "down": 1024}[SLABOF[k]]) for k in KINDS["qwen"]}
Mz = np.load(f"{HERE}/qwen/qwen_base2.npz")
def hist_all(g, kind, form="sd"):
    """weight-only cell counts (up lines, down lines) per (neuron, group) cell, over every layer, at g; per pair"""
    if g in (8, 12):
        rec = R[("qwen", g, "main")]; H = sum(np.array(rec[f"{kind}.{l}"][form]["weight"]["hist_ud"], float) for l in range(28))
        return H / (28 * NLINES[kind] * NOUT[kind])
    slab = SLABOF[kind]; a, b = {"q": (0, 2048), "k": (2048, 3072), "v": (3072, 4096), "o": (0, 1024), "gate": (0, 3072), "up": (3072, 6144), "down": (0, 1024)}[kind]
    H = np.zeros((g + 1, g + 1))
    for l in range(28):
        w = np.asarray(Mz[f"{slab}.{l}.w"])[:, a:b]; key = [1, ["qkv", "proj", "fc", "mproj", "o", "gu", "down"].index(slab), l, g]
        glines, goff = GM.grouping(NLINES[kind], g, key); up, dn = GM.plate_masks(w, form)
        _, _, Wst = GM.cells_of(up, dn, glines, goff); h = np.array(Wst["hist_ud"], float); H[:h.shape[0], :h.shape[1]] += h
    return H / (28 * NLINES[kind] * NOUT[kind])
def light_all(g, kind):
    """f(k) for k up to g: from the g = 12 run (every layer) for k <= 12, and the g = 16 sample run for k = 13..16"""
    first = SLABFIRST[SLABOF[kind]]; f12 = light_curve(R[("qwen", 12, "main")], "qwen", [first], 12)
    if g <= 12: f = f12.copy()
    else:                                                             # k = 13..16 from the sample layers, scaled to every layer by f(12)
        f16 = light_curve(REC16, "qwen", [first], 16, layers=L16); f12s = light_curve(R[("qwen", 12, "main")], "qwen", [first], 12, layers=L16)
        f = f12.copy(); f[13:17] = f16[13:17] * f12[12] / f12s[12]
    f[0] = 0.0; return f
def sd_per_value(kind):
    rec = R[("qwen", 12, "main")]; first = SLABFIRST[SLABOF[kind]]
    return sum(rec[f"{first}.{l}"]["base"]["light_sd"] for l in range(28)) / (28 * 613 * NLINES[kind])
plain_pair = {k: sum(R[("qwen", 12, "main")][f"{k}.{l}"]["base"]["plain"] for l in range(28)) / (28 * 613 * NLINES[k] * NOUT[k]) for k in KINDS["qwen"]}
check_l16 = ""
if REC16:
    f12s = light_curve(R[("qwen", 12, "main")], "qwen", ["q"], 12, layers=L16); f12a = light_curve(R[("qwen", 12, "main")], "qwen", ["q"], 12)
    f16s = light_curve(REC16, "qwen", ["q"], 16, layers=L16)
    check_l16 = (f"  (the light's curve at k <= 12: the sample layers {L16} against every layer, qkv light: largest difference "
                 f"{100 * np.nanmax(np.abs(f12s[1:13] / f12a[1:13] - 1)):.2f}%; the g = 16 run's own curve against the g = 12 run's at the same "
                 f"layers: {100 * np.nanmax(np.abs(f16s[1:13] / f12s[1:13] - 1)):.2f}%)")
log(check_l16)
log(f"  {'kind':5s} {'14B lines x neurons':>20s}  {'g':>2s} {'reads+comb/pair':>15s} {'table/pair':>10s} {'plain/pair':>10s} {'share':>7s}   at 0.6B widths (same relation)")
EST = {}
for g in (12, 16):
    totn = totd = totn06 = totd06 = 0.0
    for kind in KINDS["qwen"]:
        H = hist_all(g, kind); f = light_all(g, kind)
        rc = est_rc(H, f)                                              # per pair (H is per pair)
        sdx = sd_per_value(kind)
        N14, M14, S14 = SH14[kind]; N06, M06, S06 = SH06[kind]
        t14 = (2 ** g - 1) * sdx / (g * S14); t06 = (2 ** g - 1) * sdx / (g * S06)
        sh14 = (rc + t14) / plain_pair[kind]; sh06 = (rc + t06) / plain_pair[kind]
        totn += N14 * M14 * (rc + t14); totd += N14 * M14 * plain_pair[kind]
        totn06 += N06 * M06 * (rc + t06); totd06 += N06 * M06 * plain_pair[kind]
        EST[f"{kind}.g{g}"] = dict(rc_pair=rc, table14=t14, table06=t06, plain_pair=plain_pair[kind], share14=sh14, share06=sh06)
        log(f"  {kind:5s} {N14:9,d} x {M14:8,d}  {g:2d} {rc:15.3f} {t14:10.3f} {plain_pair[kind]:10.3f} {100 * sh14:6.2f}%   {100 * sh06:6.2f}% (table {t06:.3f})")
    EST[f"ALL.g{g}"] = dict(share14=totn / totd, share06=totn06 / totd06)
    log(f"  ALL plates, one layer's pairs weighted, g = {g}: 14B estimate {100 * totn / totd:.2f}% of plain; the same relation at 0.6B widths {100 * totn06 / totd06:.2f}%")
OUT["estimate14B"] = EST

# ---------- 6. the up/hole form (updown_groups.py): the flip that costs nothing ----------
log("\nTHE PLATE IN ITS UP/HOLE FORM (groups/updown_groups.py; the session's addition): every place of every weight up or a hole, so")
log("a group's pattern is never empty, nothing is combined, and the flip (ups and holes exchanged) is exactly the opposite pile:")
log("half the table, read as a take-away. Same groups, light and plates as the main run. machine = half table + reads + X")
log("(the light's sum, once a position) + X taken from each neuron's pile; against SD's machine on the same plates.")
UH = {}
for model in ("gpt2", "qwen"):
    P = NPOS[model]
    for g in (8, 12):
        f = f"{OUTD}/uh_{model}_g{g}.json"
        if not os.path.exists(f): log(f"  {model} g={g}: no record"); continue
        rec = json.load(open(f))["plates"]; main = R[(model, g, "main")]; T = None
        log(f"  {model.upper()} g = {g}:  {'kind':5s} {'table':>7s} {'reads':>7s} {'X, off':>7s} {'MACHINE':>8s}  {'SD machine':>10s}  {'adds/direct sd':>14s}")
        for kind in KINDS[model]:
            s = dict(plain=0, half=0, half_n=0, read=0, read_n=0, x=0, x_n=0, off=0, off_n=0, adds_signed=0, sdm=0, sdp=0)
            for l in range(LAYERS[model]):
                r = rec[f"{kind}.{l}"]; c = r["counts"]; b = r["base"]; assert r["piles_differ"] == 0 and c["odd"] == 0
                if model == "gpt2": share = 1.0
                else:
                    sib = [k for k in KINDS["qwen"] if SLABOF[k] == SLABOF[kind]]
                    share = b["plain"] / sum(rec[f"{k}.{l}"]["base"]["plain"] for k in sib)
                s["plain"] += b["plain"]; s["adds_signed"] += b["adds_signed"]
                s["half"] += share * c["build_half"]; s["half_n"] += share * c["build_half_n"]; s["x"] += share * c["xsum"]; s["x_n"] += share * c["xsum_n"]
                s["read"] += c["read"]; s["read_n"] += c["read_n"]; s["off"] += c["offset"]; s["off_n"] += c["offset_n"]
            sd_ = sums(main, model, kind, "sd"); s["sdm"] = sd_["table"] + sd_["read"] + sd_["combine"]
            mach = s["half"] + s["read"] + s["x"] + s["off"]; adds = s["half_n"] + s["read_n"] + s["x_n"] + s["off_n"]
            log(f"  {'':16s}{kind:5s} {pc(s['half'], s['plain'])} {pc(s['read'], s['plain'])} {pc(s['x'] + s['off'], s['plain'])} {pc(mach, s['plain'])}  "
                f"{pc(s['sdm'], s['plain']):>10s}  {pc(adds, s['adds_signed']):>14s}")
            T = s if T is None else {k: T[k] + s[k] for k in T}
            UH[f"{model}.g{g}.{kind}"] = dict(s, machine=mach, adds=adds)
        mach = T["half"] + T["read"] + T["x"] + T["off"]; adds = T["half_n"] + T["read_n"] + T["x_n"] + T["off_n"]
        log(f"  {'':16s}ALL   {pc(T['half'], T['plain'])} {pc(T['read'], T['plain'])} {pc(T['x'] + T['off'], T['plain'])} {pc(mach, T['plain'])}  "
            f"{pc(T['sdm'], T['plain']):>10s}  {pc(adds, T['adds_signed']):>14s}")
        UH[f"{model}.g{g}.ALL"] = dict(T, machine=mach, adds=adds)
OUT["updown"] = UH
if all(f"qwen.g{g}.q" in UH for g in (8, 12)) and REC16:
    log("\nESTIMATE AT 14B WIDTHS, UP/HOLE FORM (LABELLED: NOT A RUN). Each (neuron, place, group) is one read of a pile of all g lines:")
    log("reads a pair = 16 x r(g) / g, r(g) the mean signed digits of a read. r(8), r(12) measured on Qwen3-0.6B's own runs; r(16) =")
    log("r(12) x f(16) / f(12), f the light curve above (k = 16 from the g = 16 sample). The half table: 2^(g-1) x (the light's")
    log("signed digits a value) a group a token, shared by the slab's neurons; X and its take-away as at 0.6B (per pair).")
    log(f"  {'kind':5s}  g {'r(g)':>6s} {'reads/pair':>10s} {'table/pair':>10s} {'plain/pair':>10s} {'14B share':>9s}  {'0.6B widths':>11s}  {'measured 0.6B':>13s}")
    EU = {}
    for g in (12, 16):
        totn = totd = totn06 = totd06 = 0.0
        for kind in KINDS["qwen"]:
            u12 = UH[f"qwen.g12.{kind}"]; r12 = u12["read"] / u12["read_n"]
            if g == 12: r = r12
            else:
                f12 = light_all(12, kind); f16 = light_all(16, kind); r = r12 * f16[16] / f12[12]
            rd = 16 * r / g; sdx = sd_per_value(kind); N14, M14, S14 = SH14[kind]; N06, M06, S06 = SH06[kind]
            xo = (u12["x"] + u12["off"]) / (28 * 613 * NLINES[kind] * NOUT[kind])
            t14 = 2 ** (g - 1) * sdx / (g * S14); t06 = 2 ** (g - 1) * sdx / (g * S06)
            sh14 = (rd + t14 + xo) / plain_pair[kind]; sh06 = (rd + t06 + xo) / plain_pair[kind]
            meas = f"{100 * u12['machine'] / u12['plain']:12.2f}%" if g == 12 else ""
            log(f"  {kind:5s} {g:2d} {r:6.3f} {rd:10.3f} {t14:10.3f} {plain_pair[kind]:10.3f} {100 * sh14:8.2f}%  {100 * sh06:10.2f}%  {meas}")
            totn += N14 * M14 * (rd + t14 + xo); totd += N14 * M14 * plain_pair[kind]; totn06 += N06 * M06 * (rd + t06 + xo); totd06 += N06 * M06 * plain_pair[kind]
        EU[f"g{g}"] = dict(share14=totn / totd, share06=totn06 / totd06)
        log(f"  ALL plates, one layer's pairs weighted, g = {g}: 14B estimate {100 * totn / totd:.2f}% of plain; at 0.6B widths {100 * totn06 / totd06:.2f}%")
    OUT["estimate14B_updown"] = EU
json.dump(OUT, open(f"{OUTD}/report.json", "w"), default=float)
log("\n(done)")
