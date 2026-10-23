"""THE REBUILD SET BESIDE THE BUILDER'S RECORD, AND THE TOTALS (groups2/; reads only JSON records).

1. Plate by plate, the rebuild's main run (groups2/out/uhr_<model>_g<g>.json, seed-1 groups) against the builder's
   up/hole record (groups/out/uh_<model>_g<g>.json): plain, signed, read photons, read count, half and full table,
   X, offsets, odd. The builder ran q, k, v (and gate, up) as separate calls on the same light, so its table and X
   numbers are per call; the rebuild's are per slab: each must equal the builder's for every plate of the slab.
2. Totals, table included, as a share of plain photon meetings: machine = half table + up/hole reads + signed-digit
   reads + X + X's take-aways (+ any making of a multiple of X), for every run of the rebuild (uhr_) and of the
   per-place choice (uhp_). Per kind, a slab's table and X are shared out to its plates in proportion to their plain
   work (as groups/report.py does); the totals do not depend on that sharing.
    python3 groups2/uh_compare.py  -> groups2/out/compare.log, compare.json
"""
import os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/out"
KINDS = {"gpt2": ["qkv", "proj", "fc", "mproj"], "qwen": ["q", "k", "v", "o", "gate", "up", "down"]}
SLABOF = {"qkv": "qkv", "proj": "proj", "fc": "fc", "mproj": "mproj", "q": "qkv", "k": "qkv", "v": "qkv", "o": "o",
          "gate": "gu", "up": "gu", "down": "down"}
LOGF = open(f"{OUT}/compare.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def pc(a, b): return f"{100 * a / b:7.2f}%"
RES = {}

# ---------- 1. number for number ----------
log("1. THE REBUILD'S MAIN RUN AGAINST THE BUILDER'S UP/HOLE RECORD, PLATE BY PLATE")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        fm, fb = f"{OUT}/uhr_{model}_g{g}.json", f"{HERE}/groups/out/uh_{model}_g{g}.json"
        if not (os.path.exists(fm) and os.path.exists(fb)): log(f"  {model} g={g}: record missing"); continue
        mine = json.load(open(fm))["runs"]["main"]; theirs = json.load(open(fb))["plates"]
        n = neq = 0; bad = []
        for name, r in mine.items():
            if name.endswith(".slab"): continue
            kind, l = name.split("."); s = mine[f"{SLABOF[kind]}.{l}.slab"]; t = theirs[name]; c = t["counts"]; b = t["base"]
            pairs = [("plain", r["plain"], b["plain"]), ("signed", r["signed"], b["signed"]), ("read", r["read"], c["read"]),
                     ("read_n", r["read_n"], c["read_n"]), ("half", s["half"], c["build_half"]), ("half_n", s["half_n"], c["build_half_n"]),
                     ("full", s["full"], c["build_full"]), ("full_n", s["full_n"], c["build_full_n"]), ("xsum", s["xsum"], c["xsum"]),
                     ("xsum_n", s["xsum_n"], c["xsum_n"]), ("offset", r["offset"], c["offset"]), ("offset_n", r["offset_n"], c["offset_n"]),
                     ("odd", r["odd"], c["odd"]), ("piles_differ", r["piles_differ"], t["piles_differ"])]
            n += 1; d = [(k, a, bb) for k, a, bb in pairs if a != bb]
            if d: bad.append((name, d))
            else: neq += 1
        log(f"  {model} g = {g:2d}: {neq} of {n} plates equal in all 14 numbers" + ("" if not bad else f"; differ: {bad[:5]}"))
        RES[f"equal.{model}.g{g}"] = dict(plates=n, equal=neq, differ=bad[:20])

# ---------- 2. totals ----------
def totals(rec, model):
    """per kind and in all: plain, table, reads, signed-digit reads, X part, machine"""
    out = {}; T = None
    for kind in KINDS[model]:
        s = dict(plain=0.0, table=0.0, read=0.0, sdread=0.0, x=0.0, read_n=0.0, sdread_n=0.0, table_n=0.0, x_n=0.0, plates=0)
        for name, r in rec.items():
            if name.endswith(".slab") or name.split(".")[0] != kind: continue
            l = name.split(".")[1]; sl = rec[f"{SLABOF[kind]}.{l}.slab"]
            sib = [k for k in KINDS[model] if SLABOF[k] == SLABOF[kind] and f"{k}.{l}" in rec]
            share = r["plain"] / sum(rec[f"{k}.{l}"]["plain"] for k in sib)
            s["plain"] += r["plain"]; s["table"] += share * sl["half"]; s["table_n"] += share * sl["half_n"]
            s["read"] += r["read"]; s["read_n"] += r["read_n"]; s["sdread"] += r["sdread"]; s["sdread_n"] += r["sdread_n"]
            s["x"] += share * sl["xsum"] + r["offset"] + r["make"]; s["x_n"] += share * sl["xsum_n"] + r["offset_n"] + r["make_n"]
            s["plates"] += 1
        if not s["plates"]: continue
        s["machine"] = s["table"] + s["read"] + s["sdread"] + s["x"]; out[kind] = s
        T = dict(s) if T is None else {k: T[k] + s[k] for k in s}
    out["ALL"] = T
    return out

log("\n2. TOTALS, TABLE INCLUDED, AS A SHARE OF PLAIN PHOTON MEETINGS")
log("   up/hole = the plate in up/hole form at every place (uhr_); per-place = each place up/hole or signed digits by the")
log("   plate alone (uhp_). machine = half table + up/hole reads + signed-digit reads + X and its take-aways.")
for model in ("gpt2", "qwen"):
    for g in (8, 12):
        for form, pre in (("up/hole", "uhr"), ("per-place", "uhp")):
            f = f"{OUT}/{pre}_{model}_g{g}.json"
            if not os.path.exists(f): log(f"  {model} g={g} {form}: no record"); continue
            runs = json.load(open(f))["runs"]; tots = {run: totals(rec, model) for run, rec in runs.items() if rec}
            if "main" not in tots: continue
            m = tots["main"]
            log(f"\n  {model.upper()} g = {g}, {form} ({m['ALL']['plates']} plates)")
            log(f"    {'kind':6s} {'table':>8s} {'reads':>8s} {'sd reads':>8s} {'X':>8s} {'MACHINE':>8s}   {'seed2-main':>10s} {'shuf-main':>10s} {'order-main':>10s}")
            for kind in list(KINDS[model]) + ["ALL"]:
                if kind not in m: continue
                s = m[kind]; mach = s["machine"] / s["plain"]
                dif = []
                for run in ("seed2", "shuffle", "order"):
                    if run in tots and kind in tots[run]: dif.append(f"{100 * (tots[run][kind]['machine'] / tots[run][kind]['plain'] - mach):+10.3f}")
                    else: dif.append(f"{'-':>10s}")
                log(f"    {kind:6s} {pc(s['table'], s['plain'])} {pc(s['read'], s['plain'])} {pc(s['sdread'], s['plain'])} {pc(s['x'], s['plain'])} {pc(s['machine'], s['plain'])}   " + " ".join(dif))
            if "order" in runs:
                same = all(runs["order"][k]["read"] == runs["main"][k]["read"] and runs["order"][k]["read_n"] == runs["main"][k]["read_n"]
                           and runs["order"][k]["sdread"] == runs["main"][k]["sdread"]
                           for k in runs["main"] if not k.endswith(".slab") and k in runs["order"])
                log(f"    order-free: piles identical (asserted in the run); up/hole and signed-digit reads identical at every plate: {same}")
            RES[f"{form}.{model}.g{g}"] = {run: {k: dict(v, share=v["machine"] / v["plain"]) for k, v in t.items()} for run, t in tots.items()}
json.dump(RES, open(f"{OUT}/compare.json", "w"), default=float)
log("\n(done)")
