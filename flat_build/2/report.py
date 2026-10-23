"""flat_build/2/report.py -- the totals of machine.py's runs (and check.py's agreement), a token = the mean over the 648
refs positions. Whole numbers summed; shares printed.
    python3 flat_build/2/report.py   -> flat_build/2/out/report.log
"""
import json, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_build/2/out"; P = 648
WORK = ("light", "table", "pwrite", "reads", "sdreads", "xsum", "xoff")
KINDS = ("qkv", "proj", "fc", "mproj")
LOG = open(f"{OUT}/report.log", "w")
def log(s=""): print(s); LOG.write(s + "\n")
def w(c): return sum(c[k] for k in WORK)
def pc(a, b): return 100 * a / b

for GM in ("12", "R"):
    f = f"{OUT}/machine_g{GM}.json"
    if not os.path.exists(f): continue
    B = json.load(open(f))["banks"]
    log(f"=== g mode {GM} ({'g = 12 at every bank: pick 2 alone' if GM == '12' else 'g by readers, 11 at proj/mproj: pick 2 stacked on pick 1'}); "
        f"{len(B)} banks, every pile exact (asserted in the run) ===")
    agg = {}
    for name, r in B.items():
        kind = name.split(".")[0]
        for run in ("A", "C", "A_shuffle", "C_shuffle", "C_order"):
            a = agg.setdefault(kind, {}).setdefault(run, {}); t = agg.setdefault("ALL", {}).setdefault(run, {})
            for k, v in r[run].items(): a[k] = a.get(k, 0) + v; t[k] = t.get(k, 0) + v
        for k in ("C_reads_a_position", "C_cells"):
            agg[kind].setdefault("_", {}); agg[kind]["_"][k] = agg[kind]["_"].get(k, 0) + r[k]
        agg[kind].setdefault("_", {}).setdefault("hist", np.zeros(16, np.int64)); agg[kind]["_"]["hist"] += np.array(r["C_c_hist"])
        jr = r.get("judge_formC_reads")
        agg[kind]["_"]["judge"] = agg[kind]["_"].get("judge", 0) + (jr if jr is not None else 0)
        agg[kind]["_"]["judge_n"] = agg[kind]["_"].get("judge_n", 0) + (jr is not None)
        agg[kind]["_"]["entries_read"] = agg[kind]["_"].get("entries_read", 0) + r["C_entries_read"]
        agg[kind]["_"]["entries"] = agg[kind]["_"].get("entries", 0) + r["C_entries"]
    log("  PHOTONS a token (% of the plate kind's plain photon meetings)")
    log(f"  {'kind':6s} {'plain':>14s} | {'A (gbank)':>22s} | {'C (the run)':>22s} | {'C - A':>18s} | read work C/A | pwrite C/A")
    for kind in KINDS + ("ALL",):
        a, c = agg[kind]["A"], agg[kind]["C"]; pl = a["plain"]; assert pl == c["plain"]
        rwA = a["reads"] + a["sdreads"]; rwC = c["reads"] + c["sdreads"]
        log(f"  {kind:6s} {pl / P:14,.0f} | {w(a) / P:12,.0f} {pc(w(a), pl):7.3f}% | {w(c) / P:12,.0f} {pc(w(c), pl):7.3f}% | "
            f"{(w(c) - w(a)) / P:+11,.0f} {pc(w(c) - w(a), pl):+6.3f} | {pc(rwC, rwA):8.2f}%   | {pc(c['pwrite'], a['pwrite']):7.3f}%")
    log("  BY ITEM, ALL 48 banks, a token:")
    a, c = agg["ALL"]["A"], agg["ALL"]["C"]
    for k in WORK:
        log(f"    {k:8s} A {a[k] / P:14,.1f}   C {c[k] / P:14,.1f}   C - A {(c[k] - a[k]) / P:+14,.1f}")
    log("  ADDS a token (% of plain adds: nonzero light x plate 1s)")
    for kind in KINDS + ("ALL",):
        a, c = agg[kind]["A"], agg[kind]["C"]
        log(f"  {kind:6s} plain adds {a['plain_adds'] / P:14,.0f} | A {a['adds'] / P:12,.0f} {pc(a['adds'], a['plain_adds']):6.3f}% | "
            f"C {c['adds'] / P:12,.0f} {pc(c['adds'], c['plain_adds']):6.3f}% | C/A {pc(c['adds'], a['adds']):6.2f}%  "
            f"(reads a position: A {a['nreads'] / P:,.0f} + lit signed digits {a.get('nsd', 0) / P:,.0f}; C {c['nreads'] / P:,.0f})")
    log("  CONTROLS")
    for kind in KINDS + ("ALL",):
        A, C, As, Cs, Co = (agg[kind][r] for r in ("A", "C", "A_shuffle", "C_shuffle", "C_order"))
        pl, pls = A["plain"], As["plain"]
        log(f"  {kind:6s} shuffled light: A {pc(w(As), pls):7.3f}% ({pc(w(As), pls) - pc(w(A), pl):+.3f}), C {pc(w(Cs), pls):7.3f}% "
            f"({pc(w(Cs), pls) - pc(w(C), pl):+.3f}); C - A {pc(w(Cs) - w(As), pls):+.3f} (real {pc(w(C) - w(A), pl):+.3f}) | "
            f"order-free: reads {'=' if Co['reads'] == C['reads'] else 'DIFFER'}, read count {'=' if Co['nreads'] == C['nreads'] else 'DIFFER'}, "
            f"entries written {'=' if Co['pwrite'] == C['pwrite'] else 'DIFFER'}, X {'=' if (Co['xsum'], Co['xoff']) == (C['xsum'], C['xoff']) else 'DIFFER'}, "
            f"table {pc(Co['table'] - C['table'], pl):+.4f} point")
    log("  THE CELLS (plate only)")
    for kind in KINDS:
        u = agg[kind]["_"]; h = u["hist"]; n = h.sum()
        log(f"  {kind:6s} cells {u['C_cells']:,}; reads a cell {u['C_reads_a_position'] / u['C_cells']:.3f}; c = 15 (no sheet skipped) "
            f"{100 * h[15] / n:.1f}%, 14 {100 * h[14] / n:.1f}%, 13 {100 * h[13] / n:.1f}%, 12 {100 * h[12] / n:.1f}%, <= 11 {100 * h[:12].sum() / n:.1f}%; "
            f"reads a position {u['C_reads_a_position']:,}"
            + (f" (judge's count {u['judge']:,} over {u['judge_n']} banks: {'SAME' if u['judge'] == u['C_reads_a_position'] else 'DIFFERENT'})" if u['judge_n'] else "")
            + f"; half-table entries read {100 * u['entries_read'] / u['entries']:.2f}%")
    cf = f"{OUT}/check_g{GM}.json"
    if os.path.exists(cf):
        CJ = json.load(open(cf))
        nb = len(CJ); ok = sum(1 for v in CJ.values() if v["piles_differ"] == 0 and not v["differ"])
        log(f"  SECOND PATH (check.py, same session): {ok} of {nb} banks with every pile exact and every count the same")
    log("")
