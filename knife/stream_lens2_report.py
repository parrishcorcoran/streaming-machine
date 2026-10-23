"""THE ALL-LAYER REPORT OF knife/stream_lens2.py, from its per-layer files (whole numbers; percentages rounded to 0.1 by
integer arithmetic). Reads knife/stream_lens2_<run>_L<l>.npz for run = real, shuffled, real_order.
    python3 knife/stream_lens2_report.py   -> knife/stream_lens2_report.log
"""
import os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
LOG = open(f"{HERE}/knife/stream_lens2_report.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def pm(a, b): q = (1000 * int(a) + int(b) // 2) // int(b) if b else 0; s = "-" if q < 0 else ""; q = abs(q); return f"{s}{q // 10}.{q % 10}%"
WHOLE, P, LINES = 3666596259, 648, 768
NAMES = [f"D{k}" for k in range(2, 29, 2)] + ["all"]; NS = len(NAMES)
for run in ("real", "shuffled", "real_order"):
    files = [f"{HERE}/knife/stream_lens2_{run}_L{l}.npz" for l in range(12)]
    have = [l for l in range(12) if os.path.exists(files[l])]
    if not have: log(f"==== {run}: no files ===="); continue
    Z = {l: np.load(files[l]) for l in have}
    T = lambda l, k: Z[l][f"T_{k}"]
    FT = sum(int(T(l, "full")) for l in have); L2 = sum(int(T(l, "l2")) for l in have)
    EO = sum(int(T(l, "edge_only")) for l in have)
    log(f"==== {run}: layers {have}; fc meeting {FT:,} ({FT // P:,} a token, {pm(FT, P * WHOLE)} of the whole machine); "
        f"plain lens 2 {L2:,} ({L2 // P:,} a token, {pm(L2, FT)} of fc); settle.py's per-edge output stops {pm(EO, FT)} of fc ====")
    log("  share fixed by each stage (cumulative), all layers named")
    log("    " + " " * 36 + "".join(f"{nm:>7s}" for nm in NAMES))
    for v in ("A", "B"):
        for key, lab, n in (("FM", "stream value, lines", P * LINES), ("FX", "next light, lines", P * LINES),
                            ("FPM", "stream value, all 768 (positions)", P), ("FPX", "next light, all 768 (positions)", P)):
            h = sum(np.bincount(Z[l][f"{key}_{v}"].ravel().astype(np.int64) % 256, minlength=256)[:NS] for l in have)
            log(f"    ({v}) {lab:32s}" + "".join(f"{pm(x, n * len(have)):>7s}" for x in np.cumsum(h)))
    log("  per layer: lines fixed before the full pile (stream value / next light), positions fixed before the full pile "
        "(stream / next light), (A)")
    for l in have:
        fm, fx = Z[l]["FM_A"], Z[l]["FX_A"]; pmm, pxx = Z[l]["FPM_A"], Z[l]["FPX_A"]
        fx28 = int((fx <= NS - 2).sum()); fx26 = int((fx <= NS - 3).sum()); fx24 = int((fx <= NS - 4).sum())
        log(f"    layer {l:2d}: stream {pm((fm < NS - 1).sum(), fm.size):>6s}  next light {pm(fx28, fx.size):>6s} "
            f"(by D26 {pm(fx26, fx.size)}, by D24 {pm(fx24, fx.size)})  positions: stream {int((pmm < NS - 1).sum())}/648, "
            f"next light {int((pxx < NS - 1).sum())}/648")
    log("  work, all layers named (per token = over 648 positions); net = saved minus extra, as a share of the whole machine")
    for v in ("A", "B"):
        for r, lab in (("m", "stream value"), ("x", "next light")):
            l1 = sum(int(T(l, f"l1_{v}_{r}")) for l in have); p1 = sum(int(T(l, f"push1_{v}_{r}")) for l in have)
            l1i = sum(int(T(l, f"l1_ii_{v}_{r}")) for l in have); p2 = sum(int(T(l, f"push2_{v}_{r}")) for l in have)
            np2 = sum(int(T(l, f"npush2_{v}_{r}")) for l in have); rd = sum(int(T(l, f"reads_{v}_{r}")) for l in have)
            ref = FT if v == "A" else EO                 # (B) against settle.py's per-edge stops alone
            log(f"    ({v}) stop when the {lab} is fixed: lens 1 {pm(l1, FT)} of fc; saved against "
                f"{'the plain fc meeting' if v == 'A' else 'the per-edge stops'} {(ref - l1) // P:,} a token")
            log(f"        push (i), one at the stop: extra {p1 // P:,} a token; net {pm(ref - l1 - p1, P * WHOLE)} of the whole machine")
            log(f"        push (ii), every one-level stage: {np2 / (P * len(have)):.2f} pushes a position a layer; lens 1 {pm(l1i, FT)}; "
                f"extra {p2 // P:,} a token ({pm(p2, L2)} of lens 2); net {pm(ref - l1i - p2, P * WHOLE)} of the whole machine")
            p3 = sum(int(T(l, f"push3_{v}_{r}")) for l in have); l13 = sum(int(T(l, f"l1_iii_{v}_{r}")) for l in have)
            log(f"        push (iii), one at D28 then the full pile: lens 1 {pm(l13, FT)}; extra {p3 // P:,} a token "
                f"({pm(p3, L2)} of lens 2); net {pm(ref - l13 - p3, P * WHOLE)} of the whole machine")
            log(f"        bound reads (apart): {rd // P:,} a token")
    log("  widths at D28 / D26 / D24, (A), all layers named: share of lines whose interval is one number, and the median "
        "width (stream units; the next light in its own units)")
    for s in (NS - 2, NS - 3, NS - 4):
        row = f"    {NAMES[s]:4s}"
        for w, lab in (("m", "stream value"), ("x", "next light")):
            h = sum(Z[l][f"WID_A_{s}_{w}"] for l in have); c = np.cumsum(h); tot = int(c[-1])
            med = int(np.searchsorted(c, (tot + 1) // 2))
            rng = "0" if med == 0 else f"{1 << (med - 1)}-{(1 << med) - 1}"
            row += f"   {lab}: one number {pm(h[0], tot):>6s}, median width {rng}"
        log(row)
    rows = []
    for l in have:
        h = T(l, "reach"); c = np.cumsum(h); tot = int(c[-1]); med = int(np.searchsorted(c, (tot + 1) // 2))
        rows.append(f"L{l} med {'<1' if med == 0 else f'{1 << (med - 1)}-{(1 << med) - 1}'}, >=1 unit {pm(tot - h[0], tot)}")
    log("  one unit of an edge's output, carried by lens 2 onto a stream line (stream units, at the layer's middle sh; "
        "share of plate values reaching at least 1 unit): " + "; ".join(rows))
    nb = sum(T(l, "nblur") for l in have); ns = sum(T(l, "nsingle") for l in have)
    log("  edges still unsure in a position where the edges' water level is one value (mean), (A) / (B): " +
        ", ".join(f"{NAMES[s]} {nb[0, s] // max(ns[0, s], 1)}/{nb[1, s] // max(ns[1, s], 1)}" for s in range(NS - 6, NS)))
    log("")
