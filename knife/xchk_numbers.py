"""CHECKER: every all-layer number the knife report quotes, recomputed from settle.py's saved totals
(knife/settle_real.npz, settle_shuffled.npz). Whole numbers; percentages rounded to 0.1 by integer arithmetic.
    python3 knife/xchk_numbers.py   -> knife/xchk_numbers.log
"""
import numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
LOG = open(f"{HERE}/knife/xchk_numbers.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def pm(a, b): q = (1000 * int(a) + int(b) // 2) // int(b) if b else 0; return f"{q // 10}.{q % 10}%"
WHOLE = 3666596259; POS = 648
SN = [["L2", "L4", "L6", "L8", "L10", "L12", "all"], [f"D{k}" for k in range(2, 29, 2)] + ["all"]]
GN = ["far dark", "-4..-2", "-2..0", "0..2.66", "open"]
for name in ("real", "shuffled"):
    S = np.load(f"{HERE}/knife/settle_{name}.npz")
    hist, work, workb, full, fullb, breads = S["hist"], S["work"], S["workb"], S["full"], S["fullb"], S["breads"]
    TOT = int(hist[:, 0, 0, 0].sum()); FT = int(full.sum())
    log(f"==== {name}: {TOT:,} edges, full fc meeting {FT:,}, per token {FT // POS:,} (rem {FT % POS}), "
        f"{pm(FT, POS * WHOLE)} of the whole machine ====")
    for si in range(2):
        ns = len(SN[si])
        for bi in (1, 0):
            cum = {ci: np.cumsum(hist[:, si, bi, ci].sum((0, 1)))[:ns] for ci in range(3)}
            log(f" {['light', 'diag'][si]} {['plain', 'signed'][bi]}:")
            for ci, cn in enumerate(("side", "share", "output")):
                log(f"   {cn:6s} cum " + " ".join(f"{SN[si][s]}:{pm(cum[ci][s], TOT)}" for s in range(ns)))
                pre = int(hist[:, si, bi, ci].sum((0, 1))[:ns - 1].sum())
                wl = [int((1000 * work[l, si, bi, ci] + full[l] // 2) // full[l]) for l in range(12)]
                log(f"          before full {pre:,} ({pm(pre, TOT)}); work {pm(work[:, si, bi, ci].sum(), FT)}; per layer "
                    f"{min(wl) / 10}-{max(wl) / 10}%; saved per token {(FT - int(work[:, si, bi, ci].sum())) // POS:,}")
            sv = FT - int(work[:, si, bi, 2].sum()); rd = int(breads[:, si, bi].sum())
            log(f"   output stops: saved {sv // POS:,} per token ({pm(sv, FT)} of fc, {pm(sv, POS * WHOLE)} of whole); "
                f"bound reads {rd // POS:,} per token; net {pm(sv - rd, POS * WHOLE)} of whole")
        log(f"  {['light', 'diag'][si]} signed by group (edges; side/share/output: before full, median, work):")
        for g in range(5):
            row = f"   {GN[g]:9s} {pm(hist[:, si, 1, 0, g].sum(), TOT):>6s}"
            for ci in range(3):
                h = hist[:, si, 1, ci, g].sum(0)[:ns]; n = int(h.sum())
                med = SN[si][int(np.searchsorted(np.cumsum(h), (n + 1) // 2))] if n else "-"
                row += f" | {pm(h[:ns - 1].sum(), n)} {med} {pm(workb[:, si, 1, ci, g].sum(), fullb[:, g].sum())}"
            log(row)
    grp = hist[:, 0, 0, 0].sum(-1)                       # (12, 5) edges per layer and group
    log(f"  -2..0 per layer: " + ", ".join(pm(grp[l, 2], grp[l].sum()) for l in range(12)))
