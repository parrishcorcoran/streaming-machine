"""The staged top read, all runs side by side (from staged_<sched>_<light>.npz). Whole numbers only.
Work per position in photon meetings (millions, M). The plain top read is measured here on the same positions; the
whole machine is taken as the rest of the machine per token from earlier results (3,667M - 904M = 2,763M) plus the
top read scaled to the 904M basis (staged / plain here x 904M).
    python3 collapse_top/report.py
"""
import numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
REST, TOP = 3667 - 904, 904
def r(v): return int(round(v))
for light in ("real", "shuffled"):
    print(f"\n=== light {light}" + ("  (CONTROL: each position's light shuffled across its lines)" if light == "shuffled" else ""))
    for sc in ("stamp", "light", "diag"):
        Z = np.load(f"{D}/staged_{sc}_{light}.npz")
        S, Wk, P, E = Z["surv"], Z["work"], Z["plain"], Z["enter"]
        bad = dict(zip(Z["bad_keys"].tolist(), Z["bad"].tolist()))
        same = int((Z["top_st"] == Z["top_pl"]).all(1).sum())
        print(f"{sc:5s}: {len(S)} positions; top 5 = plain read {same}/{len(S)}; bound broken {bad['bound']}; "
              f"pieces wrong {bad['pieces']}/{bad['pieces_checked']}")
        for s, k in enumerate((4, 8, 12)):
            v = S[:, s]
            print(f"   after top {k:2d}: survivors median {r(np.median(v)):6d}  mean {r(v.mean()):6d}  min {v.min():6d}  max {v.max():6d}"
                  f"   (5 left at {int((v == 5).sum())} positions)")
        pl = P.mean(); tot = Wk.sum(1).mean()
        print("   work per stage: " + "  ".join(f"{r(Wk[:, s].mean() / 1e6)}M" for s in range(4)) +
              f"  = {r(tot / 1e6)}M of plain {r(pl / 1e6)}M -> {r(100 * tot / pl)}% of plain")
        top = TOP * tot / pl
        print(f"   on the 904M basis: top read {r(top)}M; whole machine {r(REST + top)}M of 3,667M = {r(100 * (REST + top) / 3667)}%"
              f" (saved {r(100 * (TOP - top) / 3667)}% of the whole machine)")
        print(f"   extra per position: bound adds {r((E.sum(1).mean() * 15) / 1e3)}K + tables, gauge reads {r(2 * E.sum(1).mean() / 1e3)}K, "
              f"compares {r(E.sum(1).mean() / 1e3)}K")
        src = np.load(f"{D}/light_top.npz")["src"][Z["sel"]]
        for name, m in (("refs", src == 0), ("windows", src == 1)):
            print(f"     {name:7s}: survivors after 8 median {r(np.median(S[m, 1]))}, after 12 median {r(np.median(S[m, 2]))};"
                  f" {r(100 * Wk[m].sum(1).mean() / P[m].mean())}% of plain")
