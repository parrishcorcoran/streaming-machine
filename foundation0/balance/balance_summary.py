"""Compact tables from balance_<size>.json (reads only the counts; no plates).

    python3 balance_summary.py [sizes=70m,160m]
"""
import os, sys, json, numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "70m,160m").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = [("qkv", "query/key/value"), ("attn_out", "attention out"), ("mlp_in", "MLP in"), ("mlp_out", "MLP out"),
         ("words_in", "input word table"), ("words_out", "output word table")]
STEPS = ("step0", "step143000")


def pct(x): return f"{100 * x:.2f}"


for SIZE in SIZES:
    R = json.load(open(os.path.join(HERE, f"balance_{SIZE}.json")))
    print(f"\n=================== pythia-{SIZE} ===================")
    for st in STEPS:
        # pooled over every plate: how many places sit beyond 2 and 3 coin-toss widths, against the fresh draw
        print(f"\n--- {st}: every plate pooled ---")
        for form in ("ss", "sd"):
            z = [v for r in R for v in r["steps"][st][form]["plate"]["place_z"] if v is not None]
            f = [v for r in R for v in r["steps"][st][form]["fresh"]["place_z"] if v is not None]
            z, f = np.abs(z), np.abs(f)
            print(f"  {form}: place counts {len(z)}: |z|>2 {int((z > 2).sum())} ({pct((z > 2).mean())}%), >3 {int((z > 3).sum())}, "
                  f"largest {z.max():.2f} | fresh: >2 {int((f > 2).sum())} ({pct((f > 2).mean())}%), >3 {int((f > 3).sum())}, largest {f.max():.2f}"
                  f" | fair coin: >2 4.55%, >3 0.27%")
        pz = np.array([r["steps"][st]["ss"]["plate"]["plate_z"] for r in R]); fz = np.array([r["steps"][st]["ss"]["fresh"]["plate_z"] for r in R])
        print(f"  whole-plate net, in coin-toss widths: |z|>3 on {int((np.abs(pz) > 3).sum())} of {len(pz)} plates (largest {np.abs(pz).max():.2f}); "
              f"fresh {int((np.abs(fz) > 3).sum())} (largest {np.abs(fz).max():.2f})")
        for ax, nk, lab in (("neuron", "neurons", "per neuron"), ("line", "lines", "per line")):
            w = np.array([r["shape"][0] if (ax == "neuron") != (r["plate"] == "words_in") else r["shape"][1] for r in R], float)
            for kind in ("net3", "cell3", "any3"):
                a = np.array([r["steps"][st]["sd"]["plate"][f"{ax}_{kind}"] for r in R])
                fr = np.array([r["steps"][st]["sd"]["fresh"][f"{ax}_{kind}"] for r in R])
                sh = np.array([r["steps"][st]["sd"]["shuffle"][f"{ax}_{kind}"] for r in R])
                print(f"  {lab:10s} {kind:5s} (sd form; pooled over plates, weighted by count): plate {pct((a * w).sum() / w.sum())}%  "
                      f"fresh {pct((fr * w).sum() / w.sum())}%  shuffle {pct((sh * w).sum() / w.sum())}%")
        # where the place imbalance sits: the 4 highest places that hold any 1s against the rest
        for form in ("ss", "sd"):
            lo = hi = nlo = nhi = 0
            for r in R:
                zz = r["steps"][st][form]["plate"]["place_z"]
                live = [p for p, v in enumerate(zz) if v is not None]
                top = set(live[-4:])
                for p in live:
                    big = abs(zz[p]) > 3
                    if p in top: hi += big; nhi += 1
                    else: lo += big; nlo += 1
            print(f"  {form}: places beyond 3 widths: lower places {lo} of {nlo}, the 4 highest live places {hi} of {nhi}")

    print(f"\n--- per plate kind (layers in order; sd = signed digits, ss = sign and size) ---")
    for k, label in KINDS:
        rs = [r for r in R if r["plate"].split(".")[0] == k]
        nm = ("lane", "word") if k == "words_in" else ("word", "lane") if k == "words_out" else ("neuron", "line")
        print(f"\n{label} ({k}), {len(rs)} plate(s), shape {rs[0]['shape']} as stored [out][in] (words_in: [word][lane])")
        for st in STEPS:
            g = lambda form, src, key: [r["steps"][st][form][src][key] for r in rs]
            mz = lambda form: " ".join(f"{v:.1f}@{p}" for v, p in zip(g(form, "plate", "maxz"), g(form, "plate", "maxz_place")))
            print(f"  {st:10s} largest place |z| (value@place) ss: {mz('ss')}")
            print(f"  {'':10s} largest place |z| (value@place) sd: {mz('sd')}   | fresh largest sd: {max(g('sd', 'fresh', 'maxz')):.1f}")
            print(f"  {'':10s} places beyond 3 (ss / sd): {' '.join(str(a) + '/' + str(b) for a, b in zip(g('ss', 'plate', 'n_over3'), g('sd', 'plate', 'n_over3')))}"
                  f"   of {' '.join(str(a) + '/' + str(b) for a, b in zip(g('ss', 'plate', 'n_live'), g('sd', 'plate', 'n_live')))}")
            print(f"  {'':10s} plate net z: {' '.join(f'{v:+.1f}' for v in g('ss', 'plate', 'plate_z'))}   (fresh {' '.join(f'{v:+.1f}' for v in g('ss', 'fresh', 'plate_z'))})")
            for ax, lab in (("neuron", nm[0]), ("line", nm[1])):
                print(f"  {'':10s} per {lab:6s} net>3 %: {' '.join(pct(v) for v in g('sd', 'plate', ax + '_net3'))}"
                      f"  (fresh {' '.join(pct(v) for v in g('sd', 'fresh', ax + '_net3'))}; shuffle {' '.join(pct(v) for v in g('sd', 'shuffle', ax + '_net3'))})")
                print(f"  {'':10s} per {lab:6s} place cells>3 % sd: {' '.join(pct(v) for v in g('sd', 'plate', ax + '_cell3'))}"
                      f"  ss: {' '.join(pct(v) for v in g('ss', 'plate', ax + '_cell3'))}"
                      f"  (fresh sd {' '.join(pct(v) for v in g('sd', 'fresh', ax + '_cell3'))}; shuffle sd {' '.join(pct(v) for v in g('sd', 'shuffle', ax + '_cell3'))})")
