"""THE REPORT of tips_blank/: every table read from the runs' JSON files (pythia_<size>.json, tiny_w128_L4_<r>.json,
gpt2.json, qwen.json). Nothing here is measured again; every number is a ratio or a difference of whole-number counts.

RULES FIXED BEFORE ANY RESULT WAS READ (written with this script, while the runs were going):
  - "the bulk" of a plate's diagonals = every used diagonal except the top 6 (the top 4 for the tiny models, whose
    plate side has one more sign place); "the top" = the top 6 (tiny: 4), listed one by one.
  - the task's A, per plate: within 1 point of the blank at EVERY diagonal, and within 1 point with 2^d weights.
  - a diagonal's tips "exceed their shuffle" when the tip size is at least 1.01 x the shuffle's, "fall below" at 0.99 or
    less; between is "match".
  - the tip size above the shuffle: sum over k of 2^k (tips_k - shuffled tips_k) over sum of 2^k tips_k (2^d weights),
    and the same with every diagonal counted alike (place pairs counted alike).
    python3 tips_blank/report.py > tips_blank/report.log
"""
import os, json, numpy as np, sys
OUT = "/Users/abundancemachine/streaming-machine/tips_blank"; sys.path.insert(0, OUT)
from engine import diag
KIND = lambda p: p.split(".")[0]
def pct(v): return f"{100 * v:6.2f}" if np.isfinite(v) else "   -  "

def add(dd):
    """sum tips and meetings of several plates' diag() results (same plate places)"""
    tip = sum(d["tip"] for d in dd); meet = sum(d["meet"] for d in dd); ds = sum(d["ds"] for d in dd)
    used = meet > 0; return dict(tip=tip, meet=meet, ds=ds, used=used, paired=np.where(used, 1 - tip / np.where(used, meet, 1), np.nan))

def wshare(d, which="paired"):
    ks = np.flatnonzero(d["used"]); w = np.exp2(ks.astype(float))
    return 1 - float((w * d["tip"][ks]).sum() / (w * d["meet"][ks]).sum())

def above(d, dsh, weighted=True):
    ks = np.flatnonzero(d["used"]); w = np.exp2(ks.astype(float)) if weighted else np.ones(len(ks))
    return float((w * (d["tip"][ks] - dsh["tip"][ks])).sum() / (w * d["tip"][ks]).sum())

def by_kind(plates, call):
    kinds = {}
    for p, r in plates.items(): kinds.setdefault(KIND(p), []).append(diag(r[call]))
    return {k: add(v) for k, v in kinds.items()}

def section_models(title, plates, calls, ref_name, ntop=6):
    """A/D tables (paired-off shares) and C tables (against the light shuffle) for one model"""
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
    first = plates[next(iter(plates))]
    show = calls + [c for c in ("fin_sh",) if c in first]
    K = {c: by_kind(plates, c) for c in show}
    kinds = list(K[calls[0]].keys()); coin = {KIND(p): r["coin"] for p, r in plates.items()}
    print(f"\n  PAIRED-OFF SHARE BY DIAGONAL, per plate kind (all plates of the kind together). coin = the coin toss's share.")
    for k in kinds:
        d0 = K[calls[0]][k]; ks = np.flatnonzero(d0["used"]); bulk, top = ks[:-ntop], ks[-ntop:]
        print(f"\n  {k} (N = {plates[[p for p in plates if KIND(p) == k][0]]['N']}, coin {pct(1 - coin[k])}%)")
        print(f"    {'':12s}" + "".join(f"{c:>16s}" for c in show))
        print(f"    {'bulk d' + str(bulk[0]) + '-' + str(bulk[-1]):12s}" + "".join(
            f"   {pct(np.nanmin(K[c][k]['paired'][bulk]))}-{pct(np.nanmax(K[c][k]['paired'][bulk])).strip():>5s}" for c in show))
        for kk in top: print(f"    {'d' + str(kk):12s}" + "".join(f"{pct(K[c][k]['paired'][kk]):>16s}" for c in show))
        print(f"    {'2^d weights':12s}" + "".join(f"{pct(wshare(K[c][k])):>16s}" for c in show))
        dv = lambda c1, c2: 100 * np.nanmax(np.abs(K[c1][k]['paired'][bulk] - K[c2][k]['paired'][bulk]))
        print(f"    largest difference in the bulk, points: " + ", ".join(f"fin-{c} {dv('fin', c):.3f}" for c in show if c != "fin"))
    # the task's A, plate by plate
    for ref_name in [c for c in calls if c not in ("fin", "own")]:
        n_all = n_w = 0; worst = []
        for p, r in plates.items():
            df, db = diag(r["fin"]), diag(r[ref_name]); ks = np.flatnonzero(df["used"])
            dev = np.abs(df["paired"][ks] - db["paired"][ks]); wdev = abs(wshare(df) - wshare(db))
            n_all += bool(dev.max() <= 0.01); n_w += bool(wdev <= 0.01); worst.append((dev.max(), p, int(ks[np.argmax(dev)]), wdev))
            dev_bulk = dev[:-ntop].max()
            r["_bulk_dev"] = dev_bulk
        bulk_max = max(r["_bulk_dev"] for r in plates.values())
        worst.sort(reverse=True)
        print(f"\n  FINISHED AGAINST THE {ref_name.upper()} ON THE FINISHED LIGHT, plate by plate ({len(plates)} plates):")
        print(f"    within 1 point at every diagonal: {n_all} plates; within 1 point with 2^d weights: {n_w} plates")
        print(f"    largest difference in the bulk (all but the top {ntop}) over all plates: {100 * bulk_max:.3f} points")
        print(f"    largest differences: " + "; ".join(f"{p} d{k} {100 * v:.1f} pts (2^d: {100 * w:.1f})" for v, p, k, w in worst[:6]))
    # C: tips against the light shuffle
    pairs = [(c, c + "_sh") for c in calls if c + "_sh" in plates[next(iter(plates))]]
    if pairs:
        print(f"\n  C. TIP SIZE AGAINST THE SAME PLATE MET BY THE LIGHT SHUFFLED ACROSS ITS LINES (ratio real/shuffled, per kind)")
        Ks = {c: by_kind(plates, c) for _, c in pairs}
        for k in kinds:
            d0 = K[calls[0]][k]; ks = np.flatnonzero(d0["used"]); bulk, top = ks[:-ntop], ks[-ntop:]
            print(f"\n  {k}")
            print(f"    {'':12s}" + "".join(f"{c + ' tips':>18s}{c + ' signed':>14s}" for c, _ in pairs))
            rb = lambda c, s, key: K[c][k][key][bulk] / Ks[s][k][key][bulk]
            print(f"    {'bulk':12s}" + "".join(f"   {rb(c, s, 'tip').min():6.4f}-{rb(c, s, 'tip').max():6.4f}  {rb(c, s, 'ds').min():5.3f}-{rb(c, s, 'ds').max():5.3f}" for c, s in pairs))
            for kk in top:
                print(f"    {'d' + str(kk):12s}" + "".join(f"{K[c][k]['tip'][kk] / Ks[s][k]['tip'][kk]:>18.4f}{K[c][k]['ds'][kk] / Ks[s][k]['ds'][kk]:>14.3f}" for c, s in pairs))
            print(f"    {'above shuf.':12s}" + "".join(f"{100 * above(K[c][k], Ks[s][k]):>16.2f}%{100 * above(K[c][k], Ks[s][k], False):>13.2f}%" for c, s in pairs)
                  + "   (tip size above the shuffle's: 2^d weights | diagonals alike)")
        print(f"\n  C, plate by plate: diagonals whose tips exceed (>= 1.01) or fall below (<= 0.99) their shuffle")
        for c, s in pairs:
            ex_, lo_, hit = 0, 0, {}
            for p, r in plates.items():
                d, dsh = diag(r[c]), diag(r[s]); ks = np.flatnonzero(d["used"]); rr = d["tip"][ks] / dsh["tip"][ks]
                e = ks[rr >= 1.01]; f = ks[rr <= 0.99]; ex_ += len(e); lo_ += len(f)
                for kk in e: hit[int(ks[-1] - kk)] = hit.get(int(ks[-1] - kk), 0) + 1
            depth = ", ".join(f"top-{j}: {hit[j]}" for j in sorted(hit))
            print(f"    {c:6s}: {ex_} plate-diagonals exceed, {lo_} fall below (of {sum(len(np.flatnonzero(diag(r[c])['used'])) for r in plates.values())});"
                  f" exceeding, by depth from the top diagonal (top-0 = the top): {depth if depth else 'none'}")
        tot = {c: add([diag(r[c]) for r in plates.values() if r["N"] == r["N"]]) for c in [x for pr in pairs for x in pr]}
        print("\n  the whole model (every plate together): tip size above the shuffle's, 2^d weights | diagonals alike: "
              + "; ".join(f"{c} {100 * above(tot[c], tot[s]):.2f}% | {100 * above(tot[c], tot[s], False):.2f}%" for c, s in pairs))
    print(f"\n  EVERY DIAGONAL, per kind (paired-off share, %):")
    for k in kinds:
        ks = np.flatnonzero(K[show[0]][k]["used"])
        print(f"    {k:6s} " + " ".join(f"{'d' + str(kk):>6s}" for kk in ks))
        for c in show: print(f"    {c:6s} " + " ".join(f"{100 * K[c][k]['paired'][kk]:6.2f}" for kk in ks))
    exact = all(r[c]["bad"] == 0 for r in plates.values() for c in r if isinstance(r[c], dict) and "bad" in r[c])
    of = [(p, r["orderfree_same"]) for p, r in plates.items() if "orderfree_same" in r]
    print(f"\n  EXACT: every call's piles from the tips = the direct piles at every position: {exact}; order-free: " + ", ".join(f"{p} {'same' if v else 'DIFFERENT'}" for p, v in of))
    return K

def section_tiny(r):
    snaps = r["snaps"]; steps = sorted(snaps, key=int)
    print(f"\n{'=' * 118}\nB. TINY {r['build']} ({r['positions']} positions): finished, blank (as it stands), pushes alone, on the finished light\n{'=' * 118}")
    for k in ("qkv", "proj", "fc", "mproj", "head"):
        print(f"\n  {k}: paired-off share. bulk = min-max over all but the top 4 diagonals; then the top 4; then the tip size against the shuffle at the top 2")
        print(f"    {'step':>5s} {'coin':>7s} | {'fin bulk':>15s} {'blank bulk':>15s} {'push bulk':>15s} | {'fin top4':>27s} | {'blank top4':>27s} | {'push top4':>27s} | ratio fin/bl/push at d-top, d-top-1")
        for t in steps:
            pl = {p: v for p, v in snaps[t].items() if KIND(p) == k}; coin = next(iter(pl.values()))["coin"]
            D = {c: add([diag(v[c]) for v in pl.values()]) for c in ("fin", "blank", "push", "fin_sh", "blank_sh", "push_sh")}
            ks = np.flatnonzero(D["fin"]["used"]); bulk, top = ks[:-4], ks[-4:]
            b = lambda c: f"{pct(np.nanmin(D[c]['paired'][bulk])).strip()}-{pct(np.nanmax(D[c]['paired'][bulk])).strip()}"
            tp = lambda c: " ".join(f"{100 * D[c]['paired'][kk]:6.2f}" for kk in top)
            rat = " ".join(f"{D[c]['tip'][kk] / D[c + '_sh']['tip'][kk]:.3f}" for kk in top[::-1][:2] for c in ("fin", "blank", "push"))
            print(f"    {t:>5s} {pct(1 - coin)} | {b('fin'):>15s} {b('blank'):>15s} {b('push'):>15s} | {tp('fin'):>27s} | {tp('blank'):>27s} | {tp('push'):>27s} | {rat}")
    # the real scale: does the paired-off mass grow with the pushes?
    print(f"\n  THE REAL SCALE (each meeting x 2^(its place pair + the neuron's power + the position's power)), all plates together:")
    print(f"    {'step':>5s} {'fin paired off':>16s} {'x step 0':>9s} {'fin tips':>12s} {'x step 0':>9s} {'blank paired off':>17s} {'push paired off':>16s} | paired-off share (real scale): fin / blank / push")
    base = None
    for t in steps:
        tot = {}
        for c in ("fin", "blank", "push"):
            po = tp_ = mm = 0.0
            for p, v in snaps[t].items():
                d = diag(v[c]); ks = np.flatnonzero(d["used"]); sc = np.exp2(ks + v["ex_min"] + v["e_min"])
                po += float(((d["meet_real"][ks] - d["tip_real"][ks]) * sc).sum()); tp_ += float((d["tip_real"][ks] * sc).sum()); mm += float((d["meet_real"][ks] * sc).sum())
            tot[c] = (po, tp_, mm)
        if base is None: base = tot["fin"]
        print(f"    {t:>5s} {tot['fin'][0]:16.4e} {tot['fin'][0] / base[0]:9.3f} {tot['fin'][1]:12.4e} {tot['fin'][1] / base[1]:9.3f} {tot['blank'][0]:17.4e} {tot['push'][0]:16.4e} | "
              + " / ".join(f"{100 * tot[c][0] / tot[c][2]:.3f}" for c in ("fin", "blank", "push")))
    exact = all(v[c]["bad"] == 0 for s in snaps.values() for v in s.values() for c in v if isinstance(v[c], dict) and "bad" in v[c])
    of = all(v["orderfree_same"] for s in snaps.values() for v in s.values() if "orderfree_same" in v)
    print(f"\n  EXACT at every call, every snapshot: {exact}; order-free (qkv.0 and head, every snapshot): {'same' if of else 'DIFFERENT'}")

if __name__ == "__main__":
    for size in ("70m", "160m"):
        f = f"{OUT}/pythia_{size}.json"
        if os.path.exists(f):
            R = json.load(open(f)); section_models(f"A and C. PYTHIA-{size} ({R['positions']} positions): fin = finished, blank = step-0 blank on the shared grid, fresh = a bell draw of each finished neuron's spread, all on the finished light; own = the blank on the blank machine's own light", R["plates"], ["fin", "blank", "fresh", "own"], "blank")
    for rc in ("A", "B", "C"):
        f = f"{OUT}/tiny_w128_L4_{rc}.json"
        if os.path.exists(f): section_tiny(json.load(open(f)))
    for nm, title in (("gpt2", "GPT-2"), ("qwen", "QWEN3-0.6B")):
        f = f"{OUT}/{nm}.json"
        if os.path.exists(f):
            R = json.load(open(f)); section_models(f"D and C. {title} ({R['positions']} positions): fin = finished, fresh = a bell draw of each neuron's spread, on the finished light", R["plates"], ["fin", "fresh"], "fresh")
