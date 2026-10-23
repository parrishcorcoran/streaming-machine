"""groups2/topg_report.py: the totals of groups2/topg.py's runs, the controls, the order-free check, the predictions set
beside the results, and the whole machines (an estimate, labelled). Reads only groups2/out/topg_*.{npz,json}.
    python3 groups2/topg_report.py   -> groups2/out/topg_report.log, topg_report.json
"""
import os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/out"
FN = {"uh": "up/hole", "sd": "signed digits", "pp": "per place"}; FORMS = ["uh", "sd", "pp"]
LOG = open(f"{OUT}/topg_report.log", "w"); RES = {}
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def pc(a, b): return 100.0 * a / b if b else float("nan")

def load(model, st, g, run):
    """one run's records, the Qwen form halves joined (every common array checked equal)"""
    tags = [f"topg_{model}_{st}_g{g}_{run}"] if model == "gpt2" else [f"topg_{model}_{st}_g{g}_{run}_uh-pp", f"topg_{model}_{st}_g{g}_{run}_sd"]
    if not all(os.path.exists(f"{OUT}/{t}.json") for t in tags): return None
    R = dict(forms={}, bad={}, digest={}, sha=None); common = None
    for t in tags:
        J = json.load(open(f"{OUT}/{t}.json")); Z = np.load(f"{OUT}/{t}.npz")
        com = {k: Z[k] for k in Z.files if "__" not in k}
        if common is None: common = com
        else:
            for k in com: assert np.array_equal(com[k], common[k]), (t, k)
        for f in J["forms"]:
            R["forms"][f] = {k.split("__")[1]: Z[k] for k in Z.files if k.startswith(f + "__")}
            R["bad"][f] = J["bad"][f]; R["digest"][f] = J["digest"][f]
        if model == "qwen": R["sha"] = J["sha"]
        R["G"], R["ppmask"], R["lit"] = J["G"], J["ppmask"], J["lit_by_place"]
    R.update(common); R["n"] = len(R["plain_meet"]); R["plain"] = R["plain_meet"] + R["plain_div"] + R["plain_stand"]
    return R

def refs_of(model, st):
    if model == "gpt2" and st == "refs":
        Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); P = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
        return dict(top5=P["top5"][Z["rows"]], win=P["win"][Z["rows"]])
    if model == "gpt2" and st == "windows": return dict(top5=np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5))
    if model == "qwen" and st == "machine":
        RF = json.load(open(f"{HERE}/qwen_fringes/ref.json"))["read"]; MT = json.load(open(f"{HERE}/qwen06b/light/meta.json"))
        return dict(top5=np.array(RF["top5"]), win=np.array(RF["win"]), sha=RF["sha"], shac=MT["sha"])
    if model == "qwen" and st == "words":
        W = json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))
        return dict(top5=np.array(W["top5"]), win=np.array(W["win"]), sha=W["sha"], nxt=np.array(W["next"]))

_TB = _QF = None
def plain_scores(model, st, run, q):
    """CHECK ONLY: the plain read's scores of every word at position q of a run's light (original word numbers)"""
    global _TB, _QF
    import sys
    if model == "gpt2":
        if _TB is None:
            a = sys.argv; sys.argv = ["x"]; sys.path.insert(0, f"{HERE}/topband"); import topband as T; sys.argv = a; _TB = T
        Z = np.load(f"{HERE}/fringes_machine/{'refs' if st == 'refs' else 'windows'}_top_light.npz"); X = Z["x16"].astype(np.int64)
        if run == "shuffle":
            rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])
        W = np.asarray(_TB.M["head.w"]).astype(np.int64); pile = X[q] @ W
        F = int((_TB.HEAD_E[0] + int(Z["ex"][q]) + _TB.ES) * 65536 - int(Z["LS"][q]))
        return _TB.power(pile, F) + np.asarray(_TB.HEAD_B).astype(np.int64)
    return None

def tie_lines(name, model, st, run, R):
    """where (ii)'s top 5 differs from the plain read's by the top-5 rule: is it an exact tie, with (ii)'s five the five
    best scores in order?"""
    for f in FORMS:
        if f not in R["forms"]: continue
        r = R["forms"][f]; d = np.flatnonzero(~(r["ii_top5"] == R["ptop5"]).all(1))
        for q in d:
            S = plain_scores(model, st, run, q)
            if S is None: log(f"  {name:28s} [{f}] position {q}: (no score check for this model)"); continue
            best = np.sort(S)[::-1][:5]; a, b = S[r["ii_top5"][q]], S[R["ptop5"][q]]
            log(f"  {name:28s} [{f}] position {q}: plain {R['ptop5'][q].tolist()} scores {b.tolist()}; (ii) {r['ii_top5'][q].tolist()} "
                f"scores {a.tolist()}; (ii)'s five = the five best scores in order: {bool(np.array_equal(a, best))}; an exact tie: {bool(np.array_equal(a, b))}")

def exact_lines(name, R, ref):
    n = R["n"]; ok = True
    for f in FORMS:
        if f not in R["forms"]: continue
        r = R["forms"][f]; bad = sum(R["bad"][f].values())
        s = (f"  {name:28s} [{f}] checks all 0: {bad == 0}; (i) top 5 = plain {int((r['i_top5'] == R['ptop5']).all(1).sum())}/{n}, "
             f"race {int((r['i_win'] == R['pwin']).sum())}/{n}; (ii) top 5 {int((r['ii_top5'] == R['ptop5']).all(1).sum())}/{n}, "
             f"race {int((r['ii_win'] == R['pwin']).sum())}/{n}")
        ok &= bad == 0 and (r["i_top5"] == R["ptop5"]).all() and (r["ii_top5"] == R["ptop5"]).all() and (r["i_win"] == R["pwin"]).all() and (r["ii_win"] == R["pwin"]).all()
        if ref is not None:
            s += f"; saved refs: top 5 (i) {int((r['i_top5'] == ref['top5']).all(1).sum())}, (ii) {int((r['ii_top5'] == ref['top5']).all(1).sum())}"
            if "win" in ref: s += f", race (i) {int((r['i_win'] == ref['win']).sum())}, (ii) {int((r['ii_win'] == ref['win']).sum())}"
            if "nxt" in ref: s += f", race = next greedy word (i) {int((r['i_win'] == ref['nxt']).sum())}, (ii) {int((r['ii_win'] == ref['nxt']).sum())}"
            ok &= bool((r["ii_top5"] == ref["top5"]).all())
        log(s)
    if ref is not None and R["sha"] is not None:
        log(f"  {name:28s} score rows (i), sha256 of every word's score: = qwen_base2's own read {sum(a == b for a, b in zip(R['sha'], ref['sha']))}/{n}"
            + (f", = capture's record {sum(a == b for a, b in zip(R['sha'], ref['shac']))}/{n}" if "shac" in ref else ""))
    dg = [R["digest"][f] for f in FORMS if f in R["digest"]]
    log(f"  {name:28s} every form's piles the same (digests, every position): {all(d == dg[0] for d in dg)}")
    return ok

def shares(R, f):
    r = R["forms"][f]; PL = int(R["plain"].sum()); PM = int(R["plain_meet"].sum()); B = int(R["band_total"].sum())
    return dict(i=pc(int(r["i_total"].sum()), PL), i_meet=pc(int(r["table"].sum() + r["i_reads"].sum() + r["i_light"].sum()), PM),
                ii=pc(int(r["ii_total"].sum()), PL), ii_band=pc(int(r["ii_total"].sum()), B), band=pc(B, PL),
                below=int((r["ii_total"] < R["band_total"]).sum()), n=R["n"],
                table=pc(int(r["table"].sum()), PL), i_reads=pc(int(r["i_reads"].sum()), PL),
                ii_reads=pc(int(r["ii_reads"].sum()), PL), ii_stand=pc(int(r["ii_stand"].sum()), PL),
                ii_rest=pc(int((r["ii_light"] + r["ii_bound"] + r["ii_end"]).sum()), PL), cmp=float(r["ii_cmp"].mean()))

def joined(Rs, f, key):
    """a share over several sets together (sums over all their positions)"""
    num = sum(int(R["forms"][f][key].sum()) for R in Rs); PL = sum(int(R["plain"].sum()) for R in Rs)
    return pc(num, PL)

RUNS = {}
for model, sets in (("gpt2", ("refs", "windows")), ("qwen", ("machine", "words"))):
    for st in sets:
        for g in (16, 20):
            for run in ("main", "seed2", "shuffle", "order"):
                R = load(model, st, g, run)
                if R is not None: RUNS[(model, st, g, run)] = R
log("GROUP PILES AT THE TOP TABLES (groups2/topg.py): the report")
log(f"runs found: {sorted(RUNS)}")

log("\nEXACT (every run; checks: piles = the direct product at every word, (i) every word's score = the plain read's, (ii) the bound")
log("held for every word at every test, no word set aside at or above the true 5th-best, the words read to the end score as the plain read)")
allok = True
for key, R in sorted(RUNS.items()):
    model, st, g, run = key
    allok &= exact_lines(f"{model} {st} g={g} {run}", R, refs_of(model, st) if run in ("main", "seed2", "order") else None)
log(f"  every check 0 and every top 5 by the top-5 rule identical: {allok}")
log("  where the top-5 rule's order differs (an exact tie can be ordered either way by judge5's rule, which partitions; the checks")
log("  above say every word at or above the 5th-best was kept and scored exactly):")
for key, R in sorted(RUNS.items()):
    tie_lines(f"{key[0]} {key[1]} g={key[2]} {key[3]}", *[key[0], key[1], key[3]], R)

log("\n(i) GROUP PILES FOR EVERY WORD, ALL PLACES (whole read: table + reads + X + every word's divide and standing amount), share of")
log("the plain read; in brackets the meetings part against the plain meetings")
for model, sets in (("gpt2", ("refs", "windows")), ("qwen", ("machine", "words"))):
    for g in (16, 20):
        Rs = [RUNS[(model, st, g, "main")] for st in sets if (model, st, g, "main") in RUNS]
        if not Rs: continue
        for f in FORMS:
            if f not in Rs[0]["forms"]: continue
            parts = "   ".join(f"{st} {shares(R, f)['i']:6.2f}% ({shares(R, f)['i_meet']:5.2f}%)" for st, R in zip(sets, Rs))
            s0 = shares(Rs[0], f)
            log(f"  {model} g={g} {FN[f]:14s} {parts}   all {joined(Rs, f, 'i_total'):6.2f}%   [{sets[0]}: table {s0['table']:.2f}%, reads {s0['i_reads']:.2f}%]")
            RES[f"i|{model}|{g}|{f}"] = dict(first=s0["i"], all=joined(Rs, f, "i_total"))

log("\n(ii) FROM THE TOP PLACE DOWN, THE BAND READ'S SET-ASIDE AFTER EVERY PLACE: share of the plain read; against the band read on the")
log("same light (its own per-position records); positions where it is below the band read")
for model, sets in (("gpt2", ("refs", "windows")), ("qwen", ("machine", "words"))):
    for g in (16, 20):
        Rs = [RUNS[(model, st, g, "main")] for st in sets if (model, st, g, "main") in RUNS]
        if not Rs: continue
        for f in FORMS:
            if f not in Rs[0]["forms"]: continue
            for st, R in zip(sets, Rs):
                s = shares(R, f)
                log(f"  {model} g={g} {FN[f]:14s} {st:8s} {s['ii']:6.2f}%  = {s['ii_band']:6.2f}% of the band read ({s['band']:.2f}%); below it at "
                    f"{s['below']}/{s['n']}; table {s['table']:.2f}, reads {s['ii_reads']:.2f}, standing {s['ii_stand']:.2f}, rest {s['ii_rest']:.3f}; compares {s['cmp']:,.0f} a position")
            RES[f"ii|{model}|{g}|{f}"] = dict(first=shares(Rs[0], f)["ii"], all=joined(Rs, f, "ii_total"),
                                             band_first=shares(Rs[0], f)["ii_band"])
            if len(Rs) > 1:
                bt = sum(int(R["band_total"].sum()) for R in Rs); it = sum(int(R["forms"][f]["ii_total"].sum()) for R in Rs)
                log(f"  {model} g={g} {FN[f]:14s} {'all':8s} {joined(Rs, f, 'ii_total'):6.2f}%  = {pc(it, bt):6.2f}% of the band read "
                    f"({pc(bt, sum(int(R['plain'].sum()) for R in Rs)):.2f}%)")

log("\n(ii) WHERE THE WORDS FALL AWAY (per place, main runs, g = 16; median / mean / least / most words still in after that place's test;")
log("that place's reads into the words still in, share of the plain read)")
for model, st in (("gpt2", "refs"), ("gpt2", "windows"), ("qwen", "machine")):
    R = RUNS.get((model, st, 16, "main"))
    if R is None: continue
    for f in ("pp", "uh", "sd"):
        if f not in R["forms"]: continue
        r = R["forms"][f]; PL = int(R["plain"].sum()); log(f"  {model} {st} [{f}]")
        for b in range(17, -1, -1):
            col = r["ii_surv"][:, b]; rb = pc(int(r["ii_byplace"][:, b].sum()), PL)
            if (col >= 0).all():
                log(f"    place {b:2d}: {int(np.median(col)):7d} {col.mean():10.1f} {col.min():7d} {col.max():7d}   reads {rb:6.3f}%")
            elif r["ii_byplace"][:, b].any() or b == 0:
                log(f"    place {b:2d}: read to the end: median {int(np.median(r['ii_last']))}, mean {r['ii_last'].mean():.2f}, most {int(r['ii_last'].max())}   reads {rb:6.3f}%")

log("\nCONTROLS (g = 16; refs for GPT-2, the 613 positions for Qwen; shuffle: Qwen at every 4th position, the band read's own control draw)")
for model, st in (("gpt2", "refs"), ("qwen", "machine")):
    M0 = RUNS.get((model, st, 16, "main"))
    if M0 is None: continue
    for run in ("seed2", "shuffle", "order"):
        R = RUNS.get((model, st, 16, run))
        if R is None: continue
        if run == "shuffle" and model == "qwen":
            idx = np.asarray(R["sel"])
            def sub(A): return A[idx]
        else:
            idx = np.arange(M0["n"])
            def sub(A): return A
        PL0 = int(sub(M0["plain"]).sum()); PL1 = int(R["plain"].sum()); B0 = int(sub(M0["band_total"]).sum()); B1 = int(R["band_total"].sum())
        for f in FORMS:
            if f not in R["forms"] or f not in M0["forms"]: continue
            a, b = M0["forms"][f], R["forms"][f]
            i0, i1 = pc(int(sub(a["i_total"]).sum()), PL0), pc(int(b["i_total"].sum()), PL1)
            ii0, ii1 = pc(int(sub(a["ii_total"]).sum()), PL0), pc(int(b["ii_total"].sum()), PL1)
            s = f"  {model} {run:7s} [{f}] (i) {i1:6.2f}% against {i0:6.2f}% ({i1 - i0:+.3f});  (ii) {ii1:6.2f}% against {ii0:6.2f}% ({ii1 - ii0:+.3f})"
            if run == "shuffle": s += f";  the band read on the same lights {pc(B1, PL1):.2f}% against {pc(B0, PL0):.2f}%"
            if run in ("seed2", "order"):
                s += f";  survivors identical at every test {bool(np.array_equal(sub(a['ii_surv']), b['ii_surv']))}"
                s += f"; (ii) reads identical {bool(np.array_equal(sub(a['ii_reads']), b['ii_reads']))}"
            if run == "order":
                same = {k: bool(np.array_equal(a[k], b[k])) for k in ("i_reads", "i_light", "ii_reads", "ii_surv", "ii_end", "ii_bound", "ii_cmp", "ii_light", "ii_stand", "i_byplace", "ii_byplace", "i_adds")}
                s += f"\n      every count identical except the table: " + ", ".join(f"{k} {v}" for k, v in same.items())
                s += f"\n      table {pc(int(b['table'].sum()), PL1):.4f}% against {pc(int(a['table'].sum()), PL0):.4f}%; piles identical (digests) {M0['digest'][f] == R['digest'][f]}"
                s += (f"\n      top 5 (ii) = main's in order {int((a['ii_top5'] == b['ii_top5']).all(1).sum())}/{R['n']}, as a set "
                      f"{int((np.sort(a['ii_top5'], 1) == np.sort(b['ii_top5'], 1)).all(1).sum())}/{R['n']}; race {int((a['ii_win'] == b['ii_win']).sum())}/{R['n']}")
                dif = np.flatnonzero(~(a["ii_top5"] == b["ii_top5"]).all(1))
                for q in dif[:5]: s += f"\n      position {q}: main {a['ii_top5'][q].tolist()}, order-free {b['ii_top5'][q].tolist()}"
            if run == "seed2": s += f"; piles identical (digests) {M0['digest'][f] == R['digest'][f]}"
            log(s)
            RES[f"ctl|{model}|{run}|{f}"] = dict(i=i1, i0=i0, ii=ii1, ii0=ii0)

log("\nTHE WHOLE MACHINES (an estimate, labelled: only the top read changed, every other part as its report gives it; refs / 613 positions)")
G2 = RUNS.get(("gpt2", "refs", 16, "main")); Q = RUNS.get(("qwen", "machine", 16, "main"))
if G2 is not None:
    whole, a3, plainG = 1694214555, 145099607, 3666596259
    band = int(G2["band_total"].sum()) // G2["n"]
    log(f"  GPT-2 base2_fringes2.py 46.21% = {whole:,} a token, its top read the collapsing read a3 ({a3:,})")
    log(f"    with the band read ({band:,}): {pc(whole - a3 + band, plainG):.2f}%")
    for f in FORMS:
        v = int(G2["forms"][f]["ii_total"].sum()) // G2["n"]; RES[f"whole|gpt2|{f}"] = pc(whole - a3 + v, plainG)
        log(f"    with (ii) {FN[f]} ({v:,}): {pc(whole - a3 + v, plainG):.2f}%")
if Q is not None:
    whole, band0, plainQ = 5602023037, 302579448, 16397464589
    log(f"  Qwen3-0.6B qwen_fringes2.py 34.16% = {whole:,} a token, its top read the band read ({band0:,}; here {int(Q['band_total'].sum()) // Q['n']:,})")
    for f in FORMS:
        v = int(Q["forms"][f]["ii_total"].sum()) // Q["n"]; RES[f"whole|qwen|{f}"] = pc(whole - band0 + v, plainQ)
        log(f"    with (ii) {FN[f]} ({v:,}): {pc(whole - band0 + v, plainQ):.2f}%")

log("\nPREDICTIONS (written in topg.py before it ran) AGAINST THE RESULTS (first set: GPT-2 refs, Qwen's 613 positions)")
PRED = [("i|gpt2|16|uh", 20, 23.5), ("i|gpt2|16|pp", 16, 19), ("i|gpt2|16|sd", 23, 28), ("i|qwen|16|uh", 14.5, 17.5), ("i|qwen|16|pp", 13, 16),
        ("i|qwen|16|sd", 20, 25), ("ii|gpt2|16|pp", 7.5, 10), ("ii|gpt2|16|uh", 11, 14), ("ii|gpt2|16|sd", 10.5, 14), ("ii|qwen|16|pp", 4.3, 5.8),
        ("ii|qwen|16|uh", 5.5, 7), ("ii|qwen|16|sd", 6, 8), ("ii|gpt2|20|pp", 13, 17), ("ii|qwen|20|pp", 5.5, 7.5)]
for k, lo, hi in PRED:
    if k in RES:
        v = RES[k]["first"]; log(f"  {k:16s} predicted {lo}-{hi}%: {v:.2f}% -> {'held' if lo <= v <= hi else ('lower' if v < lo else 'higher')}")
for k, lo, hi in (("i|gpt2|16|pp", 15, 22), ("i|qwen|16|pp", 15, 22), ("ii|gpt2|16|pp", 4, 8), ("ii|qwen|16|pp", 4, 8)):
    if k in RES:
        v = RES[k]["first"]; log(f"  THE TASK'S {k:16s} {lo}-{hi}%: {v:.2f}% -> {'held' if lo <= v <= hi else ('lower' if v < lo else 'higher')}")
json.dump(RES, open(f"{OUT}/topg_report.json", "w"), indent=1)
