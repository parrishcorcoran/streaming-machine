"""confirm_groups/cg_report.py -- the independent rebuild's report: exactness, work by part against the delivered machine,
another grouping, the shuffle control, the order-free run. Reads only this folder's outputs, the references
(capture's record, qwen_fringes/ref.json and words_ref.json) and, for the comparison, the delivered machine's own
records in groups2/machine/ (results only).

    python3 confirm_groups/cg_report.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = "/Users/abundancemachine/streaming-machine"; CG = f"{HERE}/confirm_groups"; O = f"{CG}/out"; DM = f"{HERE}/groups2/machine"
LOG = open(f"{O}/report.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def J(p): return json.load(open(p))
def counts(p): Z = np.load(p); return {k.replace("__", "|"): Z[k] for k in Z.files}
NP = 613; L = 28
def tok(v): return f"{v / NP:,.0f}"
def pc(a, b): return f"{a / b * 100:.4f}%"

meta = J(f"{HERE}/qwen06b/light/meta.json"); RF = J(f"{HERE}/qwen_fringes/ref.json"); WR = J(f"{HERE}/qwen_fringes/words_ref.json")
ref = J(f"{O}/ref_refs.json"); refw = J(f"{O}/ref_words.json") if os.path.exists(f"{O}/ref_words.json") else None
def dcmp(mine, other, keys=None):
    kinds = {}
    for k, v in other.items():
        if keys and k.split("|")[0] not in keys: continue
        kk = k.split("|")[0]; kinds.setdefault(kk, [0, 0]); kinds[kk][1] += 1; kinds[kk][0] += mine.get(k) != v
    return ", ".join(f"{k} {a} of {b} differ" for k, (a, b) in kinds.items())

log("THE INDEPENDENT REBUILD OF THE QWEN3-0.6B GROUP MACHINE (confirm_groups/): report")
log("")
log("EXACTNESS")
runs = {}
for tag in ("refs_s1", "refs_s7", "words_s1", "orderfree_s1_o4242"):
    if os.path.exists(f"{O}/m_{tag}.json"): runs[tag] = J(f"{O}/m_{tag}.json")
for tag, m in runs.items():
    log(f"  [machine {tag}] checks: {m['chk']}")
    if tag.startswith("refs"):
        log(f"    vs this folder's fresh qwen_base2 run (the word table's raw products 'pile|head' left out: the group machine reads the top by its own read, checked below): {dcmp(m['dig'], {k: v for k, v in ref['dig'].items() if not k.startswith('pile|head')})}")
        log(f"    vs qwen_fringes/ref.json: {dcmp(m['dig'], RF['dig'])}")
        cap = {f'pile|{k}|{t}': v for k, vs in meta['dig'].items() for t, v in enumerate(vs)}
        log(f"    piles vs capture's record: {dcmp(m['dig'], cap)}")
    elif tag.startswith("words"):
        if refw: log(f"    vs this folder's fresh qwen_base2 run on the greedy texts: {dcmp(m['dig'], {k: v for k, v in refw['dig'].items() if not k.startswith('pile|head')})}")
        log(f"    vs words_ref.json: {dcmp(m['dig'], WR['dig'])}")
    else:
        sub = {k: v for k, v in ref["dig"].items() if int(k.split("|")[-1]) < 3 and not k.startswith("pile|head")}
        log(f"    vs this folder's fresh qwen_base2 run (texts 0-2): {dcmp(m['dig'], sub)}")
        mm = runs.get("refs_s1")
        if mm: log(f"    vs the main machine run (texts 0-2): {dcmp(m['dig'], {k: v for k, v in mm['dig'].items() if int(k.split('|')[-1]) < 3})}")
tops = {}
for tag in ("refs_s1", "refs_s7", "words_s1", "orderfree_s1_o4242"):
    if os.path.exists(f"{O}/t_{tag}.npz"): tops[tag] = (dict(np.load(f"{O}/t_{tag}.npz")), J(f"{O}/t_{tag}.json"))
for tag, (t, tj) in tops.items():
    sel = t["sel"]; n = len(sel)
    log(f"  [top {tag}] checks: {tj['bad']}")
    log(f"    top 5 = the plain read's {int((t['top5'] == t['ptop5']).all(1).sum())}/{n}, race winner {int((t['win'] == t['pwin']).sum())}/{n}")
    if tag.startswith(("refs", "orderfree")):
        R5, RW, RS = ref["top5"], ref["win"], ref["sha"]
        log(f"    vs this folder's fresh qwen_base2 read: score rows {sum(tj['sha'][i] == RS[q] for i, q in enumerate(sel))}/{n}, "
            f"top 5 {sum(list(t['top5'][i]) == R5[q] for i, q in enumerate(sel))}/{n}, race winner {sum(int(t['win'][i]) == RW[q] for i, q in enumerate(sel))}/{n}")
        log(f"    vs capture's record: score rows {sum(tj['sha'][i] == meta['sha'][q] for i, q in enumerate(sel))}/{n}, "
            f"top 5 {sum(list(t['top5'][i]) == meta['top5'][q] for i, q in enumerate(sel))}/{n}, race winner {sum(int(t['win'][i]) == meta['win'][q] for i, q in enumerate(sel))}/{n}")
    else:
        log(f"    race winner = the next greedy word {sum(int(t['win'][i]) == nx for i, nx in enumerate(WR['next']))}/{n}; top 5 = words_ref's "
            f"{sum(list(t['top5'][i]) == WR['top5'][i] for i in range(n))}/{n}; score rows = words_ref's {sum(tj['sha'][i] == WR['sha'][i] for i in range(n))}/{n}")
        if refw: log(f"    vs this folder's fresh qwen_base2 read: score rows {sum(tj['sha'][i] == refw['sha'][q] for i, q in enumerate(sel))}/{n}, race winner {sum(int(t['win'][i]) == refw['win'][q] for i, q in enumerate(sel))}/{n}")

# ---------------- WORK ----------------
SL = {"qkv": ["q", "k", "v"], "o": ["o"], "gu": ["gate", "up"], "down": ["down"]}
def work(C, T, shuf=False):
    """totals over the positions (whole numbers) by part"""
    r = {}
    for s, pls in SL.items():
        r[s] = int(sum(C[f"{s}.{l}|group" + ("_shuf" if shuf else "")].sum() for l in range(L)))
        r[s + "_plain"] = int(sum(C[f"{s}.{l}|{p}|plain"].sum() for l in range(L) for p in pls))
        for it in ("light", "table", "pwrite", "xsum"): r[f"{s}_{it}"] = int(sum(C[f"{s}.{l}|{it}"].sum() for l in range(L)))
        for p in pls:
            for it in ("reads", "sdreads", "xoff", "plain", "signed"): r[f"{p}_{it}"] = int(sum(C[f"{s}.{l}|{p}|{it}"].sum() for l in range(L)))
    r["plates"] = sum(r[s] for s in SL); r["plates_plain"] = sum(r[s + "_plain"] for s in SL)
    r["arr"] = int(sum(C[f"arr.{l}|{it}"].sum() for l in range(L) for it in ("sd_scores", "sd_values", "write_q", "write_k", "write_v")))
    r["arr_plain"] = int(sum(C[f"arr.{l}|{it}"].sum() for l in range(L) for it in ("plain_scores", "plain_values")))
    if T is not None:
        for it in ("table", "pwrite", "lwrite", "ii_light", "ii_reads", "ii_bound", "ii_end", "plain_meet", "plain_div"): r[f"top_{it}"] = int(T[it].sum())
        r["top_ii"] = r["top_table"] + r["top_ii_light"] + r["top_ii_reads"] + r["top_ii_bound"] + r["top_ii_end"]
        r["top"] = r["top_ii"] + r["top_pwrite"] + r["top_lwrite"]; r["top_plain"] = r["top_plain_meet"] + r["top_plain_div"]
        r["whole"] = r["plates"] + r["top"] + r["arr"]; r["plain"] = r["plates_plain"] + r["top_plain"] + r["arr_plain"]
    return r

# the delivered machine's own records (results), to compare number for number
DC = counts(f"{DM}/qwen_machine_counts.npz"); DT = dict(np.load(f"{DM}/qwen_top_machine.npz"))
d = {}
for s, pls in SL.items():
    d[s] = int(sum(DC[f"{s}.{l}|ctl_real"].sum() for l in range(L))); d[s + "_shuf"] = int(sum(DC[f"{s}.{l}|ctl_shuffled"].sum() for l in range(L)))
    d[s + "_plain"] = int(sum(DC[f"{p}.{l}|plain"].sum() for l in range(L) for p in pls))
d["plates"] = sum(d[s] for s in SL); d["plates_plain"] = sum(d[s + "_plain"] for s in SL)
d["arr"] = int(sum(DC[f"arr.{l}|{it}"].sum() for l in range(L) for it in ("sd_scores", "sd_values", "write_q", "write_k", "write_v")))
d["arr_plain"] = int(sum(DC[f"arr.{l}|{it}"].sum() for l in range(L) for it in ("plain_scores", "plain_values")))
for it in ("table", "pwrite", "lwrite", "ii_light", "ii_reads", "ii_bound", "ii_end", "plain_meet", "plain_div"): d[f"top_{it}"] = int(DT[it].sum())
d["top"] = int(DT["ii_total"].sum() + DT["pwrite"].sum() + DT["lwrite"].sum()); d["top_plain"] = d["top_plain_meet"] + d["top_plain_div"]
d["whole"] = d["plates"] + d["top"] + d["arr"]; d["plain"] = d["plates_plain"] + d["top_plain"] + d["arr_plain"]

if os.path.exists(f"{O}/m_refs_s1.npz") and "refs_s1" in tops:
    C1 = counts(f"{O}/m_refs_s1.npz"); T1 = tops["refs_s1"][0]; w1 = work(C1, T1)
    log(""); log("WORK, a token (totals over the 613 positions / 613); the checker's counter against the delivered machine's record")
    log(f"  {'part':34s} {'plain':>16s} {'checker':>16s} {'delivered':>16s} {'difference (total)':>20s} {'share of plain':>15s}")
    for name, k, kp in [("q+k+v", "qkv", "qkv_plain"), ("o", "o", "o_plain"), ("gate+up", "gu", "gu_plain"), ("down", "down", "down_plain"),
                        ("THE 196 PLATES", "plates", "plates_plain"), ("THE TOP READ", "top", "top_plain"), ("THE ARRIVAL", "arr", "arr_plain"),
                        ("THE WHOLE MACHINE", "whole", "plain")]:
        log(f"  {name:34s} {tok(w1[kp]):>16s} {tok(w1[k]):>16s} {tok(d[k]):>16s} {w1[k] - d[k]:>20,d} {pc(w1[k], w1[kp]):>15s}")
    log(f"  plain basis: plates {tok(w1['plates_plain'])} (delivered {tok(d['plates_plain'])}), read {tok(w1['top_plain'])} ({tok(d['top_plain'])}), "
        f"arrival {tok(w1['arr_plain'])} ({tok(d['arr_plain'])}); total {tok(w1['plain'])} ({tok(d['plain'])}); totals equal: {w1['plain'] == d['plain']}")
    log(f"  whole as a share of plain: checker {pc(w1['whole'], w1['plain'])}, delivered {pc(d['whole'], d['plain'])}")
    log("  the top read by item (checker / delivered, a token):")
    for it in ("table", "pwrite", "lwrite", "ii_light", "ii_reads", "ii_bound", "ii_end"):
        same = (T1[it] == DT[it]).all()
        log(f"    {it:10s} {tok(w1['top_' + it]):>14s} {tok(d['top_' + it]):>14s}   per position equal: {same}" + ("" if same else f" (positions differing {(T1[it] != DT[it]).sum()}, total difference {w1['top_' + it] - d['top_' + it]:,})"))
    log(f"    survivors after every place equal at every position: {(T1['surv'] == DT['ii_surv']).all()}; words read to the end (median) {int(np.median(T1['n_end']))}")
    log(f"    survivors, median after places 15..1: {[int(np.median(T1['surv'][:, b])) for b in range(15, 0, -1)]}")
    # every slab item per position
    nd, tot = 0, 0; first = []
    for s, pls in SL.items():
        for l in range(L):
            pairs = [(f"{s}.{l}|{it}", f"{s}.{l}|g_{it}") for it in ("light", "table", "pwrite", "xsum")] + [(f"{s}.{l}|group", f"{s}.{l}|ctl_real")]
            for p in pls: pairs += [(f"{s}.{l}|{p}|{it}", f"{p}.{l}|g_{it}") for it in ("reads", "sdreads")] + [(f"{s}.{l}|{p}|{it}", f"{p}.{l}|{it}") for it in ("plain", "signed")]
            xo = sum(C1[f"{s}.{l}|{p}|xoff"] for p in pls); tot += 1; nd += int((xo != DC[f"{s}.{l}|g_xoff"]).any())
            for a, b in pairs:
                tot += 1
                if not np.array_equal(C1[a], DC[b]): nd += 1; first.append(a)
    for l in range(L):
        for it in ("plain_scores", "sd_scores", "write_q", "write_k", "plain_values", "sd_values", "write_v"):
            tot += 1
            if not np.array_equal(C1[f"arr.{l}|{it}"], DC[f"arr.{l}|{it}"]): nd += 1; first.append(f"arr.{l}|{it}")
    log(f"  count arrays (613 positions each) equal to the delivered record: {tot - nd} of {tot}" + (f"; differing: {first[:10]}" if nd else ""))
    log(f"  the plates by item (a token): light {tok(sum(w1[s + '_light'] for s in SL))}; table {tok(sum(w1[s + '_table'] for s in SL))}; "
        f"pwrite {tok(sum(w1[s + '_pwrite'] for s in SL))}; xsum {tok(sum(w1[s + '_xsum'] for s in SL))}; "
        f"xoff {tok(sum(w1[p + '_xoff'] for pls in SL.values() for p in pls))}; reads {tok(sum(w1[p + '_reads'] for pls in SL.values() for p in pls))}; "
        f"signed-digit reads {tok(sum(w1[p + '_sdreads'] for pls in SL.values() for p in pls))}")
    pw = sum(w1[s + "_pwrite"] for s in SL) + w1["top_pwrite"]
    log(f"  without the ready piles written: {tok(w1['whole'] - pw)} = {pc(w1['whole'] - pw, w1['plain'])}")
    if "shuf" in "".join(C1.keys()):
        sh = work(C1, T1, shuf=True)
        log(f"  SHUFFLE CONTROL (each slab's light shuffled across its lines, the checker's draw, count only): plates {tok(sh['plates'])} against "
            f"{tok(w1['plates'])}: {(sh['plates'] - w1['plates']) / w1['plain'] * 100:+.4f} points of plain; delivered's draw "
            f"{(sum(d[s + '_shuf'] for s in SL) - d['plates']) / d['plain'] * 100:+.4f}")
        for s in SL: log(f"    {s:5s} {(sh[s] - w1[s]) / w1[s] * 100:+.3f}% of its group work")
    if os.path.exists(f"{O}/m_refs_s7.npz") and "refs_s7" in tops:
        C7 = counts(f"{O}/m_refs_s7.npz"); T7 = tops["refs_s7"][0]; w7 = work(C7, T7)
        log(""); log("ANOTHER GROUPING (seed 7, the checker's own draw), a token")
        for name, k, kp in [("q+k+v", "qkv", "qkv_plain"), ("o", "o", "o_plain"), ("gate+up", "gu", "gu_plain"), ("down", "down", "down_plain"),
                            ("THE 196 PLATES", "plates", "plates_plain"), ("THE TOP READ", "top", "top_plain"), ("THE ARRIVAL", "arr", "arr_plain"),
                            ("THE WHOLE MACHINE", "whole", "plain")]:
            log(f"  {name:20s} seed 7 {tok(w7[k]):>16s} ({pc(w7[k], w7[kp])})   seed 1 {tok(w1[k]):>16s} ({pc(w1[k], w1[kp])})   "
                f"{(w7[k] - w1[k]) / w1[kp] * 100:+.4f} points")
        log(f"  top read: survivors identical at every place and position: {(T7['surv'] == T1['surv']).all()}; reads {tok(w7['top_ii_reads'])} against {tok(w1['top_ii_reads'])}; "
            f"table {tok(w7['top_table'])} against {tok(w1['top_table'])}")
    if os.path.exists(f"{O}/m_orderfree_s1_o4242.npz"):
        CO = counts(f"{O}/m_orderfree_s1_o4242.npz"); diff = []
        for k, v in CO.items():
            if k.endswith(("group_shuf",)) or k not in C1: continue
            if not np.array_equal(v, C1[k][:len(v)]): diff.append(k)
        items = sorted(set(k.split("|")[-1] for k in diff))
        log(""); log(f"ORDER-FREE (texts 0-2, 63 positions): count arrays differing from the main run {len(diff)} of {len(CO)}; items {items}")
        if diff:
            a = sum(int(CO[k].sum()) for k in diff if k.endswith("|table")); b = sum(int(C1[k][:63].sum()) for k in diff if k.endswith("|table"))
            log(f"    the half table's walk {a:,} against {b:,} ({(a - b) / (w1['plain'] / NP * 63) * 100:+.5f} points of plain)")
        if "orderfree_s1_o4242" in tops:
            TO = tops["orderfree_s1_o4242"][0]
            log(f"    the top read order-free (groups and lines reordered, words relabelled): top 5 = the main run's {int((TO['top5'] == T1['top5'][:63]).all(1).sum())}/63, "
                f"race winner {int((TO['win'] == T1['win'][:63]).sum())}/63; reads identical {(TO['ii_reads'] == T1['ii_reads'][:63]).all()}, survivors identical {(TO['surv'] == T1['surv'][:63]).all()}, "
                f"table {int(TO['table'].sum()):,} against {int(T1['table'][:63].sum()):,}")
    rules = runs["refs_s1"]["rules"]; agg = {}
    for name, r in rules.items():
        for p, sp in r.items(): agg.setdefault(p, {}); agg[p][str(sp)] = agg[p].get(str(sp), 0) + 1
    log(""); log("THE PER-PLACE RULE (signed places by plate, layers):")
    for p, v in agg.items(): log(f"    {p:5s} " + "; ".join(f"{k} at {n} layers" for k, n in v.items()))
