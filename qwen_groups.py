"""QWEN_GROUPS: THE WHOLE QWEN3-0.6B MACHINE WITH GROUP PILES, END TO END (qwen_fringes2.py's machine, 34.16% of plain,
with every slab read from random groups' ready piles and the top read by group piles from the top place down).
Whole numbers only; every pile identical to qwen_base2.py's own run. The session's step, labelled as such.

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  A token's light reaches a plate. Take a RANDOM group of g light lines. Every subset's pile is built once per token,
  and each neuron at each plate place adds one ready pile. In the up/hole form every place of a weight is a photon up
  or a hole, so by the flip (Babinet) a pattern's complement is exactly its opposite pile and half the table is free.
  Where a place of the plate is nearly dark, its lit digits meet their light directly instead (the per-place form).

WHAT IS KEPT FROM qwen_fringes2.py (its own code, imported, unchanged): qwen_base2.py's own steps (its source executed
there, only the products hooked): every water level, turn, bend, stream add, the gauge's whole-number constant, and the
arrival from int8 signed-digit planes (int64 sums, counted lit digit x lit digit plus the writing). WHAT CHANGES: the
slabs' piles (way 2's line-ups are not used) and the top read (the band read is not used).

THE RULES (fixed here, before any run; nothing in them is read from results)
  1. EVERY SLAB in the per-place group form of the confirm step (groups2/uh_perplace.py through groups2/gbank.py):
     g = 12 (the cheaper of the two confirmed sizes at every plate kind), the confirmed runs' random groups (seed 1,
     key [1, slab, layer, 12]); q, k and v meet one light and share one table a token (one grouping), gate and up
     likewise; o and down one plate a slab. Each place of each PLATE (q, k, v, o, gate, up, down) up/hole or signed
     digits by that plate alone (L_b < G x M: signed digits).
  2. THE TOP READ: groups2/topg.py's (ii) in the per-place form at g = 16 (the cheapest exact top read of the top
     step: 5.11% of the plain read on the 613 positions), on this machine's own light at the top, with the light
     written in signed digits and the ready piles written in signed digits added (topg.py leaves both out).
  3. THE READY PILES WRITTEN IN SIGNED DIGITS (left out of the confirm and top steps' counts): every read is costed at
     the ready pile's signed digits, so every half-table entry the plate reads is first written in signed digits: its
     1s, once a position (as the light's writing is costed). Counted at every slab and at the top, reported as its own
     line so the totals can be set beside the confirmed ones.
  There is no knife and no stored layer in qwen_fringes2.py, so none here.

THE WORK (the basis of qwen_fringes2.py and Phase VI-a: 16,397,464,589 a token on the 613 positions = the plates' plain
meetings 11,845,741,380 + the plain read 4,490,888,460 + the arrival's plain meetings 60,834,749, rebuilt in the same
run; the unchanged parts (water levels, turns, bends, divides of the plates and the arrival, stream adds) are counted on
neither side, as there): at every slab the group items of groups2/gbank.py (the light's writing, the half table, the
ready piles' writing, the up/hole reads, the signed-digit reads, X and its take-away); the top read's (ii) total with
its two added writings; the arrival as qwen_fringes2 counts it. Beside it (count only): each plate in signed digits on
the same light, the shuffle control, and qwen_fringes2.py's own totals by part.

CONTROLS AND CHECKS
  exact: every slab's pile at every position against the exact product (kern.c mm16, int64 sums); every pile's,
    arrival output's, stream's and the top light's sha256 against qwen_fringes/ref.json (qwen_base2.py's own steps)
    and the piles against capture's record (qwen06b/light/meta.json); the light at every slab against capture's
    record; the arrival's products against the int products; the top read: topg.py's checks, score rows' sha256
    against qwen_base2's own read and capture's record (613/613), top 5 and race winner.
  the 16 greedy words: the machine on qwen_fringes/words_ref.json's 30 greedy texts, every array against that file's
    digests, and the race winner at the 16 read places of every text = the next greedy word (480/480).
  shuffle control (the 613 positions, every slab, count only): each position's light shuffled across its lines, every
    value kept; the group work against the real light's; the shuffled piles against their own exact product. At the
    top: qwen_fringes.py's control draw (every 4th position).
  order-free (texts 0-2): the groups in a random order and each group's lines in a random order at every slab, the
    place parts summed in a random order, the arrival's lines in random orders (qwen_fringes2's); the top read with
    topg.py's order run (groups and lines reordered, words relabelled). Every pile, stream and top 5 identical.
RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the work of the finished machine, every pile exact and the words checked; not the light's 1s.
  13.2 the base is qwen_base2.py's own machine (qwen_fringes2.py's exact form); only how piles are formed changes.
  13.3 no agreement statistic: the shuffle measures work.
  13.4 fixed by the definitions: the up/hole read count, the per-place split (plate alone), the table's size. The run
       measures the ready piles' digits and the top read's set-aside on the real light, and exactness.
  13.5 the picture and the rules are the session's; the task's prediction is labelled.
  13.6 every slab, every position, every text; the top read carried to place 0 at every position.

ANALYSIS BEFORE THE RUN (the session's; from the confirm and top steps' records and qwen_fringes2's report)
  - The slabs in the per-place form at g = 12 (table and X included): 22.11% of the plates' plain meetings
    (q 20.23, k 20.24, v 20.17, o 27.43, gate 19.81, up 19.80, down 29.00; groups2/out/compare.log): about 2,619M a
    token. The ready piles' writing: on GPT-2's plates each read half-table entry carries about 1.5-1.6 x the table's
    photons (a ready pile of 12 lines has about 1.5 x as many 1s as signed digits); Qwen's table is 1.09% of the
    plates' plain, so the writing about 1.6-1.8%, about 200M. The light's writing about 1M.
  - The top read: topg (ii) per place, 229.3M a token (5.11% of the plain read), plus its ready piles written (64
    groups x 32,768 entries, about 1.6 x its table of 8.1M: about 13M) and the light's writing (about 2 thousand):
    about 242M.
  - The arrival unchanged: 34.2M.
PREDICTIONS (written before any run of this file)
  THE TASK'S: Qwen3-0.6B about 20-26% of plain.
  THE SESSION'S:
  - Exact: every pile (3,360 digests), arrival output (840), stream (1,680) and top light (30) = qwen_fringes/ref.json and
    capture's record; every slab pile = the exact product; the top read's top 5 and race winner 613/613 against the plain
    read, the reference and capture's record, score rows 613/613; greedy words: every array equal and the race winner =
    the next greedy word 480/480; every check 0.
  - By part, a token: the plates about 2,815M (23.8% of their plain; 22.1% without the ready piles' writing), the top
    read about 242M (5.4% of the plain read), the arrival 34.2M.
  - WHOLE: about 3,090M a token, about 18.9% of plain (18-20%), against qwen_fringes2's 34.16%; without the ready
    piles' writing about 17.6%. So the task's 20-26% is expected to be missed low.
  - Shuffle control: the slab group work within 0.1 point of the real light's (the confirm step: -0.01 to -0.03).
  - Order-free: every pile, stream and top 5 identical; the reads and the ready piles' writing identical, the half table
    within 0.01 point.

RESULTS (2026-10-07; groups2/machine/qwen_report.log; a token on the 613 positions; plain 16,397,464,589 rebuilt)
  - EXACT: piles 3,360/3,360, arrival outputs 840, streams 1,680, top light 30 = the reference and capture's record;
    every slab pile = the exact product; top read 613/613 (score rows, top 5, race winner, against the plain read,
    qwen_base2's own read and capture's); greedy texts: every array equal, race winner = the next greedy word 480/480;
    order-free 63/63; every check 0.
  - WHOLE: 3,105,280,101 = 18.94% of plain (qwen_fringes2 34.16%: 55.43% of it). Without the ready piles' writing
    (208.5M plates + 12.8M top): 2,883,928,487 = 17.59%. The task's 20-26%: missed low. The session's 18-20%: held.
  - By part: the plates 2,828.9M (23.88% of their plain; qkv 21.07%, o 32.25%, gate+up 20.36%, down 34.05%), the top
    read 242.2M (5.39% of the plain read), the arrival 34.2M.
  - Controls: shuffle +0.013 points (+0.07%); top read on the shuffled light 5.74% against 5.40% at the same positions.
    Order-free: only the half table's counts differ (+0.0004 point); the top read's reads and survivors identical.
  - NOT YET: an independent rebuild by a separate checker (rule 10).

    python3 qwen_groups.py machine [shuffle=0|1]     the machine on the 30 refs texts (613 positions), every slab counted
    python3 qwen_groups.py orderfree                 texts 0-2, every order random
    python3 qwen_groups.py words                     the machine on the 30 greedy texts
    python3 qwen_groups.py top set=machine|orderfree|words [light=shuffled]
    python3 qwen_groups.py report
    -> groups2/machine/qwen_*.log, .json, .npz
"""
import os, sys, json, time, glob, hashlib, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; G2 = f"{HERE}/groups2"; OUTD = f"{G2}/machine"
sys.path.insert(0, HERE); sys.path.insert(0, G2)
_argv = sys.argv; sys.argv = [sys.argv[0]]
import qwen_fringes2 as QF
import gbank as GB
from gbank import ones, naf, NPL
sys.argv = _argv
G_BANK, G_TOP = 12, 16
SLABS = GB.R.SLABS; QSEG = QF.SLABS
ST = QF.ST; cadd = QF.cadd; chk = QF.chk; rss = QF.rss
def pct(a, b): return f"{100 * a / b:.2f}%" if b else "-"
GITEMS = ("light", "table", "pwrite", "xsum", "xoff")
CTL = dict(on=False, n=0)
RULES = {}

def gproduct(x16, ex, name):
    """THE MACHINE at every slab: the per-place group form (gbank), checked against the exact product, counted"""
    x16 = np.asarray(x16, np.int64); assert np.abs(x16).max(initial=0) <= 1 << 15
    slab, l = name.split("."); l = int(l)
    if ST["light_cmp"]:
        if slab not in QF.CAPL: QF.CAPL[slab] = np.load(f"{QF.CAP}/light/{slab}_light.npy", mmap_mode="r")
        chk("light at the slabs vs capture's record (rows differing)", (np.asarray(QF.CAPL[slab][l][:len(x16)]).astype(np.int64) != x16).any(1).sum())
    W16 = np.asarray(QF.Mz[f"{slab}.{l}.w"]); assert W16.dtype == np.int16
    order = ST["order"]
    reo = None if order is None else [order, 4242, SLABS.index(slab), l]
    gp = GB.GPlate(W16.astype(np.int64), QSEG[slab], G_BANK, [1, SLABS.index(slab), l, G_BANK], reorder=reo)
    for nm in (gp.chosen or {}): RULES[f"{nm}.{l}"] = gp.chosen[nm]["signed_places"]
    po = None if order is None else np.random.default_rng([order, 77, SLABS.index(slab), l]).permutation(NPL)
    pile, c = gp.read(x16, place_order=po)
    chk("slab: group piles vs the exact product (rows differing)", (pile != QF.exact_product(x16, W16)).any(1).sum())
    P = len(x16); rows = slice(ST["row0"], ST["row0"] + P); ox, nx = ones(x16), naf(x16)
    for k in GITEMS: cadd(f"{slab}.{l}|g_{k}", rows, c[k])
    for kind, a, b in QSEG[slab]:
        si = [s[0] for s in QSEG[slab]].index(kind)
        cadd(f"{kind}.{l}|g_reads", rows, c[f"reads|{kind}"]); cadd(f"{kind}.{l}|g_sdreads", rows, c[f"sdreads|{kind}"])
        cadd(f"{kind}.{l}|plain", rows, ox @ gp.seg_wrow[si]); cadd(f"{kind}.{l}|signed", rows, nx @ gp.seg_nrow[si])
    if CTL["on"]:
        CTL["n"] += 1; xs = np.random.default_rng([31337, SLABS.index(slab), l, CTL["n"]]).permuted(x16, axis=1)
        ps, cs = gp.read(xs)
        chk("shuffle control: piles = the shuffled light's exact product (rows differing)", (ps != QF.exact_product(xs, W16)).any(1).sum())
        cadd(f"{slab}.{l}|ctl_shuffled", rows, sum(cs.values())); cadd(f"{slab}.{l}|ctl_real", rows, sum(c.values()))
    del gp
    for t, (a, b) in enumerate(ST["rows"]): ST["dig"][f"pile|{name}|{t}"] = QF.digest(pile[a:b])
    return pile
QF.B["PRODUCT"] = gproduct                                       # the hook qwen_fringes2 gave qwen_base2's source

def opener(tag):
    os.makedirs(OUTD, exist_ok=True); f = open(f"{OUTD}/{tag}.log", "w")
    def log(s=""): print(s, flush=True); f.write(s + "\n"); f.flush()
    return log

def phase_machine(ARG, order=None, tag="qwen_machine", ntext=None):
    CTL["on"] = bool(int(ARG.get("shuffle", "1" if order is None else "0")))
    log = opener(tag); t0 = time.time(); M_ = QF.META(); texts = M_["texts"][:ntext]; RF = json.load(open(f"{QF.OUT1}/ref.json"))
    log(f"QWEN_GROUPS ({tag}): every slab in the per-place group form (g = {G_BANK}), the arrival from int8 signed-digit planes; "
        f"{len(texts)} texts, {sum(map(len, texts))} positions; order seed {order}; shuffle control {CTL['on']}")
    res = QF.forward(texts, order=order, light_cmp=True, log=log)
    log("  against the reference run of qwen_base2.py's own steps (qwen_fringes/ref.json): " + QF.diff_digests(res["dig"], RF["dig"]))
    pd = sum(res["dig"][f"pile|{n}|{t}"] != M_["dig"][n][t] for n in M_["dig"] for t in range(len(texts)))
    log(f"  piles against capture's record of qwen_base2.py's own run: {pd} of {len(M_['dig']) * len(texts)} differ")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    QF.save_counts(f"{OUTD}/{tag}_counts.npz", res["cnt"])
    json.dump(dict(chk=ST["chk"], dig=res["dig"], rules=RULES), open(f"{OUTD}/{tag}.json", "w"))
    np.savez(f"{OUTD}/{tag}_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    C = res["cnt"]; NP = sum(map(len, texts))
    pl = sum(int(v.sum()) for k, v in C.items() if k.endswith("|plain"))
    gw = sum(int(v.sum()) for k, v in C.items() if "|g_" in k)
    log(f"  the plates: plain {pl / NP:,.0f} a token; the group form {gw / NP:,.0f} ({pct(gw, pl)})")
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_words(ARG):
    CTL["on"] = False; log = opener("qwen_words"); t0 = time.time(); W = json.load(open(f"{QF.OUT1}/words_ref.json"))
    log(f"QWEN_GROUPS on the 30 greedy texts of qwen_fringes/words_ref.json: {sum(map(len, W['texts']))} positions")
    res = QF.forward(W["texts"], log=log)
    log("  against the reference path on the same texts: " + QF.diff_digests(res["dig"], W["dig"]))
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    QF.save_counts(f"{OUTD}/qwen_words_counts.npz", res["cnt"])
    np.savez(f"{OUTD}/qwen_words_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_top(ARG):
    SET = ARG.get("set", "machine"); LIGHT = ARG.get("light", "real"); ORDER = SET == "orderfree"
    tag = f"qwen_top_{SET}" + ("" if LIGHT == "real" else f"_{LIGHT}"); log = opener(tag); t0 = time.time()
    src = dict(machine="qwen_machine", orderfree="qwen_orderfree", words="qwen_words")[SET]
    Z = np.load(f"{OUTD}/{src}_top_light.npz"); X, EX, LS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"]
    sel = np.arange(len(X))
    if SET == "words":
        W = json.load(open(f"{QF.OUT1}/words_ref.json")); sel = np.array(W["read_rows"]); ref = dict(top5=np.array(W["top5"]), win=np.array(W["win"]), sha=W["sha"], nxt=np.array(W["next"]))
    else:
        RF = json.load(open(f"{QF.OUT1}/ref.json"))["read"]; MT = QF.META(); n_ = len(X)
        ref = dict(top5=np.array(RF["top5"][:n_]), win=np.array(RF["win"][:n_]), sha=RF["sha"][:n_], top5c=np.array(MT["top5"][:n_]),
                   winc=np.array(MT["win"][:n_]), shac=MT["sha"][:n_])
    if LIGHT == "shuffled":
        sel = np.arange(0, len(X), 4); X = np.random.default_rng([2026, 7]).permuted(X[sel], axis=1); EX, LS = EX[sel], LS[sel]; ref = None
    else: X, EX, LS = X[sel], EX[sel], LS[sel]
    import gtop
    TG = gtop.top_module("qwen", G_TOP, ORDER)
    log(f"THE TOP READ: topg (ii), per place, g = {G_TOP}, on qwen_groups' light: set {SET}, light {LIGHT}, order {ORDER}, {len(X)} positions")
    rec, bad, pwrite, lwrite, ppmask, lab = gtop.run_top(TG, X, EX, LS, ORDER, log, batch=8)
    r = rec["pp"]; n = len(X); pl = rec["plain_meet"] + rec["plain_div"]; tot = r["ii_total"] + pwrite + lwrite
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(bad["pp"].items())))
    log(f"  (ii) top 5 = the plain read's {int((r['ii_top5'] == rec['ptop5']).all(1).sum())}/{n}, race winner {int((r['ii_win'] == rec['pwin']).sum())}/{n}")
    if ref is not None:
        msg = (f"  against qwen_base2's own read: score rows (sha256) {sum(a == b for a, b in zip(rec['sha'], ref['sha']))}/{n}, "
               f"top 5 {int((r['ii_top5'] == ref['top5']).all(1).sum())}/{n}, race winner {int((r['ii_win'] == ref['win']).sum())}/{n}")
        if "shac" in ref: msg += (f"; capture's record: score rows {sum(a == b for a, b in zip(rec['sha'], ref['shac']))}/{n}, "
                                  f"top 5 {int((r['ii_top5'] == ref['top5c']).all(1).sum())}/{n}, race winner {int((r['ii_win'] == ref['winc']).sum())}/{n}")
        if "nxt" in ref: msg += f"; race winner = the next greedy word {int((r['ii_win'] == ref['nxt']).sum())}/{n}"
        log(msg)
    log(f"  plain read {pl.sum() / n:,.0f} a position; (ii) {r['ii_total'].sum() / n:,.0f} ({pct(r['ii_total'].sum(), pl.sum())}); "
        f"+ ready piles written {pwrite.sum() / n:,.0f} + light written {lwrite.sum() / n:,.0f} = {tot.sum() / n:,.0f} ({pct(tot.sum(), pl.sum())})")
    np.savez(f"{OUTD}/{tag}.npz", sel=sel, plain=pl, ii_total=r["ii_total"], pwrite=pwrite, lwrite=lwrite, table=r["table"],
             ii_reads=r["ii_reads"], ii_light=r["ii_light"], ii_bound=r["ii_bound"], ii_end=r["ii_end"], ii_cmp=r["ii_cmp"],
             ii_surv=r["ii_surv"], ii_top5=r["ii_top5"], ii_win=r["ii_win"], ptop5=rec["ptop5"], pwin=rec["pwin"],
             plain_meet=rec["plain_meet"], plain_div=rec["plain_div"])
    json.dump(dict(bad=bad, n=int(n), ppmask=int(ppmask), sha=rec["sha"], digest=r["digest"]), open(f"{OUTD}/{tag}.json", "w"))
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

QF2 = dict(total=5602023037)                                     # qwen_fringes2/report.log, a token
def phase_report(ARG):
    log = opener("qwen_report"); M_ = QF.META(); NP = M_["npos"]; L = QF.L
    C = QF.load_counts(f"{OUTD}/qwen_machine_counts.npz"); T = np.load(f"{OUTD}/qwen_top_machine.npz")
    C2 = QF.load_counts(f"{QF.OUT}/machine_counts.npz"); T2 = np.load(f"{QF.OUT}/top_machine.npz")
    KN = ["q", "k", "v", "o", "gate", "up", "down"]; SN = ["qkv", "o", "gu", "down"]; KS = dict(qkv=["q", "k", "v"], o=["o"], gu=["gate", "up"], down=["down"])
    def S(key, src=None): src = C if src is None else src; return int(src[key].sum()) if key in src else 0
    def A(it, src=None): return sum(S(f"arr.{l}|{it}", src) for l in range(L))
    pt = lambda v: v / NP
    log("QWEN_GROUPS: THE WHOLE QWEN3-0.6B MACHINE WITH GROUP PILES (every slab in the per-place group form, g = 12; the top read")
    log(f"by topg (ii) per place, g = 16). 30 refs texts in Qwen's words, {NP} positions; photons a token.")
    log("")
    log("EXACT")
    for f in ("qwen_machine", "qwen_top_machine", "qwen_orderfree", "qwen_top_orderfree", "qwen_words", "qwen_top_words"):
        if os.path.exists(f"{OUTD}/{f}.log"):
            for line in open(f"{OUTD}/{f}.log"):
                if any(w in line for w in ("against", "checks:", "top 5 =")): log(f"  [{f}] {line.strip()}")
    log("")
    # ---- the plain basis and the work by part ----
    plates_plain = sum(S(f"{k}.{l}|plain") for k in KN for l in range(L)); top_plain = int(T["plain"].sum())
    arr_plain = A("plain_scores") + A("plain_values"); P0 = plates_plain + top_plain + arr_plain
    log(f"THE PLAIN BASIS rebuilt: plates {pt(plates_plain):,.0f}, read {pt(top_plain):,.0f}, arrival {pt(arr_plain):,.0f}; total {pt(P0):,.0f} "
        f"(Phase VI-a: {QF.PLAIN_PLATES + QF.PLAIN_TOP + QF.PLAIN_ARR:,}; same: {round(pt(P0)) == QF.PLAIN_PLATES + QF.PLAIN_TOP + QF.PLAIN_ARR})")
    GI = ("g_light", "g_table", "g_pwrite", "g_xsum", "g_xoff")
    def slab_group(s_, items=GI): return sum(S(f"{s_}.{l}|{it}") for l in range(L) for it in items)
    def kind_reads(k): return sum(S(f"{k}.{l}|g_reads") + S(f"{k}.{l}|g_sdreads") for l in range(L))
    def kind_plain(k, src=None): return sum(S(f"{k}.{l}|plain", src) for l in range(L))
    log("")
    log("WORK A TOKEN BY PART")
    log(f"  {'part':48s} {'plain':>15s} {'qwen_fringes2':>15s} {'qwen_groups':>15s} {'of plain':>9s} {'of before':>9s}")
    def row(name, plain_, old_, new_):
        log(f"  {name:48s} {pt(plain_):15,.0f} {pt(old_):15,.0f} {pt(new_):15,.0f} {100 * new_ / plain_:8.2f}% {100 * new_ / old_ if old_ else 0:8.2f}%")
    plates_new = 0; plates_old = 0
    for s_ in SN:
        pl_ = sum(kind_plain(k) for k in KS[s_]); new_ = slab_group(s_) + sum(kind_reads(k) for k in KS[s_])
        old_ = sum(S(f"{s_}.{l}|walk", C2) + S(f"{s_}.{l}|pours", C2) + S(f"{s_}.{l}|writing", C2) for l in range(L))
        row(f"{s_} (table, writings, reads, X)", pl_, old_, new_); plates_new += new_; plates_old += old_
        for k in KS[s_]:
            log(f"      {k:6s} reads {pt(kind_reads(k)):14,.0f} ({100 * kind_reads(k) / kind_plain(k):.2f}% of its plain)")
    row("THE 196 PLATES", plates_plain, plates_old, plates_new)
    top_ii, top_pw, top_lw = int(T["ii_total"].sum()), int(T["pwrite"].sum()), int(T["lwrite"].sum()); top_new = top_ii + top_pw + top_lw
    top_old = int((T2["walk"] + T2["writing"] + T2["pours"].sum(1) + T2["bound"].sum(1) + T2["end"]).sum())
    row("THE TOP READ (topg (ii) per place + its writings)", top_plain, top_old, top_new)
    arr_new = A("sd_scores") + A("sd_values") + A("write_q") + A("write_k") + A("write_v")
    row("THE ARRIVAL (signed digits, unchanged)", arr_plain, arr_new, arr_new)
    whole = plates_new + top_new + arr_new; whole_old = plates_old + top_old + arr_new
    row("THE WHOLE MACHINE", P0, whole_old, whole)
    pw = sum(S(f"{s_}.{l}|g_pwrite") for s_ in SN for l in range(L))
    log(f"  of it, the ready piles written in signed digits: plates {pt(pw):,.0f}, top {pt(top_pw):,.0f}; without them (the confirmed counts' unit) "
        f"{pt(whole - pw - top_pw):,.0f} = {100 * (whole - pw - top_pw) / P0:.2f}% of plain")
    log(f"  the plates by item: " + "; ".join(f"{it[2:]} {pt(sum(S(f'{s_}.{l}|{it}') for s_ in SN for l in range(L))):,.0f}" for it in GI)
        + f"; reads {pt(sum(S(f'{k}.{l}|g_reads') for k in KN for l in range(L))):,.0f}; signed-digit reads {pt(sum(S(f'{k}.{l}|g_sdreads') for k in KN for l in range(L))):,.0f}")
    sdm = sum(S(f"{k}.{l}|signed") for k in KN for l in range(L))
    log(f"  beside it, every plate in signed digits on the same light (meetings): {pt(sdm):,.0f} ({100 * sdm / plates_plain:.2f}% of the plates' plain)")
    log(f"  the top read: (ii) {pt(top_ii):,.0f}; ready piles written {pt(top_pw):,.0f}; light written {pt(top_lw):,.0f}; "
        + "; ".join(f"{k_} {T[k_].sum() / NP:,.0f}" for k_ in ("table", "ii_reads", "ii_light", "ii_bound", "ii_end")))
    log("")
    log("CONTROLS")
    sh = sum(S(f"{s_}.{l}|ctl_shuffled") for s_ in SN for l in range(L)); re_ = sum(S(f"{s_}.{l}|ctl_real") for s_ in SN for l in range(L))
    if re_: log(f"  shuffle (every slab, count only): group work {pt(sh):,.0f} against {pt(re_):,.0f}: {100 * (sh - re_) / P0:+.3f} points of plain ({100 * (sh - re_) / re_:+.2f}%)")
    for s_ in SN:
        a = sum(S(f"{s_}.{l}|ctl_shuffled") for l in range(L)); b = sum(S(f"{s_}.{l}|ctl_real") for l in range(L))
        if b: log(f"    {s_:5s} {100 * (a - b) / b:+.3f}% of its group work")
    if os.path.exists(f"{OUTD}/qwen_top_machine_shuffled.npz"):
        Ts = np.load(f"{OUTD}/qwen_top_machine_shuffled.npz"); sel = Ts["sel"]
        ts = int((Ts["ii_total"] + Ts["pwrite"] + Ts["lwrite"]).sum()); tr = int((T["ii_total"] + T["pwrite"] + T["lwrite"])[sel].sum())
        log(f"  top read on the shuffled light (qwen_fringes.py's draw, every 4th position): {100 * ts / int(Ts['plain'].sum()):.2f}% of the plain read against "
            f"{100 * tr / int(T['plain'][sel].sum()):.2f}% on the real light at the same positions")
    if os.path.exists(f"{OUTD}/qwen_orderfree_counts.npz"):
        O = QF.load_counts(f"{OUTD}/qwen_orderfree_counts.npz"); n3 = len(next(iter(O.values())))
        keys = [k for k in O if k in C]; od = [k for k in keys if not np.array_equal(C[k][:n3], O[k])]
        log(f"  order-free (texts 0-2, {n3} positions): count arrays differing from the main run {len(od)} of {len(keys)}; items: {sorted({k.split('|')[1] for k in od})}")
        ta = sum(int(C[k][:n3].sum()) for k in keys if k.endswith("g_table")); tb = sum(int(O[k].sum()) for k in keys if k.endswith("g_table"))
        log(f"    the half table {ta:,} against {tb:,} ({100 * (tb - ta) / sum(int(C[k][:n3].sum()) for k in keys if k.endswith('|plain')):+.4f} points of the plates' plain)")
    if os.path.exists(f"{OUTD}/qwen_top_orderfree.npz"):
        To = np.load(f"{OUTD}/qwen_top_orderfree.npz"); n3 = len(To["ii_win"])
        log(f"    the top read order-free: top 5 = the main run's {int((To['ii_top5'] == T['ii_top5'][:n3]).all(1).sum())}/{n3}, race winner "
            f"{int((To['ii_win'] == T['ii_win'][:n3]).sum())}/{n3}; (ii) reads identical {bool(np.array_equal(To['ii_reads'], T['ii_reads'][:n3]))}, "
            f"survivors identical {bool(np.array_equal(To['ii_surv'], T['ii_surv'][:n3]))}")
    log("")
    J = json.load(open(f"{OUTD}/qwen_machine.json")); rules = J["rules"]
    for k in KN:
        sp = {}
        for l in range(L): sp.setdefault(tuple(rules[f"{k}.{l}"]), []).append(l)
        log(f"  the per-place rule, {k}: " + "; ".join(f"signed places {list(a)} at {len(b)} layers" for a, b in sp.items()))
    log("")
    log("PREDICTIONS (written in this file before it ran) AGAINST THE RESULT")
    w = 100 * whole / P0
    log(f"  THE TASK'S: 20-26% of plain: {w:.2f}% -> {'held' if 20 <= w <= 26 else ('lower' if w < 20 else 'higher')}")
    log(f"  THE SESSION'S: about 18.9% (18-20%): {w:.2f}% -> {'held' if 18 <= w <= 20 else ('lower' if w < 18 else 'higher')}")
    log(f"  the plates about 23.8% of their plain: {100 * plates_new / plates_plain:.2f}%; the top read about 5.4% of the plain read: {100 * top_new / top_plain:.2f}%")

if __name__ == "__main__":
    ph = sys.argv[1]; ARG = dict(a.split("=") for a in sys.argv[2:])
    if ph == "machine": phase_machine(ARG)
    elif ph == "orderfree": phase_machine(ARG, order=7, tag="qwen_orderfree", ntext=3)
    elif ph == "words": phase_words(ARG)
    elif ph == "top": phase_top(ARG)
    elif ph == "report": phase_report(ARG)
