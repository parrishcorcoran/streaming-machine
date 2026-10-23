"""BASE2_GROUPS: THE WHOLE GPT-2 MACHINE WITH GROUP PILES, END TO END (base2_fringes2.py's machine, 46.21% of plain,
with every bank read from random groups' ready piles and the top read by group piles from the top place down).
Whole numbers only; every pile identical to base2.py's own run. The session's step, labelled as such.

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  A token's light reaches a plate. Take a RANDOM group of g light lines. Every subset's pile is built once per token,
  and each neuron at each plate place adds one ready pile. In the up/hole form every place of a weight is a photon up
  or a hole, so by the flip (Babinet) a pattern's complement is exactly its opposite pile and half the table is free.
  Where a place of the plate is nearly dark, its lit digits meet their light directly instead (the per-place form).

WHAT IS KEPT FROM base2_fringes2.py (its own code, imported, unchanged): every bend, water level, gauge, divide, bias,
stream add and the arrival (in signed digits, its level 3); layer 0 from storage (entry/ and layer0/: qkv.0's stored
word and position rows with only the -1/0/+1 carry meeting the plate, position 0's whole stream after layer 0, proj.0's
own value, fc.0's word and position parts); the plates' whole numbers rebuilt from recording/C's sheets (the machine
never opens base2_gpt2.npz for a bank plate). WHAT CHANGES: how each bank's pile is formed (below), the knife carried
down the plate's places instead of the diagonals, and the top read.

THE RULES (fixed here, before any run; nothing in them is read from results)
  1. EVERY BANK in the per-place group form of the confirm step (groups2/uh_perplace.py, through groups2/gbank.py):
     g = 12 (the cheaper of the two confirmed sizes at every plate kind), the confirmed runs' own random groups (seed 1,
     key [1, slab, layer, 12]); each place up/hole or signed digits by the plate alone (L_b < G x M: signed digits).
  2. qkv.0's CARRY (the stored rows' -1/0/+1 remainder) meets the plate in signed digits: a ready pile of the few lit
     carry lines in a group (about 3 of 12) holds about as many signed digits as those lines' own digits, so 14 reads a
     (neuron, group) cannot be cheaper than the lit lines' digits met directly. The carry's group form is counted
     beside it (count only).
  3. LAYER 0 FROM STORAGE, KEPT ONLY IF IT STILL PAYS: at qkv.0, proj.0 and fc.0 the stored form (rows read and added,
     the split, the rest of the light in the group form) is set against the whole light in the group form on the same
     positions (count only, its piles checked). The stored form is kept at a plate only if it is lower summed over
     the 648 refs positions; the windows and the order-free run follow the same choice. Position 0's whole stream
     after layer 0 (Z0, a few thousand photons against tens of millions) is kept without a test.
  4. THE KNIFE STOPS, KEPT ONLY IF THEY STILL PAY: the knife of base2_fringes2.py (its gate, its intervals at the edge's
     own scale, settle.py's test of the bend output, the frozen edges, the water level's shift range), carried down
     the plate's PLACES from the top instead of the diagonals. After the places >= b are read the unread part of every
     pile is at most B(b) x S1 at the doubled scale, S1 = the light's |x| summed, B(b) the per-place form's bound
     (groups2/gbank.bound_per_line: 2^q for each up/hole place below b, twice the non-adjacent bound of each signed
     run below b); one shared bound a position. The gate: no edge carried to its scale before (bound) x e <=
     2^(16 + smallest shift) (base2_fringes2's, at the doubled scale). Both forms are run at every fc plate and both
     counted: WHOLE (every place read, then base2's own divide, bias and bend on the whole pile) and KNIFE (the reads
     up to each edge's stop, plus the knife's own work as base2_fringes2 counts it: the pile and each new place carried
     to the edge's scale, the bound carried, the interval ends, the bends at the interval's low end, the biases, S1 and
     its multiple a place). The knife is kept at fc.l only if its total is lower summed over the 648 refs positions.
     Exactness is checked for both at every position (the knife's settled outputs and shifts = base2's).
  5. THE TOP READ: groups2/topg.py's (ii) in the per-place form at g = 16 (the cheapest exact top read of the top
     step: 7.87% of the plain read on the refs), run on this machine's own light at the top, with two counts added that
     topg.py leaves out: the light written in signed digits (the band read counts it) and the ready piles written in
     signed digits (below).
  6. THE READY PILES WRITTEN IN SIGNED DIGITS (a count the confirm and top steps leave out): every read is costed at the
     ready pile's signed digits, so every ready pile the plate reads is first written in signed digits; that costs its
     1s, once a position, as base2_fringes2 costs the light's writing (ones) at every bank. Counted at every bank and
     at the top, and reported as its own line so the totals can be set beside the confirmed ones.

THE WORK (photon meetings and photons added, the unit of every earlier result; a token on the 648 refs positions;
plain 3,666,596,259, rebuilt in the same run): at every bank the group items of groups2/gbank.py (the light's writing,
the half table, the ready piles' writing, the up/hole reads, the signed-digit reads, X and its take-away), layer 0's
stored rows and splits, the knife's or the whole form's steps at fc, and every other part as base2_fringes2 counts it.
Compares and table reads are a count, kept apart. Beside it (count only): the signed-digit form on the same light at
every bank, the carry's group form, the whole light at layer 0, and the shuffle control.

CONTROLS AND CHECKS
  exact: every pile (qkv, proj, mproj; fc's whole pile), fc's output (out16 and its shift), the stream after every
    half layer and the light at the top against base2.py's own run (recording/C_ref/plain_refs.npz on the 30 refs
    texts; fringes_machine/plain_digests_windows.json on judge5's 300 windows); the check path: every bank's piles
    against the direct product at layer 0's whole light, the true H inside every knife interval, the knife's outputs
    = the whole form's, position 0's layer 0 by base2's own steps; the top read's own checks (topg.py's).
  judge5: in the reference top 5 and the same top word at 6,000/6,000; the refs: top 5 and race winner against
    plain_refs.
  shuffle control (refs, every bank, count only): each position's light shuffled across its lines, every value kept;
    the group work against the real light's; the shuffled piles checked against their own direct product. At the top:
    topband.py's draw (count only).
  order-free (texts 0-2): the groups in a random order and each group's lines in a random order at every bank, the
    place parts summed in a random order (fc read top down, as the knife needs), fc's neurons in a random order,
    mproj's edges met in groups by their stop place in a random order; the top read with topg.py's order run (groups
    and lines reordered, words relabelled). Every pile, stream and top 5 identical; the reads identical.
RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the work of the finished machine, every pile exact and the top 5 judged; the light's sizes
       enter only through the counted adds and the shared bound S1, a relation from the light alone.
  13.2 the base is base2.py's own machine in base2_fringes2.py's exact form; only how piles are formed changes.
  13.3 no agreement statistic: the shuffle measures work; exactness is against base2's own run.
  13.4 fixed by the definitions and so not measured: the up/hole read count (one per neuron, group and up/hole place),
       the per-place split (plate alone), the table's size. The run measures the ready piles' digits, the knife's
       stops, the storage's net, and exactness.
  13.5 the picture and every rule are the session's; the task's prediction is labelled as the task's.
  13.6 every bank, every position, every text; the knife carried to place 0 at every position.

ANALYSIS BEFORE THE RUN (the session's; from the confirm step's per-plate records, base2_fringes2's report and a look
at the light's sizes at the fc plates)
  - The banks (the per-place form at g = 12, table and X included, whole light): qkv 19.66%, proj 21.12%, fc 19.54%,
    mproj 21.18% of their plain meetings (groups2/out/compare.log). With base2_fringes2's plain parts (qkv 679.7M,
    proj 248.5M, fc 902.1M of which 6.7M divides, biases and bends, mproj 914.7M): about 133.6M, 52.5M, 175.0M + 6.7M,
    193.7M.
  - The ready piles' writing: a ready pile of 12 lines has about 1.5 x as many 1s as signed digits (about 6.5-7 1s
    against the reads' 4.5 signed digits); 2,048 entries a group, 64 groups at a 768-line plate, 256 at mproj: about
    0.9M a bank and 3.6M at mproj a layer, about 75M a token (2.0% of plain).
  - Layer 0 from storage: qkv.0's whole light in the group form would be about 12.2M; stored rows plus the carry in
    signed digits about 2.1M (base2_fringes2's counts), so storage keeps paying there (-10M). proj.0's and fc.0's rest
    light is nearly as large as the whole light (their signed-digit meetings 97% and 96% of the whole light's), so
    the group form on it is a few percent lower, a saving of 0.1-0.3M each against 15-45 thousand photons of stored
    reads: kept at all three, but by little at proj.0 and fc.0.
  - The knife: on the refs light the gate (bound x e <= 2^(16 + smallest shift)) opens after place 5 or 6 at most
    positions (S1 about 0.5-1.0M, the gate's bound 1-4 x 10^7 at pile scale). But an edge's bend output is fixed only
    when its interval is narrower than the output's last kept place: that needs the low places of every edge whose
    output is not exactly 0, and only the edges far on the dark side (below the bend's table, output exactly 0)
    settle early. If 10% of edges stop after place 5 they skip 5 of 14 up/hole places: about 3-4% of fc's reads
    (5-7M a token), while carrying every unsettled edge down 6 places costs about 120-150 photons an edge a place
    (gain x part, its add, both interval ends): about 25-30M a token beyond the whole form's divides. THE KNIFE IS NOT
    EXPECTED TO PAY at any of the 12 fc plates.
  - The top read: topg (ii) per place at g = 16, 71.2M a token on the refs (7.87% of the plain read), plus its ready
    piles written (48 groups x 32,768 entries x about 6.5 1s = about 10M) and the light's writing (3.6 thousand):
    about 81M, 9.0% of the plain read.
  - Unchanged: the arrival 8.02M, the rest (water levels, bank divides, biases, stream adds) 4.56M.

PREDICTIONS (written before any run of this file)
  THE TASK'S: GPT-2 about 22-28% of plain.
  THE SESSION'S:
  - Exact: every pile, fc output and stream identical to base2.py's own run on the 30 refs texts and the 300 windows;
    judge5 6000/6000 and 6000/6000; refs top 5 = plain_refs at 647/648 or 648/648 (position 494 is an exact tie for
    5th place, words 655 and 1654: (ii) keeps both and judge5's rule may order them either way), race winner 648/648;
    every check-path count 0.
  - The rules: storage kept at qkv.0, proj.0 and fc.0 (qkv.0 by about 10M, the other two by under 0.4M each); the
    knife kept at 0 of 12 fc plates.
  - By part, a token: qkv about 133M (19.6% of its plain), proj about 62M, fc about 192M, mproj about 228M, the ready
    piles' writing about 75M of these, arrival 8.0M, rest 4.6M, the top read about 81M.
  - WHOLE: about 709-720M a token, about 19.5% of plain (18.5-21%), against base2_fringes2's 46.21%; without the ready
    piles' writing (the confirmed counts' unit) about 17.5%. So the task's 22-28% is expected to be missed low.
  - Shuffle control: the bank group work within 0.1 point of the real light's (the confirm step: -0.01 to -0.06).
  - Order-free: every pile, stream, output and top 5 identical; the reads, the ready piles' writing and every other
    count identical but the half table (its Gray walk turns other lines), within 0.01 point.

RESULTS (2026-10-07; groups2/machine/gpt2_report.log; a token on the 648 refs positions; plain 3,666,596,259 rebuilt)
  - EXACT: 96 arrays of every refs text, every array of the 300 windows (8 parts), the light at the top 300/300 windows,
    every check-path count 0 (the knife's outputs = the whole form's, every interval held the true H, the whole light
    and the carry in the group form = the machine's piles, the shuffled piles = their direct product). judge5
    6000/6000 and 6000/6000 (top 5 = judge5's reference 6000/6000). Refs: top 5 647/648, race winner 648/648; position
    494 is the exact tie (words 655 and 1654, both -9,021,364): (ii)'s five are the five best scores in order.
  - WHOLE: 711,431,068 = 19.40% of plain (base2_fringes2 46.21%: 41.99% of it). Without the ready piles' writing
    (67.5M banks + 9.5M top): 634,392,832 = 17.30%. The task's 22-28%: missed low. The session's 18.5-21%: held.
  - By part: qkv 132.7M (19.52%), proj 62.8M (25.28%), fc 190.8M (21.15%), mproj 231.9M (25.35%), arrival 8.0M, top
    read 80.7M (8.93% of the plain read), rest 4.6M.
  - Rule 3: storage kept at qkv.0 (+10.1M a token) and fc.0 (+0.19M); NOT at proj.0 (the whole light 74.0 thousand
    lower): predicted kept at all three, wrong at proj.0. Rule 2: the carry's group form is 2,186 lower than its
    signed digits (the rule's reasoning was wrong by that much; the rule stands as fixed).
  - Rule 4: the knife kept at 0 of 12 (held): 2-10% of edges stop before the last place, saving 1.17M reads a token;
    its own carrying and intervals cost more (knife 199.2M against the whole form's 174.7M).
  - Controls: shuffle +0.010 points (+0.06% of the group work); top read on the shuffled light 8.47% against 8.93%.
    Order-free: arrays differing 0; the only counts that differ are the half table's (+0.001 point); stops identical;
    the top read 62/62.
  - NOT YET: an independent rebuild by a separate checker (rule 10).

    python3 base2_groups.py refs [order=SEED texts=0,1,2] [shuffle=0|1]  -> groups2/machine/gpt2_refs[_order].json/.log/_top_light.npz
    python3 base2_groups.py windows a=0 b=40                             -> groups2/machine/gpt2_windows_a_b.npz/.log
    python3 base2_groups.py join                                         -> groups2/machine/gpt2_windows_top_light.npz
    python3 base2_groups.py top set=refs|windows [light=shuffled] [order=SEED] [part=i/n]
    python3 base2_groups.py judge
    python3 base2_groups.py report
"""
import os, sys, json, time, glob, hashlib, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; G2 = f"{HERE}/groups2"; OUTD = f"{G2}/machine"
os.environ["FR2_ADD"] = "3"                    # base2_fringes2's own helpers, the arrival in signed digits (its level 3)
sys.path.insert(0, HERE); sys.path.insert(0, G2)
_argv = sys.argv; sys.argv = [sys.argv[0]]
import base2_fringes2 as FM
import gbank as GB
from gbank import ones, naf, ones1, sd1, NPL
sys.argv = _argv
assert FM.ADD == 3
G_BANK, G_TOP = 12, 16
SLABS = GB.R.SLABS
ES = FM.ES
def rss(): return FM.rss()
def pct(a, b): return f"{100 * a / b:.2f}%" if b else "-"

# ---------- the plates, laid once a run (plate only) ----------
def gplate(kind, l, reorder=None, up=None, neuron_perm=None):
    w16, e = FM.table_values(f"{kind}.{l}"); w = w16.astype(np.int64)
    if neuron_perm is not None: w = w[:, neuron_perm]; e = e[neuron_perm]
    gp = GB.GPlate(w, [(kind, 0, w.shape[1])], G_BANK, [1, SLABS.index(kind), l, G_BANK], reorder=reorder, up=up)
    gp.e = e; gp.name = f"{kind}.{l}"
    return gp
GITEMS = ("light", "table", "pwrite", "xsum", "xoff")
GLABEL = dict(light="light written in signed digits", table="half table (by the flip)", pwrite="ready piles written in signed digits",
              xsum="X (the light summed)", xoff="X taken from each neuron", reads="reads, up/hole places", sdreads="reads, signed-digit places")
def add_group(C, who, lay, part, cnt, reads=True):
    for k in GITEMS: C.add(f"{who}|{lay}|{part}|{GLABEL[k]}", cnt[k])
    if reads:
        C.add(f"{who}|{lay}|{part}|{GLABEL['reads']}", sum(v for k, v in cnt.items() if k.startswith("reads|")))
        C.add(f"{who}|{lay}|{part}|{GLABEL['sdreads']}", sum(v for k, v in cnt.items() if k.startswith("sdreads|")))
def gtotal(cnt): return sum(cnt.values())

CTL = dict(on=False, seed=31337, n=0)                      # the shuffle control (count only)
def shuffle_ctl(C, lay, part, gp, x, cnt_real, chk):
    if not CTL["on"] or C is None: return
    CTL["n"] += 1; rng = np.random.default_rng([CTL["seed"], SLABS.index(part), int(lay[1:]), CTL["n"]])
    xs = rng.permuted(np.asarray(x, np.int64), axis=1)
    ps, cs = gp.read(xs)
    if chk is not None: chk("shuffle control: piles = the shuffled light's direct product (rows differing)", (ps != GB.plain_product(xs, gp.w16)).any(1).sum())
    C.add(f"ctl|{lay}|{part}|group work, light shuffled across its lines", gtotal(cs))
    C.add(f"ctl|{lay}|{part}|group work, the real light", gtotal(cnt_real))

def bank(C, lay, part, gp, x, plain_x=None, chk=None, place_order=None, side_sd=True):
    """a bank's pile from the group form, counted (fr), with the plain count and the signed-digit form beside it"""
    pile, cnt = gp.read(x, place_order=place_order)
    if C is not None:
        px = x if plain_x is None else plain_x
        C.add(f"plain|{lay}|{part}|meetings, plain", ones(px) @ gp.wrow)
        add_group(C, "fr", lay, part, cnt)
        if side_sd:
            C.add(f"side|{lay}|{part}|signed digits: meetings", naf(x) @ gp.nrow); C.add(f"side|{lay}|{part}|signed digits: writing", ones(x).sum())
        shuffle_ctl(C, lay, part, gp, x, cnt, chk)
    return pile

def side_whole(C, lay, part, gp, xw, want, chk, what):
    """count only: a light in the group form beside the machine's (layer 0's whole light; the carry's group form)"""
    if C is None and chk is None: return
    p, c = gp.read(xw)
    if chk is not None and want is not None: chk(f"{what}: piles = the machine's (rows differing)", (p != want).any(1).sum())
    if C is not None: C.add(f"side|{lay}|{part}|{what}: group work", gtotal(c))

# ---------- fc: the group form read place by place from the top, the knife and the whole form both ----------
def fc_read(gp, x, c0, F, bias, C, lay, chk, plain_x, chunk=96, MZ=None):
    """fc.l: x the light that meets the plate (P, N), c0 the stored part (P, M) or None, F the gain's power, bias.
    Returns out16, sh, the knife's stop (index in the place order), the whole pile (CHECK ONLY), and the light's S1."""
    x = np.asarray(x, np.int64); P, M = len(x), gp.M
    f = F & 0xFFFF; assert (f == f[:, :1]).all()
    e = FM.EXP2[f[:, 0]].astype(np.int64); s = 16 - (F >> 16); assert s.min() >= 1
    smin = s.min(1); s1 = s + 1
    Bsh = np.left_shift(np.asarray(bias, np.int64)[None, :], s1); assert np.abs(Bsh).max() < 2 ** 60
    thr2 = np.array([(1 << (16 + int(smin[i]))) // int(e[i]) for i in range(P)], np.int64)
    order = gp.read_order; NS = len(order); BND = gp.bnd; assert BND is not None
    OUT16 = np.empty((P, M), np.int64); SH = np.empty(P, np.int64); STOP = np.empty((P, M), np.int8); PILE = np.empty((P, M), np.int64)
    ob = np.ones((1, M), np.int64) * ones(bias)[None, :]
    for r0 in range(0, P, chunk):
        r = slice(r0, min(P, r0 + chunk)); xs = x[r]; Pc = len(xs)
        A, H, wr = gp.kernel(xs); X = xs.sum(1); S1 = np.abs(xs).sum(1)
        cnt = gp.counts(xs, H, wr, X)
        c0r = np.zeros((Pc, M), np.int64) if c0 is None else 2 * np.asarray(c0[r], np.int64)
        D0 = c0r + gp.kcorr[None, :] * X[:, None]
        Dfull = D0.copy()
        for b in range(NPL): Dfull += A[:, :, b] << b
        assert not (Dfull & 1).any(); pile = Dfull >> 1; PILE[r] = pile
        Fr, er, s1r, Br, thr = F[r], e[r], s1[r], Bsh[r], thr2[r]
        Ht = FM.power(pile, Fr) + bias[None, :]
        o16w, egw = FM.to16(FM.edge(Ht), -ES); shw = egw + ES
        if C is not None:
            add_group(C, "fr", lay, "fc", cnt, reads=False)
            C.add(f"fcw|{lay}|fc|{GLABEL['reads']}", sum(v for k, v in cnt.items() if k.startswith("reads|")))
            C.add(f"fcw|{lay}|fc|{GLABEL['sdreads']}", sum(v for k, v in cnt.items() if k.startswith("sdreads|")))
            C.add(f"fcw|{lay}|fc|divides", ones(pile).sum(1) * ones(er)); C.add(f"fcw|{lay}|fc|biases", Pc * ones(bias).sum())
            C.add(f"fcw|{lay}|fc|bends", (ones(Ht) * ones(FM.share_of(Ht))).sum())
            px = np.asarray(plain_x[r], np.int64)
            C.add(f"plain|{lay}|fc|meetings, plain", ones(px) @ gp.wrow)
            C.add(f"plain|{lay}|fc|divides", ones(pile).sum(1) * ones(er)); C.add(f"plain|{lay}|fc|biases", Pc * ones(bias).sum())
            C.add(f"plain|{lay}|fc|bends", (ones(Ht) * ones(FM.share_of(Ht))).sum())
            C.add(f"side|{lay}|fc|signed digits: meetings", naf(xs) @ gp.nrow); C.add(f"side|{lay}|fc|signed digits: writing", ones(xs).sum())
            shuffle_ctl(C, lay, "fc", gp, xs, cnt, chk)
            C.add(f"fck|{lay}|fc|knife: S1, the light's |x| summed", naf(xs)[:, 1:].sum())
        # ---- the knife, carried down the places from the top ----
        Dc = D0.copy(); opened = np.zeros(Pc, bool); Y = np.zeros((Pc, M), np.int64)
        st = np.full((Pc, M), -1, np.int8); FLO = np.zeros((Pc, M), np.int64); FHI = np.zeros((Pc, M), np.int64)
        V = np.zeros((Pc, M), np.int64); HC = np.zeros((Pc, M), np.int64); Hcum = np.zeros((Pc, M), np.int64)
        for si, b in enumerate(order):
            dc = A[:, :, b] << b; Dc += dc; Hcum += H[:, :, b]
            last = si == NS - 1
            R2 = np.zeros(Pc, np.int64) if last else BND[b] * S1
            if C is not None and not last: C.add(f"fck|{lay}|fc|knife: the shared bound a place (S1 x B)", sum(ones1(v) * ones1(BND[b]) for v in S1))
            uns = st < 0
            newly = ~opened & (R2 <= thr)
            if newly.any():
                m = newly[:, None] & uns
                Y[newly] = Dc[newly] * er[newly][:, None] + Br[newly]
                if C is not None:
                    C.add(f"fck|{lay}|fc|knife: pile carried to the edge's scale (gain x pile)", (ones(Dc) * ones(er)[:, None])[m].sum())
                    C.add(f"fck|{lay}|fc|biases", np.broadcast_to(ob, (Pc, M))[m].sum())
            old = opened & ~newly
            if old.any():
                m = old[:, None] & uns
                inc = dc * er[:, None]; Y = np.where(m, Y + inc, Y)
                if C is not None: C.add(f"fck|{lay}|fc|knife: newly read place carried (gain x part, and its add)", (ones(dc) * ones(er)[:, None] + ones(inc))[m].sum())
            opened |= newly
            if not opened.any(): continue
            op = np.flatnonzero(opened)
            RE = R2[op] * er[op]; assert np.abs(RE).max(initial=0) < 2 ** 61
            Yo = Y[op]; so1 = s1r[op]
            Hlo = FM.rshift(Yo - RE[:, None], so1); Hhi = FM.rshift(Yo + RE[:, None], so1)
            uo = uns[op]
            if C is not None:
                C.add(f"fck|{lay}|fc|knife: bound carried to the edge's scale (gain x bound)", (ones(R2[op]) * ones(er[op])).sum())
                C.add(f"fck|{lay}|fc|knife: interval ends (pile -/+ bound at the edge's scale)", (2 * ones(RE)[:, None] * uo).sum())
            sto = st[op]; fr = sto >= 0
            LOc = np.where(fr, FLO[op], Hlo); HIc = np.where(fr, FHI[op], Hhi)
            slo, shi = FM.sh_range(LOc, HIc)
            if chk is not None:
                Hto = Ht[op]; chk("knife: true H inside the interval", int(((Hto < Hlo) | (Hto > Hhi))[~fr].sum()))
            idx = np.flatnonzero(~fr.ravel())
            if C is not None:
                C.add(f"reads|{lay}|fc|knife: table reads (out at both ends, change counts, sh range)",
                      len(idx) * 2 + 2 * ((shi - slo + 1)[:, None] * ~fr).sum() + 2 * fr.size)
                C.add(f"cmp|{lay}|fc|knife: compares", len(idx))
            okc, v = FM.out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo, M)[idx], np.repeat(shi, M)[idx])
            new = idx[okc]
            if len(new):
                pr, nn = op[new // M], new % M
                st[pr, nn] = si; FLO[pr, nn] = Hlo.ravel()[new]; FHI[pr, nn] = Hhi.ravel()[new]; V[pr, nn] = v[okc]
                HC[pr, nn] = Hcum[pr, nn]
                if C is not None:
                    hl = Hlo.ravel()[new]; C.add(f"fck|{lay}|fc|knife: bends (at the interval's low end)", (ones(hl) * ones(FM.share_of(hl))).sum())
        assert (st >= 0).all(), "an edge not settled at the last place"
        sto, sth = FM.sh_range(FLO, FHI); assert np.array_equal(sto, sth), "the water level's shift not fixed"
        if chk is not None:
            chk("knife: settled outputs = the whole form's (edges differing)", int((V != o16w).sum()))
            chk("knife: shift = the whole form's (positions differing)", int((sto != shw).sum()))
        if C is not None:
            C.add(f"fck|{lay}|fc|knife: reads up to each edge's stop", HC.sum())
            C.add(f"side|{lay}|fc|knife: reads saved (the places below each edge's stop)", (Hcum - HC).sum())
            if MZ is not None:
                MZ.setdefault(lay, dict(hist=np.zeros(NS, np.int64), gate=np.zeros(NS + 1, np.int64), order=order))
                MZ[lay]["hist"] += np.bincount(st.ravel(), minlength=NS)
        OUT16[r], SH[r], STOP[r] = o16w, shw, st
        del A, H
    return OUT16, SH, STOP, PILE

# ---------- the machine, every text at once, layer by layer (base2_fringes2.forward's own steps) ----------
STORE_RULE = {}                                       # filled by the report from the refs (qkv.0, proj.0, fc.0); default: stored
def forward(texts, C=None, check=None, chk=None, log=print, order_seed=None, MZ=None):
    t0 = time.time(); S = FM.store(); M = FM.M; BIAS = FM.BIAS
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in lens])
    first = pos == 0; r1 = np.flatnonzero(~first); f0 = np.flatnonzero(first); NP = len(ids)
    WTE, WPE = M["wte"], M["wpe"]; A_rows, B_rows = WTE[ids], WPE[pos]; R = A_rows + B_rows
    orng = None if order_seed is None else np.random.default_rng(order_seed)
    def reo(kind, l): return None if order_seed is None else [order_seed, 4242, SLABS.index(kind), l]
    def porder(): return None if orng is None else orng.permutation(NPL)
    def keep(k, v, rows=None):
        if check is not None: check(k, v, rows)
    ck = chk; cb = FM.count_both
    if C is not None: cb(C, "L0", "rest", "entry: word + position", ones(B_rows).sum())
    STOPS = []

    def fc_and_mproj(lay, l, xlight, c0, x_plain, ex_, LS_, rows):
        FCPERM = None if orng is None else orng.permutation(3072)
        gpf = gplate("fc", l, reorder=reo("fc", l), neuron_perm=FCPERM)
        bias_fc = BIAS[f"fc.{l}"] if FCPERM is None else BIAS[f"fc.{l}"][FCPERM]
        if c0 is not None and FCPERM is not None: c0 = c0[:, FCPERM]
        Ff = (gpf.e[None, :] + ex_[:, None] + ES) * 65536 - LS_[:, None]
        if C is not None: cb(C, lay, "rest", "water levels", FM.water(x_plain))
        out16, sh, stop, fpile = fc_read(gpf, xlight, c0, Ff, bias_fc, C, lay, ck, x_plain, MZ=MZ)
        if FCPERM is not None:
            inv = np.argsort(FCPERM); out16, stop, fpile = out16[:, inv], stop[:, inv], fpile[:, inv]
        STOPS.append(stop)
        keep(f"fc pile {l}", fpile, rows); keep(f"fc out16 {l}", out16, rows); keep(f"fc sh {l}", sh, rows)
        if c0 is not None: side_whole(C, lay, "fc", gplate("fc", l), x_plain, fpile, ck, "storage: the whole light in the group form")
        del gpf, fpile
        gpm = gplate("mproj", l, reorder=reo("mproj", l)); wem, bm = gpm.e, BIAS[f"mproj.{l}"]
        if orng is None: mpile = bank(C, lay, "mproj", gpm, out16, chk=ck)
        else:                                          # ORDER-FREE: the edges met in groups by their stop place, in a random order
            mpile = np.zeros((len(out16), gpm.M), np.int64)
            for g_ in orng.permutation(int(stop.max()) + 1):
                xg = np.where(stop == g_, out16, 0)
                if xg.any(): mpile += gpm.read(xg, place_order=porder())[0]
            if C is not None: bank(C, lay, "mproj", gpm, out16, chk=ck)         # counted once, on the whole light
        keep(f"mproj pile {l}", mpile, rows)
        if C is not None: cb(C, lay, "rest", "biases", len(out16) * ones(bm).sum())
        mo = FM.rshift(mpile, -(wem[None, :] + (sh - ES)[:, None] + ES)) + bm
        if C is not None: cb(C, lay, "rest", "stream adds", ones(mo).sum())
        return mo

    # ================= layer 0 =================
    gp = gplate("qkv", 0, reorder=reo("qkv", 0)); we, b = gp.e, BIAS["qkv.0"]
    x16, ex, LS = FM.gauge_log(R); s = ex + ES
    pile = np.empty((NP, gp.M), np.int64); carry = np.zeros((NP, 768), np.int64); carried = np.zeros(NP, bool)
    stored_q = np.zeros(NP, np.int64)
    for sv in np.unique(s):
        m = np.flatnonzero(s == sv)
        if sv == 4:
            assert (pos[m] == 0).all(); z = S["ZP0"].read(ids[m]); pile[m] = z
            if C is not None: C.add("fr|L0|qkv|stored rows read and added", ones(z).sum())
        elif sv == 0:
            assert (pos[m] == 1023).all(); z = S["Z1023"].read(ids[m]); pile[m] = z
            if C is not None: C.add("fr|L0|qkv|stored rows read and added", ones(z).sum())
        elif sv in (1, 2):
            a_, b_ = FM.rshift(A_rows[m], sv), FM.rshift(B_rows[m], sv); c = x16[m] - a_ - b_; assert np.abs(c).max() <= 1
            t1, t2 = S["T"][sv].read(ids[m]), S["P"][sv].read(pos[m])
            pile[m] = t1 + t2; carry[m] = c; carried[m] = True
            stored_q[m] = ones(t1).sum(1) + ones(t2).sum(1) + ones(a_).sum(1) + ones(b_).sum(1)
            if C is not None:
                C.add("fr|L0|qkv|stored rows read and added", ones(t1).sum() + ones(t2).sum())
                C.add("fr|L0|qkv|split (stored rounded rows taken off)", ones(a_).sum() + ones(b_).sum())
        else: raise ValueError(f"qkv.0 shift {sv} never occurs (entry/survey_s.py)")
    cr = np.flatnonzero(carried); nc = np.flatnonzero(~carried)
    if len(cr):
        gpc = gplate("qkv", 0, reorder=reo("qkv", 0), up=np.zeros(gp.M, np.int64))        # rule 2: the carry in signed digits
        mp = bank(C, "L0", "qkv", gpc, carry[cr], plain_x=x16[cr], chk=ck, place_order=porder()); pile[cr] += mp
        side_whole(C, "L0", "qkv", gp, carry[cr], mp, ck, "the carry in the group form")
        side_whole(C, "L0", "qkv", gp, x16[cr], pile[cr], ck, "storage: the whole light in the group form")
        if C is not None: C.add("side|L0|qkv|storage: the stored form (rows, split, the carry's work)", stored_q[cr].sum())
        del mp, gpc
    if C is not None:
        if len(nc): C.add("plain|L0|qkv|meetings, plain", ones(x16[nc]) @ gp.wrow)
        cb(C, "L0", "rest", "water levels", FM.water(x16))
        cb(C, "L0", "rest", "bank divides", ones(pile).sum(1) * FM.E_ones(LS))
        cb(C, "L0", "rest", "biases", NP * ones(b).sum())
    keep("qkv pile 0", pile)
    QKV = FM.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, gp
    o = np.concatenate([FM.arrival(QKV[off[t]:off[t + 1]], C, lay="L0", first_counted=1, chk=ck) for t in range(len(texts))])
    R2 = np.empty_like(R)
    z0 = S["Z0"].read(ids[f0]); R2[f0] = z0
    if C is not None: C.add("fr|L0|rest|position 0: the stream after layer 0 read", ones(z0).sum())
    # proj.0 (positions 1 on): the own value's stored tip read, only o16 - own met (in the group form)
    gp = gplate("proj", 0, reorder=reo("proj", 0)); wep, bp = gp.e, BIAS["proj.0"]
    o16, eo = FM.to16(o[r1], -ES); so = eo + ES; keys = ids[r1] * 1024 + pos[r1]
    rest_o = o16.copy(); stored = np.zeros((len(r1), 768), np.int64); stored_p = np.zeros(len(r1), np.int64)
    for sv in np.unique(so):
        m = np.flatnonzero(so == sv)
        has = S["OWN"][sv].has(keys[m]) if sv in S["OWN"] else np.zeros(len(m), bool)
        st_, nf = m[has], m[~has]
        if len(st_):
            ref = FM.rshift(QKV[r1[st_], 1536:], sv); stored[st_] = S["OWN"][sv].read(keys[st_]); rest_o[st_] = o16[st_] - ref
            stored_p[st_] = ones(stored[st_]).sum(1) + ones(ref).sum(1)
            if C is not None:
                C.add("fr|L0|proj|stored rows read and added", ones(stored[st_]).sum())
                C.add("fr|L0|proj|split (own value taken off)", ones(ref).sum())
        if len(nf): FM.FALLBACK["proj shift not stored" if sv not in S["OWN"] else "proj pairs not stored"] += len(nf)
    mp = bank(C, "L0", "proj", gp, rest_o, plain_x=o16, chk=ck, place_order=porder()); ppile = stored + mp; keep("proj pile 0", ppile, r1)
    side_whole(C, "L0", "proj", gp, o16, ppile, ck, "storage: the whole light in the group form")
    if C is not None: C.add("side|L0|proj|storage: the stored form (rows, split)", stored_p.sum())
    att = FM.rshift(ppile, -(wep[None, :] + eo[:, None] + ES)) + bp; del gp, ppile, mp
    if C is not None:
        cb(C, "L0", "rest", "biases", len(r1) * ones(bp).sum()); cb(C, "L0", "rest", "stream adds", ones(att).sum())
    R1 = R[r1] + att; keep("stream a 0", R1, r1)
    # fc.0 (positions 1 on): the word's and position's stored parts read, the rest met
    x1, e1, LS1 = FM.gauge_log(R1); s1_ = e1 + ES
    rest = x1.copy(); c0 = np.zeros((len(r1), 3072), np.int64); BPROJ = BIAS["proj.0"]; stored_f = np.zeros(len(r1), np.int64)
    for sv in np.unique(s1_):
        m = np.flatnonzero(s1_ == sv)
        if sv in S["FT"]:
            a_ = FM.rshift(WTE[ids[r1[m]]], sv); b_ = FM.rshift(WPE[pos[r1[m]]] + BPROJ[None, :], sv)
            rest[m] = x1[m] - a_ - b_; c0[m] = S["FT"][sv].read(ids[r1[m]]) + S["FP"][sv].read(pos[r1[m]])
            stored_f[m] = ones(c0[m]).sum(1) + ones(a_).sum(1) + ones(b_).sum(1)
            if C is not None:
                C.add("fr|L0|fc|stored rows read and added", ones(c0[m]).sum())
                C.add("fr|L0|fc|split (stored rounded rows taken off)", ones(a_).sum() + ones(b_).sum())
        else: FM.FALLBACK["fc shift not stored"] += len(m)
    if C is not None: C.add("side|L0|fc|storage: the stored form (rows, split)", stored_f.sum())
    mo = fc_and_mproj("L0", 0, rest, c0, x1, e1, LS1, r1)
    R2[r1] = R1 + mo; del mo, rest, c0
    FM.check_position0(R, QKV, o, f0, R2, C, ck)
    keep("stream 0", R2)
    R = R2; del QKV, o, R1, R2
    log(f"    layer  0 done ({time.time() - t0:.0f}s, {rss()} MB)")
    # ================= layers 1-11 =================
    for l in range(1, 12):
        lay = f"L{l}"
        gp = gplate("qkv", l, reorder=reo("qkv", l)); we, b = gp.e, BIAS[f"qkv.{l}"]
        x16, ex, LS = FM.gauge_log(R); pile = bank(C, lay, "qkv", gp, x16, chk=ck, place_order=porder()); keep(f"qkv pile {l}", pile)
        if C is not None:
            cb(C, lay, "rest", "water levels", FM.water(x16)); cb(C, lay, "rest", "bank divides", ones(pile).sum(1) * FM.E_ones(LS))
            cb(C, lay, "rest", "biases", NP * ones(b).sum())
        QKV = FM.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, gp
        o = np.concatenate([FM.arrival(QKV[off[t]:off[t + 1]], C, lay=lay, chk=ck) for t in range(len(texts))]); del QKV
        gp = gplate("proj", l, reorder=reo("proj", l)); wep, bp = gp.e, BIAS[f"proj.{l}"]
        o16, eo = FM.to16(o, -ES); pile = bank(C, lay, "proj", gp, o16, chk=ck, place_order=porder()); keep(f"proj pile {l}", pile)
        att = FM.rshift(pile, -(wep[None, :] + eo[:, None] + ES)) + bp; del pile, gp, o
        if C is not None: cb(C, lay, "rest", "biases", NP * ones(bp).sum()); cb(C, lay, "rest", "stream adds", ones(att).sum())
        R = R + att; keep(f"stream a {l}", R); del att
        x16, ex, LS = FM.gauge_log(R)
        mo = fc_and_mproj(lay, l, x16, None, x16, ex, LS, None)
        R = R + mo; keep(f"stream {l}", R); del mo
        log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex, LS = FM.gauge_log(R)
    if C is not None: cb(C, "top", "rest", "top water level", FM.water(x16))
    return (x16, ex, LS), STOPS

# ---------- phases ----------
def opener(tag):
    os.makedirs(OUTD, exist_ok=True); f = open(f"{OUTD}/{tag}.log", "w")
    def log(s=""): print(s, flush=True); f.write(s + "\n"); f.flush()
    return log

def phase_refs(ARG):
    SEL = [int(v) for v in ARG["texts"].split(",")] if "texts" in ARG else list(range(30))
    ORDER = int(ARG["order"]) if "order" in ARG else None; CTL["on"] = bool(int(ARG.get("shuffle", "1" if ORDER is None else "0")))
    tag = "gpt2_refs" + ("" if "texts" not in ARG else "_t" + "-".join(map(str, SEL))) + ("" if ORDER is None else f"_order{ORDER}")
    log = opener(tag); t0 = time.time()
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
    ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL]); TEXTS = [FM.TEXTS[t] for t in SEL]
    assert all(len(FM.TEXTS[t]) == npos[t] for t in range(30))
    log(f"BASE2_GROUPS on refs texts {SEL if len(SEL) < 30 else 'all 30'} ({len(ROWS)} positions); order seed {ORDER}; shuffle control {CTL['on']}")
    _fcref = {}
    def fc_ref(l):
        if l not in _fcref:
            Rr = REF[f"stream_a_{l}"].astype(np.int64)[ROWS]; pl_ = REF[f"fc_pile_{l}"].astype(np.int64)[ROWS]
            x16, ex, LS = FM.gauge_log(Rr)
            H = FM.power(pl_, (FM.M[f"fc.{l}.e"][None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + FM.BIAS[f"fc.{l}"]
            o16, eo = FM.to16(FM.edge(H), -ES); _fcref.clear(); _fcref[l] = (o16, eo + ES)
        return _fcref[l]
    DIFF, CHECKED, BAD = {}, {}, {}
    def check(key, value, rows=None):
        k, l = key.rsplit(" ", 1); l = int(l)
        if k == "fc out16": ref = fc_ref(l)[0]
        elif k == "fc sh": ref = fc_ref(l)[1]
        else: ref = REF[f"{k.replace(' ', '_')}_{l}"].astype(np.int64)[ROWS]
        if rows is not None: ref = ref[rows]
        v = np.asarray(value); assert v.shape == ref.shape, (key, v.shape, ref.shape)
        bad = (v != ref) if v.ndim == 1 else (v != ref).any(1)
        DIFF[key] = int(bad.sum()); CHECKED[key] = len(ref)
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    C = FM.Count(); MZ = {}
    (x16, ex, LS), STOPS = forward(TEXTS, C, check, chk, log=log, order_seed=ORDER, MZ=MZ)
    same_top = all(np.array_equal(a, REF[k][ROWS]) for a, k in ((x16, "x16"), (ex, "ex"), (LS, "LS")))
    log(f"\nEXACT: {len(DIFF)} arrays checked against base2.py's own run; positions differing in any: {sum(DIFF.values())}; "
        f"the light at the top identical: {same_top}")
    for k in sorted(DIFF):
        if DIFF[k]: log(f"    DIFFERS: {k}: {DIFF[k]} of {CHECKED[k]}")
    log("  the check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())) + f"; fallbacks {FM.FALLBACK}")
    np.savez(f"{OUTD}/{tag}_top_light.npz", x16=x16, ex=ex, LS=LS, rows=ROWS)
    dig = {f"stop {l}": FM.digest(STOPS[l]) for l in range(12)}
    dig.update({k: FM.digest(v) for k, v in (("x16", x16), ("ex", ex), ("LS", LS))})
    json.dump(dict(counts=dict(C), npos=int(len(ROWS)), texts=SEL, diff=DIFF, bad=BAD, same_top=bool(same_top), digests=dig,
                   fallback=FM.FALLBACK, order=ORDER, shuffle=CTL["on"],
                   knife={k: dict(hist=v["hist"].tolist(), order=[int(b) for b in v["order"]]) for k, v in MZ.items()}),
              open(f"{OUTD}/{tag}.json", "w"), indent=0)
    n = len(ROWS)
    def tot(who, part=None): return sum(v for k, v in C.items() if k.split("|")[0] == who and (part is None or k.split("|")[2] == part))
    log(f"\nWORK a token (stream part; fc both forms shown):")
    for part in ("qkv", "proj", "fc", "mproj", "arrival", "rest"):
        log(f"  {part:8s} plain {tot('plain', part) / n:>14,.0f}   group form (common) {tot('fr', part) / n:>14,.0f}"
            + (f"   whole form {tot('fcw', part) / n:>12,.0f}   knife {tot('fck', part) / n:>12,.0f}" if part == "fc" else ""))
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_windows(ARG):
    a, b = int(ARG.get("a", 0)), int(ARG.get("b", 300)); tag = f"gpt2_windows_{a:03d}_{b:03d}"; log = opener(tag); t0 = time.time()
    PD = json.load(open(f"{HERE}/fringes_machine/plain_digests_windows.json"))
    ids = [w["ids"] for w in FM.WINDOWS[a:b]]; assert all(len(i) == 24 for i in ids); nw = len(ids); DIFF = {}; NCMP = [0]
    def check(key, value, rows=None):
        v = np.asarray(value); per = 24 if rows is None else 23; assert len(v) == nw * per, (key, v.shape)
        for j in range(nw):
            NCMP[0] += 1
            if FM.digest(v[j * per:(j + 1) * per]) != PD[a + j][key]: DIFF[key] = DIFF.get(key, 0) + 1
    BAD = {}
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    (x16, ex, ls), STOPS = forward(ids, None, check, chk, log=log)
    top_same = sum(int(FM.digest(np.concatenate([x16[24 * j:24 * j + 24].ravel(), ex[24 * j:24 * j + 24], ls[24 * j:24 * j + 24]]))
                       == PD[a + j]["top light"]) for j in range(nw))
    rows = np.concatenate([24 * j + np.arange(FM.FIRST, FM.LAST) for j in range(nw)])
    np.savez(f"{OUTD}/{tag}.npz", x16=x16[rows], ex=ex[rows], LS=ls[rows], windows=np.arange(a, b))
    log(f"windows {a}..{b - 1}: {NCMP[0]} window-arrays compared with base2.py's own run; differing: {sum(DIFF.values())} {DIFF}; "
        f"the light at the top identical in {top_same}/{nw} windows; check path: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items()))
        + f"; fallbacks {FM.FALLBACK} ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_join(ARG):
    log = opener("gpt2_windows_join"); Z = [np.load(p) for p in sorted(glob.glob(f"{OUTD}/gpt2_windows_[0-9]*.npz"))]
    W_ = np.concatenate([z["windows"] for z in Z]); assert np.array_equal(W_, np.arange(300)), "windows missing"
    X, EX, LS = (np.concatenate([z[k] for z in Z]) for k in ("x16", "ex", "LS"))
    np.savez(f"{OUTD}/gpt2_windows_top_light.npz", x16=X, ex=EX, LS=LS)
    P = np.load(f"{HERE}/fringes_machine/windows_top_light.npz")
    log(f"joined {len(Z)} parts: the light at the top of the 6,000 judged positions = base2's own: "
        f"{bool(np.array_equal(X, P['x16']) and np.array_equal(EX, P['ex']) and np.array_equal(LS, P['LS']))}")

# ---------- the top read: groups2/topg.py's (ii), per place, g = 16, on this machine's light ----------
from gtop import run_top, top_module as _top_module
def top_module(order): return _top_module("gpt2", G_TOP, order)

def phase_top(ARG):
    SET = ARG.get("set", "refs"); LIGHT = ARG.get("light", "real"); ORDER = "order" in ARG
    PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/"))
    tag = f"gpt2_top_{SET}" + ("" if LIGHT == "real" else f"_{LIGHT}") + ("_order" if ORDER else "") + ("" if PN == 1 else f"_p{PI}of{PN}")
    log = opener(tag); t0 = time.time()
    if SET == "refs":
        Z = np.load(f"{OUTD}/gpt2_refs_top_light.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
        ref5, refw = REF["top5"], REF["win"]
    elif SET == "orderfree":
        Z = np.load(f"{OUTD}/gpt2_refs_t0-1-2_order7_top_light.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
        ref5, refw = REF["top5"][Z["rows"]], REF["win"][Z["rows"]]
    else:
        Z = np.load(f"{OUTD}/gpt2_windows_top_light.npz"); ref5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); refw = None
    X, EX, LS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"]
    cut = np.linspace(0, len(X), PN + 1).astype(np.int64); part = np.arange(cut[PI - 1], cut[PI])
    X, EX, LS = X[part], np.asarray(EX)[part], np.asarray(LS)[part]
    if LIGHT == "shuffled": rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])   # topband.py's draw
    TG = top_module(ORDER)
    log(f"THE TOP READ: topg (ii), per place, g = {G_TOP}, on base2_groups' light: set {SET}, light {LIGHT}, order {ORDER}, part {PI}/{PN}, {len(X)} positions")
    rec, bad, pwrite, lwrite, ppmask, lab = run_top(TG, X, EX, LS, ORDER, log)
    r = rec["pp"]; n = len(X)
    pl = rec["plain_meet"] + rec["plain_div"] + rec["plain_stand"]
    tot = r["ii_total"] + pwrite + lwrite
    log(f"  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(bad["pp"].items())))
    log(f"  (ii) top 5 = the plain read's {int((r['ii_top5'] == rec['ptop5']).all(1).sum())}/{n}, race winner {int((r['ii_win'] == rec['pwin']).sum())}/{n}")
    if LIGHT == "real":
        log(f"  against the saved references: top 5 {int((r['ii_top5'] == ref5[part]).all(1).sum())}/{n}"
            + ("" if refw is None else f", race winner {int((r['ii_win'] == refw[part]).sum())}/{n}"))
    log(f"  plain read {pl.sum() / n:,.0f} a position; (ii) {r['ii_total'].sum() / n:,.0f} ({pct(r['ii_total'].sum(), pl.sum())}); "
        f"+ ready piles written {pwrite.sum() / n:,.0f} + light written {lwrite.sum() / n:,.0f} = {tot.sum() / n:,.0f} ({pct(tot.sum(), pl.sum())})")
    np.savez(f"{OUTD}/{tag}.npz", part=part, plain=pl, ii_total=r["ii_total"], pwrite=pwrite, lwrite=lwrite, table=r["table"],
             ii_reads=r["ii_reads"], ii_light=r["ii_light"], ii_bound=r["ii_bound"], ii_end=r["ii_end"], ii_stand=r["ii_stand"],
             ii_cmp=r["ii_cmp"], ii_surv=r["ii_surv"], ii_top5=r["ii_top5"], ii_win=r["ii_win"], ptop5=rec["ptop5"], pwin=rec["pwin"],
             plain_meet=rec["plain_meet"], plain_div=rec["plain_div"], plain_stand=rec["plain_stand"])
    json.dump(dict(bad=bad, n=int(n), ppmask=int(ppmask), digest=r["digest"]), open(f"{OUTD}/{tag}.json", "w"))
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_judge(ARG):
    log = opener("gpt2_judge"); t0 = time.time()
    parts = sorted(glob.glob(f"{OUTD}/gpt2_top_windows_p*of*.npz")) or [f"{OUTD}/gpt2_top_windows.npz"]
    Z = [np.load(p) for p in parts]; sel = np.concatenate([z["part"] for z in Z]); assert np.array_equal(sel, np.arange(6000))
    WIN = np.concatenate([z["ii_win"] for z in Z]); T5 = np.concatenate([z["ii_top5"] for z in Z])
    _cwd = os.getcwd(); os.chdir(HERE); import judge5 as J; os.chdir(_cwd)
    REF = J._ref(); it = iter(range(len(J.WINDOWS))); same_ref = [0]
    def fwd(ids):
        i = next(it); assert ids == J.WINDOWS[i]["ids"]; r = slice(20 * i, 20 * i + 20)
        same_ref[0] += int((T5[r] == REF[i]).all(1).sum()); return WIN[r]
    inn, same, tot = J.judge(forward=fwd)
    log(f"base2_groups: judge5.judge in the reference top 5 {inn}/{tot}, same top word {same}/{tot}; the machine's top 5 = judge5's "
        f"reference (set and order) {same_ref[0]}/{tot}  ({len(parts)} parts; {time.time() - t0:.0f}s)")

# ---------- the report: the rules applied (from the refs), the work by part, the controls, the checks ----------
FR2 = dict(qkv=362167885, proj=142256231, fc=488518533, mproj=543593368, arrival=8020119, top=145099607, rest=4558811, total=1694214555)
GKEYS = [GLABEL[k] for k in ("light", "table", "pwrite", "reads", "sdreads", "xsum", "xoff")]
def phase_report(ARG):
    log = opener("gpt2_report"); J = json.load(open(f"{OUTD}/gpt2_refs.json")); C = J["counts"]; n = J["npos"]
    T = np.load(f"{OUTD}/gpt2_top_refs.npz")
    def S(pred): return sum(v for k, v in C.items() if pred(k.split("|", 3)))
    def item(who, lay, part, it): return C.get(f"{who}|{lay}|{part}|{it}", 0)
    def grp(who, lay, part): return sum(item(who, lay, part, g) for g in GKEYS)
    pt = lambda v: v / n
    log("BASE2_GROUPS: THE WHOLE GPT-2 MACHINE WITH GROUP PILES (every bank in the per-place group form, g = 12; the top read")
    log("by topg (ii) per place, g = 16). A token on the 648 positions of the 30 refs texts; photon meetings and photons added.")
    log("")
    log("EXACT")
    for f in ("gpt2_refs", "gpt2_windows_join", "gpt2_top_refs", "gpt2_judge", "gpt2_refs_t0-1-2", "gpt2_refs_t0-1-2_order7", "gpt2_top_orderfree_order"):
        if os.path.exists(f"{OUTD}/{f}.log"):
            for line in open(f"{OUTD}/{f}.log"):
                if any(w in line for w in ("EXACT", "check path", "checks:", "top 5", "judge5", "joined")): log(f"  [{f}] {line.strip()}")
    wl = sorted(glob.glob(f"{OUTD}/gpt2_windows_[0-9]*.log")); wd = 0; wn = 0; wt = 0
    for f in wl:
        t = open(f).read(); import re
        m = re.search(r"compared with base2.py's own run; differing: (\d+)", t); wd += int(m.group(1)) if m else 10 ** 9
        m = re.search(r"identical in (\d+)/(\d+) windows", t); wt += int(m.group(1)) if m else 0; wn += int(m.group(2)) if m else 0
    log(f"  [windows] {len(wl)} parts: window-arrays differing from base2.py's own run {wd}; the light at the top identical in {wt}/{wn} windows")
    log("")
    # ---- rule 3: layer 0 from storage, kept only if it pays ----
    log("RULE 3, LAYER 0 FROM STORAGE (fixed before the run): the stored form kept at a plate only if it is lower on the refs")
    keep_store = {}
    for part in ("qkv", "proj", "fc"):
        stored_rows = item("side", "L0", part, "storage: the stored form (rows, split, the carry's work)" if part == "qkv" else "storage: the stored form (rows, split)")
        work = grp("fr", "L0", part) if part != "fc" else (grp("fr", "L0", "fc") + item("fcw", "L0", "fc", GLABEL["reads"]) + item("fcw", "L0", "fc", GLABEL["sdreads"]))
        stored = stored_rows + work; whole = item("side", "L0", part, "storage: the whole light in the group form: group work")
        keep_store[part] = stored < whole
        log(f"  {part}.0: stored form {pt(stored):>12,.0f} (rows and split {pt(stored_rows):,.0f}, the rest of the light met {pt(work):,.0f}); "
            f"the whole light {pt(whole):>12,.0f}: {'KEPT' if stored < whole else 'NOT KEPT'} ({pt(whole - stored):+,.0f} a token)")
    if "qkv" in keep_store:
        cg = item("side", "L0", "qkv", "the carry in the group form: group work")
        log(f"  rule 2 beside it: qkv.0's carry in signed digits {pt(grp('fr', 'L0', 'qkv')):,.0f}; the carry in the group form {pt(cg):,.0f} (count only)")
    log("")
    # ---- rule 4: the knife, kept only if it pays ----
    log("RULE 4, THE KNIFE (fixed before the run): kept at fc.l only if its total is lower than the whole form's on the refs")
    knife = {}; fcw_tot = 0; fck_tot = 0
    for l in range(12):
        lay = f"L{l}"; w_ = S(lambda k: k[0] == "fcw" and k[1] == lay); k_ = S(lambda k: k[0] == "fck" and k[1] == lay)
        saved = item("side", lay, "fc", "knife: reads saved (the places below each edge's stop)")
        knife[lay] = k_ < w_; fcw_tot += w_; fck_tot += k_
        h = J["knife"].get(lay, {}).get("hist"); o = J["knife"].get(lay, {}).get("order")
        early = (sum(h[:-1]) / sum(h)) if h else 0
        log(f"  fc.{l:<2d}: whole form {pt(w_):>12,.0f}; knife {pt(k_):>12,.0f} (its reads saved {pt(saved):>10,.0f}); {'KEPT' if k_ < w_ else 'NOT KEPT'}; "
            f"edges stopped before the last place {100 * early:5.2f}%")
    own = {}
    for k, v in C.items():
        w_, L, pp_, it = k.split("|", 3)
        if w_ == "fck": own[it] = own.get(it, 0) + v
    log("  the knife's work at all 12 (a token): " + "; ".join(f"{it} {pt(v):,.0f}" for it, v in sorted(own.items(), key=lambda z: -z[1])))
    log(f"  the whole form's at all 12: {pt(fcw_tot):,.0f}; the knife's {pt(fck_tot):,.0f}")
    log("")
    # ---- the work by part ----
    def part_fr(part):
        v = S(lambda k: k[0] == "fr" and k[2] == part)
        if part == "fc": v += sum(S(lambda k, L=L: k[0] == ("fck" if knife[L] else "fcw") and k[1] == L) for L in knife)
        return v
    for part, keepit in keep_store.items():                      # a stored form not kept: the whole light's group work in its place
        if not keepit: log(f"  (rule 3 sends {part}.0 to the whole light: counted below with its whole-light group work)")
    def store_adj(part):
        if keep_store.get(part, True): return 0
        stored_rows = item("side", "L0", part, "storage: the stored form (rows, split, the carry's work)" if part == "qkv" else "storage: the stored form (rows, split)")
        work = grp("fr", "L0", part) if part != "fc" else (grp("fr", "L0", "fc") + item("fcw", "L0", "fc", GLABEL["reads"]) + item("fcw", "L0", "fc", GLABEL["sdreads"]))
        return item("side", "L0", part, "storage: the whole light in the group form: group work") - stored_rows - work
    plain = {p_: S(lambda k, p_=p_: k[0] == "plain" and k[2] == p_) for p_ in ("qkv", "proj", "fc", "mproj", "arrival", "rest")}
    mine = {p_: part_fr(p_) + store_adj(p_) for p_ in plain}
    top_plain = int(T["plain"].sum()); top_ii = int(T["ii_total"].sum()); top_pw = int(T["pwrite"].sum()); top_lw = int(T["lwrite"].sum())
    plain["top"] = top_plain; mine["top"] = top_ii + top_pw + top_lw
    P0 = sum(plain.values()); M0 = sum(mine.values())
    pw_stream = S(lambda k: k[0] == "fr" and k[3] == GLABEL["pwrite"])
    log(f"THE PLAIN MACHINE rebuilt in the same run: {pt(P0):,.0f} a token (the unit of all earlier results 3,666,596,259; same: {round(pt(P0)) == 3666596259})")
    log("")
    log("WORK A TOKEN BY PART")
    log(f"  {'part':44s} {'plain':>14s} {'base2_fringes2':>15s} {'base2_groups':>14s} {'of plain':>9s} {'of fringes2':>11s}")
    for p_ in ("qkv", "proj", "fc", "mproj", "arrival", "top", "rest"):
        log(f"  {p_:44s} {pt(plain[p_]):14,.0f} {FR2[p_]:15,} {pt(mine[p_]):14,.0f} {100 * mine[p_] / plain[p_]:8.2f}% {100 * pt(mine[p_]) / FR2[p_]:10.2f}%")
    log(f"  {'THE WHOLE MACHINE':44s} {pt(P0):14,.0f} {FR2['total']:15,} {pt(M0):14,.0f} {100 * M0 / P0:8.2f}% {100 * pt(M0) / FR2['total']:10.2f}%")
    log(f"  of it, the ready piles written in signed digits: banks {pt(pw_stream):,.0f}, top {pt(top_pw):,.0f}; without them (the confirmed counts' unit) "
        f"{pt(M0 - pw_stream - top_pw):,.0f} = {100 * (M0 - pw_stream - top_pw) / P0:.2f}% of plain")
    log("")
    log("THE BANKS BY ITEM (all 12 layers, a token; fc: the form rule 4 chose)")
    for p_ in ("qkv", "proj", "fc", "mproj"):
        items = {}
        for k, v in C.items():
            w_, L, pp_, it = k.split("|", 3)
            if pp_ != p_: continue
            if w_ == "fr" or (w_ == "fck" and knife.get(L)) or (w_ == "fcw" and not knife.get(L)): items[it] = items.get(it, 0) + v
        log(f"  {p_}: " + "; ".join(f"{it} {pt(v):,.0f}" for it, v in sorted(items.items(), key=lambda z: -z[1])))
        sdm = S(lambda k, p_=p_: k[0] == "side" and k[2] == p_ and k[3].startswith("signed digits"))
        log(f"      beside it, the signed-digit form on the same light (meetings + writing): {pt(sdm):,.0f} ({100 * sdm / plain[p_]:.2f}% of plain)")
    log("")
    dpos = np.flatnonzero(~(T["ii_top5"] == T["ptop5"]).all(1))
    if len(dpos):
        a_ = sys.argv; sys.argv = ["x"]; sys.path.insert(0, f"{HERE}/topband"); import topband as TB; sys.argv = a_
        Zl = np.load(f"{OUTD}/gpt2_refs_top_light.npz"); Wh = np.asarray(TB.M["head.w"]).astype(np.int64)
        for q in dpos:
            xq = Zl["x16"][q].astype(np.int64); Fp = int((TB.HEAD_E[0] + int(Zl["ex"][q]) + TB.ES) * 65536 - int(Zl["LS"][q]))
            Sq = TB.power(xq @ Wh, Fp) + np.asarray(TB.HEAD_B).astype(np.int64); best5 = np.sort(Sq)[::-1][:5]
            log(f"  refs position {q}: (ii)'s top 5 {T['ii_top5'][q].tolist()} scores {Sq[T['ii_top5'][q]].tolist()}; the plain read's (judge5's rule) "
                f"{T['ptop5'][q].tolist()} scores {Sq[T['ptop5'][q]].tolist()}; the five best scores {best5.tolist()}: (ii)'s five are the five best in order: "
                f"{bool(np.array_equal(Sq[T['ii_top5'][q]], best5))}")
    log("THE TOP READ (topg (ii), per place, g = 16, on this machine's light): " + f"plain {pt(top_plain):,.0f}; (ii) {pt(top_ii):,.0f} "
        f"({100 * top_ii / top_plain:.2f}%); ready piles written {pt(top_pw):,.0f}; light written {pt(top_lw):,.0f}; total {100 * mine['top'] / top_plain:.2f}% of the plain read")
    for k_ in ("table", "ii_reads", "ii_light", "ii_bound", "ii_end", "ii_stand"):
        log(f"    {k_:10s} {T[k_].sum() / n:14,.0f}")
    log("")
    log("CONTROLS")
    sh = S(lambda k: k[0] == "ctl" and k[3].endswith("shuffled across its lines")); re_ = S(lambda k: k[0] == "ctl" and k[3].endswith("the real light"))
    log(f"  shuffle (every bank, the light shuffled across its lines, count only): group work {pt(sh):,.0f} against the real light's {pt(re_):,.0f}: "
        f"{100 * (sh - re_) / P0:+.3f} points of plain ({100 * (sh - re_) / re_:+.2f}% of the group work)")
    for p_ in ("qkv", "proj", "fc", "mproj"):
        a = S(lambda k, p_=p_: k[0] == "ctl" and k[2] == p_ and k[3].endswith("shuffled across its lines")); b = S(lambda k, p_=p_: k[0] == "ctl" and k[2] == p_ and k[3].endswith("the real light"))
        log(f"    {p_:6s} {100 * (a - b) / b:+.3f}% of its group work")
    if os.path.exists(f"{OUTD}/gpt2_top_refs_shuffled.npz"):
        Ts = np.load(f"{OUTD}/gpt2_top_refs_shuffled.npz"); ts = int((Ts["ii_total"] + Ts["pwrite"] + Ts["lwrite"]).sum())
        log(f"  top read on the shuffled light (topband.py's draw): {100 * ts / int(Ts['plain'].sum()):.2f}% of the plain read against {100 * mine['top'] / top_plain:.2f}%")
    if os.path.exists(f"{OUTD}/gpt2_refs_t0-1-2.json") and os.path.exists(f"{OUTD}/gpt2_refs_t0-1-2_order7.json"):
        A_ = json.load(open(f"{OUTD}/gpt2_refs_t0-1-2.json")); B_ = json.load(open(f"{OUTD}/gpt2_refs_t0-1-2_order7.json"))
        ca, cb_ = A_["counts"], B_["counts"]; keys = set(ca) | set(cb_)
        diff = sorted(k for k in keys if ca.get(k) != cb_.get(k))
        kinds = sorted({k.split("|", 3)[3] for k in diff})
        log(f"  order-free (texts 0-2): arrays differing {sum(B_['diff'].values())}; count keys differing {len(diff)} of {len(keys)}, items: {kinds}")
        tab_a = sum(v for k, v in ca.items() if k.endswith(GLABEL['table'])); tab_b = sum(v for k, v in cb_.items() if k.endswith(GLABEL['table']))
        pa = sum(v for k, v in ca.items() if k.startswith('plain|'))
        log(f"    the half table {tab_a:,} against {tab_b:,} ({100 * (tab_b - tab_a) / pa:+.4f} points of the stream's plain); stops identical: "
            f"{all(A_['digests'][f'stop {l}'] == B_['digests'][f'stop {l}'] for l in range(12))}")
    log("")
    log("PREDICTIONS (written in this file before it ran) AGAINST THE RESULT")
    w = 100 * M0 / P0
    log(f"  THE TASK'S: 22-28% of plain: {w:.2f}% -> {'held' if 22 <= w <= 28 else ('lower' if w < 22 else 'higher')}")
    log(f"  THE SESSION'S: about 19.5% (18.5-21%): {w:.2f}% -> {'held' if 18.5 <= w <= 21 else ('lower' if w < 18.5 else 'higher')}")
    log(f"  the knife kept at 0 of 12 fc plates: kept at {sum(knife.values())}; storage kept at qkv.0, proj.0, fc.0: {keep_store}")

if __name__ == "__main__":
    ph = sys.argv[1]; ARG = dict(a.split("=") for a in sys.argv[2:])
    dict(refs=phase_refs, windows=phase_windows, join=phase_join, top=phase_top, judge=phase_judge, report=phase_report)[ph](ARG)
