"""GROUP PILES AT THE TOP TABLES (groups2/, the session's own step; the task's question, the session's rule and analysis,
labelled as such). Whole numbers only; every word's pile checked digit for digit against the direct product.

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  A token's light reaches a plate. Take a RANDOM group of g light lines. Every subset's pile is built once per token,
  and each neuron at each plate place adds one ready pile instead of meeting its lit lines one at a time. In the
  up/hole form every place of every weight is a photon up or a hole, so by the flip (Babinet) a pattern's complement
  is exactly its opposite pile and half the table comes free.
  At the top table the plate is the word table and every word is a neuron: GPT-2's 50,257 words and Qwen3-0.6B's
  151,936 words all meet ONE light. So one table a group a token is read by every word at every place, and a large
  group costs almost nothing a word.
  The band read (topband/, qwen_fringes2.py) pours every exposure from its top places down, and after every band the
  words that can no longer reach the top 5 are set aside by one shared bound on what is still unpoured. Every word
  must receive the top bands before any word can be set aside; those are most of its work.

WHAT IS MEASURED
  (i)  group piles for EVERY word, all places, in three forms, and every word's score made from its pile as the plain
       read makes it (every word's divide and standing amount, as in the plain read's count):
         uh  up/hole: every place of every weight up or hole (w + 2^15 in 16 places), the half table by the flip;
         sd  signed digits: up- and down-subsets at each place, the full table (the subset flip costs a take-away);
         pp  each place up/hole or signed digits, chosen by the plate alone (the rule below).
  (ii) the same group piles read from the TOP PLACE DOWN, one place at a time, each place only into the words still in;
       after every place but the last, the band read's own set-aside (below), with one shared bound from the light
       alone; after the last place the words still in are finished exactly. This is group piles for the places every
       word must receive, then the band read's exact set-aside for the rest.

THE RULE (fixed here, before any run; nothing in it is read from results)
  g:   16 and 20, the task's two sizes, both run, both reported; neither is chosen from results. The relation of
       section E says beforehand where a table can pay: a group's table is read 16 M times a token in (i) (M words)
       against 2^(g-1) entries in its half: GPT-2 804,112 against 32,768 (g = 16) and 524,288 (g = 20); Qwen 2,430,976
       against the same. At g = 20 the GPT-2 table is 65% of the reads it serves, so it is not expected to pay there.
       Tables at g = 20 are 8 MB a group a core: under 3 GB (Qwen's g = 20 runs hold one or two forms' patterns at a
       time).
  pp:  the confirm step's rule (groups2/uh_perplace.py), applied to the top table: place b is read in signed digits
       when L_b < U_b, L_b = the table's lit signed digits at place b (its own signed digits), U_b = G x M (the up/hole
       reads the place costs, G = ceil(N / g)); otherwise up/hole. Each weight written once, bottom place to top:
       binary digits at the up/hole places, signed digits elsewhere, offset C = 2^(top up/hole place); then
       2 x.w = sum over up/hole places of 2^b (ready pile) + 2 (signed part) + (K - 2C) X. The form of (ii) is pp;
       uh and sd are run beside it.
  (ii) read order and set-aside:
       - places from the top down (the form's places that the table lights at all, a plate-only list); each place
         read into the words still in;
       - after every place but the last, the set-aside of topband.py (GPT-2) and qwen_fringes2.band_read (Qwen),
         carried to the doubled scale: GPT-2 V_w = (read part) + 2 beta_w, set aside when V_w <= v5 - (2 R2 + 2 mu);
         Qwen per shift class k, the 5th-best low L5 among the classes' top 5 (V << k) - (R2 << k), set aside when
         (V << K_w) <= L5 - ((R2 << k) + 2 mu). mu, beta and the shifts exactly as those files make them;
       - ONE SHARED BOUND from the light alone: R2 = B_f(b) x S1, S1 = the sum of |x| over the lines, B_f(b) the
         largest the form's unread places below b can hold per unit of light at the doubled scale: uh 2^b - 1; sd
         2 t_b with t_n = (2^(n+1) - 1) // 3 (non-adjacent digits); pp the up/hole places below b, 2^p each, plus twice
         the non-adjacent bound of each signed run below b;
       - X (the light summed, the up/hole forms' offset) is one number shared by every word, so it is folded into the
         shared threshold (GPT-2: it cancels; Qwen: one add a shift class), and taken off only the words read to the
         end;
       - after the last place: the words still in are finished (X off, halved, the standing amount off, the divide,
         the bias), and the top 5 and race winner taken among them, as the band read does.

THE WORK COUNTED (photon meetings, the unit of every earlier result)
  table: uh and pp the half table by the flip, sd the full table, each Gray step one light value's signed digits
    (counted by the walk's relation from each line's digits: in the half table line t turns 2^(g-2-t) times plus the
    group total's g - 1 adds; in the full table 2^(g-1-t) times); reads: the ready pile's signed digits (a shift adds
    none); combines (sd, both subsets lit): the down pile's digits; signed-digit places of pp: each lit digit meets its
    own light value (that value's digits); X and S1: N - 1 adds of light values each; X's take-away: X's digits a word.
  (i)  table + reads + X (uh, pp) + every word's divide and standing amount as the plain read counts them; against
       the plain read (topband.py's 904,017,313 a position on the refs; qwen_fringes2.py's 4,490,888,460), and its
       meetings part against the plain meetings.
  (ii) table + light sums + reads into the words still in + the bound's photons + standing amounts (GPT-2, as
       topband counts them) + the words read to the end (as topband / qwen_fringes2 count them); compares kept apart as
       a count. Against the plain read and against the band read at the same positions (topband/top_*.npz,
       qwen_fringes2/top_*.npz, the band read's own per-position records).

CHECK PATH (never used by the machine, never counted): the direct product of the light and the table as stored (topg.c,
int64 sums, every line met one by one); every word's doubled pile from its place parts must equal twice the product;
every word's score from its group pile must equal the plain read's score (Qwen: the score row's sha256 against
qwen_base2's own read and capture's record); (ii): the bound must hold for EVERY word at every test, every word set aside
must score below the true 5th-best, the words read to the end must score as the plain read; top 5 (set and order) and
race winner against the plain read and the saved references (GPT-2 refs: base2.py's run, recording/C_ref/plain_refs.npz;
windows: judge5_ref.npz; Qwen: qwen_fringes/ref.json and qwen06b/light/meta.json; the greedy-word positions:
qwen_fringes/words_ref.json, race winner = the next greedy word).

CONTROLS (each a separate run)
  seed2    a second random grouping
  shuffle  each position's light shuffled across its lines, every value kept: GPT-2 topband.py's draw (all 648 refs
           positions), Qwen qwen_fringes.py's control draw (every 4th position, 154), so the band read on the very same
           shuffled light is on disk to set beside it; exact against the shuffled light's own direct product
  order    order-free: the groups in a random order, each group's lines in a random order, the words relabelled

RULE 13, CHECKED BEFORE THE RUN
  13.1 not the light's 1s: the measure is the work of the finished top read (every pile exact; the top 5 and race
       winner exact); the light's sizes enter only as the shared bound S1, a relation from the light alone.
  13.2 the base is the machine's own: base2.py's word table and its own light at the top (refs and judge5 windows);
       qwen_base2's word table, shifts and capture's light. The score steps (power, bias, shifts, race, top 5) are the
       machines' own, unchanged.
  13.3 no agreement statistic: the shuffle measures work, and exactness is against the product.
  13.4 fixed by the definitions, and so not measured: the up/hole read count (16 a word a group), the table's size,
       the per-place split (plate alone), and in (ii) the survivors under seed2 and order (the read part after places
       >= b is sum_i x_i (digits >= b), the same for any grouping): those runs CHECK that they are identical. The run
       measures the ready piles' digits, the survivors on the real light, and exactness.
  13.5 the picture and the rule are the session's (the predictions marked THE TASK'S are the task's text).
  13.6 every word, every position; (ii) carried to place 0 at every position.

ANALYSIS BEFORE THE RUN (the session's; plate-only counts and the light's sizes, and earlier logs on disk)
  The tables' digits (plate only): GPT-2's word table has one power of two, so its weights sit low: lit signed-digit
  shares by place 0.50 0.25 0.38 0.31 0.34 0.33 0.34 0.33 0.33 0.33 0.27 (place 10) 0.079 0.016 0.008 0.006 0.000
  (place 15). Qwen's words each have their own power of two: 0.33 at places 1-11, 0.26 at 12, 0.10 at 13, 0.009 at 14,
  0.001 at 15. pp's rule (lit share below G/N = 1/16 at g = 16, 39/768 and 52/1024 at g = 20) sends GPT-2 places 12-15
  and Qwen places 14-15 to signed digits at both g. In pp's written form the up/hole window is |w| < 2^11 (GPT-2) and
  2^13 (Qwen), so the 4.35% (GPT-2) and 3.0% (Qwen) of weights outside it light the places above: about 0.06 and 0.04
  lit digits a weight there.
  The light (sizes only): GPT-2 a typical value 2^9-2^10 (mean |x| 742; rms 2,052 because about 11 lines a position
  stand at 2^12-2^15); Qwen mean |x| 623, rms 1,288, about 11 lines at 2^12 and up. A ready up/hole pile of 16 lines is
  about 4 x the group's rms: 2^11.5-2^12 in most groups (about 4.4-4.5 signed digits), 2^14-2^15 in the groups holding a
  large line (about 5.3; 21% of GPT-2's groups, 15% of Qwen's): about 4.7 (GPT-2) and 4.4 (Qwen) a read.
  So one up/hole place read by every word costs 48 x 4.7 x 50,257 = 11.3M photons, 1.25% of GPT-2's plain read (23.4
  photons a pair), and 64 x 4.4 x 151,936 = 42.8M, 0.95% of Qwen's (28.9 a pair). The half table: 48 x 32,768 x 3.7 =
  5.8M (0.64%) and 64 x 32,768 x 3.85 = 8.1M (0.18%). At g = 20 the reads fall by about a sixth but the half table
  grows 16-fold: 8.2% of GPT-2's plain read, 2.3% of Qwen's.
  (i) uh: GPT-2 16 x 1.25 + 0.64 + 0.05 (X) + 0.83 (every word's divide and standing amount) = about 21.5%; Qwen
      16 x 0.95 + 0.18 + 0.03 + 0.41 = about 15.8%. pp drops the dark top places: GPT-2 12 x 1.25 + 0.9 (signed) +
      0.64 + 0.88 = about 17.4%; Qwen 14 x 0.95 + 0.5 + 0.18 + 0.44 = about 14.4%. sd: at every lit place a read and
      (at 88% of cells) a combine, about 7.6 photons a (word, group, place) against up/hole's 4.7: about 26% (GPT-2),
      22% (Qwen).
  (ii) the shared bound after the places >= b are read is, at the pile's scale, (2^b - 1) S1 / 2: about 2.85e5 x 2^b
      (GPT-2, S1 about 5.7e5) and 3.2e5 x 2^b (Qwen, S1 about 6.4e5). Where words fall away, from the logs on disk:
      GPT-2 (topband, fine_staged_top): bound 2e8 sets none aside; 6e7 leaves 42,535 (median); 1.6e7 leaves 418;
      1.26e7 leaves 70 (median; mean 760). Qwen (qwen_fringes2's band read): 2.7e8 leaves 26%; 1.7e7 leaves 9.
      GPT-2 pp: the signed places (0.9%) and up/hole places 11, 10, 9, 8 go to every word (4 x 1.25%); after place 8
      (bound 7.3e7) about 90% still in, after 7 (3.6e7) 10-30%, after 6 (1.8e7) 1-3%, after 5 under 200 words; reads
      about 7.4%, + table 0.64 + standing amounts 0.57 = about 8.5%.
      Qwen pp: the signed places (0.5%) and up/hole places 13-10 to every word (4 x 0.95%); after place 10 (3.3e8)
      20-40% still in, after 9 (1.6e8) 3-10%, after 8 under 1.5%, after 7 under 200 words; reads about 4.6%, + table
      0.18 = about 4.9%.
      uh reads its four dark top places for every word (GPT-2 +5 points, Qwen +1.9); sd pays about 1.6 times up/hole's
      photons a place and its bound (non-adjacent digits, 2/3 of 2^b a line) is 4/3 of up/hole's.

PREDICTIONS (written before this file ran)
  THE TASK'S: (i) about 15-22% of the plain read, (ii) about 4-8%.
  THE SESSION'S:
  - Exact everywhere: every pile of every word identical to the direct product, in every form, both g, every run;
    (i) and (ii) top 5 (set and order) and race winner = the plain read's at 6,648/6,648 GPT-2 positions (648 refs +
    6,000 judge5 windows) and 613/613 Qwen, = the saved references (plain_refs 648/648, judge5_ref 6,000/6,000, Qwen's
    score rows 613/613 in (i)); Qwen greedy-word positions: race winner = the next greedy word 480/480; the bound held
    for every word at every test; no word set aside at or above the true 5th-best.
  - (i), g = 16, whole read as a share of the plain read:
      GPT-2  uh about 21.5% (20-23.5)   pp about 17.4% (16-19)   sd about 26% (23-28)
      Qwen   uh about 15.8% (14.5-17.5) pp about 14.4% (13-16)   sd about 22% (20-25)
    (i), g = 20: GPT-2 does not pay (uh about 25.6%, pp about 22.4%, sd about 37%); Qwen about even with g = 16 (uh
    about 15.4%, pp about 14.3%, sd about 25%).
  - (ii), g = 16, as a share of the plain read (against the band read's 10.88% / 6.74%):
      GPT-2  pp about 8.5% (7.5-10), about 78% of the band read; uh about 12.6% (11-14); sd about 12% (10.5-14)
      Qwen   pp about 4.9% (4.3-5.8), about 73% of the band read; uh about 6.2% (5.5-7); sd about 7% (6-8)
    (ii), g = 20: GPT-2 pp about 15% (13-17), worse than the band read; Qwen pp about 6.4% (5.5-7.5).
    Survivors (pp, median): GPT-2 all words through place 9; after place 8 most (85-95%); after 7 10-30%; after 6
    1-3%; after 5 under 200; 5-6 at the end. Qwen all through place 11; after 10 20-40%; after 9 3-10%; after 8 under
    1.5%; after 7 under 200.
  - Whole machines (an estimate, labelled; nothing else changed): GPT-2's 46.21% machine reads the top with the
    collapsing read a3 (145,099,607); with the band read it would be 44.93%, with (ii) pp about 44.3%. Qwen's 34.16%
    machine has the band read (302,579,448): with (ii) pp about 33.7%.
  - Controls: seed2 (i) within 0.05 point, (ii) survivors identical at every test and every count within 0.05 point;
    shuffled light (i) within 0.1 point, (ii) within 1 point, moving the same way as the band read's control (GPT-2
    10.81 against 10.88, Qwen 7.76 against 6.81 at the same positions); order-free: every pile identical, every read and
    survivor count identical, the table within 0.01 point, top 5 identical as a set except at an exact tie for 5th
    place (GPT-2 refs position 494: words 655 and 1654), race winner identical.

    python3 groups2/topg.py model=gpt2|qwen set=refs|windows|machine|words g=16|20 run=main|seed2|shuffle|order
                            [forms=uh,sd,pp] [batch=10] [part=i/n]
    -> groups2/out/topg_<model>_<set>_g<g>_<run>[_<forms>][_p<i>of<n>].npz / .json / .log
"""
import os, sys, json, time, ctypes, subprocess, hashlib, resource, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; G2 = f"{HERE}/groups2"; OUT = f"{G2}/out"
ARG = dict(a.split("=") for a in sys.argv[1:])
MODEL, SET, G_, RUN = ARG.get("model", "gpt2"), ARG.get("set", "refs"), int(ARG.get("g", 16)), ARG.get("run", "main")
FORMS = [f for f in ("uh", "sd", "pp") if f in ARG.get("forms", "uh,sd,pp").split(",")]
BATCH = int(ARG.get("batch", 10 if MODEL == "gpt2" else 8)); PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/"))
OUT = ARG.get("out", OUT)                                                       # smoke tests of the code go elsewhere
FI = {"uh": 0, "sd": 1, "pp": 2}; NPL = {"uh": 16, "sd": 16, "pp": 18}; NPMAX = 18

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def sdn(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")
def sd1(v): a = abs(int(v)); return bin(a ^ (3 * a)).count("1")
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def pct(a, b): return f"{100 * a / b:.2f}%" if b else "-"

# ---------- topg.c ----------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{G2}/topg.c", f"{G2}/topg.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
        _lib.top_groups.argtypes = [I] * 6 + [V] * 3 + [I] * 3 + [V] * 4 + [I] + [V] * 4 + [V] * 6 + [V]; _lib.top_groups.restype = None
        _lib.direct_product.argtypes = [I, I, I, V, V, V]; _lib.direct_product.restype = None
        _lib.direct_product_f.argtypes = [I, I, I, V, V, V]; _lib.direct_product_f.restype = None
    return _lib

# ---------- the model: word table, scores, light sets ----------
if MODEL == "gpt2":
    _argv = sys.argv; sys.argv = [sys.argv[0]]; sys.path.insert(0, f"{HERE}/topband"); import topband as TB; sys.argv = _argv
    WT = np.ascontiguousarray(np.asarray(TB.M["head.w"])); assert WT.dtype == np.int16
    BIAS = np.asarray(TB.HEAD_B).astype(np.int64); BIAS_ONES = ones(BIAS); BMAX = int(np.abs(BIAS).max())
    KW = np.zeros(WT.shape[1], np.int64); power, race, top5f, EXP2, ES = TB.power, TB.race, TB.top5, TB.EXP2, TB.ES
    def top5(S): return top5f(S[None])[0]
else:
    _argv = sys.argv; sys.argv = [sys.argv[0]]; sys.path.insert(0, HERE); import qwen_fringes2 as QF; sys.argv = _argv
    WT = QF.member("head.w"); assert WT.dtype == np.int16 and WT.flags["F_CONTIGUOUS"]       # stored word by word
    HE = np.asarray(QF.Mz["head.e"]).astype(np.int64); EMIN = int(HE.min()); KW = HE - EMIN; assert KW.max() <= 3
    power, race, EXP2, ES = QF.power, QF.race, QF.EXP2, QF.ES
    def top5(S): return QF.top5(S)[0]
N, M = WT.shape

def light_set():
    """the light at the top, the references, and the band read's own per-position records on the same light"""
    if MODEL == "gpt2" and SET == "refs":
        Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); X, EX, LS, ROWS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"], Z["rows"]
        REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
        assert np.array_equal(X, REF["x16"][ROWS]) and np.array_equal(EX, REF["ex"][ROWS]) and np.array_equal(LS, REF["LS"][ROWS])
        ref = dict(top5=REF["top5"][ROWS], win=REF["win"][ROWS]); band = [np.load(f"{HERE}/topband/top_refs.npz")]
        if RUN == "shuffle":
            rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])     # topband.py's draw
            band = [np.load(f"{HERE}/topband/top_refs_shuffled.npz")]; ref = None
        sel = np.arange(len(X))
    elif MODEL == "gpt2" and SET == "windows":
        Z = np.load(f"{HERE}/fringes_machine/windows_top_light.npz"); X, EX, LS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"]
        ref = dict(top5=np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5), win=None)
        band = [np.load(f"{HERE}/topband/top_windows_p1of2.npz"), np.load(f"{HERE}/topband/top_windows_p2of2.npz")]
        sel = np.arange(len(X)); assert RUN == "main"
    elif MODEL == "qwen" and SET == "machine":
        Z = np.load(f"{HERE}/qwen_fringes2/machine_top_light.npz"); X, EX, LS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"]
        RF = json.load(open(f"{HERE}/qwen_fringes/ref.json"))["read"]; MT = json.load(open(f"{HERE}/qwen06b/light/meta.json"))
        ref = dict(top5=np.array(RF["top5"]), win=np.array(RF["win"]), sha=RF["sha"], top5c=np.array(MT["top5"]), winc=np.array(MT["win"]), shac=MT["sha"])
        band = [np.load(f"{HERE}/qwen_fringes2/top_machine.npz")]; sel = np.arange(len(X))
        if RUN == "shuffle":
            sel = np.arange(0, len(X), 4); X = np.random.default_rng([2026, 7]).permuted(X[sel], axis=1)   # qwen_fringes.py's control draw
            EX, LS = EX[sel], LS[sel]; band = [np.load(f"{HERE}/qwen_fringes/top_control.npz")]; ref = None
            return X, EX, LS, ref, band, sel
    elif MODEL == "qwen" and SET == "words":
        Z = np.load(f"{HERE}/qwen_fringes2/words_top_light.npz"); W = json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))
        sel = np.array(W["read_rows"]); X, EX, LS = Z["x16"].astype(np.int64)[sel], Z["ex"][sel], Z["LS"][sel]
        ref = dict(top5=np.array(W["top5"]), win=np.array(W["win"]), sha=W["sha"], nxt=np.array(W["next"]))
        band = [np.load(f"{HERE}/qwen_fringes2/top_words.npz")]; assert RUN == "main"
        return X, EX, LS, ref, band, sel
    else:
        raise SystemExit(f"no set {SET} for {MODEL}")
    return X[sel] if MODEL == "qwen" else X, EX, LS, ref, band, sel

def band_records(band):
    """the band read's per-position total and plain read, as its own report counts them"""
    tot, pl = [], []
    for B in band:
        t = B["walk"] + B["writing"] + B["pours"].sum(1) + B["bound"].sum(1) + B["end"]
        if MODEL == "gpt2": t = t + B["standing_written"] + B["standing_poured"]; p = B["plain_meet"] + B["plain_div"] + B["plain_stand"]
        else: p = B["plain_meet"] + B["plain_div"]
        tot.append(t); pl.append(p)
    return np.concatenate(tot).astype(np.int64), np.concatenate(pl).astype(np.int64)

# ---------- grouping ----------
def make_groups(seed, reorder=None):
    order = np.random.default_rng([seed, 99, N, G_]).permutation(N); runs = [order[s:s + G_] for s in range(0, N, G_)]
    if reorder is not None:
        r = np.random.default_rng(reorder); runs = [runs[i] for i in r.permutation(len(runs))]; runs = [r.permutation(u) for u in runs]
    sizes = np.array([len(u) for u in runs], np.int64)
    return np.concatenate(runs).astype(np.int32), np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

def table_coeffs(glines, goff):
    """photons of the tables by the Gray walk's relation: per line, how many times its light value is added.
    half (uh, pp): the group total (lines 1..gk-1 once) + Gray steps over lines 0..gk-2 (line t: 2^(gk-2-t));
    full (sd): Gray steps over all gk lines (line t: 2^(gk-1-t))."""
    ch = np.zeros(N, np.int64); cf = np.zeros(N, np.int64); nh = nf = 0
    for k in range(len(goff) - 1):
        L = glines[goff[k]:goff[k + 1]]; gk = len(L)
        for t in range(gk):
            ch[L[t]] = (1 if t >= 1 else 0) + ((1 << (gk - 2 - t)) if t <= gk - 2 else 0)
            cf[L[t]] = 1 << (gk - 1 - t)
        nh += (gk - 1) + (1 << (gk - 1)) - 1; nf += (1 << gk) - 1
    return ch, cf, nh, nf

# ---------- the table's digits: the per-place rule, the forms' patterns ----------
PL16 = np.arange(16, dtype=np.int64)
def naf_masks(W):
    a = np.abs(W); t = 3 * a; up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = W < 0
    up, dn = np.where(neg, dn, up), np.where(neg, up, dn); assert np.array_equal(up - dn, W); return up, dn

def lit_by_place(CH=4096):
    L = np.zeros(18, np.int64)
    for j0 in range(0, M, CH):
        up, dn = naf_masks(np.asarray(WT[:, j0:j0 + CH], np.int64)); lit = up | dn
        for b in range(18): L[b] += int(((lit >> b) & 1).sum())
    assert not L[16:].any(); return L

def expand_pp(Wc, ppmask):
    """pp's written form: bottom place to top, a binary digit at an up/hole place, a signed digit (2 - (r mod 4) where r
    is odd) elsewhere, offset C = 2^(top up/hole place). Returns the binary digits E and the signed digits (t, j, b, s)."""
    hi = ppmask.bit_length() - 1; C = 1 << hi; r = Wc + C; E = np.zeros(Wc.shape, np.int64); parts = []
    for b in range(NPMAX):
        isup = b < 16 and (ppmask >> b) & 1
        odd = (r & 1) == 1
        if isup:
            e = odd.astype(np.int64); E |= e << b; r = (r - e) >> 1
        else:
            d = np.where(odd, 2 - (r & 3), 0); nz = np.nonzero(d)
            if len(nz[0]): parts.append((nz[0], nz[1], np.full(len(nz[0]), b), d[nz]))
            r = (r - d) >> 1
    assert not r.any(), "digits left above place 17"
    back = E - C
    if parts:
        t, j, b, s = (np.concatenate(z) for z in zip(*parts)); np.add.at(back, (t, j), s << b)
    else:
        t = j = b = s = np.zeros(0, np.int64)
    assert np.array_equal(back, Wc), "pp's digits do not give the table back"
    return E, (t, j, b, s)

def pack(src, dt):
    """src (gk, C) bit fields -> (C, 16): bit t of place b = bit b of src[t]"""
    bits = (src[:, :, None] >> PL16[None, None, :]) & 1
    return (bits << np.arange(src.shape[0], dtype=np.int64)[:, None, None]).sum(0).astype(dt)

def build(glines, goff, forms, ppmask, wide, perm=None, CH=4096):
    """every form's patterns (G, M, 16), word chunk by word chunk; pp's signed digits by word (CSR). perm: the words
    relabelled (order-free run): column j of the stored table is word perm[j]."""
    G = len(goff) - 1; dt = np.uint32 if wide else np.uint16; P = {}
    for f in forms:
        if f == "sd": P["u"] = np.zeros((G, M, 16), dt); P["d"] = np.zeros((G, M, 16), dt)
        else: P[f] = np.zeros((G, M, 16), dt)
    sl = []; fmt_sd = np.zeros(16, bool); fmt_pp = np.zeros(NPMAX, bool)
    groups = [glines[goff[k]:goff[k + 1]] for k in range(G)]
    for j0 in range(0, M, CH):
        cols = np.arange(j0, min(M, j0 + CH)) if perm is None else perm[j0:j0 + CH]
        Wc = np.asarray(WT[:, cols], np.int64); src = {}
        if "uh" in forms: src["uh"] = Wc + 32768
        if "sd" in forms:
            src["u"], src["d"] = naf_masks(Wc); orall = int(np.bitwise_or.reduce((src["u"] | src["d"]).ravel()))
            fmt_sd |= ((orall >> PL16) & 1).astype(bool)
        if "pp" in forms:
            src["pp"], (t, j, b, s) = expand_pp(Wc, ppmask); sl.append((t, j + j0, b, s)); fmt_pp[np.unique(b)] = True
        for key, S in src.items():
            for k, L in enumerate(groups): P[key][k, j0:j0 + len(cols)] = pack(S[L], dt)
        del src, Wc
    if "pp" in forms:
        t, j, b, s = (np.concatenate(z) for z in zip(*sl)); o = np.argsort(j, kind="stable")
        P["toff"] = np.concatenate([[0], np.cumsum(np.bincount(j, minlength=M))]).astype(np.int64)
        P["tline"] = np.ascontiguousarray(t[o], np.int32); P["tplace"] = np.ascontiguousarray(b[o], np.int8)
        P["tsign"] = np.ascontiguousarray(s[o], np.int8)
        for bb in range(16):
            if (ppmask >> bb) & 1: fmt_pp[bb] = True
    P["fmt"] = dict(uh=np.ones(16, bool), sd=fmt_sd, pp=fmt_pp)
    return P

def bound_per_line(form, ppmask):
    """B_f(b): the most the form's places below b can add, per unit of light, at the doubled scale"""
    tn = lambda n: ((1 << (n + 1)) - 1) // 3 if n > 0 else 0
    out = []
    for b in range(NPMAX + 1):
        if form == "uh": out.append((1 << min(b, 16)) - 1)
        elif form == "sd": out.append(2 * tn(min(b, 16)))
        else:
            v = sum(1 << q for q in range(min(b, 16)) if (ppmask >> q) & 1)
            runs, q = [], 0
            while q < NPMAX:                                                   # the signed runs [a, c)
                if q < 16 and (ppmask >> q) & 1: q += 1; continue
                a = q
                while q < NPMAX and not (q < 16 and (ppmask >> q) & 1): q += 1
                runs.append((a, q))
            for a, c in runs:
                if a < b: v += 2 * (1 << a) * tn(min(c, b) - a)
            out.append(v)
    return out

def run_kernel(Xb, glines, goff, P, ppmask, wide):
    Xb = np.ascontiguousarray(Xb, np.int32); B = len(Xb); G = len(goff) - 1; gmax = int(np.diff(goff).max())
    A = {f: np.empty((B, M, NPL[f]), np.int32) for f in FORMS}; H = {f: np.empty((B, M, NPL[f]), np.uint16) for f in FORMS}
    NA = np.zeros((B, 3, NPMAX), np.int64)
    pt = lambda k: P[k].ctypes.data if k in P else None
    g = lambda d, f: d[f].ctypes.data if f in d else None
    lib().top_groups(B, N, M, G, gmax, int(wide), Xb.ctypes.data, glines.ctypes.data, goff.ctypes.data,
                     int("uh" in FORMS), int("sd" in FORMS), int("pp" in FORMS), pt("uh"), pt("u"), pt("d"), pt("pp"), int(ppmask),
                     pt("toff"), pt("tline"), pt("tplace"), pt("tsign"), g(A, "uh"), g(A, "sd"), g(A, "pp"), g(H, "uh"), g(H, "sd"),
                     g(H, "pp"), NA.ctypes.data)
    return A, H, NA

def direct(Xb, Wt):
    Xb = np.ascontiguousarray(Xb, np.int32); out = np.empty((len(Xb), M), np.int64)
    if Wt.flags["C_CONTIGUOUS"]: lib().direct_product(len(Xb), N, M, Xb.ctypes.data, Wt.ctypes.data, out.ctypes.data)
    else:
        assert Wt.flags["F_CONTIGUOUS"]; lib().direct_product_f(len(Xb), N, M, Xb.ctypes.data, Wt.ctypes.data, out.ctypes.data)
    return out

# ---------- one position: the plain read (check path), (i) and (ii) for every form ----------
KCORR = {}
def position(q, x, ex, ls, PILE, ctx, bad, rec):
    bias, bias1, KWp = ctx["bias"], ctx["bias1"], ctx["kw"]
    def nb(f, k, v): bad[f][k] = bad[f].get(k, 0) + int(v)
    X = int(x.sum()); S1 = int(np.abs(x).sum()); sx = sdn(x); light_sum = int(sx[1:].sum())   # N - 1 adds a sum
    if MODEL == "gpt2":
        Fp = int((TB.HEAD_E[0] + ex + ES) * 65536 - ls); e = int(EXP2[Fp & 0xFFFF]); ge = ones1(e)
        F2 = -Fp; I = Fp >> 16; s = 16 - I; I2 = F2 >> 16; e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - I2; ge2 = ones1(e2)
        beta = power(bias, F2); twos = Fraction(2) ** s; twot = Fraction(2) ** (s + s2)
        gg = Fraction(1, 2) + BMAX * twos * abs(e * e2 - twot) / (e * twot)
        gamma = int(-(-gg.numerator // gg.denominator)); delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator))))
        mu = 2 * gamma + delta
        S = power(PILE, Fp) + bias
        rec["plain_meet"][q] = int(ones(x) @ ctx["pph"]); rec["plain_div"][q] = int(ones(PILE).sum()) * ge
        rec["plain_stand"][q] = int(bias1.sum()); stand = int(bias1.sum()) * ge2 + int(ones(beta).sum())
    else:
        Fp = int((EMIN + ex + ES) * 65536 - ls); e = int(EXP2[Fp & 0xFFFF]); ge = ones1(e); s = 16 - (Fp >> 16)
        twos = Fraction(2) ** s; delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator)))); mu = 2 * 1 + delta
        S = power(PILE << KWp, Fp)
        rec["plain_meet"][q] = int(ones(x) @ ctx["pph"]); rec["plain_div"][q] = int(ones(PILE).sum()) * ge; rec["plain_stand"][q] = 0
        stand = 0; rec["sha"][q] = hashlib.sha256(np.ascontiguousarray(S[ctx["inv"]], np.int64).tobytes()).hexdigest()
    NW = M; s5 = np.partition(S, NW - 5)[NW - 5]
    rec["ptop5"][q] = ctx["lab"][top5(S)]; rec["pwin"][q] = ctx["lab"][race(S)]
    rec["X"][q] = X; rec["S1"][q] = S1; rec["mu"][q] = mu
    tab_h, tab_f = int(sx @ ctx["ch"]), int(sx @ ctx["cf"])
    for f in FORMS:
        A, H = ctx["A"][f], ctx["H"][f]; kc = KCORR[f]; npl = NPL[f]
        Cb = A.astype(np.int64) << np.arange(npl, dtype=np.int64)[None, :]
        CUM = np.cumsum(Cb[:, ::-1], axis=1)[:, ::-1]                             # CUM[:, b] = the read part, places >= b
        Dbl = CUM[:, 0] + kc * X
        nb(f, "piles differing from the direct product (words)", (Dbl != 2 * PILE).sum())
        pile = Dbl >> 1
        tab = tab_f if f == "sd" else tab_h
        xs = (light_sum + NW * sd1(kc * X)) if kc else 0
        reads = int(H.sum(dtype=np.int64))
        r = rec[f]
        r["table"][q] = tab; r["i_reads"][q] = reads; r["i_light"][q] = xs
        r["i_total"][q] = tab + reads + xs + rec["plain_div"][q] + rec["plain_stand"][q]
        r["i_byplace"][q, :npl] = H.sum(0, dtype=np.int64); r["i_adds"][q] = ctx["NA"][FI[f]]
        Sg = power(pile, Fp) + bias if MODEL == "gpt2" else power(pile << KWp, Fp)
        nb(f, "(i) scores differing from the plain read (words)", (Sg != S).sum())
        r["i_top5"][q] = ctx["lab"][top5(Sg)]; r["i_win"][q] = ctx["lab"][race(Sg)]
        r["digest"][q] = digest(pile[ctx["inv"]])
        # ---- (ii): from the top place down, each place into the words still in; the band read's set-aside ----
        places = [b for b in range(npl - 1, -1, -1) if ctx["fmt"][f][b]]
        BND = ctx["bnd"][f]; alive = np.arange(NW); rd = 0; bph = 0; cmpc = 0
        base = 2 * beta if MODEL == "gpt2" else None
        for i, b in enumerate(places):
            hb = H[alive, b].astype(np.int64); rd += int(hb.sum()); r["ii_byplace"][q, b] = int(hb.sum())
            if i == len(places) - 1: break
            R2 = BND[b] * S1                                                       # unread: the places below b
            nb(f, "(ii) bound broken (any word, check)", (np.abs(CUM[:, 0] - CUM[:, b]) > R2).sum())
            na = len(alive)
            if MODEL == "gpt2":
                V = CUM[alive, b] + base[alive]; v5 = np.partition(V, na - 5)[na - 5]
                thr = 2 * R2 + 2 * mu; keep = V > v5 - thr
                bph += ones1(S1) * ones1(BND[b]) + ones1(2 * mu) + ones1(thr); cmpc += 2 * na
            else:
                K = KWp[alive]; Vs = (CUM[alive, b] + kc * X) << K; lows = []; ph = ones1(S1) * ones1(BND[b]); cm = 2 * na
                for k in range(4):
                    Vk = Vs[K == k]
                    if len(Vk) == 0: continue
                    m = min(5, len(Vk)); t5 = np.partition(Vk, len(Vk) - m)[len(Vk) - m:]
                    lows += list(t5 - (R2 << k)); ph += m * (ones1(R2 << k) + (sd1(kc * X << k) if kc else 0))
                lows = np.sort(np.array(lows, np.int64)); L5 = int(lows[-5]); cm += len(lows); THR = np.zeros(4, np.int64)
                for k in range(4):
                    if (K == k).any():
                        thr = (R2 << k) + 2 * mu; THR[k] = L5 - thr; ph += ones1(2 * mu) + ones1(thr) + (sd1(kc * X << k) if kc else 0)
                keep = Vs > THR[K]; bph += ph; cmpc += cm
            nb(f, "(ii) a word set aside that is not below the 5th-best", (S[alive[~keep]] >= s5).sum())
            alive = alive[keep]; r["ii_surv"][q, b] = len(alive)
        pa = pile[alive]
        if MODEL == "gpt2":
            sc = power(pa, Fp) + bias[alive]
            end = (sd1(kc * X) * len(alive) if kc else 0) + int(ones(beta[alive]).sum()) + int(ones(pa).sum()) * ge + int(bias1[alive].sum())
        else:
            Ve = pa << KWp[alive]; sc = power(Ve, Fp); end = (sd1(kc * X) * len(alive) if kc else 0) + int(ones(Ve).sum()) * ge
        nb(f, "(ii) words read to the end vs the plain read", (sc != S[alive]).sum())
        Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
        r["ii_top5"][q] = ctx["lab"][top5(Sp)]; r["ii_win"][q] = ctx["lab"][race(Sp)]
        lsum = light_sum * (2 if kc else 1)                                        # S1, and X when the form has an offset
        r["ii_reads"][q] = rd; r["ii_light"][q] = lsum; r["ii_bound"][q] = bph; r["ii_cmp"][q] = cmpc; r["ii_end"][q] = end
        r["ii_stand"][q] = stand; r["ii_total"][q] = tab + lsum + rd + bph + end + stand; r["ii_last"][q] = len(alive)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    tag = f"topg_{MODEL}_{SET}_g{G_}_{RUN}" + ("" if FORMS == ["uh", "sd", "pp"] else "_" + "-".join(FORMS)) + ("" if PN == 1 else f"_p{PI}of{PN}")
    LOG = open(f"{OUT}/{tag}.log", "w")
    def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    X, EX, LS, ref, band, sel = light_set()
    cut = np.linspace(0, len(X), PN + 1).astype(np.int64); part = np.arange(cut[PI - 1], cut[PI])
    X, EX, LS = X[part], np.asarray(EX)[part], np.asarray(LS)[part]; NP_ = len(X)
    btot, bpl = band_records(band); btot, bpl = btot[part], bpl[part]
    seed = 2 if RUN == "seed2" else 1
    glines, goff = make_groups(seed, reorder=[4242, N, G_] if RUN == "order" else None)
    G = len(goff) - 1; wide = int(np.diff(goff).max()) > 16
    perm = np.random.default_rng([4343, M]).permutation(M) if RUN == "order" else None
    log(f"GROUP PILES AT THE TOP TABLE: {MODEL} {SET}, g = {G_} ({G} groups, the last of {int(np.diff(goff)[-1])} lines), run {RUN} "
        f"(grouping seed {seed}{', groups and lines reordered, words relabelled' if RUN == 'order' else ''}), forms {FORMS}, "
        f"{NP_} positions (part {PI}/{PN}); table {N} lines x {M} words")
    Lb = lit_by_place(); U = G * M; ppmask = sum(1 << b for b in range(16) if not Lb[b] < U)
    log(f"  the table's lit signed digits by place (plate only): {Lb[:16].tolist()}; U = G x M = {U:,}")
    log(f"  pp's rule: signed-digit places {[b for b in range(16) if Lb[b] < U]}, up/hole places {[b for b in range(16) if (ppmask >> b) & 1]}")
    hi = ppmask.bit_length() - 1
    KCORR.update(uh=-1, sd=0, pp=ppmask - 2 * (1 << hi))                         # K - 2C
    tb = time.time(); P = build(glines, goff, FORMS, ppmask, wide, perm=perm)
    log(f"  patterns built ({time.time() - tb:.0f}s, {rss()} MB); pp signed digits {len(P.get('tline', [])):,}; "
        f"form places: " + ", ".join(f"{f} {np.flatnonzero(P['fmt'][f]).tolist()}" for f in FORMS))
    ch, cf, nh, nf = table_coeffs(glines, goff)
    bnd = {f: bound_per_line(f, ppmask) for f in FORMS}
    log("  shared bound per unit of light (doubled scale) by place: " + "; ".join(f"{f} " + ",".join(str(v) for v in bnd[f][:NPL[f] + 1]) for f in FORMS))
    Wt = WT if perm is None else (np.ascontiguousarray if WT.flags["C_CONTIGUOUS"] else np.asfortranarray)(np.asarray(WT)[:, perm])
    lab = np.arange(M) if perm is None else perm; inv = np.argsort(lab)
    ctx = dict(bias=BIAS[lab] if MODEL == "gpt2" else None, bias1=BIAS_ONES[lab] if MODEL == "gpt2" else None, kw=KW[lab],
               ch=ch, cf=cf, bnd=bnd, fmt=P["fmt"], lab=lab, inv=inv)
    pph = np.zeros(N, np.int64)                                                 # the plain read's 1s per line
    for j0 in range(0, M, 8192): pph += ones(np.asarray(Wt[:, j0:j0 + 8192])).sum(1)
    ctx["pph"] = pph
    Z = lambda *s: np.zeros(s, np.int64)
    rec = dict(plain_meet=Z(NP_), plain_div=Z(NP_), plain_stand=Z(NP_), X=Z(NP_), S1=Z(NP_), mu=Z(NP_), ptop5=Z(NP_, 5), pwin=Z(NP_),
               sha=[""] * NP_)
    for f in FORMS:
        rec[f] = dict(table=Z(NP_), i_reads=Z(NP_), i_light=Z(NP_), i_total=Z(NP_), i_byplace=Z(NP_, NPMAX), i_adds=Z(NP_, NPMAX),
                      i_top5=Z(NP_, 5), i_win=Z(NP_), digest=[""] * NP_, ii_reads=Z(NP_), ii_light=Z(NP_), ii_bound=Z(NP_),
                      ii_cmp=Z(NP_), ii_end=Z(NP_), ii_stand=Z(NP_), ii_total=Z(NP_), ii_surv=np.full((NP_, NPMAX), -1, np.int64),
                      ii_byplace=Z(NP_, NPMAX), ii_top5=Z(NP_, 5), ii_win=Z(NP_), ii_last=Z(NP_))
    bad = {f: {} for f in FORMS}; tl = time.time()
    for c0 in range(0, NP_, BATCH):
        Xb = X[c0:c0 + BATCH]; Bn = len(Xb)
        A, H, NA = run_kernel(Xb, glines, goff, P, ppmask, wide)
        PILE = direct(Xb, Wt)                                                      # CHECK PATH
        for p in range(Bn):
            ctx["A"] = {f: A[f][p] for f in FORMS}; ctx["H"] = {f: H[f][p] for f in FORMS}; ctx["NA"] = NA[p]
            position(c0 + p, Xb[p], int(EX[c0 + p]), int(LS[c0 + p]), PILE[p], ctx, bad, rec)
        del A, H, NA, PILE
        if (c0 // BATCH) % 10 == 0 or c0 + BATCH >= NP_:
            el = time.time() - tl
            log(f"    {min(NP_, c0 + BATCH)}/{NP_} positions ({el:.0f}s, {rss()} MB)")
    # ---------- summary ----------
    pl = rec["plain_meet"] + rec["plain_div"] + rec["plain_stand"]; PL = int(pl.sum()); PM = int(rec["plain_meet"].sum())
    log(f"\nTHE PLAIN READ (mean a position): {PL / NP_:,.0f} (meetings {PM / NP_:,.0f}); its record in the band read's own files "
        f"{int(bpl.sum()) / NP_:,.0f}: positions equal {int((pl == bpl).sum())}/{NP_}")
    log(f"THE BAND READ on the same light: {int(btot.sum()) / NP_:,.0f} = {pct(int(btot.sum()), PL)} of the plain read")
    labf = {"uh": "up/hole", "sd": "signed digits", "pp": "per place"}
    for f in FORMS:
        r = rec[f]; I = int(r["i_total"].sum()); II = int(r["ii_total"].sum())
        log(f"\n[{f}] {labf[f]}: checks: " + ", ".join(f"{k} {v}" for k, v in sorted(bad[f].items())))
        same_i = int((r["i_top5"] == rec["ptop5"]).all(1).sum()); same_ii = int((r["ii_top5"] == rec["ptop5"]).all(1).sum())
        log(f"  EXACT: (i) top 5 = the plain read's {same_i}/{NP_}, race winner {int((r['i_win'] == rec['pwin']).sum())}/{NP_}; "
            f"(ii) top 5 {same_ii}/{NP_}, race winner {int((r['ii_win'] == rec['pwin']).sum())}/{NP_}")
        if ref is not None:
            msg = f"  against the saved references: (i) top 5 {int((r['i_top5'] == ref['top5'][part]).all(1).sum())}/{NP_}, (ii) top 5 {int((r['ii_top5'] == ref['top5'][part]).all(1).sum())}/{NP_}"
            if ref.get("win") is not None: msg += f"; race winner (i) {int((r['i_win'] == ref['win'][part]).sum())}/{NP_}, (ii) {int((r['ii_win'] == ref['win'][part]).sum())}/{NP_}"
            if "sha" in ref: msg += f"; score rows (i) {sum(a == b for a, b in zip(rec['sha'], [ref['sha'][i] for i in part]))}/{NP_}"
            if "shac" in ref: msg += f"; capture's record: score rows {sum(a == b for a, b in zip(rec['sha'], [ref['shac'][i] for i in part]))}/{NP_}, top 5 (ii) {int((r['ii_top5'] == ref['top5c'][part]).all(1).sum())}/{NP_}, race (ii) {int((r['ii_win'] == ref['winc'][part]).sum())}/{NP_}"
            if "nxt" in ref: msg += f"; race winner = the next greedy word (i) {int((r['i_win'] == ref['nxt'][part]).sum())}/{NP_}, (ii) {int((r['ii_win'] == ref['nxt'][part]).sum())}/{NP_}"
            log(msg)
        tab = int(r["table"].sum()); rd = int(r["i_reads"].sum()); xl = int(r["i_light"].sum())
        log(f"  (i)  every word, all places: {I / NP_:,.0f} = {pct(I, PL)} of the plain read (table {pct(tab, PL)}, reads {pct(rd, PL)}, "
            f"X {pct(xl, PL)}, divides and standing amounts {pct(int((rec['plain_div'] + rec['plain_stand']).sum()), PL)}); "
            f"meetings part {pct(tab + rd + xl, PM)} of the plain meetings")
        log(f"       reads by place, share of the plain read: " + " ".join(f"{b}:{100 * r['i_byplace'][:, b].sum() / PL:.2f}" for b in range(NPL[f] - 1, -1, -1) if r['i_byplace'][:, b].any()))
        log(f"  (ii) from the top place down, set aside after every place: {II / NP_:,.0f} = {pct(II, PL)} of the plain read; "
            f"{pct(II, int(btot.sum()))} of the band read (band read {pct(int(btot.sum()), PL)}); below the band read at {int((r['ii_total'] < btot).sum())}/{NP_} positions")
        log(f"       table {pct(tab, PL)}, light sums {pct(int(r['ii_light'].sum()), PL)}, reads {pct(int(r['ii_reads'].sum()), PL)}, "
            f"bound {pct(int(r['ii_bound'].sum()), PL)}, standing amounts {pct(int(r['ii_stand'].sum()), PL)}, read to the end {pct(int(r['ii_end'].sum()), PL)}; "
            f"compares (a count) {r['ii_cmp'].sum() / NP_:,.0f} a position")
        sv = r["ii_surv"]
        log("       words still in after each place (median / mean / least / most) and that place's reads (share of the plain read):")
        for b in range(NPL[f] - 1, -1, -1):
            if not P["fmt"][f][b]: continue
            col = sv[:, b]; rb = 100 * r["ii_byplace"][:, b].sum() / PL
            if (col >= 0).all():
                log(f"         place {b:2d}: {int(np.median(col)):7d} {col.mean():10.1f} {col.min():7d} {col.max():7d}   reads {rb:6.3f}%")
            else:
                log(f"         place {b:2d}: (read to the end)                         reads {rb:6.3f}%   words read to the end: median {int(np.median(r['ii_last']))}, most {int(r['ii_last'].max())}")
    np.savez(f"{OUT}/{tag}.npz", part=part, sel=np.asarray(sel)[part], band_total=btot, band_plain=bpl,
             **{k: v for k, v in rec.items() if k not in FORMS and k != "sha"},
             **{f"{f}__{k}": np.asarray(v) for f in FORMS for k, v in rec[f].items() if k != "digest"})
    json.dump(dict(model=MODEL, set=SET, g=G_, run=RUN, forms=FORMS, part=[PI, PN], n=NP_, G=G, ppmask=ppmask, kcorr=KCORR,
                   lit_by_place=Lb.tolist(), bad=bad, sha=rec["sha"], digest={f: rec[f]["digest"] for f in FORMS},
                   plain=PL, band=int(btot.sum()), i_total={f: int(rec[f]["i_total"].sum()) for f in FORMS},
                   ii_total={f: int(rec[f]["ii_total"].sum()) for f in FORMS}, table_adds=dict(half=nh, full=nf)),
              open(f"{OUT}/{tag}.json", "w"))
    log(f"\n({time.time() - t0:.0f}s, max memory {rss()} MB)")
