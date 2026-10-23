"""VERIFY base2_entry.py AGAINST base2.py: every pile, the stream, and the words.

Every product base2.py makes (each bank's pile, the arrival's scores and shares x values, the top read's piles at
every position) is recorded in order, both ways; the entry machine's qkv.0 pile is recorded at the same point. A guard
stops the run if the entry machine ever meets the qkv.0 plate with the full light.

PREDICTIONS (written before the run):
  A. 30 refs texts (ids + ref, every position): every pile, the stream after every half-layer, the top piles and
     words at every position identical, 30/30. The same with the literal pour (MEET = "lit") and with every carry
     forced through the float64 fallback (LIT_MAX = 0). Order-free: the carry's lines poured one at a time in a
     random order and the parts added in a random order give the same pile at every position.
  C. 2,000 random (word, position) pairs over the whole vocabulary and all 1,024 positions: every qkv.0 pile and
     value identical. Every stored row (170,315 rows: all six tables) used at a pair that needs it: identical.
     1,000 single-token and 1,000 two-token sequences of random words: every pile, stream and word identical.
     CONTROL: reading a random other word's stored row: no pile identical (unless two words' rows were equal).
  R. Independent rebuild: 300 rows of each table recomputed in int64 whole-number arithmetic (no float) with a
     separately written rounding: identical. Every file's sha256 and size equal the manifest.
  B. 300 judge windows: every pile and the stream identical, 300/300.
  J. judge5.judge(forward=base2_entry.scores_all): 6000/6000 in the top 5, 6000/6000 the same top word.
    python3 entry/verify.py parts=A        (parts: A, C, R, B, J; windows=0:150 for part of B)
"""
import sys, os, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARGS = dict(a.split("=") for a in sys.argv[1:])
PARTS = ARGS.get("parts", "A,C,R,B,J").split(","); WIN = [int(x) for x in ARGS.get("windows", "0:300").split(":")]
import base2_entry as E
J = E.J
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def say(*a): print(*a, f"   ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)
WTE = J.M["wte"]; WPE = J.M["wpe"]; WQ = J.W["qkv.0"][0]; P16 = J.M["qkv.0.w"].astype(np.int64)

# ---------- recording every product; the guard ----------
REC = []; GUARD = [False]; _product = J.product; GUARD_HITS = [0]
def rec_product(x16, w16):
    if GUARD[0] and w16 is WQ:
        GUARD_HITS[0] += 1; raise RuntimeError("the entry machine met the qkv.0 plate with the full light")
    p = _product(x16, w16); REC.append(p); return p
J.product = rec_product
E.ON_PILE = lambda p: REC.append(p)

def base2_stream(ids, wte, rec):                     # base2.py's forward, line for line, the stream kept
    ids = np.asarray(ids, np.int64); R = wte[ids] + WPE[:len(ids)]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}"); rec.append(R)
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}"); rec.append(R)
    return R
def top_all(R):                                      # base2's read at the top, at every position
    w, we, b = J.W["head"]; x16, ex, LS = J.gauge_log(R)
    S = J.power(J.product(x16, w), (we[None, :] + ex[:, None] + J.ES) * 65536 - LS[:, None]) + b
    return S, np.array([J.race(S[i]) for i in range(len(S))])
def run_both(ids):
    REC.clear(); sb = []; Rb = base2_stream(ids, WTE, sb); Sb, wb = top_all(Rb); rb = list(REC)
    REC.clear(); se = []; GUARD[0] = True
    try: Re = E.stream(ids, WTE, se)
    finally: GUARD[0] = False
    Se, we_ = top_all(Re); re_ = list(REC); REC.clear()
    piles = len(rb) == len(re_) and all(np.array_equal(a, b) for a, b in zip(rb, re_))
    stream = len(sb) == len(se) and all(np.array_equal(a, b) for a, b in zip(sb, se))
    return piles, stream, np.array_equal(Sb, Se) and np.array_equal(wb, we_), len(rb)
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]

def part_A():                                        # the 30 refs texts, three ways of meeting the carry; order-free
    GUARD[0] = True; fired = False
    try: J.gauged_bank(WTE[[464]] + WPE[:1], "qkv.0")
    except RuntimeError: fired = True
    GUARD[0] = False; GUARD_HITS[0] = 0; REC.clear()
    say(f"guard self-test: base2's own qkv.0 bank, run under the guard, is stopped: {fired}")
    for how, litmax in (("narrow", 512), ("lit", 512), ("narrow", 0)):
        E.MEET, E.LIT_MAX = how, litmax; ok = [0, 0, 0]; npiles = 0
        for ids in TEXTS:
            p, s, w, n = run_both(ids); ok[0] += p; ok[1] += s; ok[2] += w; npiles += n
        say(f"A. 30 refs texts ({sum(map(len, TEXTS))} positions), carry met '{how}'{' (every position through the float64 fallback)' if litmax == 0 else ''}: "
            f"every pile identical {ok[0]}/30 ({npiles:,} piles each way), stream {ok[1]}/30, top piles and words at every position {ok[2]}/30")
    E.MEET, E.LIT_MAX = "narrow", 512
    rng = np.random.default_rng(11); ok = tr = 0
    for ids in TEXTS:
        ids = np.asarray(ids); A = WTE[ids]; B = WPE[:len(ids)]; x16, ex, LS = J.gauge_log(A + B)
        REC.clear(); want = J.product(x16, WQ); REC.clear()
        for i in range(len(ids)):
            s = int(ex[i] + J.ES)
            if s == 4: parts = [E.ZP0.read([ids[i]])[0]]
            elif s == 0: parts = [E.Z1023.read([ids[i]])[0]]
            else:
                c = x16[i] - J.rshift(A[i], s) - J.rshift(B[i], s); acc = np.zeros(2304, np.int64)
                for ln in rng.permutation(768):
                    if c[ln]: acc = acc + c[ln] * P16[ln]
                parts = [E.T[s].read([ids[i]])[0], E.P[s].read([i])[0], acc]
            tot = np.zeros(2304, np.int64)
            for k in rng.permutation(len(parts)): tot = tot + parts[k]
            ok += np.array_equal(tot, want[i]); tr += 1
    say(f"   order-free: carry lines poured one at a time in a random order, parts added in a random order: identical {ok}/{tr} positions")

def bank_both(w, p, words_read=None):                # the qkv.0 bank on single light rows, both ways
    w = np.asarray(w, np.int64); p = np.asarray(p, np.int64); A = WTE[w]; B = WPE[p]; R = A + B
    REC.clear(); qb = J.gauged_bank(R, "qkv.0"); pb = REC[-1]; REC.clear(); qe = None
    GUARD[0] = True
    try:
        if words_read is None: qe = E.entry_bank(R, A, B, w, p); pe = REC[-1]
        else: x16, ex, LS = J.gauge_log(R); pe = E.entry_pile(x16, ex, A, B, words_read, p)
    finally: GUARD[0] = False
    REC.clear(); x16, ex, _ = J.gauge_log(R)
    return (pb == pe).all(1), (None if qe is None else (qb == qe).all(1)), ex + J.ES
def shifts(m):                                       # s = bit length of the largest size - 15, floored at 0
    s = np.zeros(m.shape, np.int64)
    for k in range(15, 22): s += (m >= (1 << k))
    return s
def part_C():
    rng = np.random.default_rng(20261005)
    w2k = rng.integers(0, 50257, 2000); p2k = rng.integers(0, 1024, 2000)
    got = [bank_both(w2k[i:i + 500], p2k[i:i + 500]) for i in range(0, 2000, 500)]
    same_p, same_q, s2k = (np.concatenate([g[k] for g in got]) for k in range(3))
    say(f"C. 2,000 random (word, position) pairs (words {w2k.min()}..{w2k.max()}, positions {p2k.min()}..{p2k.max()}, "
        f"{len(set(w2k.tolist()))} distinct words, {len(set(p2k.tolist()))} distinct positions; shifts "
        f"{dict(zip(*[x.tolist() for x in np.unique(s2k, return_counts=True)]))}): "
        f"qkv.0 pile identical {int(same_p.sum())}/2000, qkv.0 values identical {int(same_q.sum())}/2000")
    stored_ids = {0: np.load(f"{HERE}/entry/Z1023_ids.npy"), 1: np.load(f"{HERE}/entry/T1_ids.npy"),
                  2: np.load(f"{HERE}/entry/T2_ids.npy"), 4: np.load(f"{HERE}/entry/ZP0_ids.npy")}
    sh = np.array([w2k[i] for i in range(2000)])
    for i in range(2000):                            # another word that has a stored row for this pair's shift
        while sh[i] == w2k[i]: sh[i] = rng.choice(stored_ids[int(s2k[i])])
    same_c = np.concatenate([bank_both(w2k[i:i + 500], p2k[i:i + 500], words_read=sh[i:i + 500])[0] for i in range(0, 2000, 500)])
    say(f"   CONTROL, a random other word's stored row read (same light, same carry): pile identical {int(same_c.sum())}/2000")
    # every stored row, at a pair that needs it
    S = np.load(f"{HERE}/entry/survey_s.npz"); wmask, pmask = S["wmask"], S["pmask"]
    wte32 = WTE.astype(np.int32); wpe32 = WPE.astype(np.int32)
    pos_for = {1: np.full(50257, -1), 2: np.full(50257, -1)}
    for i in range(0, 50257, 8):                     # for each word: a random position where it takes s = 1, and s = 2
        s = shifts(np.abs(wte32[i:i + 8, None, :] + wpe32[None, :, :]).max(2))
        for j in range(s.shape[0]):
            for v in (1, 2):
                hit = np.flatnonzero(s[j] == v)
                if len(hit): pos_for[v][i + j] = rng.choice(hit)
    word_for = {1: np.full(1024, -1), 2: np.full(1024, -1)}
    for v in (1, 2):                                 # for each position: a random word with which it takes s = v
        for p in np.flatnonzero((pmask >> v) & 1):
            while word_for[v][p] < 0:
                cand = rng.integers(0, 50257, 256); hit = np.flatnonzero(shifts(np.abs(wte32[cand] + wpe32[p][None, :]).max(1)) == v)
                if len(hit): word_for[v][p] = cand[hit[0]]
    del wte32
    t1w = np.flatnonzero((wmask >> 1) & 1); p2 = np.flatnonzero((pmask >> 2) & 1); p1 = np.flatnonzero((pmask >> 1) & 1)
    cover = {"ZP0": (np.arange(50257), np.zeros(50257, np.int64)),
             "Z1023": (np.flatnonzero(wmask & 1), np.full(int((wmask & 1).sum()), 1023)),
             "T2": (np.arange(50257), pos_for[2]), "T1": (t1w, pos_for[1][t1w]),
             "P2": (word_for[2][p2], p2), "P1": (word_for[1][p1], p1)}
    want_s = {"ZP0": 4, "Z1023": 0, "T2": 2, "T1": 1, "P2": 2, "P1": 1}
    def remap():                                     # fresh maps, so the pages already read are let go (memory under 3 GB)
        E.ZP0, E.Z1023 = E.Table("ZP0"), E.Table("Z1023"); E.T = {1: E.Table("T1"), 2: E.Table("T2")}; E.P = {1: E.Table("P1"), 2: E.Table("P2")}
    tot_rows = tot_same = 0
    for name, (ws, ps) in cover.items():
        assert (ws >= 0).all() and (ps >= 0).all(); n_same = 0; s_ok = True
        for i in range(0, len(ws), 1024):
            sp, sq, sv = bank_both(ws[i:i + 1024], ps[i:i + 1024]); n_same += int((sp & sq).sum()); s_ok &= bool((sv == want_s[name]).all())
            remap()
        tot_rows += len(ws); tot_same += n_same
        say(f"   every row of {name:5s} ({len(ws):6,} rows, each at a pair whose shift is {want_s[name]}: {s_ok}): pile and value identical {n_same:,}/{len(ws):,}")
    say(f"   every stored row: {tot_same:,}/{tot_rows:,} identical")
    ok = [0, 0, 0]
    for k in range(2000):                            # sequences of one and two random words, the whole machine both ways
        ids = rng.integers(0, 50257, 1 if k < 1000 else 2).tolist()
        p, s, w, _ = run_both(ids); ok[0] += p; ok[1] += s; ok[2] += w
        if k in (999, 1999):
            say(f"   {'single' if k == 999 else 'two'}-token sequences of random words (1,000): every pile identical {ok[0]}/1000, "
                f"stream {ok[1]}/1000, top piles and words {ok[2]}/1000"); ok = [0, 0, 0]

def part_R():                                        # independent rebuild in whole-number arithmetic; fingerprints
    rng = np.random.default_rng(5); MAN = json.load(open(f"{HERE}/entry/manifest.json"))
    def half_up(x, s): return x if s == 0 else (x + (1 << (s - 1))) >> s        # written apart from base2.rshift
    light = {"ZP0": lambda i: half_up(WTE[i] + WPE[0], 4), "Z1023": lambda i: WTE[i] + WPE[1023],
             "T2": lambda i: half_up(WTE[i], 2), "T1": lambda i: half_up(WTE[i], 1),
             "P2": lambda i: half_up(WPE[i], 2), "P1": lambda i: half_up(WPE[i], 1)}
    for name, t in MAN["tables"].items():
        ids = np.load(f"{HERE}/entry/{t['ids']}").astype(np.int64); pick = np.sort(rng.choice(len(ids), min(300, len(ids)), replace=False))
        lo = np.load(f"{HERE}/entry/{name}_lo.npy", mmap_mode="r"); hi = np.load(f"{HERE}/entry/{name}_hi.npy", mmap_mode="r")
        stored = hi[pick].astype(np.int64) * (1 << 32) + lo[pick].astype(np.int64)
        rebuilt = np.dot(light[name](ids[pick]).astype(np.int64), P16)              # int64 whole-number products and sums
        say(f"R. independent rebuild (int64, no float) of {len(pick)} rows of {name}: identical {int((stored == rebuilt).all(1).sum())}/{len(pick)}")
    bad = []
    for f, meta in MAN["files"].items():
        h = hashlib.sha256()
        with open(f"{HERE}/entry/{f}", "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
        if h.hexdigest() != meta["sha256"] or os.path.getsize(f"{HERE}/entry/{f}") != meta["bytes"]: bad.append(f)
    say(f"   sha256 and size of all {len(MAN['files'])} files against the manifest: {'all match' if not bad else 'MISMATCH ' + str(bad)}")

def part_B():
    ok = [0, 0, 0]; n = 0
    for wdw in J.WINDOWS[WIN[0]:WIN[1]]:
        p, s, w, _ = run_both(wdw["ids"]); ok[0] += p; ok[1] += s; ok[2] += w; n += 1
    say(f"B. judge windows {WIN[0]}..{WIN[1] - 1}: every pile identical {ok[0]}/{n}, stream {ok[1]}/{n}, top piles and words at every position {ok[2]}/{n}")

def part_J():
    E.ON_PILE = None
    def guard_only(x16, w16):                        # no recording; the guard stays on
        if w16 is WQ: GUARD_HITS[0] += 1; raise RuntimeError("the entry machine met the qkv.0 plate with the full light")
        return _product(x16, w16)
    J.product = guard_only
    inn, same, tot = J.judge(forward=E.scores_all, n=300)
    say(f"J. judge5.judge(forward=base2_entry.scores_all, n=300): in top 5 {inn}/{tot}, same top word {same}/{tot}")

for name in PARTS: {"A": part_A, "C": part_C, "R": part_R, "B": part_B, "J": part_J}[name]()
say(f"guard: times the entry machine met the qkv.0 plate with the full light: {GUARD_HITS[0]}")
