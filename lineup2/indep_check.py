"""INDEPENDENT CHECK OF WAY 2 (a separate checker; it never imports lineup2/way2.py, walk_rc, check.py or run_refs.py).

Way 2, rebuilt here from the task's words only: for each line i of a plate, that line's weights over all neurons put
in order of size once (the order stored: which neuron at each rank). Per token the light on line i is one whole
number x_i. Walk the ranks: r_0 = x_i * w_(0); r_k = r_(k-1) + x_i * gap_k; add r_k into the pile of the neuron at
rank k.

Built at three banks (qkv.7, fc.2, mproj.11) and at the top table, on 10 refs texts (1, 4, 7, ..., 28), inside
base2.py's own machine (base2.py's own steps exec'd from its source; at the three banks the product is replaced by the
walk, so everything after them runs on the walk's piles).
  EXACT: every pile of all 48 banks, the stream after every half layer, the light at the top, against
         recording/C_ref/plain_refs.npz (base2.py's saved run); at the three banks and the top table the walk's piles
         also against base2's own product of the same light; top 5, race winner, score rows (sha256).
  ORDER-FREE: fc.2 again with the lines in a random order and the walk made DOWN from the top weight (must close at 0).
  SHUFFLE: each line's gaps taken in a random neuron order (own seed): piles still exact, gap meetings near the
           signed-digit meetings, pours unchanged.
  RECOUNT (my own counters; signed digits by the step-by-step rule as a table built in Python integers, plain 1s by
           Python's bin(); the pours counted from the walk's own running products, 16 places at a time through the
           table): pairs, plain, signed, gaps, pours, pours in single waves, writing, every position, against the
           session's refs_main_counts.npz at the same positions.
  AGGREGATES: every reported share recomputed from the session's saved counts (all 648 positions, 49 tables).
  ORDER FLOOR: per line floor(log2(N!/prod ties!)) in Python integers at the three plates and the top table, against
           storage.json.
    python3 lineup2/indep_check.py   -> lineup2/indep_check.log
"""
import sys, os, json, time, math, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"
LOG = open(f"{OUT}/indep_check.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
FAIL = []
def need(ok, what):
    if not ok: FAIL.append(what); log(f"  *** DOES NOT HOLD: {what}")

# ---------- base2.py's own steps (its source, unchanged except the torch/transformers import and the npz path) ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_src = _src.replace('np.load("base2_gpt2.npz")', f'np.load("{HERE}/base2_gpt2.npz")')
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
M, W, ES = B["M"], B["W"], B["ES"]
log(f"base2.py's steps loaded ({time.time() - t0:.0f}s, {rss()} MB)")

# ---------- my counters: tables built in Python integers ----------
SPAN = 1 << 18
def naf_step(v):                                   # signed digits, the step rule: odd -> digit 2 - (v mod 4)
    n = 0
    while v:
        if v & 1: d = 2 - (v & 3); n += 1; v -= d
        v >>= 1
    return n
NAF_T = np.array([naf_step(v) for v in range(SPAN)], np.int64)
ONE_T = np.array([bin(v).count("1") for v in range(1 << 16)], np.int64)
def ONES(A):                                       # plain 1s of |A|, 16 places at a time through the table
    a = np.abs(np.asarray(A, np.int64)); n = np.zeros(a.shape, np.int64)
    while a.any(): n += ONE_T[a & 0xFFFF]; a = a >> 16
    return n
def NAF(A):                                        # small values: the step-rule table
    a = np.abs(np.asarray(A, np.int64)); assert a.max(initial=0) < SPAN; return NAF_T[a]
def NAF_BIG(A):                                    # any value: lit places of a ^ 3a (checked against the step rule below)
    a = np.abs(np.asarray(A, np.int64)); return ONES(a ^ (3 * a))
_probe = np.arange(SPAN, dtype=np.int64)
need(np.array_equal(NAF_BIG(_probe), NAF_T), "the a^3a rule equals the step rule for every |v| < 2^18")
need(np.array_equal(ONES(_probe), np.array([bin(v).count("1") for v in range(SPAN)])), "chunked 1s equal bin() for |v| < 2^18")

# ---------- my line-up ----------
CHOSEN = ["qkv.7", "fc.2", "mproj.11"]
class LineUp:
    def __init__(self, name, seed=None):
        w = np.asarray(M[name + ".w"]).astype(np.int16); L, N = w.shape; self.L, self.N = L, N; assert N < 65536
        self.O = np.empty((L, N), np.uint16); self.G = np.empty((L, N), np.int32); self.TOP = np.empty(L, np.int64)
        self.GN = np.empty(L, np.int64); self.WONES = np.empty(L, np.int64); self.WNAF = np.empty(L, np.int64)
        self.exact_record = True; self.zero_steps = 0; rng = np.random.default_rng(seed) if seed is not None else None
        for i0 in range(0, L, 32):
            wb = w[i0:i0 + 32].astype(np.int64); Lb, _ = wb.shape
            if seed is None:                                       # each line by size; ties by neuron (a unique key)
                O = np.argsort(wb * 65536 + np.arange(N)[None, :], axis=1)
            else:                                                  # SHUFFLE: a random neuron order per line
                O = np.argsort(rng.random((Lb, N)), axis=1)
            WS = np.take_along_axis(wb, O, 1); G = np.empty_like(WS); G[:, 0] = WS[:, 0]; G[:, 1:] = WS[:, 1:] - WS[:, :-1]
            if seed is None: assert (G[:, 1:] >= 0).all()
            back = np.zeros_like(wb); np.put_along_axis(back, O, np.cumsum(G, axis=1), 1)
            self.exact_record &= bool(np.array_equal(back, wb))
            self.O[i0:i0 + Lb] = O; self.G[i0:i0 + Lb] = G; self.TOP[i0:i0 + Lb] = WS[:, -1]
            self.GN[i0:i0 + Lb] = NAF(G).sum(1); self.WONES[i0:i0 + Lb] = ONES(wb).sum(1); self.WNAF[i0:i0 + Lb] = NAF(wb).sum(1)
            self.zero_steps += int((G[:, 1:] == 0).sum()); del O, WS, G, back, wb
        self.w = w
    def walk(self, x, line_order=None, down=False):
        P = x.shape[0]; pile = np.zeros((P, self.N), np.int64); pours = np.zeros(P, np.int64); pours_sw = np.zeros(P, np.int64)
        opened = 0
        for i in (range(self.L) if line_order is None else line_order):
            xi = x[:, i:i + 1].astype(np.int64)
            if not xi.any(): continue
            steps = xi * self.G[i].astype(np.int64)[None, :]       # each gap meets the light
            if not down:
                r = np.cumsum(steps, axis=1)                       # r_k = r_(k-1) + x gap_k
            else:                                                  # from the top weight down: r = x top; r -= x gap_k
                c = np.cumsum(steps[:, ::-1], axis=1)
                r = (xi * self.TOP[i] - np.concatenate([np.zeros((P, 1), np.int64), c[:, :-1]], 1))[:, ::-1]
                opened += int((xi[:, 0] * self.TOP[i] - c[:, -1] != 0).sum())
            pile[:, self.O[i]] += r
            pours += ONES(r).sum(1); pours_sw += NAF_BIG(r).sum(1)
        return pile, pours, pours_sw, opened
    def counts(self, x):
        nx, ox = NAF(x), ONES(x)
        return dict(pairs=(x != 0).sum(1) * self.N, plain=ox @ self.WONES, signed=nx @ self.WNAF, gaps=nx @ self.GN, writing=ox.sum(1))

REC = {n: LineUp(n) for n in CHOSEN + ["head"]}
for n, r in REC.items(): need(r.exact_record, f"{n}: the record gives the plate back")
log(f"line-ups laid: {', '.join(CHOSEN)}, head; every record gives its plate back: {all(r.exact_record for r in REC.values())} "
    f"({time.time() - t0:.0f}s, {rss()} MB)")
SHUF = {n: LineUp(n, seed=2026 + k) for k, n in enumerate(CHOSEN + ["head"])}

# ---------- a literal walk in Python integers on a few lines, against the vectorised one ----------
def literal(name, xv, i):
    r_ = REC[name]; pile = {}; r = 0; pours = 0
    for k in range(r_.N):
        r += int(xv) * int(r_.G[i, k]); j = int(r_.O[i, k]); pile[j] = pile.get(j, 0) + r; pours += bin(abs(r)).count("1")
    return pile, pours

# ---------- the machine: base2's forward, the walk at the chosen banks ----------
_REFZ = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = _REFZ["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
REFS = json.load(open(f"{HERE}/refs_many.json")); SEL = list(range(1, 30, 3))
_MYROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL]); _LOC = {int(r): k for k, r in enumerate(_MYROWS)}
class _Sliced:                                                     # base2's saved run, only my texts' rows kept
    def __init__(self):
        self.d = {k: _REFZ[k][_MYROWS] for k in _REFZ.files if k != "npos" and len(_REFZ[k]) == int(offs[-1])}
    def __getitem__(self, k): return self.d[k]
_RS = _Sliced()
class _RefView:                                                    # REF[key][global rows] -> the kept rows
    def __getitem__(self, k):
        class _A:
            def __getitem__(s, rows): return _RS[k][np.array([_LOC[int(r)] for r in rows])]
        return _A()
REF = _RefView()
log(f"base2's saved run sliced to my {len(_MYROWS)} positions: {len(_RS.d)} arrays ({time.time() - t0:.0f}s, {rss()} MB)")
CN = np.load(f"{OUT}/refs_main_counts.npz"); CROWS = CN["rows"]; where = {int(r): k for k, r in enumerate(CROWS)}
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
ID = {id(W[n][0]): n for n in BANKS}
orig_product = B["product"]
CUR = {}; NOW = {}; PILE_BAD = {n: 0 for n in BANKS}; PILE_N = {n: 0 for n in BANKS}
def product_walk(x16, w):
    n = ID.get(id(w))
    if n is None: return orig_product(x16, w)
    ref = orig_product(x16, w)
    if n in CHOSEN:
        pile, pours, psw, _ = REC[n].walk(x16)
        CUR.setdefault(n, []).append(dict(x=x16.copy(), pile_same=int((pile == ref).all(1).sum()), P=len(x16), pours=pours, psw=psw))
    else: pile = ref
    kind, l = n.split("."); saved = REF[f"{kind}_pile_{l}"][NOW["rows"]]
    PILE_BAD[n] += int((pile != saved).any(1).sum()); PILE_N[n] += len(pile)
    return pile                                                    # the walk's pile goes on through the machine
B["product"] = product_walk
gauged_bank, plain_bank, attention, edge, gauge_log, power, race = (B[k] for k in
    ("gauged_bank", "plain_bank", "attention", "edge", "gauge_log", "power", "race"))
HW, HE, HB = W["head"]; WTE, WPE = M["wte"], M["wpe"]
arr_diff = 0; arr_checked = 0; top_ok = dict(light=0, pile=0, sha=0, t5=0, win=0, n=0); TOPC = []
for t in SEL:
    ids = REFS[t]["ids"] + REFS[t]["ref"]; P = len(ids); assert P == int(npos[t]); rows = np.arange(offs[t], offs[t + 1])
    NOW["rows"] = rows; R = WTE[ids] + WPE[:P]
    def cmp(key, v):
        global arr_diff, arr_checked
        d = int((np.asarray(v) != REF[key][rows]).any(1).sum()); arr_diff += d; arr_checked += 1
        if d: log(f"    DIFFERS text {t}: {key} at {d} positions")
    for l in range(12):
        QKV = gauged_bank(R, f"qkv.{l}")
        att = plain_bank(attention(QKV), f"proj.{l}"); R = R + att; cmp(f"stream_a_{l}", R)
        H = gauged_bank(R, f"fc.{l}"); mo = plain_bank(edge(H), f"mproj.{l}"); R = R + mo; cmp(f"stream_{l}", R)
    # the banks' piles: base2's piles are checked through the stream; the chosen banks' walk piles directly
    x16, ex, LS = gauge_log(R)
    top_ok["light"] += int(((x16 == REF["x16"][rows]).all(1) & (ex == REF["ex"][rows]) & (LS == REF["LS"][rows])).sum())
    hp, hpo, hps, _ = REC["head"].walk(x16); href = orig_product(x16, HW)
    top_ok["pile"] += int((hp == href).all(1).sum())
    S = power(hp, (HE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + HB
    sha = [hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).digest() for s in S]
    top_ok["sha"] += sum(bytes(a) == bytes(b) for a, b in zip(sha, REF["score_sha"][rows]))
    t5 = np.argsort(-S, axis=1, kind="stable")[:, :5]
    top_ok["t5"] += int((t5 == REF["top5"][rows]).all(1).sum()); top_ok["win"] += int(sum(race(s) == w for s, w in zip(S, REF["win"][rows])))
    top_ok["n"] += P; TOPC.append(dict(x=x16, pours=hpo, psw=hps))
    log(f"  text {t:2d}: {P} positions ({time.time() - t0:.0f}s, {rss()} MB)")
ROWS = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in SEL]); NP = len(ROWS); CR = np.array([where[int(r)] for r in ROWS])
log(f"\nEXACT: {arr_checked} stream arrays (after every half layer, 24 per text) against base2's saved run: positions differing {arr_diff}")
need(arr_diff == 0, "the stream after every half layer equals base2's saved run")
for n in CHOSEN:
    same = sum(c["pile_same"] for c in CUR[n]); tot = sum(c["P"] for c in CUR[n])
    log(f"  {n:8s}: the walk's piles = base2's product of the same light {same}/{tot}"); need(same == tot, f"{n} walk piles exact")
bad48 = sum(PILE_BAD.values()); log(f"  every pile of the 48 banks (the walk's at {', '.join(CHOSEN)}) against base2's SAVED piles: "
    f"{len(BANKS)} banks x {PILE_N['qkv.0']} positions, positions differing {bad48}")
need(bad48 == 0 and all(PILE_N[n] == PILE_N['qkv.0'] for n in BANKS), "every bank pile = saved")
log(f"  top table: light (x16, ex, LS) = saved {top_ok['light']}/{top_ok['n']}; walk piles = base2's product {top_ok['pile']}/{top_ok['n']}; "
    f"score rows (sha256) {top_ok['sha']}/{top_ok['n']}; top 5 {top_ok['t5']}/{top_ok['n']}; race winner {top_ok['win']}/{top_ok['n']}")
for k in ("light", "pile", "sha", "t5", "win"): need(top_ok[k] == top_ok["n"], f"top table {k}")

# ---------- literal Python-integer walk on sample lines ----------
rng = np.random.default_rng(99); lit_bad = 0; lit_n = 0
for n in CHOSEN:
    c = CUR[n][3]; x = c["x"]
    for _ in range(3):
        p = int(rng.integers(len(x))); i = int(rng.choice(np.flatnonzero(x[p] != 0)))
        pl, po = literal(n, x[p, i], i); want = {j: int(x[p, i]) * int(REC[n].w[i, j]) for j in range(REC[n].N)}
        lit_bad += int(pl != want); lit_n += 1
        rv = np.cumsum(x[p, i] * REC[n].G[i]); lit_bad += int(po != int(ONES(rv).sum()))
log(f"  literal walk in Python integers (r += x gap, poured; {lit_n} lines): each neuron gets exactly x w, and the pours "
    f"count agrees: {lit_bad == 0}"); need(lit_bad == 0, "literal walk")

# ---------- ORDER-FREE (fc.2: lines in a random order, walked DOWN) and SHUFFLE ----------
of_bad = 0; opened = 0; sh_bad = 0; sh_pour_same = 0
SHC = {n: dict(gaps=0, plain=0, pours=0) for n in CHOSEN + ["head"]}
for n in CHOSEN + ["head"]:
    caps = CUR[n] if n != "head" else [dict(x=c["x"], pours=c["pours"]) for c in TOPC]
    for t, c in zip(SEL, caps):
        x = c["x"]; ref = orig_product(x, W[n][0])
        if n == "fc.2":
            lo = np.random.default_rng(7 + t).permutation(REC[n].L); pd, pod, _, op = REC[n].walk(x, line_order=lo, down=True)
            of_bad += int((pd != ref).any(1).sum()) + int((pod != c["pours"]).sum()); opened += op
        ps, pos_, _, _ = SHUF[n].walk(x); sh_bad += int((ps != ref).any(1).sum()); sh_pour_same += int((pos_ == c["pours"]).sum())
        cc = SHUF[n].counts(x); SHC[n]["gaps"] += int(cc["gaps"].sum()); SHC[n]["plain"] += int(cc["plain"].sum()); SHC[n]["pours"] += int(pos_.sum())
log(f"\nORDER-FREE (fc.2, lines in a random order, every walk DOWN from the top weight): positions differing (piles or pours) {of_bad}; "
    f"walks not closing at 0: {opened}"); need(of_bad == 0 and opened == 0, "order-free")
log(f"SHUFFLE (random neuron order per line, seeds 2026+): pile positions differing {sh_bad}; pours identical to the line-up's at "
    f"{sh_pour_same}/{4 * NP} position-tables"); need(sh_bad == 0, "shuffle piles exact")

# ---------- RECOUNT, every position, against the session's saved counts ----------
log(f"\nRECOUNT at {NP} positions (texts {SEL}): mine against refs_main_counts.npz, position by position")
MY = {}
for n in CHOSEN + ["head"]:
    caps = CUR[n] if n != "head" else TOPC
    xs = np.concatenate([c["x"] for c in caps]); cc = REC[n].counts(xs)
    cc["pours"] = np.concatenate([c["pours"] for c in caps]); cc["pours_single_wave"] = np.concatenate([c["psw"] for c in caps])
    MY[n] = cc
    for item in ("pairs", "plain", "signed", "gaps", "pours", "pours_single_wave", "writing"):
        theirs = CN[f"{n}|{item}"][CR]; eq = int((theirs == cc[item]).sum())
        need(eq == NP, f"{n} {item} per position")
        if eq != NP: log(f"    {n} {item}: equal at {eq}/{NP}; mine {int(cc[item].sum()):,} theirs {int(theirs.sum()):,}")
    pl = int(cc["plain"].sum()); g = int(cc["gaps"].sum()); po = int(cc["pours"].sum()); wr = int(cc["writing"].sum()); sg = int(cc["signed"].sum())
    pr = int(cc["pairs"].sum())
    log(f"  {n:8s}: all 7 counts equal at every position: {all(int((CN[f'{n}|{k}'][CR] == cc[k]).sum()) == NP for k in ('pairs','plain','signed','gaps','pours','pours_single_wave','writing'))}"
        f" | plain {pl:,}; signed {100*sg/pl:.2f}%; gaps {100*g/pl:.2f}%; pours {100*po/pl:.2f}%; writing {100*wr/pl:.3f}%; way 2 {100*(g+po+wr)/pl:.2f}% "
        f"| per pair: plain {pl/pr:.2f}, walk {g/pr:.2f}, pour {po/pr:.2f} | SHUFFLED gaps {100*SHC[n]['gaps']/SHC[n]['plain']:.2f}% "
        f"(their count-only shuffle at these positions {100*int(CN[f'{n}|gaps_shuffled'][CR].sum())/pl:.2f}%)")

# ---------- AGGREGATES from the session's saved counts ----------
log(f"\nAGGREGATES recomputed from refs_main_counts.npz (all {len(CROWS)} positions)")
ALLT = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]
S_ = {t: {k: int(CN[f"{t}|{k}"].sum()) for k in ("pairs", "plain", "signed", "gaps", "gaps_shuffled", "pours", "pours_single_wave", "writing")} for t in ALLT}
n648 = len(CROWS); pt = lambda v: (v + n648 // 2) // n648
def agg(kinds): return {k: sum(S_[t][k] for t in ALLT if t.split(".")[0] in kinds) for k in S_["head"]}
bk = agg({"qkv", "proj", "fc", "mproj"}); hd = agg({"head"})
w2b = bk["gaps"] + bk["pours"] + bk["writing"]; w2h = hd["gaps"] + hd["pours"] + hd["writing"]
log(f"  48 banks a token: plain {pt(bk['plain']):,}; signed {pt(bk['signed']):,} ({100*bk['signed']/bk['plain']:.2f}%); way 2 {pt(w2b):,} "
    f"({100*w2b/bk['plain']:.2f}%) = gaps {100*bk['gaps']/bk['plain']:.2f}% + pours {100*bk['pours']/bk['plain']:.2f}% + writing "
    f"{100*bk['writing']/bk['plain']:.3f}%; single-wave pours {100*bk['pours_single_wave']/bk['plain']:.2f}%")
need(pt(bk["plain"]) == 2737920452 and pt(hd["plain"]) == 896556125 and pt(bk["signed"]) == 1605763357, "plain/signed rebuilt")
need(pt(w2b) == 1537897796 and pt(w2h) == 341344387, "way-2 totals")
for k in ("qkv", "proj", "fc", "mproj"):
    a = agg({k}); log(f"  {k:6s}: way 2 {100*(a['gaps']+a['pours']+a['writing'])/a['plain']:.2f}%  gaps {100*a['gaps']/a['plain']:.2f}%  pours "
                      f"{100*a['pours']/a['plain']:.2f}%  per pair walk {a['gaps']/a['pairs']:.2f} pour {a['pours']/a['pairs']:.2f} plain {a['plain']/a['pairs']:.2f}")
log(f"  top table: {pt(w2h):,} = {100*w2h/hd['plain']:.2f}% (gaps {100*hd['gaps']/hd['plain']:.2f}%, pours {100*hd['pours']/hd['plain']:.2f}%); per pair walk "
    f"{hd['gaps']/hd['pairs']:.3f} pour {hd['pours']/hd['pairs']:.2f}; pairs a token banks {pt(bk['pairs']):,} top {pt(hd['pairs']):,}")
PLAIN_TOTAL = 3666596259; rest = PLAIN_TOTAL - 2737920452 - 896556125
whole = pt(w2b + w2h) + rest; wb = pt(w2b) + 896556125 + rest
log(f"  whole machine: rest {rest:,}; way 2 everywhere {whole:,} = {100*whole/PLAIN_TOTAL:.2f}%; banks only {100*wb/PLAIN_TOTAL:.2f}%; "
    f"fringes top read 191,031,613 against the plain read's meetings {100*191031613/896556125:.2f}% (against its report's 904,017,313: "
    f"{100*191031613/904017313:.2f}%)")

# ---------- the session's shuffle and order-free runs, against its main run at the same positions ----------
for mode in ("shuffle", "order"):
    Z = np.load(f"{OUT}/refs_{mode}_counts.npz"); zr = Z["rows"]; zi = np.array([where[int(r)] for r in zr]); same = 0; tot_ = 0
    items = ("pours", "pours_single_wave", "pairs", "plain", "signed", "writing") if mode == "shuffle" else \
            ("pairs", "plain", "signed", "gaps", "pours", "pours_single_wave", "writing")
    for t in ALLT:
        for k in items: same += int(np.array_equal(Z[f"{t}|{k}"], CN[f"{t}|{k}"][zi])); tot_ += 1
    zb = {k: sum(int(Z[f"{t}|{k}"].sum()) for t in ALLT[:-1]) for k in ("gaps", "pours", "writing", "plain")}
    zh = {k: int(Z[f"head|{k}"].sum()) for k in ("gaps", "pours", "writing", "plain")}
    log(f"  session's {mode} run ({len(zr)} positions): count arrays equal to the main run's at the same positions {same}/{tot_} "
        f"({', '.join(items)}); banks gaps {100*zb['gaps']/zb['plain']:.2f}%, banks work {100*(zb['gaps']+zb['pours']+zb['writing'])/zb['plain']:.2f}%, "
        f"top work {100*(zh['gaps']+zh['pours']+zh['writing'])/zh['plain']:.2f}%")
    need(same == tot_, f"session's {mode} run counts = main run's")
    if mode == "shuffle":
        for k in ("qkv", "proj", "fc", "mproj", "head"):
            g = sum(int(Z[f"{t}|gaps"].sum()) for t in ALLT if t.split(".")[0] == k); p = sum(int(Z[f"{t}|plain"].sum()) for t in ALLT if t.split(".")[0] == k)
            g0 = sum(int(CN[f"{t}|gaps_shuffled"].sum()) for t in ALLT if t.split(".")[0] == k); p0 = sum(int(CN[f"{t}|plain"].sum()) for t in ALLT if t.split(".")[0] == k)
            log(f"    {k:6s}: shuffled gaps {100*g/p:.2f}% (main run's count-only shuffle {100*g0/p0:.2f}%)")
        g = sum(int(Z[f"{t}|gaps"].sum()) for t in ALLT[:-1]); p = sum(int(Z[f"{t}|plain"].sum()) for t in ALLT[:-1])
        g0 = sum(int(CN[f"{t}|gaps_shuffled"].sum()) for t in ALLT[:-1]); p0 = sum(int(CN[f"{t}|plain"].sum()) for t in ALLT[:-1])
        log(f"    48 banks: shuffled gaps {100*g/p:.2f}% (main run's count-only shuffle {100*g0/p0:.2f}%)")

# ---------- ORDER FLOOR at the chosen plates and the top table ----------
SJ = json.load(open(f"{OUT}/storage.json"))["plates"]; FCT = {}
def fact(n):
    if n not in FCT: FCT[n] = math.factorial(n)
    return FCT[n]
log(f"\nORDER FLOOR (per line floor(log2(N!/prod ties!)), Python integers) and zero steps")
for n in CHOSEN + ["head"]:
    r_ = REC[n]; WS = np.sort(r_.w, axis=1); tot = 0; FN = fact(r_.N)
    for i in range(r_.L):
        _, cnt = np.unique(WS[i], return_counts=True); den = 1
        for c in cnt.tolist(): den *= fact(c)
        k = FN.bit_length() - den.bit_length()                    # 2^k <= N!/den < 2^(k+1)
        while (den << k) > FN: k -= 1
        while (den << (k + 1)) <= FN: k += 1
        tot += k
    log(f"  {n:8s}: {tot:,} bits ({tot/(r_.L*r_.N):.3f} a weight); storage.json {SJ[n]['order_floor_bits']:,}; zero steps {r_.zero_steps:,} "
        f"(storage.json {SJ[n]['zero_gaps']:,}; {100*r_.zero_steps/(r_.L*(r_.N-1)):.1f}%)")
    need(tot == SJ[n]["order_floor_bits"] and r_.zero_steps == SJ[n]["zero_gaps"], f"{n} order floor / zero steps")
for N_ in (768, 2304, 3072):                                       # the floor with no ties at all, per weight
    F_ = fact(N_); log(f"  no ties at all: floor(log2({N_}!)) / {N_} = {(F_.bit_length() - 1) / N_:.3f} bits a weight")

# ---------- "no two lines hold the same neuron at the same rank": count it ----------
for n in CHOSEN:
    O = REC[n].O; L, N = O.shape; shared_ranks = 0; pairs_same = 0
    for k in range(N):
        _, cnt = np.unique(O[:, k], return_counts=True); shared_ranks += int((cnt > 1).any()); pairs_same += int((cnt * (cnt - 1) // 2).sum())
    log(f"  {n:8s}: ranks at which two or more lines hold the same neuron {shared_ranks}/{N}; line pairs holding the same neuron at the "
        f"same rank {pairs_same:,} (by chance about C(L,2) = {L * (L - 1) // 2:,})")

log(f"\n{'ALL HOLD' if not FAIL else 'DOES NOT HOLD: ' + '; '.join(FAIL)} ({time.time() - t0:.0f}s, max memory {rss()} MB)")
