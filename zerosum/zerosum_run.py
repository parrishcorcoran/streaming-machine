"""THE EXACT CUT -- part 3: the machines run with the zerosum files, every pile, light and water level compared.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

GPT-2: base2.py's machine loaded the way judge5.py loads it (judge5's own namespace), on the 30 refs texts (ids + ref,
every position, 648): first with base2_gpt2.npz, then with the zerosum file in place of the write lines. Every call of
the cut (to16), the meetings (product), the water level (gauge_log), the turn back (power) and the bend is digested row
by row (a row = a position); a position is identical when every row of every call it has is. The top scores at every
position: race winner and top 5. judge=1: judge5's 300 windows (6,000 positions) with the zerosum file.
Pythia: foundation0/pythia_base2.py's own machine (its namespace), the same digests on its 30 texts (ids + ref[:15]);
the head piles at the last 16 positions hashed exactly as pythia_base2.py hashes them (piles_sha256), against the plain
run and against foundation0/words_pythia-<size>.json.
perm=1 (ORDER-FREE CHECK): the stream's lanes put in a random order, every plate and write line permuted to match;
every head score and every gauge must be the same whole number as the plain-order run of the same file.
Also kept at every read: the cut light's sum S1 = 0 (the only place the water level could skip the level exactly);
GPT-2: whether the water level's log would stay the same with S1 left out (768 S2 alone).

PREDICTIONS (written before the first run):
  cut (the grain that passes (i) and (ii)):
    Pythia-14m (37,639 word rows, four biases) and -70m (two last biases): every pile, light and water level identical
      at every position except where the top 1 of a position's largest lane moves (0-2 positions); 30/30 head piles.
    GPT-2 (proj.11.b, -2048 = a whole number of fc.11's and the top's grains): the cut light at fc.11 and the top
      moves by a whole 2-8 units on every lane, the water levels stay identical, but (iii) leaks: fc.11's piles change
      at about 95% of columns and its outputs at about 1%, the top's scores at about 5-30% of words; words the same at
      648/648; so NOT identical at nearly every position (over 600 of 648).
  fine (the format's finest grain, every line):
    GPT-2: differs from the first read on (qkv.0's cut light) at every one of the 648 positions; race winner the same at
      at least 640/648; judge5 6,000/6,000 in the top 5, top word the same at 5,960 or more.
    Pythia: layer 0's reads identical (its cut is at grain 1 there and the word rows' constants are whole stream units)
      while the first write's per-lane rounding carries; differences from layer 1's read on at nearly every position;
      head piles identical 0/30; words the same at 475/480 or more.
  ctl (same constants, half the lanes up, half down; no level): more change than fine: top-word changes at least 3x
      fine's in GPT-2's judge5 and in the race winners.
  S1 = 0 after the cut: under 5% of reads in any run; GPT-2's log unchanged without S1 at under 50% of reads in the
      plain machine, nearly all in fine.
  Order-free check: every head score and gauge identical to the plain-order run of the same file.
    python3 zerosum/zerosum_run.py machine=gpt2|14m|70m|160m variant=fine|cut|ctl [judge=1] [perm=1]
"""
import sys, os, json, time, resource, hashlib, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"
A = dict(a.split("=") for a in sys.argv[1:]); MACH, VAR = A.get("machine", "gpt2"), A.get("variant", "fine")
JUDGE, PERM = A.get("judge", "0") == "1", A.get("perm", "0") == "1"
TAG = "gpt2" if MACH == "gpt2" else f"pythia-{MACH}"
NPZ = f"{OUT}/zerosum_{TAG}{'' if VAR == 'fine' else '_' + VAR}.npz"
NAME = f"{MACH}_{VAR}{'_perm' if PERM else ''}{'_judge' if JUDGE else ''}"
t0 = time.time(); LOG = open(f"{OUT}/zerosum_run_{NAME}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
Z = np.load(NPZ); log(f"{MACH}, file {os.path.basename(NPZ)}: {sorted(k for k in Z.files if not k.startswith(('k.', 'kb.', 'G.', 'gb.', 'g.')))[:6]}...")

# ---------- the recorder: every call digested row by row ----------
REC, ON, STAGE, ST = [], [True], ["-"], {}
def hrows(x):
    if isinstance(x, tuple): parts = [np.asarray(p) for p in x if np.ndim(p) > 0]; parts = parts or [np.asarray(x[0])]
    else: parts = [np.asarray(x)]
    a = parts[0]
    if a.ndim == 2 and all(p.ndim >= 1 and p.shape[0] == a.shape[0] for p in parts):
        return tuple(hashlib.blake2b(b"".join(np.ascontiguousarray(p[i]).tobytes() for p in parts), digest_size=8).digest() for i in range(a.shape[0]))
    return (hashlib.blake2b(b"".join(np.ascontiguousarray(p).tobytes() for p in parts), digest_size=8).digest(),)
READ_STAGES = ("qkv.", "fc.", "top") if MACH == "gpt2" else ("ln1.", "ln2.", "lnf")
def is_read(st): return st.startswith(READ_STAGES) and not st.startswith("top-out")
def wrap(ns, nm):
    fn = ns[nm]
    def w(*a, **k):
        out = fn(*a, **k)
        if ON[0]:
            st = STAGE[0]
            if nm in ("to16", "gauge_log") and is_read(st):   # the cut light carries the level: keep its level-free part,
                x16 = out[0]                                  # its power, and (gauge_log) the water level, separately
                REC.append((st, nm + "~", hrows((x16 - x16[:, :1],) + tuple(out[1:]))))
                REC.append((st, "cut power", hrows(np.asarray(out[1])[:, None])))
            elif st.startswith("arrival") and isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[0] == out.shape[1]:
                REC.append((st, nm, tuple(hashlib.blake2b(np.ascontiguousarray(out[i, :i + 1]).tobytes(), digest_size=8).digest() for i in range(len(out)))))
            else:                                             # (scores: each position's row only up to itself; the rest is never read)
                REC.append((st, nm, hrows(out) if not isinstance(out, (int, np.integer)) else (int(out).to_bytes(16, "little", signed=True),)))
        return out
    ns[nm] = w
def stat(k, v): ST[k] = ST.get(k, 0) + int(v)

# ================= GPT-2 =================
if MACH == "gpt2":
    sys.path.insert(0, HERE); argv = sys.argv; sys.argv = ["x"]; import judge5 as J; sys.argv = argv
    G = J.__dict__; M, W, ES = J.M, J.W, J.ES
    log(f"loaded the way judge5.py loads it ({time.time() - t0:.0f}s, {rss()} MB)")
    _gl = G["gauge_log"]
    def gauge_log_kept(R):                            # the water level, plus what S1 does in it
        x16, ex, LS = _gl(R)
        if ON[0]:
            S1, S2 = x16.sum(1), (x16 * x16).sum(1)
            eps = np.array([J.EPS40 >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else J.EPS40 << -(40 + 2 * int(e)) for e in ex])
            LS0 = np.right_shift(J.log2fix(768 * S2 + eps), 1) - J.LOG768 + ex * 65536
            stat("reads", len(R)); stat("S1=0", (S1 == 0).sum()); stat("log same without S1", (LS0 == LS).sum())
        return x16, ex, LS
    G["gauge_log"] = gauge_log_kept
    for nm in ("to16", "product", "gauge_log", "power", "edge"): wrap(G, nm)
    BANKS = [f"{b}.{l}" for l in range(12) for b in ("proj", "mproj")]
    WTE0, WPE0 = M["wte"], M["wpe"]
    def forward(ids, wte, wpe, first=0, last=None):   # base2.py's forward / judge5's scores_all, every position at the top
        P = len(ids); STAGE[0] = "entry"; R = wte[ids] + wpe[:P]
        for l in range(12):
            STAGE[0] = f"qkv.{l}"; Q = J.gauged_bank(R, f"qkv.{l}")
            STAGE[0] = f"arrival.{l}"; At = J.attention(Q)
            STAGE[0] = f"proj.{l}"; R = R + J.plain_bank(At, f"proj.{l}")
            STAGE[0] = f"fc.{l}"; H = J.gauged_bank(R, f"fc.{l}")
            STAGE[0] = f"bend.{l}"; E = J.edge(H)
            STAGE[0] = f"mproj.{l}"; R = R + J.plain_bank(E, f"mproj.{l}")
            if ON[0]: REC.append((f"stream.{l}", "R~", hrows(R - R[:, :1])))
        STAGE[0] = "top"; w, we, b = W["head"]; Rj = R[first:last]; x16, ex, LS = J.gauge_log(Rj)
        STAGE[0] = "top-out"; S = J.power(J.product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        return S
    refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
    def run_all(wte, wpe, keepS=False):
        res = []
        for ids in TEXTS:
            REC.clear(); S = forward(ids, wte, wpe)
            res.append(dict(rec=list(REC), win=S.argmax(1), top5=J.top5(S), sh=hashlib.sha256(S.tobytes()).hexdigest(), S=S if keepS else None))
        return res
    def install(Zf, perm=None):                       # the zerosum write lines into judge5's W; perm: the lanes reordered
        for n in BANKS:
            w0, we, b0 = W[n]
            w = Zf[n + ".w"].astype(np.float64) if n + ".w" in Zf.files else w0
            b = Zf[n + ".b"] if n + ".b" in Zf.files else b0
            if perm is not None: w, we, b = w[:, perm], we[perm], b[perm]
            W[n] = (w, we, b)
        if perm is not None:
            for n in [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["head"]:
                w, we, b = W[n]; W[n] = (w[perm], we, b)
        wte = Zf["wte"].astype(np.int64) if "wte" in Zf.files else WTE0
        wpe = Zf["wpe"].astype(np.int64) if "wpe" in Zf.files else WPE0
        return (wte, wpe) if perm is None else (wte[:, perm], wpe[:, perm])
    first_last = None

# ================= Pythia =================
else:
    path = f"{HERE}/foundation0/pythia_base2.py"; src = open(path).read(); src = src[:src.index('if __name__ == "__main__":')]
    argv = sys.argv; sys.argv = ["x", f"size={MACH}"]; G = {"__file__": path, "__name__": "pyb2"}; exec(src, G); sys.argv = argv
    PL, NORM, ES, L = G["PL"], G["NORM"], G["ES"], G["L"]
    log(f"loaded foundation0/pythia_base2.py's machine, size {MACH} ({time.time() - t0:.0f}s, {rss()} MB)")
    _to16 = G["to16"]
    for nm in ("to16", "product", "power", "log2fix", "edge", "turn", "gauge"): wrap(G, nm)
    def kept_gauge(R, name):
        if ON[0]:
            x16, ex = _to16(R); stat("reads", len(R)); stat("S1=0", (x16.sum(1) == 0).sum())
        STAGE[0] = name; return G["gauge"](R, name)
    BANKS = [f"{b}.{l}" for l in range(L) for b in ("proj", "mproj")]
    EMB0 = G["EMB_F"]
    def forward(ids, emb):
        ids = np.asarray(ids); STAGE[0] = "entry"
        R = G["rshift"](emb[ids].astype(np.int64), -(G["EMB_E"][ids][:, None] + ES))
        for l in range(L):
            STAGE[0] = f"ln1.{l}"; g1 = kept_gauge(R, f"ln1.{l}")
            STAGE[0] = f"qkv.{l}"; Q = G["plain"](g1, f"qkv.{l}")
            STAGE[0] = f"arrival.{l}"; At = G["attention"](Q)
            STAGE[0] = f"proj.{l}"; At = G["plain"](At, f"proj.{l}")
            STAGE[0] = f"ln2.{l}"; g2 = kept_gauge(R, f"ln2.{l}")
            STAGE[0] = f"fc.{l}"; H = G["plain"](g2, f"fc.{l}")
            STAGE[0] = f"bend.{l}"; E = G["edge"](H)
            STAGE[0] = f"mproj.{l}"; Mo = G["plain"](E, f"mproj.{l}")
            R = R + At + Mo
            if ON[0]: REC.append((f"stream.{l}", "R~", hrows(R - R[:, :1])))
        gf = kept_gauge(R, "lnf"); STAGE[0] = "top-out"; y16, ey = G["to16"](gf); w, e, _, wmax = PL["head"]
        return G["rshift"](G["product"](y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
    refs = json.load(open(f"{HERE}/foundation0/refs_pythia-{MACH}.json"))
    TEXTS = [r["ids"] + r["ref"][:15] for r in refs]
    saved = {x["prompt"]: x["piles_sha256"] for x in json.load(open(f"{HERE}/foundation0/words_pythia-{MACH}.json"))["texts_detail"]}
    def run_all(emb, keepS=False):
        res = []
        for r, ids in zip(refs, TEXTS):
            REC.clear(); S = forward(ids, emb); lg = S[-16:]
            res.append(dict(rec=list(REC), win=S.argmax(1), top5=np.argsort(-S, axis=1, kind="stable")[:, :5], sh=hashlib.sha256(S.tobytes()).hexdigest(),
                            piles16=hashlib.sha256(lg.tobytes()).hexdigest(), saved=saved.get(r["prompt"]), S=S if keepS else None))
        return res
    def install(Zf, perm=None):
        for n in BANKS:
            w0, e, b0, _ = PL[n]
            w = Zf[n + ".w"].astype(np.float64) if n + ".w" in Zf.files else w0
            b = Zf[n + ".b"] if n + ".b" in Zf.files else b0
            if perm is not None: w, e, b = w[:, perm], e[perm], b[perm]
            PL[n] = (w, e, b, float(np.abs(w).max()))
        if perm is not None:
            for n in [f"{b}.{l}" for l in range(L) for b in ("qkv", "fc")] + ["head"]:
                w, e, b, wm = PL[n]; PL[n] = (w[perm], e, b, wm)
            for n in list(NORM):
                g, ge, b = NORM[n]; NORM[n] = (g[perm], ge, b[perm])
        emb = Zf["embed.w"] if "embed.w" in Zf.files else EMB0
        return emb if perm is None else emb[:, perm]

# ================= run: plain, then the file =================
POW = {}; UNEXPL = [0]
def compare(base, new, label):
    POW.clear(); UNEXPL[0] = 0; npos = sum(len(x["win"]) for x in base); same_pos = 0; first = {}; calls = diff_calls = 0; by_stage = {}
    for b, n in zip(base, new):
        P = len(b["win"]); assert len(b["rec"]) == len(n["rec"]), "the two runs made different calls"
        bad = np.zeros(P, bool); fdone = False
        for (st, fn, hb), (st2, fn2, hn) in zip(b["rec"], n["rec"]):
            assert (st, fn) == (st2, fn2); calls += 1
            if fn == "cut power":
                if hb != hn: moved = [i for i, (x, y) in enumerate(zip(hb, hn)) if x != y]; POW.setdefault(id(b), set()).update(moved)
                continue
            if hb != hn:
                diff_calls += 1; key = st.split(".")[0] if st != "top" else "top"; by_stage[f"{st}/{fn}"] = by_stage.get(f"{st}/{fn}", 0) + 1
                if not fdone: first[f"{st} {fn}"] = first.get(f"{st} {fn}", 0) + 1; fdone = True
                if len(hb) == P: bad |= np.array([x != y for x, y in zip(hb, hn)])
        same_pos += int((~bad).sum())
        mv = POW.get(id(b), set()); fm = min(mv) if mv else P
        UNEXPL[0] += int(bad[:fm].sum())
    win = sum(int((b["win"] == n["win"]).sum()) for b, n in zip(base, new))
    t5 = sum(int((b["top5"] == n["top5"]).all(1).sum()) for b, n in zip(base, new))
    t5set = sum(int((np.sort(b["top5"], 1) == np.sort(n["top5"], 1)).all(1).sum()) for b, n in zip(base, new))
    scores = sum(int(b["sh"] == n["sh"]) for b, n in zip(base, new))
    log(f"\n{label}")
    log(f"  calls digested {calls:,}: identical {calls - diff_calls:,}, different {diff_calls:,}")
    log(f"  (the cut light at each read and the stream compared without their level: x - x[lane 0]; the cut's power apart)")
    log(f"  positions with every per-position row of every call identical: {same_pos}/{npos}")
    log(f"  positions where some read's cut power (the top 1 of the largest lane) moved: {sum(len(v) for v in POW.values())}; "
        f"positions that differ BEFORE the first such position in their text: {UNEXPL[0]}")
    log(f"  texts whose top scores (every position, every word) are the same whole numbers: {scores}/{len(base)}")
    log(f"  race winner the same {win}/{npos}; top 5 same in order {t5}/{npos}, same as a set {t5set}/{npos}")
    if MACH != "gpt2":
        p16 = sum(int(b["piles16"] == n["piles16"]) for b, n in zip(base, new))
        log(f"  head piles at the last 16 positions (pythia_base2.py's piles_sha256 form) the same: {p16}/{len(base)}")
    if first: log("  first different call per text: " + ", ".join(f"{k} x{v}" for k, v in sorted(first.items(), key=lambda x: -x[1])))
    if by_stage:
        st_sum = {}
        for k, v in by_stage.items(): s = k.split("/")[0]; st_sum[s] = st_sum.get(s, 0) + v
        log("  different calls by stage: " + ", ".join(f"{k} {v}" for k, v in st_sum.items()))
    return dict(calls=calls, diff_calls=diff_calls, same_positions=same_pos, power_moved=sum(len(v) for v in POW.values()), differ_before_power_moved=UNEXPL[0], positions=npos, scores_same_texts=scores, win_same=win, top5_same=t5, top5_set_same=t5set, first=first)

ST.clear(); base = run_all(*((WTE0, WPE0) if MACH == "gpt2" else (EMB0,))); stb = dict(ST)
log(f"plain run: {len(base)} texts ({time.time() - t0:.0f}s, {rss()} MB)")
if MACH != "gpt2":
    ok = sum(int(x["piles16"] == x["saved"]) for x in base)
    log(f"  plain head piles = foundation0/words_pythia-{MACH}.json's piles_sha256 (the machine's own run): {ok}/{len(base)}")
log(f"  reads {stb['reads']}: S1 = 0 at {stb['S1=0']}" + (f"; the water level's log the same with S1 left out at {stb['log same without S1']} ({100 * stb['log same without S1'] / stb['reads']:.1f}%)" if MACH == "gpt2" else ""))
ST.clear(); arrays = install(Z); new = run_all(*(arrays if MACH == "gpt2" else (arrays,)), keepS=PERM); stn = dict(ST)
log(f"zerosum run ({time.time() - t0:.0f}s, {rss()} MB)")
log(f"  reads {stn['reads']}: S1 = 0 at {stn['S1=0']}" + (f"; the water level's log the same with S1 left out at {stn['log same without S1']} ({100 * stn['log same without S1'] / stn['reads']:.1f}%)" if MACH == "gpt2" else ""))
R1 = compare(base, new, f"THE ZEROSUM FILE ({VAR}) AGAINST THE PLAIN MACHINE")
out = dict(machine=MACH, variant=VAR, plain_stats=stb, zerosum_stats=stn, compare=R1)

if PERM:
    rng = np.random.default_rng(777); D = 768 if MACH == "gpt2" else int(G["d"]); perm = rng.permutation(D)
    ON[0] = False; arrays = install(Z, perm); ON[0] = True
    ST.clear(); pn = run_all(*(arrays if MACH == "gpt2" else (arrays,)), keepS=True)
    same_scores = sum(int(np.array_equal(a["S"], b["S"])) for a, b in zip(new, pn))
    gauges = 0; tot = 0
    for a, b in zip(new, pn):
        ga = [h for (st, fn, h) in a["rec"] if fn == ("gauge_log" if MACH == "gpt2" else "log2fix")]
        gb = [h for (st, fn, h) in b["rec"] if fn == ("gauge_log" if MACH == "gpt2" else "log2fix")]
        if MACH == "gpt2":                            # x16 rows are permuted; compare the order-free parts (ex, LS) instead
            ga = [h for (st, fn, h) in a["rec"] if fn == "power" and st.startswith(("qkv", "fc", "top"))]
            gb = [h for (st, fn, h) in b["rec"] if fn == "power" and st.startswith(("qkv", "fc", "top"))]
        tot += len(ga); gauges += sum(int(x == y) for x, y in zip(ga, gb))
    log(f"\nORDER-FREE CHECK (lanes in a random order, every plate permuted to match): top scores the same whole numbers in "
        f"{same_scores}/{len(new)} texts; {'gauged read outputs' if MACH == 'gpt2' else 'water-level logs'} the same {gauges}/{tot}")
    out["order_free"] = dict(scores_same=same_scores, texts=len(new), gauges_same=gauges, gauges=tot)

if JUDGE and MACH == "gpt2":
    assert not PERM, "judge=1 runs on the plain lane order only"
    ON[0] = False
    inn, same, tot = J.judge(forward=lambda ids: forward(ids, arrays[0], arrays[1], J.FIRST, J.LAST))
    log(f"\nJUDGE5 with the zerosum file ({VAR}): in the reference top 5 {inn}/{tot}, top word the same {same}/{tot} ({time.time() - t0:.0f}s)")
    out["judge5"] = dict(in_top5=inn, top1_same=same, judged=tot)
json.dump(out, open(f"{OUT}/zerosum_run_{NAME}.json", "w"), indent=1, default=int)
log(f"\ndone ({time.time() - t0:.0f}s, peak {rss()} MB)")
