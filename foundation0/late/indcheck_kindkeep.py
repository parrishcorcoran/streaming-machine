"""INDEPENDENT CHECK (rule 10) of the plate look's kinds at 160m and the keep case at 70m and 160m. 2026-10-06.

HIS SENTENCE (his exact words, the only ones here that are his): "It should be 100% removable after training."
  "Any failure is not a pivot."
SESSION'S RESTATEMENT under test (the session's words, not his): the finished plate W = what survives of the random
start, s x W0 with one s per plate, + what was built on it. "Removing the foundation" = W - s x W0.

WHAT IS CHECKED (numbers as stated by plate_look.py and keep_late.py, base-2 machine, 30 texts, three draws):
  look, 160m, by kind (top-1 of 622; the draws are cb@20261006, cb@2, cb@3):
    b:K=qkv   72.0 (76.5, 72.0, 74.1)  late level moved layers 8-11: 28.3 45.9 52.4 61.6; first word 46.8 40.1 39.5 28.2
    b:K=fc    71.5 (74.4, 75.9, 77.8)  late level moved 11.0 12.6 12.6 13.6
    b:K=mproj 87.1 (89.9, 91.0, 92.4)
  keep (K = the qkv plates of the last third of the layers kept; every other plate and both word tables lose s x W0):
    160m  k 59.2 (ck 63.8, 63.7, 68.2), late level moved 19.2 20.6 19.5 21.8, first word 56.0 59.1 64.4 48.6, 0/30 texts
          b 50.5, late level moved 35.8 51.8 55.6 66.1
    70m   k 61.1 (ck 60.5, 63.2, 63.8), late level moved 18.6 22.2, first word 58.7 36.4, 0/30 texts; b 52.6
  The base-2 numbers on the first 10 texts, read from plate_look_<size>.json / keep_late_<size>.json before this ran:
    160m  qkv kind 74.5 (77.5, 72.5, 77.0); fc kind 75.0 (77.0, 79.0, 82.5); mproj kind 91.0 (91.5, 93.0, 96.0);
          k 61.5 (ck 64.0, 70.0, 71.5); b 49.5
    70m   qkv kind 66.0 (68.0, 62.0, 64.5); k 60.5 (ck 58.0, 63.0, 62.5); b 54.5

HOW (no code shared with the base-2 scripts or keep_checker.py): Pythia's own fp16 files (step143000 finished, step0
start) from ../../models/; s recomputed here from those files, s = (W . W0) / (W0 . W0) per plate and table; each
case's plates W - s x W0 made here in float64 and held in float32; a forward pass written here in numpy (parallel
arrival + bend, layer norm, turns on the first quarter of each head, arrival with its full exponential, the erf bend
via torch.special.erf only), the stream in float64, plate products in float32 (fp32 Pythia), scores in float64.
Scrambled starts, two kinds:
  cb/ck@seed   the base-2 scripts' draw for that seed, rebuilt here from its rule (numpy default_rng(seed +
               crc32(name)), one permutation of the plate's values in the npz layout (in x out; word table words x
               lanes)), the values kept exactly (the base-2 rounding where a value lands on a coarser neuron is not
               done; it touches only places below 2^-16 of that neuron)
  mb/mk@n      this checker's own draws, a different rule (default_rng([n, 777, adler32(name)]), torch layout)
Measured on every case against this checker's own finished machine, the same fed-in words (each text's own words, then
the fp32 greedy 16): top-1, top-5, rows whose water level moves (first position at the row's highest score; rows with
one position left out; every head), the text's first word's share of the levels, knife edges that change side (the
bend's input above 0), on all 30 texts and on the first 10; the case's own 16 greedy words on the first 10 texts.
Position by position against the base-2 run of the same case (read from its JSON).

PREDICTIONS, written before the first run of this script (results go to indcheck_kindkeep_<size>.log/.json):
  P0 rebuild: this finished machine's word = fp32's greedy word at 480/480 continuation positions and = the base-2
     finished word at 622/622 positions, both sizes. s within 1e-3 (relative) of ../foundation_share.json's at every
     plate and table. The npz start digits x 2^e equal the step0 fp16 start to within half a grid step (layout check
     for the rebuilt draws).
  P1 position agreement with the base-2 case word: 97% or more for b, k, b:K=<kind>; 95% or more for the rebuilt
     cb/ck draws.
  P2 top-1 on 30 texts within 2 points of the base-2 number for every case above; on the first 10 texts within 3.
  P3 late level moved (30 texts) within 3 points of the base-2 numbers above; first word's share within 3 points.
  P4 the orderings as stated, rebuilt draws (cb/ck @20261006, @2): 160m fc kind and mproj kind below both draws;
     qkv kind not below both by more than 1 point; k below both ck draws; 70m k inside or within 3 points of them.
  P5 own draws (mb/mk @1, @2): 160m fc kind below both; qkv kind within 3 points of the two draws' spread; 160m k
     below both mk draws; 70m k within 3 points of the mk spread.
  P6 greedy, first 10 texts: 0 of 10 identical for b, k and the qkv and fc kinds at both sizes; this checker's greedy
     words equal the base-2 case's greedy words at 85% or more of the 160 places for the real-foundation cases.
A stated number "does not hold" here if P2 or P3 misses by more than twice its allowance, or an ordering in P4 reverses.

  python3 indcheck_kindkeep.py size=160m [gt=10] [cases=all|<list>]
"""
import os, sys, json, time, zlib, resource
import numpy as np, torch
from safetensors.numpy import load_file

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "160m"); GT = int(A.get("gt", 10)); WORDS = 16
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE); MD = os.path.join(HERE, "models")
LOGF = open(os.path.join(LATE, f"indcheck_kindkeep_{SIZE}{('-' + A['tag']) if A.get('tag') else ''}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
t0 = time.time()

cfg = json.load(open(os.path.join(MD, f"pythia-{SIZE}-step143000", "config.json")))
d, L, NH = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
HS = d // NH; ROT = int(HS * cfg["rotary_pct"]); BASE = float(cfg["rotary_emb_base"]); EPS = cfg["layer_norm_eps"]
KINDS = ("qkv", "proj", "fc", "mproj")
PLATES = [f"{k}.{l}" for l in range(L) for k in KINDS]
TABLES = ["embed", "head"]
KEEP = {f"qkv.{l}" for l in range((2 * L) // 3, L)}
LATEL = list(range((2 * L) // 3, L))
def key(name):
    if name == "embed": return "gpt_neox.embed_in.weight"
    if name == "head": return "embed_out.weight"
    k, l = name.split(".")
    return f"gpt_neox.layers.{l}." + {"qkv": "attention.query_key_value", "proj": "attention.dense",
                                       "fc": "mlp.dense_h_to_4h", "mproj": "mlp.dense_4h_to_h"}[k] + ".weight"

F16 = load_file(os.path.join(MD, f"pythia-{SIZE}-step143000", "model.safetensors"))
S16 = load_file(os.path.join(MD, f"pythia-{SIZE}-step0", "model.safetensors"))
FIN = {n: F16[key(n)].astype(np.float32) for n in PLATES + TABLES}           # fp16 values held exactly
BIAS = {n: F16[key(n)[:-len("weight")] + "bias"].astype(np.float64) for n in PLATES}
NORM = {}
for l in range(L):
    for nm, kk in (("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm")):
        NORM[f"{nm}.{l}"] = (F16[f"gpt_neox.layers.{l}.{kk}.weight"].astype(np.float64), F16[f"gpt_neox.layers.{l}.{kk}.bias"].astype(np.float64))
NORM["lnf"] = (F16["gpt_neox.final_layer_norm.weight"].astype(np.float64), F16["gpt_neox.final_layer_norm.bias"].astype(np.float64))
START = {n: S16[key(n)] for n in PLATES + TABLES}                           # fp16, torch layout (out x in; words x lanes)
del F16, S16

# ---------------------------------------------------------------- s, recomputed here
def chunks(n, step): return [(a, min(n, a + step)) for a in range(0, n, step)]
S = {}
for n in PLATES + TABLES:
    W, W0 = FIN[n], START[n]; num = den = 0.0
    for a, b in chunks(W.shape[0], 4096):
        w0 = W0[a:b].astype(np.float64); num += float((W[a:b].astype(np.float64) * w0).sum()); den += float((w0 * w0).sum())
    S[n] = num / den
SH = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"][SIZE]["plates"]
rel = {n: abs(S[n] - SH[n]["s"]) / abs(SH[n]["s"]) for n in PLATES + TABLES}
worst = max(rel, key=rel.get)
log(f"indcheck_kindkeep.py size {SIZE}: d {d} L {L} heads {NH} head {HS} turned {ROT}; keep set {sorted(KEEP, key=lambda x: int(x.split('.')[1]))}")
log(f"  s recomputed from the fp16 files: largest relative difference from foundation_share.json {rel[worst]:.2e} ({worst}: "
    f"{S[worst]:.6f} vs {SH[worst]['s']:.6f}); s range plates {min(S[n] for n in PLATES):.4f}..{max(S[n] for n in PLATES):.4f}, "
    f"embed {S['embed']:.4f}, head {S['head']:.4f}")

# layout check for the rebuilt draws: npz start digits x 2^e against the fp16 start, in the npz layout
NPZ = np.load(os.path.join(HERE, f"base2_pythia-{SIZE}.npz"))
def npz_layout(n, X): return X if n == "embed" else X.T
for n in ["qkv.0", "proj.0", "fc.0", f"mproj.{L-1}", "embed", "head"]:
    DG = NPZ[n + ".s"]; e = NPZ[n + ".e"].astype(np.float64); ST = npz_layout(n, START[n]); worst = 0.0
    for a, b in chunks(DG.shape[0], 512):                                # in row blocks: keeps the check small
        step = 2.0 ** (e[a:b, None] if n == "embed" else e[None, :])
        worst = max(worst, float((np.abs(DG[a:b].astype(np.float64) * step - ST[a:b].astype(np.float64)) / step).max()))
    log(f"  layout check {n:8s}: npz start digit x 2^e vs fp16 start, largest difference {worst:.3f} grid steps")
    del DG, ST
del NPZ

# ---------------------------------------------------------------- the starts for each case
def start_of(n, how, seed):
    W0 = START[n]
    if how == "real": return W0
    if how == "theirs":                                                  # the base-2 scripts' rule, rebuilt here
        X = np.ascontiguousarray(npz_layout(n, W0)).ravel()
        P = np.random.default_rng(seed + zlib.crc32(n.encode())).permutation(X.size)
        Y = X[P].reshape(npz_layout(n, W0).shape); del X, P
        return Y if n == "embed" else np.ascontiguousarray(Y.T)
    if how == "mine":                                                    # this checker's own rule
        X = W0.ravel(); P = np.random.default_rng([seed, 777, zlib.adler32(n.encode())]).permutation(X.size)
        Y = X[P].reshape(W0.shape); del P
        return Y
    raise ValueError(how)

def parse(case):
    """-> (set of names that lose the start, how the start is drawn, seed)"""
    if case == "fin": return set(), "real", None
    c, _, seed = case.partition("@"); seed = int(seed) if seed else None
    if c == "b": return set(PLATES + TABLES), "real", None
    if c == "k": return set(PLATES + TABLES) - KEEP, "real", None
    if c == "ck": return set(PLATES + TABLES) - KEEP, "theirs", seed
    if c == "mk": return set(PLATES + TABLES) - KEEP, "mine", seed
    pre, _, kind = c.partition(":K=")
    names = {p for p in PLATES if p.split(".")[0] == kind}
    return names, {"b": "real", "cb": "theirs", "mb": "mine"}[pre], seed

def build(case):
    names, how, seed = parse(case); D = dict(FIN)
    for n in names:
        W0 = start_of(n, how, seed); W = FIN[n]; out = np.empty_like(W)
        for a, b in chunks(W.shape[0], 4096):
            out[a:b] = (W[a:b].astype(np.float64) - S[n] * W0[a:b].astype(np.float64)).astype(np.float32)
        D[n] = out; del W0
    return D

# ---------------------------------------------------------------- the forward pass, written here
SQ2 = np.sqrt(2.0)
def lnorm(x, g, b):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + EPS) * g + b
def mm(x, W):                                         # macOS's product raises stray flags: checked finite here
    with np.errstate(all="ignore"): y = x.astype(np.float32) @ W.T
    assert np.isfinite(y).all(); return y.astype(np.float64)
def turn(X, c, s):
    r = ROT // 2; a, b = X[..., :r], X[..., r:ROT]
    return np.concatenate([a * c - b * s, b * c + a * s, X[..., ROT:]], -1)

def forward(D, ids, last=None, rec=None):
    ids = np.asarray(ids); P = len(ids)
    x = D["embed"][ids].astype(np.float64)
    ang = np.arange(P, dtype=np.float64)[:, None] / BASE ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT)[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    live = np.tril(np.ones((P, P), dtype=bool))[None]
    for l in range(L):
        qkv = (mm(lnorm(x, *NORM[f"ln1.{l}"]), D[f"qkv.{l}"]) + BIAS[f"qkv.{l}"]).reshape(P, NH, 3 * HS)
        q, k, v = turn(qkv[..., :HS], c, s), turn(qkv[..., HS:2 * HS], c, s), qkv[..., 2 * HS:]
        sc = np.where(live, np.einsum("phd,qhd->hpq", q, k) / np.sqrt(HS), -np.inf)
        if rec is not None: rec["wl"].append(sc.argmax(-1))                   # first position at the row's highest score
        a = np.exp(sc - sc.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        at = mm(np.einsum("hpq,qhd->phd", a, v).reshape(P, d), D[f"proj.{l}"]) + BIAS[f"proj.{l}"]
        H = mm(lnorm(x, *NORM[f"ln2.{l}"]), D[f"fc.{l}"]) + BIAS[f"fc.{l}"]
        if rec is not None: rec["ke"].append(H > 0)
        g = H * 0.5 * (1.0 + torch.special.erf(torch.from_numpy(H / SQ2)).numpy())
        x = x + at + mm(g, D[f"mproj.{l}"]) + BIAS[f"mproj.{l}"]
    y = lnorm(x if last is None else x[-last:], *NORM["lnf"])
    with np.errstate(all="ignore"): lg = y.astype(np.float32) @ D["head"].T
    assert np.isfinite(lg).all(); return lg

# ---------------------------------------------------------------- the runs
refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))
TEXTS = len(refs)
PL = json.load(open(os.path.join(LATE, f"plate_look_{SIZE}.json")))["cases"]
KL = json.load(open(os.path.join(LATE, f"keep_late_{SIZE}.json")))["cases"]
def b2(case):
    if case.startswith(("b:K=", "cb:K=")): return PL.get(case)
    if case in ("fin", "b", "k") or case.startswith("ck@"): return KL.get(case)
    return None

KC = ["qkv", "fc", "mproj"] if SIZE == "160m" else ["qkv"]
ALL = ["fin", "b"]
for k in KC:
    ALL += [f"b:K={k}", f"cb:K={k}@20261006", f"cb:K={k}@2"] + ([f"mb:K={k}@1", f"mb:K={k}@2"] if k in ("qkv", "fc") else [])
ALL += ["k", "ck@20261006", "ck@2", "mk@1", "mk@2"]
CASES = ALL if A.get("cases", "all") == "all" else ["fin"] + [c for c in A["cases"].split(",") if c != "fin"]
log(f"  cases {CASES}; texts {TEXTS} (fed words), greedy on the first {GT}")

def pct(a, b): return 100.0 * a / b if b else float("nan")
RES = {}; F = None
for case in CASES:
    tc = time.time(); D = build(case); per = []
    for t, r in enumerate(refs):
        seq = r["ids"] + r["ref"][:WORDS - 1]; n0 = len(r["ids"]) - 1
        rec = dict(wl=[], ke=[]); lg = forward(D, seq, rec=rec)
        top1 = lg.argmax(1); top5 = np.argsort(-lg, axis=1, kind="stable")[:, :5]
        x = dict(top1=top1.tolist(), top5=top5.tolist(), wl=rec["wl"], ke=rec["ke"], n0=n0)
        if t < GT:
            cont = x["top1"][n0:n0 + WORDS]; bad = [j for j in range(WORDS) if cont[j] != r["ref"][j]]
            own = list(r["ref"]) if not bad else r["ref"][:bad[0]] + [cont[bad[0]]]
            while len(own) < WORDS: own.append(int(forward(D, r["ids"] + own, last=1)[0].argmax()))
            x["greedy"] = own[:WORDS]
        per.append(x)
    del D
    if case == "fin":
        F = per
        ok = sum(int(a == b) for x, r in zip(per, refs) for a, b in zip(x["top1"][x["n0"]:x["n0"] + WORDS], r["ref"]))
        log(f"  fin: this checker's finished word = fp32's greedy word at {ok} of {TEXTS * WORDS} continuation positions; "
            f"greedy texts identical to fp32's {sum(x['greedy'] == r['ref'] for x, r in zip(per[:GT], refs))} of {GT}")
    o = {}
    for tag, T in (("30", TEXTS), ("10", 10)):
        n = t1 = t5 = 0; wl = np.zeros((L, 2)); fw = np.zeros((L, 2)); ke = np.zeros((L, 2))
        for t in range(T):
            x, f = per[t], F[t]
            for p in range(len(x["top1"])):
                n += 1; t1 += x["top1"][p] == f["top1"][p]; t5 += f["top1"][p] in x["top5"][p]
            for l in range(L):
                mv = x["wl"][l][:, 1:] != f["wl"][l][:, 1:]; wl[l] += [mv.sum(), mv.size]
                fw[l] += [(x["wl"][l][:, 1:] == 0).sum(), mv.size]
                ke[l] += [(x["ke"][l] != f["ke"][l]).sum(), x["ke"][l].size]
        o[tag] = dict(top1=pct(t1, n), top5=pct(t5, n), positions=n, wl=(100 * wl[:, 0] / wl[:, 1]).round(2).tolist(),
                      fw=(100 * fw[:, 0] / fw[:, 1]).round(2).tolist(), ke=(100 * ke[:, 0] / ke[:, 1]).round(2).tolist())
    g = [x["greedy"] for x in per[:GT]]
    o["greedy_texts"] = sum(a == r["ref"] for a, r in zip(g, refs)); o["greedy_words"] = sum(int(u == v) for a, r in zip(g, refs) for u, v in zip(a, r["ref"]))
    B = b2(case)
    if B is not None:
        bt = B["top1"]; o["agree_b2"] = pct(sum(int(u == v) for x, y in zip(per, bt) for u, v in zip(x["top1"], y)), sum(len(y) for y in bt))
        bf = (PL if case.startswith(("b:K=", "cb:K=")) else KL)["fin"]["top1"]
        o["b2_top1_30"] = 100 * B["top1_all"]
        o["b2_top1_10"] = pct(sum(int(u == v) for y, z in zip(bt[:10], bf[:10]) for u, v in zip(y, z)), sum(len(z) for z in bf[:10]))
        o["b2_wl"] = [round(100 * v, 2) for v in B["wl_moved"]]; o["b2_fw"] = [round(100 * v, 2) for v in B["first_word"]]
        o["b2_greedy_texts_10"] = sum(a == r["ref"] for a, r in zip(B["greedy"][:GT], refs))
        o["greedy_same_as_b2_words"] = sum(int(u == v) for a, bg in zip(g, B["greedy"][:GT]) for u, v in zip(a, bg))
        o["greedy_same_as_b2_texts"] = sum(a == bg for a, bg in zip(g, B["greedy"][:GT]))
    o["seconds"] = round(time.time() - tc, 1)
    RES[case] = o
    for x in per:
        if case != "fin": x.pop("wl"); x.pop("ke")
    f1 = lambda v: " ".join(f"{a:5.1f}" for a in v)
    log(f"  {case:18s} top-1 30t {o['30']['top1']:5.1f} 10t {o['10']['top1']:5.1f} top-5 {o['30']['top5']:5.1f} | greedy 10t texts {o['greedy_texts']} words {o['greedy_words']}/{GT * WORDS}"
        + (f" | base-2: top-1 30t {o['b2_top1_30']:5.1f} 10t {o['b2_top1_10']:5.1f}, same word {o['agree_b2']:5.1f}%, greedy texts {o['b2_greedy_texts_10']}, "
           f"greedy same {o['greedy_same_as_b2_words']}/{GT * WORDS} words {o['greedy_same_as_b2_texts']}/{GT} texts" if B is not None else "")
        + f" | {o['seconds']}s {mem():.2f} GB")
    log(f"  {'':18s} late level moved 30t {f1([o['30']['wl'][l] for l in LATEL])}  10t {f1([o['10']['wl'][l] for l in LATEL])}"
        + (f"  base-2 30t {f1([o['b2_wl'][l] for l in LATEL])}" if B is not None else ""))
    log(f"  {'':18s} late first word  30t {f1([o['30']['fw'][l] for l in LATEL])}  10t {f1([o['10']['fw'][l] for l in LATEL])}"
        + (f"  base-2 30t {f1([o['b2_fw'][l] for l in LATEL])}" if B is not None else ""))
    log(f"  {'':18s} level moved every layer 30t {f1(o['30']['wl'])} | edges changed side 30t {f1(o['30']['ke'])}")
    json.dump(dict(size=SIZE, texts=TEXTS, greedy_texts=GT, s=S, s_rel_diff=rel, cases=RES),
              open(LOGF.name[:-4] + ".json", "w"))

log("\nSUMMARY (top-1 of all positions against this checker's finished machine; draws as min..max)")
def rng(cs, tag):
    v = [RES[c][tag]["top1"] for c in cs if c in RES]
    return f"{min(v):5.1f}..{max(v):5.1f}" if v else "-"
for b, th, mi in [(f"b:K={k}", [f"cb:K={k}@20261006", f"cb:K={k}@2"], [f"mb:K={k}@1", f"mb:K={k}@2"]) for k in KC] + \
                 [("k", ["ck@20261006", "ck@2"], ["mk@1", "mk@2"])]:
    if b not in RES: continue
    for tag in ("30", "10"):
        log(f"  {b:8s} {tag}t  b {RES[b][tag]['top1']:5.1f}  rebuilt draws {rng(th, tag)}  own draws {rng(mi, tag)}")
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
