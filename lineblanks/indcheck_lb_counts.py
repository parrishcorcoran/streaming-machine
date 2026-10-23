"""INDEPENDENT CHECKER (a separate session's own code, rule 10) of lineblanks way2.py, pairs.py and totals.py at 70m.
2026-10-06. Nothing here is imported from lineblanks/common.py, way2.py, pairs.py or totals.py; it reads only the
stored plates (../foundation0/base2_pythia-70m.npz), the captured light and writes (light_70m_*.npz), and, to compare,
the JSON the three scripts wrote.

HIS WORDS (2026-10-06), only these are his: "Can you line up the blanks and see if it all cancels?"

WHAT IS RECOUNTED, by other arithmetic:
 1. WAY 2 at three plates, qkv.5, mproj.2 and head (check.py used proj.0, qkv.3, mproj.5), blank and finished on both
    lights (Lf, Lb), and MY OWN fresh bell draws (fresh_b, fresh_f; my seeds, my spread, through fp16 onto the grid)
    on Lf. 1s from a 16-bit lookup table built with Python's bin(); signed digits from a NAF table built digit by digit;
    pours as popcounts of every product x w formed directly; zero gaps; and the neurons-in-random-order control with
    my own permutations. A walk by cumulative sums (r_k = sum of x gap) checked against x @ w at qkv.5.
    Also: are the stored plates fp16 numbers (q 2^e exactly representable in fp16)?
 2. UPS PAIRED WITH DOWNS at four plates, proj.3, fc.4, head, embed: rows and columns, real sizes from float64 ldexp,
    left = sum over k of |max(s_desc_k, 0) - max(-s_asc_k, 0)| on each line sorted once (a different formula from
    pairs.py's two masked sorts). Blank and finished compared digit for digit; my fresh draws and my plate-wide
    re-deals compared as ratios. The head's lanes: share of the 50,304 words on the lane's majority sign.
 3. EVERY WRITE LINE ADDED: word rows (E), attention-out rows (A), MLP-out rows (M), blank and finished: per-line rms z,
    lane-total rms z (and % beyond 3, largest), r(E, M) and cancel ratio, my own 2,000 lane orders for r's range;
    the finished biases; Q of the machine's own writes (both runs), my own lane and position scrambles; the standing /
    word split of Q - 1 for the finished run (M2-M5, M2-M4, A5-M5, M5's and M2's share).

PREDICTIONS (written before this script's first run):
  - every count of blank and finished in part 1 (plain, walk, pours, writing, zero gaps) and part 2 (left, whole,
    |net|) equal to the stored JSON, digit for digit.
  - my fresh draws: way 2 share within 0.1 point of the stored fresh draw at qkv.5 and mproj.2, within 0.3 at the head;
    the blank within 0.1 point of my fresh_b at every one of the three plates.
  - my fresh / re-dealt pairing leftovers within 5% of the stored ones; the head's lanes: finished median lane >= 99%
    of words on one sign, blank about 50%.
  - part 3: rms z per line and per lane equal to the log at its printed decimals (E 0.96 / 14.90, A 0.98 / 1.03,
    M 0.98 / 1.97, ALL 0.97 / 13.53; per line 1.004 / 0.215, 1.010 / 0.207, 0.998 / 0.204), r(E, M) +0.029 / +0.085,
    cancel 1.015 / 1.010, my lane-order range within 0.01 of [-0.086, +0.087]; Q 0.980 / 0.980, scrambles at 1.00
    (lanes) and about 0.98 / 1.34 (positions); M2-M5 -0.229, M2-M4 -0.204, A5-M5 +0.375.

    python3 indcheck_lb_counts.py        (writes indcheck_lb_counts_70m.log / .json)
"""
import os, sys, json, math, time, resource
import numpy as np

SIZE = "70m"
HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
LOGF = open(os.path.join(HERE, f"indcheck_lb_counts_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()
OUT = {}; FAILS = []
def same(label, mine, theirs):
    ok = mine == theirs
    if not ok: FAILS.append(label)
    log(f"    {'SAME' if ok else 'DIFF'} {label}: mine {mine}  stored {theirs}")
    return ok

# ------------------------------------------------------------------------------------------------- my own digit tables
POP16 = np.array([bin(v).count("1") for v in range(1 << 16)], np.int64)
def my_ones(X):
    a = np.abs(np.asarray(X, np.int64)); assert a.max(initial=0) < 1 << 48
    return POP16[a & 0xFFFF] + POP16[(a >> 16) & 0xFFFF] + POP16[(a >> 32) & 0xFFFF]
def naf_count(v):
    c = 0
    while v:
        if v & 1:
            dgt = 2 - (v & 3); c += 1; v -= dgt
        v >>= 1
    return c
NAF = np.array([naf_count(v) for v in range((1 << 16) + 1)], np.int64)
assert naf_count(3) == 2 and naf_count(7) == 2 and naf_count(5) == 2 and naf_count(0xFFFF) == 2
def my_naf(X):
    a = np.abs(np.asarray(X, np.int64)); assert a.max(initial=0) <= 1 << 16; return NAF[a]

Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L_ = int(Z["L"]); d = int(Z["d"])
LIGHT = {"Lf": np.load(os.path.join(HERE, f"light_{SIZE}_finished.npz")), "Lb": np.load(os.path.join(HERE, f"light_{SIZE}_start.npz"))}
W2J = json.load(open(os.path.join(HERE, f"way2_{SIZE}.json")))
PJ = json.load(open(os.path.join(HERE, f"pairs_{SIZE}.json")))
TJ = json.load(open(os.path.join(HERE, f"totals_{SIZE}.json")))

def my_fresh(q, e, axis, seed):
    """my bell draw: each neuron (one power of two each, along `axis`) with its own population spread in grains; the real
    value g 2^e written as fp16, divided back by 2^e and rounded onto the grid, held inside +-32767"""
    rng = np.random.default_rng(seed); qf = q.astype(np.float64)
    sig = qf.std(axis=1 - axis)                                    # axis 1: e per column -> spread over rows
    shape = q.shape
    g = rng.standard_normal(shape) * (sig[None, :] if axis == 1 else sig[:, None])
    sc = np.ldexp(1.0, (e[None, :] if axis == 1 else e[:, None]).astype(np.int64))
    v = np.round((g * sc).astype(np.float16).astype(np.float64) / sc)
    return np.clip(v, -32767, 32767).astype(np.int16)

# ================================================================================ 0. are the stored plates fp16 numbers?
log(f"INDEPENDENT CHECK of way2 / pairs / totals, pythia-{SIZE}")
log("\n0. the stored plates as fp16 numbers: share of q 2^e exactly representable in fp16")
for name, axis in (("proj.0", 1), ("fc.3", 1), ("embed", 0)):
    for key in (".w", ".s"):
        q = np.asarray(Z[name + key]).astype(np.float64); e = np.asarray(Z[name + ".e"]).astype(np.int64)
        r = np.ldexp(q, e[None, :] if axis == 1 else e[:, None]); ok = (r.astype(np.float16).astype(np.float64) == r).mean()
        log(f"    {name}{key}: {ok:.6f}")
        OUT.setdefault("fp16", {})[name + key] = float(ok)

# ================================================================================ 1. way 2
log("\n1. WAY 2 recounted (shares of plain; blank and finished compared digit for digit with way2_70m.json)")
PLATES1 = [f"{k}.{l}" for l in range(L_) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
def pours_count(x, w):
    P, Ln = x.shape; N = w.shape[1]; tot = 0
    b = max(1, (1 << 24) // (P * N)); x32 = x.astype(np.int32); w32 = w.astype(np.int32)
    for i0 in range(0, Ln, b):
        pr = x32[:, i0:i0 + b, None] * w32[None, i0:i0 + b, :]
        tot += int(np.bitwise_count(np.abs(pr)).sum(dtype=np.int64))
    return tot
# the popcount used for the pours checked against my table on a sample
_s = np.random.default_rng(1).integers(-(1 << 30) + 1, 1 << 30, 100000)
assert (np.bitwise_count(np.abs(_s)).astype(np.int64) == my_ones(_s)).all()

R1 = {}
for name in ("qkv.5", "mproj.2", "head"):
    pi = PLATES1.index(name); tp = time.time()
    e = np.asarray(Z[name + ".e"]).astype(np.int64)
    vers = {"blank": np.asarray(Z[name + ".s"]), "finished": np.asarray(Z[name + ".w"])}
    vers["fresh_b"] = my_fresh(vers["blank"], e, 1, [4242, pi, 1])
    vers["fresh_f"] = my_fresh(vers["finished"], e, 1, [4242, pi, 2])
    R1[name] = {}
    for v, w in vers.items():
        w64 = w.astype(np.int64); Ln, N = w.shape
        srt = np.sort(w64, axis=1); gaps = np.diff(srt, axis=1); assert (gaps >= 0).all() and gaps.max() < 1 << 16
        GN = my_naf(srt[:, 0]) + my_naf(gaps).sum(1); ZG = int((gaps == 0).sum())
        rng = np.random.default_rng([4242, pi, 3, list(vers).index(v)]); sh = rng.permuted(w64, axis=1)
        GNS = my_naf(sh[:, 0]) + my_naf(np.diff(sh, axis=1)).sum(1); del sh
        W1 = my_ones(w64).sum(1)
        r = R1[name][v] = dict(zero_gaps=ZG, gaps_total=Ln * (N - 1))
        for lt in (("Lf", "Lb") if v in ("blank", "finished") else ("Lf",)):
            x = np.asarray(LIGHT[lt]["light." + name]).astype(np.int64)
            ox, nx = my_ones(x), my_naf(x)
            c = dict(plain=int((ox @ W1).sum()), walk=int((nx @ GN).sum()), writing=int(ox.sum()), pours=pours_count(x, w64),
                     walk_sh=int((nx @ GNS).sum()))
            c["way2"] = c["walk"] + c["pours"] + c["writing"]
            c["share"] = c["way2"] / c["plain"]; c["share_sh"] = (c["walk_sh"] + c["pours"] + c["writing"]) / c["plain"]
            r[lt] = c
            if name == "qkv.5" and v in ("blank", "finished") and lt == "Lf":      # a walk by cumulative sums against x @ w
                o = np.argsort(w64, axis=1, kind="stable"); pile = np.zeros((x.shape[0], N), np.int64)
                lowv = srt[:, 0]
                for i in range(Ln):
                    run = x[:, i:i + 1] * lowv[i] + np.cumsum(x[:, i:i + 1] * np.concatenate([[0], gaps[i]])[None, :], axis=1)
                    pile[:, o[i]] += run
                c["walk_piles_equal_direct"] = bool(np.array_equal(pile, x @ w64)); del pile
        for lt in r:
            if lt in ("Lf", "Lb") and v in ("blank", "finished"):
                st = W2J[name][v][lt]
                for k in ("plain", "walk", "pours", "writing", "way2"):
                    same(f"{name} {v} {lt} {k}", r[lt][k], st[k])
        if v in ("blank", "finished"):
            same(f"{name} {v} zero gaps", ZG, W2J[name][v]["zero_gaps"])
        del w64
    st = W2J[name]
    log(f"  {name}: way 2 share on Lf  mine: " + "  ".join(f"{v} {R1[name][v]['Lf']['share']:.4%}" for v in vers)
        + "  | stored: " + "  ".join(f"{v} {st[v]['Lf']['way2'] / st[v]['Lf']['plain']:.4%}" for v in vers))
    log(f"  {name}: way 2 share on Lb  mine: blank {R1[name]['blank']['Lb']['share']:.4%} finished {R1[name]['finished']['Lb']['share']:.4%}")
    log(f"  {name}: zero gaps mine: " + "  ".join(f"{v} {R1[name][v]['zero_gaps'] / R1[name][v]['gaps_total']:.2%}" for v in vers)
        + "  | stored: " + "  ".join(f"{v} {st[v]['zero_gaps'] / st[v]['gaps_total']:.2%}" for v in vers))
    log(f"  {name}: neurons in a random order, Lf (mine / stored): " + "  ".join(
        f"{v} {R1[name][v]['Lf']['share_sh']:.2%} / {(st[v]['Lf']['walk_neurons_shuffled'] + st[v]['Lf']['pours'] + st[v]['Lf']['writing']) / st[v]['Lf']['plain']:.2%}" for v in vers))
    if name == "qkv.5":
        log(f"  qkv.5 walk by cumulative sums equals x @ w: blank {R1[name]['blank']['Lf']['walk_piles_equal_direct']}, finished {R1[name]['finished']['Lf']['walk_piles_equal_direct']}")
    log(f"  ({time.time() - tp:.0f}s, {rss()} MB)")
    json.dump(dict(OUT, part1=R1), open(os.path.join(HERE, f"indcheck_lb_counts_{SIZE}.json"), "w"))
OUT["part1"] = R1
# the pooled ALL shares as printed, from the stored JSON (arithmetic of the pooling only)
for lt in ("Lf", "Lb"):
    sh = {v: sum(W2J[n][v][lt]["way2"] for n in PLATES1) / sum(W2J[n][v][lt]["plain"] for n in PLATES1) for v in ("blank", "finished", "fresh_b", "fresh_f")}
    log(f"  pooled from the stored JSON, {lt}: " + "  ".join(f"{v} {s:.2%}" for v, s in sh.items()))

# ================================================================================ 2. ups paired with downs
log("\n2. UPS PAIRED WITH DOWNS by size (one sort a line; blank and finished compared with pairs_70m.json)")
def leftover(V):
    """V: lines as rows (float64 whole numbers). left, whole, |net| summed over lines, as Python ints"""
    left = whole = absnet = 0
    bs = max(1, (1 << 22) // V.shape[1])
    for a in range(0, V.shape[0], bs):
        s = np.sort(V[a:a + bs], axis=1)
        lf = np.abs(np.maximum(s[:, ::-1], 0) - np.maximum(-s, 0)).sum(1)
        left += int(lf.sum()); whole += int(np.abs(s).sum()); absnet += int(np.abs(s.sum(1)).sum())
    return left, whole, absnet
R2 = {}
for name, axis in (("proj.3", 1), ("fc.4", 1), ("head", 1), ("embed", 0)):
    tp = time.time(); e = np.asarray(Z[name + ".e"]).astype(np.int64); emin = int(e.min())
    pi = [n for n in PLATES1] .index(name) if name in PLATES1 else 99
    base = {"blank": np.asarray(Z[name + ".s"]), "finished": np.asarray(Z[name + ".w"])}
    vers = dict(base)
    vers["fresh_b"] = my_fresh(base["blank"], e, axis, [5151, pi, 1]); vers["fresh_f"] = my_fresh(base["finished"], e, axis, [5151, pi, 2])
    R2[name] = {}
    for v in ("blank", "finished", "fresh_b", "fresh_f", "redeal_b", "redeal_f"):
        if v.startswith("redeal"):
            V = np.ldexp(base["blank" if v.endswith("b") else "finished"].astype(np.float64), (e - emin)[None, :] if axis == 1 else (e - emin)[:, None])
            V = np.random.default_rng([5151, pi, 3, v[-1] == "f"]).permutation(V.ravel()).reshape(V.shape)
        else:
            V = np.ldexp(vers[v].astype(np.float64), (e - emin)[None, :] if axis == 1 else (e - emin)[:, None])
        rows = leftover(V); cols = leftover(np.ascontiguousarray(V.T))
        R2[name][v] = dict(rows=rows, cols=cols)
        if v in ("blank", "finished"):
            for along, got in (("rows", rows), ("cols", cols)):
                st = PJ[name][v][along]; same(f"{name} {v} {along} (left, whole, |net|)", got, (st["left"], st["whole"], st["absnet"]))
        if name == "head" and v in ("blank", "finished"):
            pos = (V > 0).sum(1); neg = (V < 0).sum(1); maj = np.maximum(pos, neg) / V.shape[1]
            col_mean = V.mean(1)                                   # the mean word column (over the 50,304 words), per lane
            typical = math.sqrt(float((V * V).sum(0).mean()))      # rms size of a word column
            R2[name][v]["sign"] = dict(median_lane_majority=float(np.median(maj)), mean_col_over_typical=float(np.linalg.norm(col_mean) / typical))
            log(f"    head {v}: median lane has {np.median(maj):.4%} of words on its majority sign; |mean word column| / rms word column {np.linalg.norm(col_mean) / typical:.4f}")
        del V
    st = PJ[name]; f = lambda r: r[0] / r[1]
    smap = {"redeal_b": "shuffled_b", "redeal_f": "shuffled_f"}
    for along, ai in (("rows", "rows"), ("cols", "cols")):
        log(f"  {name} {along}: left/whole mine: " + "  ".join(f"{v} {f(R2[name][v][ai]):.3%}" for v in R2[name])
            + "  | stored: " + "  ".join(f"{v} {st[smap.get(v, v)][along]['left'] / st[smap.get(v, v)][along]['whole']:.3%}" for v in R2[name]))
    log(f"  {name}: finished / blank rows {f(R2[name]['finished']['rows']) / f(R2[name]['blank']['rows']):.2f}x  cols {f(R2[name]['finished']['cols']) / f(R2[name]['blank']['cols']):.2f}x"
        f"  ({time.time() - tp:.0f}s, {rss()} MB)")
OUT["part2"] = {n: {v: {k: (list(x) if isinstance(x, tuple) else x) for k, x in r.items()} for v, r in R2[n].items()} for n in R2}
json.dump(OUT, open(os.path.join(HERE, f"indcheck_lb_counts_{SIZE}.json"), "w"))

# ================================================================================ 3. every write line added
log("\n3A. EVERY WRITE LINE ADDED (real values from ldexp, one scale)")
WPL = [("embed", 0, "E")] + [(f"proj.{l}", 1, "A") for l in range(L_)] + [(f"mproj.{l}", 1, "M") for l in range(L_)]
EMIN = min(int(np.asarray(Z[n + ".e"]).min()) for n, _, _ in WPL)
def z_of(t, w2):
    z = t / np.sqrt(np.maximum(w2, 1)); return dict(rms=float(np.sqrt((z * z).mean())), beyond3=float((np.abs(z) > 3).mean()), max=float(np.abs(z).max()))
R3 = {}
for v, key in (("blank", ".s"), ("finished", ".w")):
    T = {g: np.zeros(d) for g in "EAM"}; W2 = {g: np.zeros(d, np.int64) for g in "EAM"}; PL = {g: [] for g in "EAM"}
    for name, axis, g in WPL:
        q = np.asarray(Z[name + key]).astype(np.int64); e = np.asarray(Z[name + ".e"]).astype(np.int64)
        mult = (2 ** (e - EMIN)).astype(np.int64)
        V = q * (mult[None, :] if axis == 1 else mult[:, None]); assert np.abs(V).max() < 1 << 21
        Vf = V.astype(np.float64)
        PL[g].append(z_of(Vf.sum(1), (V * V).sum(1).astype(np.float64)) | dict(t=Vf.sum(1), w2=(V * V).sum(1).astype(np.float64)))
        T[g] += Vf.sum(0); W2[g] += (V * V).sum(0)
    r = R3[v] = {"per_line": {}, "lanes": {}}
    for g in "EAM":
        t = np.concatenate([p["t"] for p in PL[g]]); w2 = np.concatenate([p["w2"] for p in PL[g]]); r["per_line"][g] = z_of(t, w2)
    for gs in ("E", "A", "M", "AM", "EAM"):
        r["lanes"][gs] = z_of(sum(T[g] for g in gs), sum(W2[g] for g in gs).astype(np.float64))
    a, b = T["E"], T["M"]
    r["rEM"] = float(a @ b / math.sqrt((a @ a) * (b @ b))); r["cancelEM"] = float(((a + b) @ (a + b)) / (a @ a + b @ b))
    rng = np.random.default_rng([6161, v == "finished"])
    nul = np.array([a @ b[rng.permutation(d)] / math.sqrt((a @ a) * (b @ b)) for _ in range(2000)])
    r["rEM_range"] = [float(np.quantile(nul, .025)), float(np.quantile(nul, .975))]
    r["T_equal_stored"] = {g: bool(np.array_equal(T[g].astype(np.int64), np.array(TJ["A"][v]["T"][g], np.int64))) for g in "EAM"}
    if v == "finished":
        bias = np.zeros(d); b2 = np.zeros(d)
        for l in range(L_):
            for k in ("proj", "mproj"):
                bb = np.asarray(Z[f"{k}.{l}.b"]).astype(np.float64) * 2.0 ** (-16 - EMIN); bias += bb; b2 += bb * bb
        allw2 = sum(W2[g] for g in "EAM").astype(np.float64)
        r["bias"] = z_of(bias, allw2); r["with_bias"] = z_of(sum(T[g] for g in "EAM") + bias, allw2 + b2)
    log(f"  {v:8s}: per line rms z  E {r['per_line']['E']['rms']:.3f}  A {r['per_line']['A']['rms']:.3f}  M {r['per_line']['M']['rms']:.3f}")
    log(f"  {v:8s}: lane totals rms z (beyond 3, max): " + "  ".join(f"{gs} {r['lanes'][gs]['rms']:.2f} ({r['lanes'][gs]['beyond3']:.2%}, {r['lanes'][gs]['max']:.1f})" for gs in ("E", "A", "M", "AM", "EAM")))
    log(f"  {v:8s}: r(E, M) {r['rEM']:+.4f}  cancel {r['cancelEM']:.4f}  my lane-order 95% range [{r['rEM_range'][0]:+.3f}, {r['rEM_range'][1]:+.3f}];  lane totals equal to stored T: {r['T_equal_stored']}")
    if v == "finished":
        log(f"  finished biases: rms z {r['bias']['rms']:.3f}; all write lines + biases rms z {r['with_bias']['rms']:.2f} ({r['with_bias']['beyond3']:.2%} beyond 3)")
    stA = TJ["A"][v]
    log(f"  stored  : per line E {stA['per_line']['E']['rms']:.3f} A {stA['per_line']['A']['rms']:.3f} M {stA['per_line']['M']['rms']:.3f}; lanes " +
        "  ".join(f"{gs} {stA['lanes'][gs]['rms']:.2f}" for gs in ("E", "A", "M", "AM", "EAM")) + f"; r(E,M) {stA['agree']['EM']['r']:+.4f} cancel {stA['agree']['EM']['cancel']:.4f}")
    for gs in ("E", "A", "M", "AM", "EAM"):
        if abs(r["lanes"][gs]["rms"] - stA["lanes"][gs]["rms"]) > 1e-9: FAILS.append(f"3A lanes {v} {gs}")
    for g in "EAM":
        if abs(r["per_line"][g]["rms"] - stA["per_line"][g]["rms"]) > 1e-9: FAILS.append(f"3A per line {v} {g}")
    if abs(r["rEM"] - stA["agree"]["EM"]["r"]) > 1e-9: FAILS.append(f"3A rEM {v}")
OUT["part3A"] = R3

log("\n3B. THE MACHINE'S OWN WRITES: Q = sum (top)^2 / sum of every write^2")
R3B = {}
for run in ("start", "finished"):
    W = np.load(os.path.join(HERE, f"light_{SIZE}_{run}.npz"))["writes"].astype(np.float64); P, K, _ = W.shape
    Q = lambda X: float((X.sum(1) ** 2).sum() / (X ** 2).sum())
    rng = np.random.default_rng([7171, run == "finished"])
    Wl = W.copy()
    for k in range(1, K):
        Wl[:, k] = np.take_along_axis(W[:, k], np.argsort(rng.random((P, d)), axis=1), axis=1)
    Wp = W.copy()
    for k in range(1, K): Wp[:, k] = W[rng.permutation(P), k]
    S = W.mean(0); Tt = W - S[None]; CS = P * (S @ S.T); CT = np.einsum("pkd,pjd->kj", Tt, Tt); C = CS + CT; tot = np.trace(C)
    names = ["word"] + [f"{k}{l}" for l in range(L_) for k in ("A", "M")]; ix = {n: i for i, n in enumerate(names)}
    R3B[run] = dict(Q=Q(W), Q_lanes=Q(Wl), Q_pos=Q(Wp), standing=float((CS.sum() - np.trace(CS)) / tot), word=float((CT.sum() - np.trace(CT)) / tot),
                    M2M5=float(2 * CT[ix["M2"], ix["M5"]] / tot), M2M4=float(2 * CT[ix["M2"], ix["M4"]] / tot), A5M5_standing=float(2 * CS[ix["A5"], ix["M5"]] / tot),
                    share_M5=float(C[ix["M5"], ix["M5"]] / tot), share_M2=float(C[ix["M2"], ix["M2"]] / tot), share_word=float(C[0, 0] / tot))
    b = R3B[run]
    log(f"  {run:8s}: Q {b['Q']:.4f}  lanes scrambled {b['Q_lanes']:.4f}  positions scrambled {b['Q_pos']:.4f} | standing pairs {b['standing']:+.3f}  word pairs {b['word']:+.3f}"
        f" | M2-M5 {b['M2M5']:+.3f}  M2-M4 {b['M2M4']:+.3f}  A5-M5 standing {b['A5M5_standing']:+.3f} | shares M5 {b['share_M5']:.3f} M2 {b['share_M2']:.3f} word {b['share_word']:.4f}")
    log(f"  stored  : Q {TJ['B'][run]['Q']:.4f}  lanes scrambled {TJ['B'][run]['Q_lanes_scrambled']:.4f}  positions scrambled {TJ['B'][run]['Q_positions_scrambled']:.4f}")
    if abs(b["Q"] - TJ["B"][run]["Q"]) > 1e-9: FAILS.append(f"3B Q {run}")
    del W, Wl, Wp
OUT["part3B"] = R3B
json.dump(OUT, open(os.path.join(HERE, f"indcheck_lb_counts_{SIZE}.json"), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log(f"\nexact comparisons that differ: {len(FAILS)} {FAILS}")
log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
