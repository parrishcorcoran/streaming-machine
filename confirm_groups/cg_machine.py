"""confirm_groups/cg_machine.py -- AN INDEPENDENT REBUILD OF THE QWEN3-0.6B GROUP MACHINE (qwen_groups.py), by a separate
checker (rule 10). Every computation is this folder's own (cg.c, cg_common.py); nothing is imported or copied from
qwen_groups.py, base2_groups.py, groups/ or groups2/. From their docstrings only the rules were taken (g = 12 at the
slabs, the per-place rule, the expansion from w + C, the counted items, g = 16 and the set-aside at the top), and from
their code only the grouping keys (the slab index list ["qkv","proj","fc","mproj","o","gu","down"] and the top's
key [seed, 99, 1024, 16]), which the task allows.

THE MACHINE (the session's picture as docs/09 restates it; the checker's words, not his)
  A token's light reaches a plate. Take a random group of 12 light lines; every pattern's ready pile is built once a
  token, half of them by the flip (an up/hole pattern's complement is exactly its opposite pile). Each neuron, at each
  up/hole place of its plate, adds one ready pile shifted to that place; where a place of the plate is nearly dark its
  lit signed digits meet their own light values directly. That choice is made from the plate alone. At the top, one
  light meets every word, so one group table of 16 lines is read by every word, from the top place down, and after
  each place the words that can no longer reach the top 5 are set aside.
  Everything else is qwen_base2.py's own steps (its source executed): water levels, turns, the arrival, the gated bend,
  stream adds, the read's last step. One float in its path is replaced by the same whole number: the gauge's half of
  log2(n) for n = 1,024 and 128 (powers of two), checked equal to qwen_base2's value at load.

THE RULES (as the delivered docstrings give them; fixed before any run)
  slabs: q|k|v one light and one table a token, gate|up likewise, o and down one plate a slab; g = 12; groups from
    numpy's generator keyed [seed, slab index, layer, 12], a random line order cut into runs of 12 (the last short);
    seed 1 = the delivered grouping. Each plate's place b (0..15) is read in signed digits when L_b < G x M, else
    up/hole. The weight is written from w + C, C = 2^(top up/hole place), bottom place up; 2 x.w = sum over up/hole
    places 2^b V + 2 (signed part) + (K - 2C) X; the pile is the doubled pile halved.
  the top: the word table in the same per-place form at g = 16 (key [seed, 99, 1024, 16]); places read from the top
    down, each into the words still in; after each place but the last, one shared bound from the light alone,
    R2 = B(b) S1 (S1 = sum |x|; B(b) = the up/hole places below b, 2^q each, plus 2 x 2^a x t(len) for each signed run
    [a, a + len) below b, t_n = (2^(n+1) - 1) // 3); the words carried at one scale (pile << k_w, k_w = the word's power
    less the smallest); X x kcorr folded in; L5 = the 5th-best low (V + kX - R2) << k; a word is set aside when
    (V + kX + R2) << k <= L5 - 2 mu, mu = 2 + max(1, ceil(2^s / e)) (s = 16 - I, e = the power's multiplier: the
    band read's own mu). The words still in after place 0 are finished: X off, halved, qwen_base2's power; the top 5
    (by score, ties by word number) and the race winner among them.
  No knife and no stored layer (qwen_fringes2's machine has none).

THE COUNTING CONVENTIONS (the delivered machine's, as its docstrings state them; photons a position)
  plain: the plates' plain meetings ones(x) x ones(w) per line pair; the plain read ones(x) . ones(word table) per line +
    every word's divide (ones(pile) x ones(e)); the arrival's plain meetings (q16.k16 causal and shares x values,
    ones x ones).
  each slab: light = ones(x) once a slab; table = the half table's walk (group total: lines 1..gk-1 once; each Gray step
    one light value's signed digits), counted by the walk itself and checked against the relation; pwrite = ones of
    every half-table entry the slab's plates read; xsum = N - 1 adds of light values (naf, line 0 the start); per plate:
    reads = naf of each ready pile read, sdreads = naf of the light value at each lit signed digit, xoff = naf(kcorr) x
    naf(X) a neuron.
  the arrival (unchanged, qwen_fringes' count): lit digit x lit digit for q16.k16 (causal) and shares x values, plus
    q16 written (ones, each head), k16 and the values written (ones, once a key/value head). Here the arrival's
    products are qwen_base2's own int products (the count does not depend on how the same numbers are formed).
  the top read: table (walk), pwrite (ones of every entry the word table reads), lwrite (ones of the light), the light
    sums X and S1 (2 x the naf of lines 1..N-1), reads into the words still in (naf of each ready pile read; naf of
    the light value at each lit signed digit); THE CHECKER'S OWN CONVENTION for two small items the docstrings do not
    fix: the bound per test = ones(S1) x (naf(B(b)) - 1) [R2's adds] + per shift class present: ones(kX) [the X fold] +
    min(5, its words) x ones(R2) [the lows] + ones(2 mu) + ones(thr_k) [the threshold]; the words read to the end =
    naf(X) x naf(kcorr) [X off] + ones(pile) x ones(e) [the divide] a word.

PREDICTION (the checker's, written before any run of this file; the layer-0 counts of cg_common.Slab on capture's
light had been compared with the delivered record before this was written: every item equal at all 613 positions)
  - EXACT: every slab pile = the direct product at every position; every pile (3,360 digests), arrival output (840),
    stream (1,680) and top light (30) = this folder's fresh run of qwen_base2.py (cg_ref.py), capture's record and
    qwen_fringes/ref.json; every word's doubled pile from its group parts = twice the direct product; the bound holds
    for every word at every test; no word set aside at or above the 5th-best score; the words read to the end score as
    the plain read; top 5 and race winner 613/613; on the greedy texts every array equal and the race winner = the next
    greedy word 480/480.
  - WORK (a token, 613 positions): the plain basis 16,397,464,589 rebuilt exactly; every slab item equal to the
    delivered record at every position, so the plates 2,828,920,694 exactly; the top read's table, pwrite, lwrite,
    light sums, reads and survivors equal to the delivered record; its bound and end items within a few thousand
    photons a token of the delivered 6,135 and 718 (conventions not fixed by the docstrings); the arrival 34,208,215.
    WHOLE: 3,105,280,101 +- 5,000, 18.94% of plain.
  - ANOTHER GROUPING (seed 7, the checker's own): exact everywhere; the plates within 0.02 point of seed 1's share of
    plain; the top read's survivors identical (the read part after places >= b does not depend on the grouping), its
    reads within 0.05 point of the plain read; the whole within 0.02 point.
  - SHUFFLE (each slab's light shuffled across its lines, seed-1 groups, count only): within 0.1 point of plain.
  - ORDER-FREE (texts 0-2: groups in a random order, each group's lines in a random order, the reads taken group by
    group in a random order, the signed digits in a random order; at the top the same, and the words relabelled): every
    pile, stream and top 5 identical; reads, sdreads, pwrite, light and X identical; only the table's walk changes, by
    under 0.001 point of plain.

RESULTS (2026-10-07; confirm_groups/out/report.log; a token on the 613 positions)
  - EXACT, everywhere: every slab pile = the direct product at every position (seed 1, seed 7, greedy texts,
    order-free, and the shuffled light against its own product); piles 3,360/3,360, arrival outputs 840/840, streams
    1,680/1,680, top light 30/30 = the fresh qwen_base2 run (cg_ref.py), qwen_fringes/ref.json and capture's record;
    every word's pile from its group parts = the direct product; bound held for every word at every test; none set
    aside at or above the 5th-best; top 5 and race winner 613/613 (seed 1 and seed 7), score rows 613/613 against the
    fresh run and capture's; greedy texts: every array equal, race winner = the next greedy word 480/480.
  - WORK: plain 16,397,464,589 rebuilt exactly. Plates 2,828,920,694 (23.88%), the arrival 34,208,215: equal number for
    number, all 1,652 count arrays (613 positions each) equal to the delivered record. The top read 242,149,204 against
    242,151,192: every item equal at every position (table, pwrite, lwrite, light sums, reads, end, survivors) except
    the bound's count (the checker's convention 4,148 a token, the delivered 6,135; -1,987 a token, 0.00006% of the
    whole). WHOLE 3,105,278,113 against 3,105,280,101: 18.9376% both. Prediction held (within its +-5,000).
  - Seed 7: whole 18.9381% (+0.0006 point); survivors identical. Shuffle: +0.0122 point. Order-free: every pile, stream,
    top 5 identical; only the half table's walk moved (-0.00047 point of plain).

    python3 confirm_groups/cg_machine.py machine set=refs|words seed=1 [shuffle=1]
    python3 confirm_groups/cg_machine.py machine set=orderfree seed=1 reorder=4242
    python3 confirm_groups/cg_machine.py top set=refs|words|orderfree seed=1 [reorder=4242 wordperm=1]
    -> confirm_groups/out/m_<set>_s<seed>[_o<reorder>].{log,json,npz}, t_<...>.{log,json,npz}
"""
import os, sys, json, time, math, hashlib, numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_common import *
A = dict(a.split("=") for a in sys.argv[2:]); PHASE = sys.argv[1]
SET, SEED = A.get("set", "refs"), int(A.get("seed", 1)); REORDER = int(A["reorder"]) if "reorder" in A else None
SHUF = A.get("shuffle", "0") == "1"; WPERM = A.get("wordperm", "0") == "1"
TAG = f"{SET}_s{SEED}" + (f"_o{REORDER}" if REORDER is not None else "")
os.makedirs(f"{CG}/out", exist_ok=True)
LOG = open(f"{CG}/out/{'m' if PHASE == 'machine' else 't'}_{TAG}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------------- qwen_base2.py's own steps; the products hooked ----------------
class LazyW(dict):
    def __getitem__(self, name):
        if name not in self.keys(): dict.__setitem__(self, name, (None, np.asarray(member(name + ".e")).astype(np.int64)))
        return dict.__getitem__(self, name)
def _half(n):
    n = int(n); assert n & (n - 1) == 0; return ((n.bit_length() - 1) << 16) // 2
assert all(_half(n) == int(round(math.log2(n) * 65536)) // 2 for n in (1024, 128))      # check only: the same whole number
ST = dict(cnt={}, dig={}, rows=None, P=0, row0=0, text=0, chk={}, vkey=None, caplight=None)
def chk(k, v): ST["chk"][k] = ST["chk"].get(k, 0) + int(v)
def cadd(k, idx, v):
    C = ST["cnt"]
    if k not in C: C[k] = np.zeros(ST["P"], np.int64)
    C[k][idx] += v

def PRODUCT(x16, ex, name):
    slab, l = name.split("."); l = int(l); assert slab != "head", "the machine reads the top by its own group read"
    x16 = np.ascontiguousarray(np.asarray(x16, np.int64))
    S = Slab(slab, l, seed=SEED, reorder=None if REORDER is None else [REORDER, SIDX[slab], l])
    piles, c = S.run(x16)
    chk("slab piles vs the direct product (rows differing)", (piles != mm16(x16, S.Wc)).any(1).sum())
    chk("table: the walk's count vs the relation (positions differing)", (c["table"] != c["table_rel"]).sum())
    if ST["caplight"] is not None:
        chk("light at the slabs vs capture's record (rows differing)", (np.asarray(ST["caplight"][slab][l][:len(x16)]).astype(np.int64) != x16).any(1).sum())
    for k, v in c.items(): ST["cnt"][f"{name}|{k}"] = np.asarray(v, np.int64)
    ST["cnt"][f"{name}|group"] = S.group_work(c)
    if SHUF:
        xs = np.random.default_rng([555, SIDX[slab], l]).permuted(x16, axis=1)
        ps, cs = S.run(xs); chk("shuffle: piles vs the shuffled light's direct product (rows differing)", (ps != mm16(xs, S.Wc)).any(1).sum())
        ST["cnt"][f"{name}|group_shuf"] = S.group_work(cs)
    ST["rules"][name] = {p["name"]: [b for b in range(16) if b not in p["uh"]] for p in S.plates}
    for t, (a, b) in enumerate(ST["rows"]): ST["dig"][f"pile|{name}|{t}"] = digest(piles[a:b])
    return piles

def QK(q16, k16, h, l):
    r = slice(ST["row0"], ST["row0"] + len(q16)); oq, ok_, nq, nk = ones(q16), ones(k16), naf(q16), naf(k16)
    cadd(f"arr.{l}|plain_scores", r, np.tril(oq @ ok_.T).sum(1)); cadd(f"arr.{l}|sd_scores", r, np.tril(nq @ nk.T).sum(1))
    cadd(f"arr.{l}|write_q", r, oq.sum(1))
    if h % 2 == 0: cadd(f"arr.{l}|write_k", r, ok_.sum(1))
    return q16 @ k16.T
def AV(a, v, p, h, l):
    key = (l, h // 2, ST["text"])
    if ST["vkey"] != key: ST["vkey"] = key; ST["vo"] = ones(v).sum(1); ST["vn"] = naf(v).sum(1)
    r = ST["row0"] + p
    cadd(f"arr.{l}|plain_values", r, int(ones(a) @ ST["vo"][:p + 1])); cadd(f"arr.{l}|sd_values", r, int(naf(a) @ ST["vn"][:p + 1]))
    if h % 2 == 0: cadd(f"arr.{l}|write_v", r, int(ST["vo"][p]))
    return a[None, :] @ v[:p + 1]

src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a, b in [('M = np.load("qwen_base2.npz")', 'M = _M'),
             ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = _W'),
             ("x16, ex = to16(R); pile = x16 @ w", "x16, ex = to16(R); pile = PRODUCT(x16, ex, name)"),
             ("x16, ex = to16(X); return rshift(x16 @ w,", "x16, ex = to16(X); return rshift(PRODUCT(x16, ex, name),"),
             ("sc = rshift(q16 @ k16.T,", "sc = rshift(QK(q16, k16, h, l),"),
             ("out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],", "out[p, h * HD:(h + 1) * HD] = power(AV(a, v, p, h, l),"),
             ("half = int(round(math.log2(n) * 65536)) // 2", "half = _half(n)")]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
Q = dict(_M=np.load(NPZ), _W=LazyW(), PRODUCT=PRODUCT, QK=QK, AV=AV, _half=_half)
_argv = sys.argv; sys.argv = ["x"]; exec(src, Q); sys.argv = _argv
assert (Q["GAUGE"], Q["TURN"], Q["BEND"], Q["ARRIVAL"]) == ("int",) * 4

def texts_of(SET):
    if SET in ("refs", "orderfree"):
        T = json.load(open(f"{HERE}/qwen06b/light/meta.json"))["texts"]; return T if SET == "refs" else T[:3]
    return json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))["texts"]

def forward(texts):
    t0 = time.time(); lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    rows = [(int(off[i]), int(off[i + 1])) for i in range(len(texts))]; P = int(off[-1])
    ST.update(cnt={}, dig={}, rows=rows, P=P, row0=0, text=0, chk={}, vkey=None, rules={})
    if SET in ("refs", "orderfree"):
        ST["caplight"] = {k: np.load(f"{HERE}/qwen06b/light/{k}_light.npy", mmap_mode="r") for k in ("qkv", "o", "gu", "down")}
    EMB = member("embed"); R = np.asarray(EMB[np.concatenate([np.asarray(t, np.int64) for t in texts])]).astype(np.int64)
    gs, ps, att, bend, to16, gauge = (Q[k] for k in ("gauged_slab", "plain_slab", "attention", "bend", "to16", "gauge"))
    D = ST["dig"]
    for l in range(Q["L"]):
        QKV = gs(R, f"qkv.{l}"); outs = []
        for t, (a, b) in enumerate(rows): ST["row0"], ST["text"] = a, t; outs.append(att(QKV[a:b], l))
        ST["row0"] = 0; o = np.concatenate(outs); del QKV, outs
        for t, (a, b) in enumerate(rows): D[f"arrival|{l}|{t}"] = digest(o[a:b])
        R = R + ps(o, f"o.{l}"); del o
        for t, (a, b) in enumerate(rows): D[f"stream|{l}.0|{t}"] = digest(R[a:b])
        GU = gs(R, f"gu.{l}"); R = R + ps(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}"); del GU
        for t, (a, b) in enumerate(rows): D[f"stream|{l}.1|{t}"] = digest(R[a:b])
        log(f"  layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB) checks so far {ST['chk']}")
    x16, ex = to16(R); LS = np.asarray(gauge(x16, ex, Q["D"]), np.int64); ex = np.asarray(ex, np.int64)
    for t, (a, b) in enumerate(rows): D[f"top light|{t}"] = digest(np.concatenate([x16[a:b], ex[a:b, None], LS[a:b, None]], 1))
    return x16, ex, LS

# ---------------- the top read: group piles from the top place down, the band read's set-aside ----------------
def top5(s): return np.lexsort((np.arange(len(s)), -s))[:5]

def top_read(tab, X16, EX, LS, sel, batch=6):
    power, race, EXP2 = Q["power"], Q["race"], Q["EXP2"]
    NP_ = len(sel); M = tab.M; inv = np.argsort(tab.wperm); np_ = len(tab.places)
    keys = ["table", "pwrite", "lwrite", "ii_light", "ii_reads", "ii_bound", "ii_end", "ii_cmp", "plain_meet", "plain_div", "n_end", "mu"]
    out = {k: np.zeros(NP_, np.int64) for k in keys}
    out.update(surv=np.full((NP_, 18), -1, np.int64), top5=np.zeros((NP_, 5), np.int64), win=np.zeros(NP_, np.int64),
               ptop5=np.zeros((NP_, 5), np.int64), pwin=np.zeros(NP_, np.int64), reads_by_place=np.zeros((NP_, 18), np.int64))
    sha = []; bad = {}; t0 = time.time()
    def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
    for c0 in range(0, NP_, batch):
        idx = np.asarray(sel[c0:c0 + batch]); xb = np.ascontiguousarray(X16[idx].astype(np.int64)); n = len(xb)
        PART, HC, wk, pw = tab.parts(xb)
        PILE = mm16(xb, tab.Wc)                                                # CHECK PATH: the direct product, every word
        for i in range(n):
            q = c0 + i; x = xb[i]; X = int(x.sum()); S1 = int(np.abs(x).sum()); kX = tab.kcorr * X
            F0 = (tab.EMIN + int(EX[idx[i]]) + 16) * 65536 - int(LS[idx[i]])
            I = F0 >> 16; e = int(EXP2[F0 & 0xFFFF]); s = 16 - I; ge = ones1(e)
            twos = Fraction(2) ** s; delta = max(1, -((-twos.numerator) // (e * twos.denominator))); mu = 2 + delta; mu2 = 2 * mu
            out["mu"][q] = mu; out["table"][q] = wk[i]; out["pwrite"][q] = pw[i]; out["lwrite"][q] = int(ones(x).sum())
            out["ii_light"][q] = 2 * int(naf(x[1:]).sum())
            # CHECK PATH (never used by the read): every word's doubled pile from its parts, and every word's score
            CUM = np.cumsum(PART[i], axis=1)
            nb("piles from the group parts vs the direct product (words)", (CUM[:, -1] + kX != 2 * PILE[i]).sum())
            Sl = power(PILE[i], F0 + tab.KW * 65536); So = Sl[inv]
            sha.append(hashlib.sha256(np.ascontiguousarray(So, np.int64).tobytes()).hexdigest())
            out["ptop5"][q] = top5(So); out["pwin"][q] = int(race(So)); s5 = np.sort(So)[-5]
            out["plain_meet"][q] = int(ones(x) @ tab.PPH); out["plain_div"][q] = int(ones(PILE[i]).sum()) * ge
            # THE READ
            alive = np.arange(M); V = np.zeros(M, np.int64); K = tab.KW.copy(); reads = 0; bph = 0; cmp = 0
            for ii, b in enumerate(tab.places):
                r_ = int(HC[i][alive, ii].sum()); reads += r_; out["reads_by_place"][q, b] = r_
                V = V + PART[i][alive, ii]
                if ii == np_ - 1: break
                R2 = tab.B[b] * S1
                nb("bound broken (any word, check)", (np.abs(CUM[:, -1] - CUM[:, ii]) > R2).sum())
                Vx = V + kX
                low = (Vx << K) - (R2 << K)
                L5 = int(np.partition(low, len(low) - 5)[len(low) - 5])
                keep = (Vx << K) + (R2 << K) > L5 - mu2
                bph += ones1(S1) * (naf1(tab.B[b]) - 1)
                for k in range(4):
                    nk = int((K == k).sum())
                    if nk: bph += ones1(kX) + min(5, nk) * ones1(R2) + ones1(mu2) + ones1((R2 << k) + mu2)
                cmp += 2 * len(alive) + sum(min(5, int((K == k).sum())) for k in range(4))
                nb("a word set aside that is not below the 5th-best", (Sl[alive[~keep]] >= s5).sum())
                alive, V, K = alive[keep], V[keep], K[keep]; out["surv"][q, b] = len(alive)
            Dend = V + kX; nb("a finished doubled pile is odd", (Dend & 1).sum()); pe = Dend >> 1
            nb("words read to the end: pile vs the direct product", (pe != PILE[i][alive]).sum())
            sc = power(pe, F0 + K * 65536)
            nb("words read to the end vs the plain read's score", (sc != Sl[alive]).sum())
            out["ii_reads"][q] = reads; out["ii_bound"][q] = bph; out["ii_cmp"][q] = cmp; out["n_end"][q] = len(alive)
            out["ii_end"][q] = len(alive) * naf1(X) * naf1(tab.kcorr) + int(ones(pe).sum()) * ge
            Sp = np.full(M, -(1 << 62), np.int64); Sp[tab.wperm[alive]] = sc
            out["top5"][q] = top5(Sp); out["win"][q] = int(race(Sp))
        del PART, HC, PILE
        if (c0 // batch) % 10 == 0 or c0 + batch >= NP_: log(f"    {min(NP_, c0 + batch)}/{NP_} positions read ({time.time() - t0:.0f}s, {rss()} MB) {bad}")
    out["sha"] = sha; out["bad"] = bad; out["sel"] = np.asarray(sel)
    return out

if PHASE == "machine":
    texts = texts_of(SET)
    log(f"THE GROUP MACHINE (checker's rebuild) on {SET}: {len(texts)} texts, {sum(map(len, texts))} positions; seed {SEED}, "
        f"reorder {REORDER}, shuffle control {SHUF}")
    x16, ex, LS = forward(texts)
    log(f"  checks: {ST['chk']}")
    np.savez(f"{CG}/out/m_{TAG}.npz", **{k.replace("|", "__"): v for k, v in ST["cnt"].items()})
    np.savez(f"{CG}/out/m_{TAG}_top_light.npz", x16=x16, ex=ex, LS=LS)
    json.dump(dict(dig=ST["dig"], chk=ST["chk"], rows=ST["rows"], rules=ST["rules"]), open(f"{CG}/out/m_{TAG}.json", "w"))
    log(f"done, max memory {rss()} MB")
elif PHASE == "top":
    L_ = np.load(f"{CG}/out/m_{TAG}_top_light.npz"); X16, EX, LS = L_["x16"], L_["ex"], L_["LS"]
    if SET == "words": sel = json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))["read_rows"]
    else: sel = list(range(len(X16)))
    wp = np.random.default_rng([4343, REORDER or 0]).permutation(151936) if WPERM else None
    log(f"THE TOP READ (checker's rebuild) on {SET}: {len(sel)} positions; seed {SEED}, reorder {REORDER}, words relabelled {WPERM}")
    tab = TopTable(seed=SEED, reorder=None if REORDER is None else [REORDER, 99], word_perm=wp, log=log)
    log(f"  places read, top down: {tab.places}; B(b) {[tab.B[b] for b in tab.places]}")
    o = top_read(tab, X16, EX, LS, sel)
    log(f"  checks: {o['bad']}")
    log(f"  top 5 = the plain read's {int((o['top5'] == o['ptop5']).all(1).sum())}/{len(sel)}, race winner {int((o['win'] == o['pwin']).sum())}/{len(sel)}")
    np.savez(f"{CG}/out/t_{TAG}.npz", **{k: v for k, v in o.items() if k not in ("sha", "bad")})
    json.dump(dict(sha=o["sha"], bad=o["bad"], places=tab.places, uh=tab.uh, sp=tab.sp, B={str(k): v for k, v in tab.B.items()}), open(f"{CG}/out/t_{TAG}.json", "w"))
    log(f"done, max memory {rss()} MB")
