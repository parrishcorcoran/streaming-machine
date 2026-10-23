"""CHECK_TAGSKIP: a separate checker's rebuild of tagskip.py's numbers (rule 10). Nothing is imported or copied from
tagskip.py. base2.py's own steps are exec'd from its source (as judge5.py loads them); everything else is written here
again: the bank steps, the arrival (checked against base2's own attention), the tags, the stage walk and the counts.

THE TAG, as tagskip's docstring states it, written here in the edge's own terms rather than through a threshold T:
  a gate's pile lies in [c - Lo, c + Up] after the top diagonals read (c: what is read; Up, Lo: the signed bound of
  knife/settle.py on the unread place pairs). power() never runs backward, so
     FLAT  as soon as power(c + Up, F) + b < 0      (the whole interval on the dark side, H < 0)
     OPEN  as soon as power(c - Lo, F) + b >= 0     (the whole interval on the bright side)
  checked before each diagonal is read, 28 down to 0; a gate never proved is finished in full.
  The skip: a FLAT gate's output is 0 (glow=zero) or its stored glow (glow=stored, rebuilt here from the exact machine
  on the 30 refs texts and compared with tagskip/glow_store.npz).

    python3 tagskip/check_tagskip.py part=A     exact machine on the refs; arrival vs base2's own; stored glow rebuilt;
                                                the walk at layer 11 on the exact light; the layer-11-only skip's token
    python3 tagskip/check_tagskip.py part=B     the skip at all 12 layers on the refs: the walk at every layer, the
                                                FLAT shares, the reads, fc and mproj recounted, the token
    python3 tagskip/check_tagskip.py part=C     greedy words of 5 refs texts, skip at all layers, zero and stored glow
    python3 tagskip/check_tagskip.py part=D     judge5 windows 0-49: none, all, L11, L6-11, all+glow, shuffled tags
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tagskip"
ARG = dict(a.split("=", 1) for a in sys.argv[1:]); PART = ARG.get("part", "A")
LOGF = open(f"{OUT}/check_{PART}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)

_s = open(f"{HERE}/base2.py").read(); _s = _s[:_s.index("tok = GPT2Tokenizer")]
_s = _s.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_b = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'; assert _b in _s
_s = _s.replace(_b, "[]")
B2 = {}; _a, _c = sys.argv, os.getcwd(); sys.argv = ["x"]; os.chdir(HERE); exec(_s, B2); sys.argv = _a; os.chdir(_c)
M, ES, SC, PHI, ARR, EXP2 = (B2[k] for k in ("M", "ES", "SC", "PHI", "ARR", "EXP2"))
rshift, power, to16, gauge_log, product, log2fix, race, b2_attention, b2_edge = (B2[k] for k in
    ("rshift", "power", "to16", "gauge_log", "product", "log2fix", "race", "attention", "edge"))
PLT = {f"{k}.{l}": (M[f"{k}.{l}.w"], M[f"{k}.{l}.e"].astype(np.int64), M[f"{k}.{l}.b"].astype(np.int64))
       for l in range(12) for k in ("qkv", "proj", "fc", "mproj")}
WTE, WPE = M["wte"], M["wpe"]; HW, HE, HB = M["head.w"], M["head.e"], M["head.b"]; HWF = None
REFS = json.load(open(f"{HERE}/refs_many.json")); WINS = json.load(open(f"{HERE}/judge5_texts.json"))
N = 3072
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def gones(F): return ones(EXP2[np.bitwise_and(np.asarray(F, np.int64), 0xFFFF)])
def pct(a, b): return f"{100.0 * a / b:.2f}%" if b else "-"

def arrival(QKV):                                       # base2's arrival, every row of one text at once
    P = len(QKV); out = np.zeros((P, 768), np.int64); tri = np.tril(np.ones((P, P), bool))
    for h in range(12):
        q, k, v = (QKV[:, o + h * 64:o + (h + 1) * 64] for o in (0, 768, 1536))
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        top = np.array([sc[p, :p + 1].max() for p in range(P)])
        d = top[:, None] - sc
        a = np.where(tri & (d < len(ARR)), ARR[np.clip(d, 0, len(ARR) - 1)], 0)
        out[:, h * 64:(h + 1) * 64] = power(product(a, v.astype(np.float64)), -log2fix(a.sum(1))[:, None])
    return out

def share(H):
    i = rshift(H, ES - 12) + 32768
    return np.where(i < 0, 0, np.where(i > 65535, 65536, PHI[np.clip(i, 0, 65535)]))

_WB = {}
def plate_bits(l):                                      # the fc plate by place: signed and unsigned bits, and counts
    if l not in _WB:
        w = PLT[f"fc.{l}"][0].astype(np.int32); aw = np.abs(w)
        bits = [((aw >> b) & 1) for b in range(15)]
        _WB.clear()
        _WB[l] = (np.stack([(np.sign(w) * bt).astype(np.float32) for bt in bits]), np.stack([bt.astype(np.float32) for bt in bits]),
                  np.stack([(bt * (w > 0)).sum(0) for bt in bits]).astype(np.int64), np.stack([(bt * (w < 0)).sum(0) for bt in bits]).astype(np.int64))
    return _WB[l]

def walk(l, x16, pile, F, b, chunk=162):
    """every gate's stage (0..28 proved, 29 never), kind (0 FLAT, 1 OPEN, 2 in full), reads (meetings of the diagonals
    read before the tag), full meetings; the bound-built photons (tagskip's cost model); checks"""
    Ws, Wu, mP, mM = plate_bits(l); P = len(x16)
    stage = np.full((P, N), 29, np.int8); kind = np.full((P, N), 2, np.int8)
    reads = np.zeros((P, N), np.int64); fullm = np.zeros((P, N), np.int64); built = np.zeros((P, N), np.int64)
    POP = np.bitwise_count(np.arange(1 << 14)).astype(np.int64)
    bad = dict(diag_sum=0, meet_sum=0, outside=0, flat_not_dark=0, open_not_bright=0)
    for r0 in range(0, P, chunk):
        r1 = min(P, r0 + chunk); xc = x16[r0:r1].astype(np.int64); ax = np.abs(xc); Pc = r1 - r0
        D = np.zeros((29, Pc, N), np.int32); MD = np.zeros((29, Pc, N), np.int32)
        Ub = np.zeros((29, Pc, N), np.int32); Lb = np.zeros((29, Pc, N), np.int32)
        for a in range(15):
            bit = (ax >> a) & 1; Xs = (np.sign(xc) * bit).astype(np.float32); Xu = bit.astype(np.float32)
            nP = (bit * (xc > 0)).sum(1); nM = (bit * (xc < 0)).sum(1)
            for bb in range(15):
                D[a + bb] += np.rint(Xs @ Ws[bb]).astype(np.int32); MD[a + bb] += np.rint(Xu @ Wu[bb]).astype(np.int32)
                pp = np.minimum(nP[:, None], mP[bb][None, :]); mm = np.minimum(nM[:, None], mM[bb][None, :])
                pq = np.minimum(nP[:, None], mM[bb][None, :]); mq = np.minimum(nM[:, None], mP[bb][None, :])
                Ub[a + bb] += (pp + mm).astype(np.int32); Lb[a + bb] += (pq + mq).astype(np.int32)
                built[r0:r1] += POP[pp] + POP[mm] + POP[pq] + POP[mq]
        for j in range(29): built[r0:r1] += POP[Ub[j]] + POP[Lb[j]]
        pc = pile[r0:r1]; Fc = F[r0:r1]; s = 16 - np.right_shift(Fc, 16)
        assert int(s.min()) >= 0, "power would shift left"         # so X * EXP2 < 2^40 * 2^17 stays inside 63 bits
        tot = np.zeros((Pc, N), np.int64)
        for j in range(29): tot += D[j].astype(np.int64) << j
        bad["diag_sum"] += int((tot != pc).sum())
        fm = MD.sum(0, dtype=np.int64); fullm[r0:r1] = fm
        bad["meet_sum"] += int((fm != np.rint(ones(xc).astype(np.float64) @ ones(PLT[f"fc.{l}"][0]).astype(np.float64)).astype(np.int64)).sum())
        c = np.zeros((Pc, N), np.int64); cm = np.zeros((Pc, N), np.int64); und = np.ones((Pc, N), bool)
        U = np.zeros((Pc, N), np.int64); L = np.zeros((Pc, N), np.int64)
        for j in range(29): U += Ub[j].astype(np.int64) << j; L += Lb[j].astype(np.int64) << j
        st, kd, rd = stage[r0:r1], kind[r0:r1], reads[r0:r1]
        for k in range(29):
            hi, lo = c + U, c - L
            bad["outside"] += int(((pc > hi) | (pc < lo)).sum())
            fl = und & (power(hi, Fc) + b[None, :] < 0); op = und & (power(lo, Fc) + b[None, :] >= 0)
            st[fl | op] = k; kd[fl] = 0; kd[op] = 1; rd[fl] = cm[fl]; und &= ~(fl | op)
            j = 28 - k; c += D[j].astype(np.int64) << j; U -= Ub[j].astype(np.int64) << j; L -= Lb[j].astype(np.int64) << j; cm += MD[j]
        Hc = power(pc, Fc) + b[None, :]
        bad["flat_not_dark"] += int(((kd == 0) & (Hc >= 0)).sum()); bad["open_not_bright"] += int(((kd == 1) & (Hc < 0)).sum())
        del D, MD, Ub, Lb
    return stage, kind, reads, fullm, built, bad

def tag28(l, x16, pile, F, b):                          # the same tags from stage 28 alone (the intervals only shrink)
    Ws, Wu, mP, mM = plate_bits(l); xc = x16.astype(np.int64); bit = np.abs(xc) & 1
    D0 = np.rint((np.sign(xc) * bit).astype(np.float32) @ Ws[0]).astype(np.int64)
    nP = (bit * (xc > 0)).sum(1); nM = (bit * (xc < 0)).sum(1)
    up = np.minimum(nP[:, None], mP[0][None, :]) + np.minimum(nM[:, None], mM[0][None, :])
    dn = np.minimum(nP[:, None], mM[0][None, :]) + np.minimum(nM[:, None], mP[0][None, :])
    c = pile - D0
    return power(c + up, F) + b[None, :] < 0, power(c - dn, F) + b[None, :] >= 0

def run(texts, skip=(), glow=None, rows="all", shuffle=None, walk_at=(), count=False, keep=None, cst=None):
    """the machine on whole texts; skip: layers tagged and skipped; glow: dict l -> stored glow (None: zero);
    shuffle: an rng to shuffle each position's tags across the gates; walk_at: layers where the full walk sets the tags
    (others: stage 28); count: the fc and mproj recount; keep: dict to fill with per-layer records; cst: dict to fill
    with the exact dark sums (stored glow rebuild)"""
    global HWF
    n = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(n)]); P = off[-1]
    ids = np.concatenate([np.asarray(t) for t in texts]); pos = np.concatenate([np.arange(k) for k in n])
    R = WTE[ids] + WPE[pos]; C = {k: 0 for k in ("fc tags", "fc open", "mproj", "plain fc", "plain mproj", "lines met", "lines", "built", "flat unread")}
    for l in range(12):
        w, we, b = PLT[f"qkv.{l}"]; x16, ex, LS = gauge_log(R)
        QKV = power(product(x16, w.astype(np.float64)), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]]) for t in range(len(texts))])
        w, we, b = PLT[f"proj.{l}"]; o16, eo = to16(o, -ES)
        Ra = R + rshift(product(o16, w.astype(np.float64)), -(we[None, :] + eo[:, None] + ES)) + b
        w, we, b = PLT[f"fc.{l}"]; x16, ex, LS = gauge_log(Ra); pile = product(x16, w.astype(np.float64))
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]; H = power(pile, F) + b; sh = share(H); G = rshift(H * sh, 16)
        if cst is not None:
            dk = H < 0; cst.setdefault(l, [np.zeros(N, np.int64), np.zeros(N, np.int64)])
            cst[l][0] += np.where(dk, G, 0).sum(0); cst[l][1] += dk.sum(0)
        Gs = G
        if l in skip:
            if l in walk_at:
                stage, kind, reads, fullm, built, bad = walk(l, x16, pile, F, b); flat = kind == 0
                f28, o28 = tag28(l, x16, pile, F, b)
                bad["walk_vs_stage28_flat"] = int((f28 != flat).sum()); bad["walk_vs_stage28_open"] = int((o28 != (kind == 1)).sum())
            else:
                flat, o28 = tag28(l, x16, pile, F, b); kind = np.where(flat, 0, np.where(o28, 1, 2))
            if keep is not None:
                K = keep.setdefault(l, {}); K["P"] = flat.size; K["flat"] = int(flat.sum()); K["open"] = int((kind == 1).sum())
                K["full"] = int((kind == 2).sum()); K["dark"] = int((H < 0).sum()); K["flat_eq_dark"] = int((flat != (H < 0)).sum())
                if l in walk_at:
                    K["bad"] = bad; K["reads_flat"] = int(reads[flat].sum()); K["fullm_flat"] = int(fullm[flat].sum()); K["fullm"] = int(fullm.sum())
                    K["hist"] = np.bincount(stage[flat].astype(np.int64), minlength=30).tolist(); K["built"] = int(built.sum())
                K["glow_skipped"] = int(np.abs(G[flat]).sum()); K["out_all"] = int(np.abs(G).sum())
            if shuffle is not None: flat = np.stack([f[shuffle.permutation(N)] for f in flat])
            Gs = np.where(flat, glow[l][None, :], G) if glow is not None else np.where(flat, 0, G)
            if count:
                pg = (np.rint(ones(x16).astype(np.float64) @ ones(w).astype(np.float64)).astype(np.int64)
                      + ones(pile) * gones(-LS)[:, None] + ones(b)[None, :] + ones(H) * ones(sh))
                C["fc tags"] += int(reads[flat].sum()); C["fc open"] += int(pg[~flat].sum()); C["plain fc"] += int(pg.sum())
                C["built"] += int(built.sum()); C["flat unread"] += int((fullm - reads)[flat].sum())
                C["lines met"] += int((~flat).sum()); C["lines"] += flat.size
        w, we, b = PLT[f"mproj.{l}"]; g16, eg = to16(Gs, -ES)
        mo = rshift(product(g16, w.astype(np.float64)), -(we[None, :] + eg[:, None] + ES)) + b
        if l in skip and (count or keep is not None):
            gx, egx = to16(G, -ES); mox = rshift(product(gx, w.astype(np.float64)), -(we[None, :] + egx[:, None] + ES)) + b
            if count:
                pw = ones(w).sum(1).astype(np.float64)
                C["mproj"] += int(np.rint(ones(g16).astype(np.float64) @ pw).sum()); C["plain mproj"] += int(np.rint(ones(gx).astype(np.float64) @ pw).sum())
            if keep is not None: keep[l]["push"] = int(np.abs(mox - mo).sum()); keep[l]["write"] = int(np.abs(mox).sum())
        R = Ra + mo
    rr = {"all": np.arange(P), "last": off[1:] - 1, "judged": np.concatenate([np.arange(off[t] + 3, off[t] + 23) for t in range(len(texts))])}[rows]
    x16, ex, LS = gauge_log(R[rr]); Fp = (int(HE[0]) + ex + ES) * 65536 - LS
    if HWF is None: HWF = HW.astype(np.float64)
    T5, WN = [], []
    for p0 in range(0, len(rr), 64):
        S = power(product(x16[p0:p0 + 64], HWF), Fp[p0:p0 + 64, None]) + HB[None, :]
        i5 = np.argpartition(-S, 5, axis=1)[:, :5]; T5.append(np.take_along_axis(i5, np.argsort(-np.take_along_axis(S, i5, 1), axis=1), 1))
        WN.append(np.array([race(s) for s in S]))
    return np.concatenate(T5), np.concatenate(WN), C

REFT = [r["ids"] + r["ref"] for r in REFS]

# ======================================================================================================== A
if PART == "A":
    log("PART A: the exact machine on the refs, the arrival against base2's own, the stored glow rebuilt, layer 11's walk")
    # the arrival against base2's own attention (its per-position race), on 3 texts, all 12 layers, exact machine
    nbad = 0
    for t in (0, 13, 29):
        R = WTE[np.array(REFT[t])] + WPE[:len(REFT[t])]
        for l in range(12):
            w, we, b = PLT[f"qkv.{l}"]; x16, ex, LS = gauge_log(R)
            QKV = power(product(x16, w.astype(np.float64)), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
            o = arrival(QKV); nbad += int((o != b2_attention(QKV)).sum())
            w, we, b = PLT[f"proj.{l}"]; o16, eo = to16(o, -ES); Ra = R + rshift(product(o16, w.astype(np.float64)), -(we[None, :] + eo[:, None] + ES)) + b
            w, we, b = PLT[f"fc.{l}"]; x16, ex, LS = gauge_log(Ra); pile = product(x16, w.astype(np.float64))
            H = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; G = rshift(H * share(H), 16)
            nbad += int((G != b2_edge(H)).sum())
            w, we, b = PLT[f"mproj.{l}"]; g16, eg = to16(G, -ES); R = Ra + rshift(product(g16, w.astype(np.float64)), -(we[None, :] + eg[:, None] + ES)) + b
    log(f"  this file's arrival and bend against base2's own attention and edge (3 texts, 12 layers): values differing {nbad}")
    # the exact machine, all 648 positions, with the stored glow rebuilt
    cst = {}; E5, EW, _ = run(REFT, cst=cst)
    CST = {l: np.where(cst[l][1] > 0, (2 * cst[l][0] + cst[l][1]) // np.maximum(2 * cst[l][1], 1), 0) for l in range(12)}
    gs = np.load(f"{OUT}/glow_store.npz")
    log(f"  stored glow rebuilt vs tagskip/glow_store.npz: gates differing {sum(int((CST[l] != gs[f'c{l}']).sum()) for l in range(12))}/{12 * N}; "
        f"dark counts differing {sum(int((cst[l][1] != gs[f'n{l}']).sum()) for l in range(12))}")
    np.savez(f"{OUT}/check_A.npz", E5=E5, EW=EW, **{f"c{l}": CST[l] for l in range(12)})
    wa = json.load(open(f"{OUT}/work_none_real.json"))
    log(f"  exact machine's words vs tagskip skip=none run: top 5 differing {int((np.array(wa['top5']) != E5).any(1).sum())}/648, words differing {int((np.array(wa['win']) != EW).sum())}")
    # the skip at layer 11 only: the walk on the exact light, the token, zero and stored glow
    for name, glow in (("zero", None), ("stored glow", CST)):
        keep = {}; T5, WN, C = run(REFT, skip=(11,), glow=glow, walk_at=(11,), count=True, keep=keep)
        K = keep[11]; h = np.cumsum(K["hist"]) / max(K["flat"], 1)
        log(f"  skip at L11, {name}: FLAT {pct(K['flat'], K['P'])}, OPEN {pct(K['open'], K['P'])}, in full {pct(K['full'], K['P'])}, "
            f"dark {pct(K['dark'], K['P'])}, FLAT != dark at {K['flat_eq_dark']} gates; gates finished in full {K['full']} (never proved); checks {K['bad']}")
        log(f"     FLAT proved by stage (cumulative): " + ", ".join(f"D{k} {100 * h[k]:.1f}%" for k in (12, 14, 16, 18, 20, 24, 28)))
        log(f"     the tags read {pct(K['reads_flat'], K['fullm_flat'])} of the FLAT gates' meetings, {pct(K['reads_flat'], K['fullm'])} of the fc meetings")
        log(f"     FLAT outputs {pct(K['glow_skipped'], K['out_all'])} of the gates' summed size; push {pct(K['push'], K['write'])} of the write")
        log(f"     a token (L11 only): fc tags {C['fc tags'] / 648:,.0f}, fc open {C['fc open'] / 648:,.0f}, bound built {C['built'] / 648:,.0f}, "
            f"mproj (L11, this glow's lines as met) {C['mproj'] / 648:,.0f} of plain {C['plain mproj'] / 648:,.0f}")
        log(f"     the token, teacher-forced vs the exact machine: same top word {int((WN == EW).sum())}/648, in top 5 {int((E5 == WN[:, None]).any(1).sum())}/648, "
            f"top 5 set and order {int((T5 == E5).all(1).sum())}/648")
    log(f"({el()}s, max memory {mb()} MB)")

# ======================================================================================================== B
if PART == "B":
    log("PART B: the skip at all 12 layers on the refs (zero), the full walk at every layer, fc and mproj recounted")
    A = np.load(f"{OUT}/check_A.npz"); E5, EW = A["E5"], A["EW"]
    keep = {}; T5, WN, C = run(REFT, skip=set(range(12)), walk_at=set(range(12)), count=True, keep=keep)
    ja = json.load(open(f"{OUT}/work_all_real.json"))
    for l in range(12):
        K = keep[l]
        log(f"  L{l:2d}: FLAT {pct(K['flat'], K['P'])}  OPEN {pct(K['open'], K['P'])}  full {pct(K['full'], K['P'])}  FLAT!=dark {K['flat_eq_dark']}  "
            f"reads/FLAT {pct(K['reads_flat'], K['fullm_flat'])}  FLAT out {pct(K['glow_skipped'], K['out_all'])}  push {pct(K['push'], K['write'])}  "
            f"| tagskip: FLAT {ja['layers'][str(l)]['flat']} mine {K['flat']}; reads {ja['layers'][str(l)]['reads_flat']} mine {K['reads_flat']}; "
            f"bad {sum(K['bad'].values())}; in full {K['full']} (tagskip {ja['layers'][str(l)]['full']})")
    tot = lambda k: sum(keep[l][k] for l in range(12))
    hs = np.sum([keep[l]["hist"] for l in range(12)], 0); cs = np.cumsum(hs) / hs.sum()
    log(f"  all layers: FLAT {pct(tot('flat'), tot('P'))}, reads {pct(tot('reads_flat'), tot('fullm_flat'))} of FLAT meetings, "
        f"{pct(tot('reads_flat'), tot('fullm'))} of fc meetings; proved by " + ", ".join(f"D{k} {100 * cs[k]:.1f}%" for k in (12, 14, 16, 18, 24)))
    log(f"  checks (all layers): " + ", ".join(f"{k} {sum(keep[l]['bad'][k] for l in range(12))}" for k in keep[0]["bad"]))
    ktag = [k for k in ja['counts'] if k.startswith('fc: the tags')][0]
    log(f"  A TOKEN: fc tags {C['fc tags'] / 648:,.0f} (tagskip {ja['counts'][ktag] / 648:,.0f}); "
        f"fc open {C['fc open'] / 648:,.0f} (tagskip {ja['counts']['fc: OPEN gates finished (meetings, divide, bias, bend)'] / 648:,.0f}); "
        f"bound built {C['built'] / 648:,.0f} (tagskip {ja['counts']['tag own: bound built'] / 648:,.0f})")
    log(f"           mproj {C['mproj'] / 648:,.0f} (tagskip {ja['counts']['mproj'] / 648:,.0f}); plain fc on this light {C['plain fc'] / 648:,.0f} "
        f"(tagskip {ja['counts']['plain fc on this light'] / 648:,.0f}); plain mproj on this light {C['plain mproj'] / 648:,.0f} "
        f"(tagskip {ja['counts']['plain mproj on this light'] / 648:,.0f}); lines met {pct(C['lines met'], C['lines'])}")
    log(f"           exact sums equal: fc tags {C['fc tags'] == ja['counts'][[k for k in ja['counts'] if k.startswith('fc: the tags')][0]]}, "
        f"fc open {C['fc open'] == ja['counts']['fc: OPEN gates finished (meetings, divide, bias, bend)']}, mproj {C['mproj'] == ja['counts']['mproj']}, "
        f"built {C['built'] == ja['counts']['tag own: bound built']}, plain fc {C['plain fc'] == ja['counts']['plain fc on this light']}, "
        f"plain mproj {C['plain mproj'] == ja['counts']['plain mproj on this light']}")
    log(f"           the FLAT gates' diagonals NOT read (what the tag lets them skip): {C['flat unread'] / 648:,.0f} a token")
    log(f"  THE TOKEN teacher-forced vs the exact machine: same top word {int((WN == EW).sum())}/648, in top 5 {int((E5 == WN[:, None]).any(1).sum())}/648; "
        f"top 5 vs tagskip's own run, positions differing {int((np.array(ja['top5']) != T5).any(1).sum())}")
    log(f"({el()}s, max memory {mb()} MB)")

# ======================================================================================================== C
if PART == "C":
    A = np.load(f"{OUT}/check_A.npz"); CST = {l: A[f"c{l}"] for l in range(12)}
    pick = [0, 7, 14, 19, 29]
    log(f"PART C: greedy, 16 words, refs texts {pick}, against tagskip's judge runs' words")
    for name, skip, glow, jf in (("none", (), None, "judge_none_real"), ("all, zero", set(range(12)), None, "judge_all_real"),
                                 ("all, stored glow", set(range(12)), CST, "judge_all_real_glowstored"), ("L11, zero", (11,), None, "judge_L11_real")):
        gen = [[] for _ in pick]; seq = [list(REFS[t]["ids"]) for t in pick]
        for s in range(16):
            _, WN, _ = run([q + g for q, g in zip(seq, gen)], skip=skip, glow=glow, rows="last")
            for i, w in enumerate(WN): gen[i].append(int(w))
        theirs = json.load(open(f"{OUT}/{jf}.json"))["gen"]; base = json.load(open(f"{OUT}/judge_none_real.json"))["gen"]
        same = sum(gen[i] == theirs[t] for i, t in enumerate(pick))
        log(f"  {name:18s}: texts identical to tagskip's words {same}/5; words equal to base2's own greedy "
            f"{sum(int(a == b) for i, t in enumerate(pick) for a, b in zip(gen[i], base[t]))}/80 (tagskip's for these texts: "
            f"{sum(int(a == b) for t in pick for a, b in zip(theirs[t], base[t]))}/80)")
    log(f"({el()}s, max memory {mb()} MB)")

# ======================================================================================================== D
if PART == "D":
    A = np.load(f"{OUT}/check_A.npz"); CST = {l: A[f"c{l}"] for l in range(12)}
    ref5 = np.load(f"{HERE}/judge5_ref.npz")["top5"][:50].reshape(-1, 5); texts = [w["ids"] for w in WINS[:50]]
    log("PART D: judge5 windows 0-49 (1,000 judged positions); tagskip's first log line beside each")
    for name, skip, glow, shuf, theirs in (
            ("none", (), None, None, "1000 1000 1000"), ("all, zero", set(range(12)), None, None, "372 151 0"),
            ("L11, zero", (11,), None, None, "942 709 54"), ("L6-11, zero", set(range(6, 12)), None, None, "663 339 0"),
            ("all, stored glow", set(range(12)), CST, None, "943 692 36"), ("L11, stored glow", (11,), CST, None, "1000 933 413"),
            ("all, zero, SHUFFLED TAGS", set(range(12)), None, np.random.default_rng(424242), "293 116 0 (its own shuffle)"),
            ("all, glow, SHUFFLED TAGS", set(range(12)), CST, np.random.default_rng(171717), "73 19 0 (its own shuffle)")):
        T5, WN, _ = run(texts, skip=skip, glow=glow, rows="judged", shuffle=shuf)
        log(f"  {name:26s}: in top 5 {int((ref5 == WN[:, None]).any(1).sum())}, same top word {int((ref5[:, 0] == WN).sum())}, "
            f"set and order {int((ref5 == T5).all(1).sum())}   | tagskip: {theirs}   ({el()}s, {mb()} MB)")
    log(f"({el()}s, max memory {mb()} MB)")
