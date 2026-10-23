"""AN INDEPENDENT REBUILD OF BASE2_FRINGES2's NEW NUMBERS. It shares no code with base2_fringes.py or base2_fringes2.py:
it reads base2_gpt2.npz (not the record), base2.py's own steps, and base2.py's own saved run (recording/C_ref/
plain_refs.npz, fringes_machine/plain_windows_top_light.npz), and writes every value in signed digits with its own digit
loop (n odd: digit 2 - (n mod 4), then halve), never the machine's 3n XOR n.

part=top      ADDITIONS 1 AND 2 on every rstep-th refs position and every wstep-th judge position (default 20 and 300):
              every (a, b) count over the lines by brute force, word by word; the start rule, the bounds and the set-aside
              stage by stage from the docstring's relations; the meetings of the words still in. Compared with the
              saved runs (top_a1_*, top_a2_*): the start stage, the words still in after every stage, the meetings at
              every stage, the top 5 and the race winner.
part=arrival  ADDITION 3 on all 30 refs texts: the arrival's lit-digit meetings (q.k and shares x values) and the writing,
              layer by layer, from base2's own qkv piles; compared with refs_counts_a3.json.
part=shared   ADDITION 4 on all 30 refs texts, layers 1-11, every plate: the shared order (its own sort), the gaps, the
              running sums, the three counts; compared with refs_counts_a4.json (layer 0 meets stored parts, not rebuilt).
    python3 fringes_machine2/checker.py part=top|arrival|shared   -> fringes_machine2/checker_<part>.log
"""
import sys, os, json, time, resource, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/fringes_machine2"
PART = dict(a.split("=") for a in sys.argv[1:])["part"]
LOG = open(f"{OUT}/checker_{PART}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_src = _src.replace('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')
_src = _src.replace('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")')
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
M = B["M"]; ES, SC, ARR, EXP2 = B["ES"], B["SC"], B["ARR"], B["EXP2"]
power, rshift, to16, gauge_log, product, race, log2fix, edge = (B[k] for k in ("power", "rshift", "to16", "gauge_log", "product", "race", "log2fix", "edge"))
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
_POP = np.array([bin(i).count("1") for i in range(256)], np.int64)
def ones(A):                                                  # 1s of |A|, byte by byte
    A = np.abs(np.asarray(A, np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFF]; A = A >> 8
    return n
def digits(V, n):                                             # signed digits, place by place: (n, *V.shape) of -1/0/+1
    V = np.asarray(V, np.int64).copy(); out = np.zeros((n,) + V.shape, np.int8)
    for i in range(n):
        d = np.where(V & 1, 2 - (V & 3), 0); out[i] = d; V = (V - d) >> 1
    assert not V.any(), "more places needed"
    return out
def lit(V):                                                   # lit digits of each value
    V = np.asarray(V, np.int64).copy(); n = np.zeros(V.shape, np.int64)
    while V.any():
        d = np.where(V & 1, 2 - (V & 3), 0); n += d != 0; V = (V - d) >> 1
    return n
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")

if PART == "top":
    TT = np.asarray(M["head.w"]); assert np.abs(TT.astype(np.int32)).max() < 1 << 15; TT = TT.astype(np.int16)   # (768, 50257)
    bias = M["head.b"].astype(np.int64); HE = M["head.e"]; assert (HE == HE[0]).all()
    NW = TT.shape[1]; BMAX = int(np.abs(bias).max())
    PW = np.load(f"{HERE}/fringes_machine/plain_windows_top_light.npz")
    ARGS = dict(a.split("=") for a in sys.argv[1:]); RSTEP, WSTEP = int(ARGS.get("rstep", 20)), int(ARGS.get("wstep", 300))
    SR, SWN = np.arange(0, 648, RSTEP), np.arange(0, 6000, WSTEP)
    XALL = np.concatenate([REF["x16"][SR], PW["x16"][SWN]]).astype(np.int64)
    EXALL = np.concatenate([REF["ex"][SR], PW["ex"][SWN]]); LSALL = np.concatenate([REF["LS"][SR], PW["LS"][SWN]])
    WHERE = [("refs", int(q)) for q in SR] + [("windows", int(j)) for j in SWN]
    log(f"top read rebuilt at {len(SR)} refs and {len(SWN)} judge positions (every {RSTEP}th and every {WSTEP}th)")
    def stage_of(a, b): return max(0, -(-(27 - (a + b)) // 2))           # pairs with a + b >= 27 - 2s are read by stage s
    for writing, tag in (("plain", "a1"), ("signed", "a2")):
        tpl = 15 if writing == "plain" else 16
        Z = {"refs": np.load(f"{OUT}/top_{tag}_refs.npz"), 1: np.load(f"{OUT}/top_{tag}_windows_p1of2.npz"), 2: np.load(f"{OUT}/top_{tag}_windows_p2of2.npz")}
        Z = {k: {f: v[f] for f in ("start", "surv", "work", "top5", "win")} for k, v in Z.items()}
        SM = np.zeros((tpl * tpl, 15), np.int64); SW = np.zeros((tpl * tpl, 15), np.int64)
        for a in range(tpl):
            for b in range(tpl): s = stage_of(a, b); SM[a * tpl + b, s] = 1 << (a + b); SW[a * tpl + b, s] = 1
        STOF = np.array([[stage_of(a, b) for b in range(tpl)] for a in range(tpl)])
        mism = {}; starts = []; work_ck = work_run = 0
        def bad(k, v): mism[k] = mism.get(k, 0) + int(v)
        for b0 in range(0, len(XALL), 64):
            X = XALL[b0:b0 + 64]; EX = EXALL[b0:b0 + 64]; LSA = LSALL[b0:b0 + 64]; NP_ = len(X)
            if writing == "plain": XD = np.stack([np.sign(X) * ((np.abs(X) >> a) & 1) for a in range(tpl)], 1).astype(np.int8)
            else: XD = np.moveaxis(digits(X, tpl), 0, 1)                    # (P, tpl, 768)
            XL = np.abs(XD); nA = XL.sum(2).astype(np.int64)                    # (P, tpl) the light's lit lines
            DCS = np.zeros((15, NP_, NW), np.int64); MT = np.zeros((15, NP_, NW), np.int64); MWB = np.zeros((NW, tpl), np.int64)
            XDf = XD.reshape(NP_ * tpl, 768).astype(np.float32); XLf = XL.reshape(NP_ * tpl, 768).astype(np.float32)
            for c0 in range(0, NW, 1024):
                T = TT[:, c0:c0 + 1024].astype(np.int64); Wc = T.shape[1]
                if writing == "plain": WD = np.stack([np.sign(T) * ((np.abs(T) >> b) & 1) for b in range(tpl)], 1).astype(np.int8)
                else: WD = np.moveaxis(digits(T, tpl), 0, 1)                    # (768, tpl, Wc)
                WL = np.abs(WD); MWB[c0:c0 + Wc] = WL.sum(0).T
                Cn = (XDf @ WD.reshape(768, tpl * Wc).astype(np.float32)).astype(np.int64).reshape(NP_, tpl, tpl, Wc)
                Ln = (XLf @ WL.reshape(768, tpl * Wc).astype(np.float32)).astype(np.int64).reshape(NP_, tpl, tpl, Wc)
                for A_, Wt, D_ in ((Cn, SM, DCS), (Ln, SW, MT)):            # every product <= 2^40, every sum < 2^49: exact
                    Y = A_.reshape(NP_, tpl * tpl, Wc).transpose(0, 2, 1).reshape(NP_ * Wc, tpl * tpl).astype(np.float64) @ Wt.astype(np.float64)
                    D_[:, :, c0:c0 + Wc] = Y.astype(np.int64).reshape(NP_, Wc, 15).transpose(2, 0, 1)
                del Cn, Ln, WD, WL
            MBb = MWB.max(0); CUM = np.cumsum(DCS, 0, out=DCS)
            for c0 in range(0, NW, 8192):
                assert np.array_equal(CUM[14, :, c0:c0 + 8192], (X @ TT[:, c0:c0 + 8192].astype(np.float64)).astype(np.int64)), "the pieces do not add to the pile"
            for i in range(NP_):
                kind, q = WHERE[b0 + i]
                z = Z["refs"] if kind == "refs" else Z[1 if q < 3000 else 2]; q = q if kind == "refs" else q % 3000
                F = int((HE[0] + EX[i] + ES) * 65536 - LSA[i]); beta = power(bias, -F)
                I = F >> 16; e = int(EXP2[F & 0xFFFF]); s_ = 16 - I; F2 = -F; e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - (F2 >> 16)
                gam = Fraction(1, 2) + BMAX * Fraction(2) ** s_ * abs(e * e2 - Fraction(2) ** (s_ + s2)) / (e * Fraction(2) ** (s_ + s2))
                mu = 2 * int(-(-gam.numerator // gam.denominator)) + max(1, -(-(2 ** s_) // e))
                mins = np.minimum(nA[i][None, :, None], MWB[:, None, :]).reshape(NW, tpl * tpl)   # (NW, a b): min(n_a, m_wb)
                GS = (mins.astype(np.float64) @ SM.astype(np.float64)).astype(np.int64)            # each stage's part of the bound
                RT = np.cumsum(GS[:, ::-1], 1)[:, ::-1]                                           # RT[:, s]: stages s.. unread
                alive = np.arange(NW); start = -1; surv = []; work = []
                for st in range(14):
                    work.append(int(MT[st, i, alive].sum()))
                    un = STOF > st
                    if start < 0:
                        U = CUM[st, i] + beta; srt = np.sort(U)
                        Rbar = int(sum(int(min(nA[i][a], MBb[b])) << (a + b) for a in range(tpl) for b in range(tpl) if un[a, b]))
                        if srt[-5] - srt[0] < 2 * Rbar + mu: surv.append(NW); continue
                        start = st
                    Rw = RT[alive, st + 1]
                    hi = CUM[st, i, alive] + beta[alive] + Rw; lo = hi - 2 * Rw
                    L5 = np.sort(lo)[-5]; alive = alive[hi > L5 - mu]; surv.append(len(alive))
                work.append(int(MT[14, i, alive].sum()))
                sc = power(CUM[14, i, alive], F) + bias[alive]
                Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
                idx = np.argpartition(-Sp, 5)[:5]; t5 = idx[np.argsort(-Sp[idx])]
                bad("start stage", start != int(z["start"][q])); bad("words still in, any stage", not np.array_equal(surv, z["surv"][q]))
                bad("meetings, any stage", not np.array_equal(work, z["work"][q])); bad("top 5", not np.array_equal(t5, z["top5"][q]))
                bad("race winner", race(Sp) != int(z["win"][q]))
                starts.append(start); work_ck += sum(work); work_run += int(z["work"][q].sum())
                if b0 + i == 0: log(f"  {writing}: refs position {q}: start stage {start}, words still in {surv[:9]}, meetings {sum(work):,}")
            del DCS, MT, CUM, mins, GS, RT
        log(f"  {writing} ({tag}): positions rebuilt {len(XALL)}; differing from the run: {mism}; start stages "
            f"{dict(zip(*np.unique(starts, return_counts=True)))}; meetings rebuilt {work_ck:,} against the run's {work_run:,} "
            f"({time.time() - t0:.0f}s, {mb()} MB)")

elif PART in ("arrival", "shared"):
    RC = json.load(open(f"{OUT}/refs_counts_{'a3' if PART == 'arrival' else 'a4'}.json"))["counts"]
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    npos = REF["npos"]; off = np.concatenate([[0], np.cumsum(npos)])
    ids = np.concatenate([np.asarray(t, np.int64) for t in TEXTS]); pos = np.concatenate([np.arange(n) for n in npos])
    mism = {}; tot = {}
    def cmp(key, v):
        r = RC.get(key, 0); tot[key] = (int(v), r)
        if int(v) != r: mism[key] = (int(v), r)
    def light_in(l): return (M["wte"][ids] + M["wpe"][pos]) if l == 0 else REF[f"stream_{l - 1}"].astype(np.int64)
    def qkv_of(l):
        x16, ex, LS = gauge_log(light_in(l))
        return power(REF[f"qkv_pile_{l}"].astype(np.int64), (M[f"qkv.{l}.e"][None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + M[f"qkv.{l}.b"]
    def heads(QKV, h): return QKV[:, h * 64:(h + 1) * 64], QKV[:, 768 + h * 64:768 + (h + 1) * 64], QKV[:, 1536 + h * 64:1536 + (h + 1) * 64]
    for l in range(12):
        QKV = qkv_of(l); fc_ = 1 if l == 0 else 0
        if PART == "arrival":
            n_sc = n_mix = n_wr = 0
            for t in range(30):
                Q = QKV[off[t]:off[t + 1]]; P = len(Q)
                for h in range(12):
                    q, k, v = heads(Q, h); q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
                    lq, lk, lv = lit(q16), lit(k16), lit(v).sum(1)
                    sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
                    for p in range(fc_, P):
                        n_sc += int((lq[p][None, :] * lk[:p + 1]).sum())
                        row = sc[p, :p + 1]; d = row[race(row)] - row
                        a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
                        n_mix += int((lit(a) * lv[:p + 1]).sum())
                    n_wr += int(ones(q16[fc_:]).sum() + ones(k16).sum() + ones(v).sum())
            cmp(f"fr|L{l}|arrival|scores, signed digits", n_sc); cmp(f"fr|L{l}|arrival|shares x values, signed digits", n_mix)
            cmp(f"fr|L{l}|arrival|q, k and values written in signed digits", n_wr)
        else:
            if l == 0: continue
            o = np.concatenate([B["attention"](QKV[off[t]:off[t + 1]]) for t in range(30)])
            o16 = to16(o, -ES)[0]
            xa, ea, La = gauge_log(REF[f"stream_a_{l}"].astype(np.int64))
            H = power(REF[f"fc_pile_{l}"].astype(np.int64), (M[f"fc.{l}.e"][None, :] + ea[:, None] + ES) * 65536 - La[:, None]) + M[f"fc.{l}.b"]
            g16 = to16(edge(H), -ES)[0]
            for part, x in (("qkv", gauge_log(light_in(l))[0]), ("proj", o16), ("fc", xa), ("mproj", g16)):
                w = M[f"{part}.{l}.w"].astype(np.int64); Ln = w.shape[0]; rs = w.sum(1)
                order = np.array(sorted(range(Ln), key=lambda i: (-int(rs[i]), i)))
                wp = w[order]; gaps = wp - np.vstack([wp[1:], np.zeros((1, w.shape[1]), np.int64)])
                S = np.cumsum(x[:, order], 1)
                cmp(f"side|L{l}|{part}|shared order: gaps x running sum, signed digits", int((lit(S) * lit(gaps).sum(1)[None, :]).sum()))
                cmp(f"side|L{l}|{part}|shared order: pours into the one running sum", int(ones(x).sum()))
                cmp(f"side|L{l}|{part}|shared order: running sum written in signed digits", int(ones(S).sum()))
        log(f"  layer {l}: done ({time.time() - t0:.0f}s, {mb()} MB)")
    log(f"{PART}: {len(tot)} counts rebuilt; differing from the run: {len(mism)} {mism}")
    log(f"  totals: rebuilt {sum(v for v, r in tot.values()):,} against the run's {sum(r for v, r in tot.values()):,}")
log(f"({time.time() - t0:.0f}s, max memory {mb()} MB)")
