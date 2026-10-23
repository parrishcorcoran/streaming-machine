"""AUDIT 3 (QWEN3-0.6B): THE PARTS THE QWEN TOTALS LEAVE OUT, COUNTED IN PHOTONS; AND QWEN'S PLAIN REBUILT.
The counting-convention audit (the session's; its words, not his). Written for this audit: no import of, or code from,
base2_groups.py, qwen_groups.py, groups/, groups2/, qwen_fringes*.py or capture.py. qwen/qwen_base2.py's own steps
(to16, gauge, power, head gauge, turn, arrival, gated bend, plain slab) are written again in whole numbers from its
lines; its tables from qwen/qwen_base2.npz. The machine is run from the word rows on the 30 refs texts in Qwen's words
(qwen06b/light/meta.json, 613 positions), every text layer by layer together, the arrival text by text.

WHAT IS COUNTED (a token = the mean over the 613 positions), in the convention base2_fringes2.py uses for GPT-2 (a
product a x b costs ones(a) x ones(b); an add costs the 1s of the value added; table reads, shifts and compares nothing):
  water levels   every gauge's squares (qkv.l, gu.l and the top: sum of ones(x)^2; RMS has no S1)
  head water     the per-head water levels on q and k (16 + 8 heads): the squares, the gain x value, the divide
  turns          every pair turned: a.c, b.s, b.c, a.s (each ones x ones) and the two adds (ones of b.s and a.s)
  divides        every gauged slab's pile x its gain (qkv.l, gu.l): ones(pile) x ones(gain); o and down are shifts
  bends          the gated bend: g x share (ones x ones) and (g share) x u (ones x ones)
  arrival        the shares' total (ones of every share) and the divide (ones(numerator) x ones(gain))
  stream adds    ones of each half layer's output added to the stream
These are qwen_groups.py's "unchanged parts", counted on neither side there. GPT-2's machine counts their GPT-2
equivalents on both sides. Also rebuilt here: Qwen's plain (the plates' meetings, the plain read with every word's
divide, the arrival's meetings) and the same pairs in signed digits (light written once a slab; lit digit x lit digit;
the arrival's q, k and values written once; the read's light written once).
CHECKS: the light at every slab of every layer and at the top = capture's record (x16, its power, the top's LS); every
score row's sha256 at the top = capture's; plain rebuilt = 16,397,464,589 a token; the arrival's meetings, plain and in
signed digits, = qwen_machine_counts' records.

PREDICTION (written before this file ran; sizes from the shapes and typical 1s: 16-bit values about 8 1s, cos/sin
30-bit about 15, piles about 20, gains about 8):
  - Exact: every light identical (28 x 4 slabs + the top), score rows 613/613; plain 16,397,464,589 a token exactly.
  - The left-out parts, a token: turns about 20-25M, head water about 15-25M, divides about 40-50M, bends about 10-15M,
    arrival totals and divides about 5-10M, water levels about 3-5M, stream adds about 1M: about 90-130M in all, 0.55-0.8%
    of plain.
  - Counted on both sides: (3,105,280,101 + Z) / (16,397,464,589 + Z) = about 19.4-19.6% (from 18.94%).
  - The same pairs in signed digits: plates about 59-60% of their plain (docs/06's 59.48%), the read about 60-63%, the
    arrival as qwen_fringes2 counts it (34.2M): the whole about 59-61% of plain; qwen_groups' 3,105M is then about
    31-32% of the signed-digit plain (before the left-out parts).

    python3 confirm_groups/audit_qwen_parts.py     -> confirm_groups/audit_qwen_parts.json/.log
"""
import os, sys, json, time, hashlib, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, rss, ones, naf, member, product, rshift, to16, log2fix, power, gain_ones

NPZ = f"{HERE}/qwen/qwen_base2.npz"; LIGHT = f"{HERE}/qwen06b/light"
ES, SC, L, D, HD = 16, 10, 28, 1024, 128
LOG2, EXP2, ARR, SIG, COS, SIN = (np.asarray(member(NPZ, k)) for k in ("log2", "exp2", "arrival", "sig", "cos", "sin"))
EPS40 = int(member(NPZ, "eps40"))
QSEG = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
        "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
def pct(a, b): v = (20000 * a + b) // (2 * b); return f"{v // 100}.{v % 100:02d}%"
def half_log(n): return ((int(n).bit_length() - 1) << 16) // 2                  # log2(n) / 2 at 2^16, n a power of two

def gauge(x16, ex, n):
    V = (x16 * x16).sum(1)
    eps = np.array([(EPS40 * n) >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else (EPS40 * n) << -(40 + 2 * int(e)) for e in ex], np.int64)
    return np.right_shift(log2fix(V + eps, LOG2), 1) - half_log(n) + ex * 65536

def main():
    t0 = time.time(); log_f = open(f"{CG}/audit_qwen_parts.log", "w")
    def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    META = json.load(open(f"{LIGHT}/meta.json")); TEXTS = META["texts"]; NP = META["npos"]
    lens = [len(t) for t in TEXTS]; off = np.concatenate([[0], np.cumsum(lens)]); assert off[-1] == NP
    ids = np.concatenate([np.asarray(t, np.int64) for t in TEXTS])
    log(f"AUDIT 3 (Qwen3-0.6B): qwen_base2.py's steps rebuilt on {len(TEXTS)} texts, {NP} positions; the left-out parts counted")
    CAP = {s: np.load(f"{LIGHT}/{s}_light.npy", mmap_mode="r") for s in ("qkv", "o", "gu", "down")}
    CAPE = {s: np.load(f"{LIGHT}/{s}_ex.npy") for s in ("qkv", "o", "gu", "down")}
    BAD = {}
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    K = {k: 0 for k in ("water levels", "head water: squares", "head water: gain x value", "head water: divides", "turns: products",
                        "turns: adds", "divides (qkv, gu)", "bends", "arrival: shares' total", "arrival: divides", "stream adds")}
    PL = {k: 0 for k in ("plates", "arrival", "top")}; SD = {k: 0 for k in ("plates", "arrival", "top")}
    ARRC = {k: 0 for k in ("plain_scores", "plain_values", "sd_scores", "sd_values", "write_q", "write_k", "write_v")}
    E = np.asarray(member(NPZ, "embed")[ids]).astype(np.int64); R = E
    def light_check(slab, l, x16, ex):
        chk(f"light vs capture's record (rows)", (x16 != np.asarray(CAP[slab][l]).astype(np.int64)).any(1).sum() + (ex != CAPE[slab][l]).sum())
    def plate_meet(slab, x16, w):
        o_, n_ = ones(x16), naf(x16)
        for name, a, b in QSEG[slab]:
            wb = np.asarray(w[:, a:b]).astype(np.int64)
            PL["plates"] += int((o_ @ ones(wb).sum(1)).sum()); SD["plates"] += int((n_ @ naf(wb).sum(1)).sum())
        SD["plates"] += int(o_.sum())                                         # the light written once a slab
    def head_gauge(X, g, ge):
        x16, ex = to16(X, -ES); LSh = gauge(x16, ex, HD); y = x16 * g[None, :]
        K["head water: squares"] += int((ones(x16) ** 2).sum()); K["head water: gain x value"] += int((ones(x16) * ones(g)[None, :]).sum())
        K["head water: divides"] += int((ones(y).sum(1) * gain_ones(-LSh, EXP2)).sum())
        return power(y, (ge + ex[:, None] + ES) * 65536 - LSh[:, None], EXP2)
    def turn(X, pos):
        a, b = X[:, :64], X[:, 64:]; c, s = COS[pos], SIN[pos]
        ac, bs, bc, as_ = a * c, b * s, b * c, a * s
        K["turns: products"] += int((ones(a) * ones(c) + ones(b) * ones(s) + ones(b) * ones(c) + ones(a) * ones(s)).sum())
        K["turns: adds"] += int((ones(bs) + ones(as_)).sum())
        return np.concatenate([rshift(ac - bs, 30), rshift(bc + as_, 30)], 1)
    def attention(QKV, l):
        P = QKV.shape[0]; pos = np.arange(P); out = np.zeros((P, 16 * HD), np.int64)
        qg, qge, kg, kge = (np.asarray(member(NPZ, f"{k}.{l}")).astype(np.int64) for k in ("qg", "qge", "kg", "kge"))
        Kh = [turn(head_gauge(QKV[:, 2048 + j * HD:2048 + (j + 1) * HD], kg, int(kge)), pos) for j in range(8)]
        Vh = [QKV[:, 3072 + j * HD:3072 + (j + 1) * HD] for j in range(8)]
        K16 = [to16(k, -ES) for k in Kh]
        for j in range(8):
            ARRC["write_k"] += int(ones(K16[j][0]).sum()); ARRC["write_v"] += int(ones(Vh[j]).sum())
        tri = np.tril(np.ones((P, P), bool)); NEG = np.int64(-(1 << 62))
        for h in range(16):
            q = turn(head_gauge(QKV[:, h * HD:(h + 1) * HD], qg, int(qge)), pos)
            k16, ek = K16[h // 2]; v = Vh[h // 2]; q16, eq = to16(q, -ES)
            sc = rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] + SC))
            ARRC["plain_scores"] += int((ones(q16) @ ones(k16).T)[tri].sum()); ARRC["sd_scores"] += int((naf(q16) @ naf(k16).T)[tri].sum())
            ARRC["write_q"] += int(ones(q16).sum())
            scm = np.where(tri, sc, NEG); mx = scm.max(1); d = mx[:, None] - sc
            ok = tri & (d < len(ARR)); A = np.where(ok, ARR[np.clip(d, 0, len(ARR) - 1)], 0)
            num = A @ v; Lp = log2fix(A.sum(1), LOG2)
            out[:, h * HD:(h + 1) * HD] = power(num, -Lp[:, None], EXP2)
            ARRC["plain_values"] += int((ones(A) @ ones(v).sum(1)).sum()); ARRC["sd_values"] += int((naf(A) @ naf(v).sum(1)).sum())
            K["arrival: shares' total"] += int(ones(A).sum()); K["arrival: divides"] += int((ones(num).sum(1) * gain_ones(-Lp, EXP2)).sum())
        return out
    for l in range(L):
        # qkv: the gauged slab
        x16, ex = to16(R, -ES); light_check("qkv", l, x16, ex); LS = gauge(x16, ex, D)
        K["water levels"] += int((ones(x16) ** 2).sum())
        w = member(NPZ, f"qkv.{l}.w"); we = np.asarray(member(NPZ, f"qkv.{l}.e")).astype(np.int64); plate_meet("qkv", x16, w)
        pile = product(x16, np.asarray(w)); K["divides (qkv, gu)"] += int((ones(pile).sum(1) * gain_ones(-LS, EXP2)).sum())
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2); del pile
        o = np.concatenate([attention(QKV[off[t]:off[t + 1]], l) for t in range(len(TEXTS))]); del QKV
        x16, ex = to16(o, -ES); light_check("o", l, x16, ex)
        w = member(NPZ, f"o.{l}.w"); we = np.asarray(member(NPZ, f"o.{l}.e")).astype(np.int64); plate_meet("o", x16, w)
        att = rshift(product(x16, np.asarray(w)), -(we[None, :] + ex[:, None] + ES)); K["stream adds"] += int(ones(att).sum()); R = R + att
        # gate and up: the gauged slab, then the gated bend
        x16, ex = to16(R, -ES); light_check("gu", l, x16, ex); LS = gauge(x16, ex, D)
        K["water levels"] += int((ones(x16) ** 2).sum())
        w = member(NPZ, f"gu.{l}.w"); we = np.asarray(member(NPZ, f"gu.{l}.e")).astype(np.int64); plate_meet("gu", x16, w)
        pile = product(x16, np.asarray(w)); K["divides (qkv, gu)"] += int((ones(pile).sum(1) * gain_ones(-LS, EXP2)).sum())
        GU = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2); del pile
        G, U = GU[:, :3072], GU[:, 3072:]
        idx = rshift(G, ES - 12) + 65536; sh = np.where(idx < 0, 0, np.where(idx > 131071, 65536, SIG[np.clip(idx, 0, 131071)]))
        t_ = rshift(G * sh, 16); y = rshift(t_ * U, ES)
        K["bends"] += int((ones(G) * ones(sh) + ones(t_) * ones(U)).sum())
        x16, ex = to16(y, -ES); light_check("down", l, x16, ex)
        w = member(NPZ, f"down.{l}.w"); we = np.asarray(member(NPZ, f"down.{l}.e")).astype(np.int64); plate_meet("down", x16, w)
        mo = rshift(product(x16, np.asarray(w)), -(we[None, :] + ex[:, None] + ES)); K["stream adds"] += int(ones(mo).sum()); R = R + mo
        log(f"  layer {l:2d}: lights checked ({time.time() - t0:.0f}s, {rss()} MB)")
    # the top: the water level and the plain read
    x16, ex = to16(R, -ES); LS = gauge(x16, ex, D)
    chk("the top light vs capture's record (rows)", (x16 != np.load(f"{LIGHT}/top_light.npy").astype(np.int64)).any(1).sum()
        + (ex != np.load(f"{LIGHT}/top_ex.npy")).sum() + (LS != np.load(f"{LIGHT}/top_LS.npy")).sum())
    K["water levels"] += int((ones(x16) ** 2).sum())
    Wh = np.asarray(member(NPZ, "head.w")); He = np.asarray(member(NPZ, "head.e")).astype(np.int64); NW = Wh.shape[1]
    pph = np.zeros(D, np.int64); nph = np.zeros(D, np.int64)
    for j0 in range(0, NW, 8192):
        blk = np.asarray(Wh[:, j0:j0 + 8192]).astype(np.int64); pph += ones(blk).sum(1); nph += naf(blk).sum(1)
    sha_same = 0; div = 0
    for p0 in range(0, NP, 32):
        PILE = product(x16[p0:p0 + 32], Wh); Lc, ec = LS[p0:p0 + 32], ex[p0:p0 + 32]
        div += int((ones(PILE).sum(1) * gain_ones(-Lc, EXP2)).sum())
        S = power(PILE, (He[None, :] + ec[:, None] + ES) * 65536 - Lc[:, None], EXP2)
        for i in range(len(S)): sha_same += hashlib.sha256(np.ascontiguousarray(S[i], np.int64).tobytes()).hexdigest() == META["sha"][p0 + i]
        del PILE, S
    chk("score rows' sha256 vs capture's record (positions differing)", NP - sha_same)
    PL["top"] = int((ones(x16) @ pph).sum()) + div; SD["top"] = int((naf(x16) @ nph).sum()) + int(ones(x16).sum()) + div
    PL["arrival"] = ARRC["plain_scores"] + ARRC["plain_values"]
    SD["arrival"] = ARRC["sd_scores"] + ARRC["sd_values"] + ARRC["write_q"] + ARRC["write_k"] + ARRC["write_v"]
    # against the builder's records
    Z = np.load(f"{HERE}/groups2/machine/qwen_machine_counts.npz"); Q = {k.replace("__", "|"): Z[k] for k in Z.files}
    theirs_arr = {k: int(sum(Q[f"arr.{l}|{k}"].sum() for l in range(L))) for k in ARRC}
    T = np.load(f"{HERE}/groups2/machine/qwen_top_machine.npz")
    theirs_plain = int(sum(v.sum() for k, v in Q.items() if k.endswith("|plain"))) + int(T["plain"].sum()) + theirs_arr["plain_scores"] + theirs_arr["plain_values"]
    log("")
    log("EXACT: " + ", ".join(f"{k} {v}" for k, v in BAD.items()))
    log("the arrival's counts against qwen_machine_counts: " + ", ".join(f"{k} {ARRC[k] / NP:,.0f} vs {theirs_arr[k] / NP:,.0f} ({ARRC[k] == theirs_arr[k]})" for k in ARRC))
    P0 = sum(PL.values()); S0 = sum(SD.values())
    log(f"PLAIN rebuilt: plates {PL['plates'] / NP:,.0f}, read {PL['top'] / NP:,.0f}, arrival {PL['arrival'] / NP:,.0f}; whole {P0 / NP:,.0f} a token "
        f"(the builder's {theirs_plain / NP:,.0f}; identical over {NP} positions: {P0 == theirs_plain}; read's divides {div / NP:,.0f}, meetings {(PL['top'] - div) / NP:,.0f})")
    log(f"THE SAME PAIRS IN SIGNED DIGITS: plates {SD['plates'] / NP:,.0f} ({pct(SD['plates'], PL['plates'])}), read {SD['top'] / NP:,.0f} ({pct(SD['top'], PL['top'])}), "
        f"arrival {SD['arrival'] / NP:,.0f} ({pct(SD['arrival'], PL['arrival'])}); whole {S0 / NP:,.0f} = {pct(S0, P0)} of plain")
    log("")
    Zt = sum(K.values())
    log("THE PARTS LEFT OUT ON BOTH SIDES, a token:")
    for k, v in K.items(): log(f"    {k:28s} {v / NP:14,.0f}   {pct(v, P0)} of plain")
    log(f"    {'ALL':28s} {Zt / NP:14,.0f}   {pct(Zt, P0)} of plain")
    MQ = 3105280101 * NP
    log(f"qwen_groups' delivered 3,105,280,101 a token: {pct(MQ, P0)} of plain; with the left-out parts on both sides {pct(MQ + Zt, P0 + Zt)}")
    log(f"   against the same pairs in signed digits: {pct(MQ, S0)}; with the left-out parts on both sides {pct(MQ + Zt, S0 + Zt)}")
    json.dump(dict(npos=NP, left_out=K, plain=PL, sd=SD, arrival=ARRC, theirs_arrival=theirs_arr, theirs_plain=theirs_plain, bad=BAD, plain_total=P0, sd_total=S0),
              open(f"{CG}/audit_qwen_parts.json", "w"), indent=0)
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

if __name__ == "__main__":
    main()
