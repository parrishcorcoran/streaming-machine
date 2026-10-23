"""AUDIT 1 (GPT-2): THE PLAIN COUNT REBUILT INDEPENDENTLY, PART BY PART, AND THE SAME PAIRS IN SIGNED DIGITS BESIDE IT.
The counting-convention audit of base2_groups.py's 19.40% (the session's audit; its words, not his). Written for this
audit: no import of, or code from, base2_groups.py, qwen_groups.py, groups/, groups2/ or base2_fringes2.py. The
reference is base2.py's own run (recording/C_ref/plain_refs.npz: every bank pile and stream on the 30 refs texts, the
light at the top and the score rows' sha256) and base2_gpt2.npz's tables.

WHAT IS COUNTED (a token = the mean over the 648 refs positions):
  PLAIN, as base2_groups.py says it counts it: at every bank the light's 1s x the plate's 1s per line (each 1 of the
  light meets each 1 of the weight on its line once); the arrival's q.k and shares x values the same way, its shares'
  total and its divides; every water level, bank divide, bias, bend, stream add, the entry's position row; the top
  read's meetings, every word's divide and every word's bias. Compares and table reads are not photons on either side.
  BESIDE IT, THE SAME PAIRS IN SIGNED DIGITS ("plain in signed digits"): every light that meets a plate written in
  signed digits once (its 1s) and each lit digit of the light meeting each lit digit of the plate; the plate written
  in signed digits is plate only (free, as the group machine's own plate rewrite is). Every other part unchanged.
  ALSO BESIDE IT: the plate in signed digits and the light left binary (ones(x) x naf(w), no writing).
WHY: the group machine costs every read at the ready pile's signed digits and pays the light's and the ready piles'
  rewriting; plain costs every meeting at binary 1s. This puts both sides in one digit form, to show how much of the
  80.6% removed is the signed-digit form itself and how much the groups.
CHECKS: every bank pile = plain_refs' (rows differing), every stream after each half layer, the light at the top, and
  every score row's sha256 at the top; my plain total = the delivered plain (groups2/machine/gpt2_refs.json plain keys
  + gpt2_top_refs.npz plain) to the photon, part by part.

PREDICTION (written before this file ran):
  - Exact: 0 rows differing in all 48 piles and 24 streams, top light identical, score rows 648/648.
  - Plain rebuilt = 3,666,596,259 a token exactly (sum over 648 positions identical to the photon), each part identical:
    qkv 679,699,985, proj 248,461,379, fc 902,064,996 (meetings + divides + biases + bends), mproj 914,738,348,
    arrival 13,053,756, top read 904,017,313, rest 4,560,481.
  - Plain in signed digits: banks about 58-59% of their plain (docs/06's 58.65% on the 48 banks), the arrival about
    61-62% (base2_fringes2's 8.02M was this form), the top read about 55-62% of the plain read (the word table's NAF
    digits against its 1s, the light's NAF against its 1s): the whole about 57-60% of plain, about 2.1-2.2 billion a
    token. Then base2_groups' 711,431,068 is about 32-34% of the signed-digit plain.
  - Plate-only signed digits (light binary): banks about 75-80% of plain.

    python3 confirm_groups/audit_gpt2_plain.py     -> confirm_groups/audit_gpt2_plain.json/.log, audit_gpt2_lights.npz
"""
import os, sys, json, time, hashlib, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, rss, ones, naf, ones1, member, product, rshift, to16, log2fix, power, gain_ones

NPZ = f"{HERE}/base2_gpt2.npz"; ES, SC = 16, 10
PHI, ARR, LOG2, EXP2 = (np.asarray(member(NPZ, k)) for k in ("phi", "arrival", "log2", "exp2"))
LOG768, EPS40 = int(member(NPZ, "log2_768")), int(member(NPZ, "eps40"))
def wplate(name): return np.ascontiguousarray(member(NPZ, f"{name}.w")), np.asarray(member(NPZ, f"{name}.e")), np.asarray(member(NPZ, f"{name}.b")).astype(np.int64)
def pct(a, b): v = (20000 * a + b) // (2 * b); return f"{v // 100}.{v % 100:02d}%"

def gauge_log(R):
    x16, ex = to16(R, -ES); S1, S2 = x16.sum(1), (x16 * x16).sum(1); V = 768 * S2 - S1 * S1
    V = V + np.array([EPS40 >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else EPS40 << -(40 + 2 * int(e)) for e in ex], np.int64)
    return x16, ex, np.right_shift(log2fix(V, LOG2), 1) - LOG768 + ex * 65536
def water(x16):                         # the gauge's sums: S1 adds, S2 squares, 768 S2 - S1^2 (base2_fringes2's convention)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
def share_of(H):
    idx = rshift(H, ES - 12) + 32768; return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))

def arrival(QKV, C, l):
    """base2.py's arrival on one text, whole numbers; counts plain and signed digits"""
    P = QKV.shape[0]; out = np.zeros((P, 768), np.int64)
    for h in range(12):
        q, k, v = QKV[:, h * 64:(h + 1) * 64], QKV[:, 768 + h * 64:768 + (h + 1) * 64], QKV[:, 1536 + h * 64:1536 + (h + 1) * 64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] - 3 + SC))
        C["plain|arrival"] += int(np.tril(ones(q16) @ ones(k16).T).sum())
        C["sd|arrival"] += int(np.tril(naf(q16) @ naf(k16).T).sum()) + int(ones(q16).sum() + ones(k16).sum() + ones(v).sum())
        ov, ovn = ones(v).sum(1), naf(v).sum(1)
        for p in range(P):
            row = sc[p, :p + 1]; d = row.max() - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = a @ v[:p + 1]; Lp = int(log2fix([a.sum()], LOG2)[0])
            out[p, h * 64:(h + 1) * 64] = power(num, -Lp, EXP2)
            both = int(ones(a).sum()) + int(ones(num).sum()) * int(gain_ones(-Lp, EXP2))      # shares' total, divides
            C["plain|arrival"] += both + int((ones(a) * ov[:p + 1]).sum())
            C["sd|arrival"] += both + int((naf(a) * ovn[:p + 1]).sum())
    return out

def main():
    t0 = time.time(); log_f = open(f"{CG}/audit_gpt2_plain.log", "w")
    def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = REF["npos"]
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    assert [len(t) for t in TEXTS] == list(npos), "texts and plain_refs disagree"
    off = np.concatenate([[0], np.cumsum(npos)]); NP = int(off[-1])
    ids = np.concatenate([np.asarray(t, np.int64) for t in TEXTS]); pos = np.concatenate([np.arange(n) for n in npos])
    log(f"AUDIT 1 (GPT-2): plain rebuilt part by part on the 30 refs texts ({NP} positions), the same pairs in signed digits beside it")
    C = {k: 0 for k in ("plain|qkv", "plain|proj", "plain|fc", "plain|mproj", "plain|arrival", "plain|rest", "plain|top",
                        "sd|qkv", "sd|proj", "sd|fc", "sd|mproj", "sd|arrival", "sd|rest", "sd|top",
                        "platesd|qkv", "platesd|proj", "platesd|fc", "platesd|mproj", "platesd|top")}
    BAD = {}
    def chk(name, n): BAD[name] = BAD.get(name, 0) + int(n)
    WTE = member(NPZ, "wte"); WPE = member(NPZ, "wpe")
    R0 = np.asarray(WTE[ids]).astype(np.int64) + np.asarray(WPE[pos]).astype(np.int64)
    C["plain|rest"] += int(ones(np.asarray(WPE[pos]).astype(np.int64)).sum()); C["sd|rest"] += int(ones(np.asarray(WPE[pos]).astype(np.int64)).sum())
    LIGHTS = {}
    def bank_meet(kind, x, w):
        o_, n_ = ones(x), naf(x); wr, nr = ones(w).sum(1), naf(w).sum(1)
        C[f"plain|{kind}"] += int((o_ @ wr).sum()); C[f"sd|{kind}"] += int((n_ @ nr).sum() + o_.sum()); C[f"platesd|{kind}"] += int((o_ @ nr).sum())
    def both(kind, v): C[f"plain|{kind}"] += int(v); C[f"sd|{kind}"] += int(v)
    for l in range(12):
        Rin = R0 if l == 0 else REF[f"stream_{l - 1}"].astype(np.int64)
        # qkv
        w, we, b = wplate(f"qkv.{l}"); x16, ex, LS = gauge_log(Rin); pile = product(x16, w)
        chk("qkv piles vs plain_refs (rows)", (pile != REF[f"qkv_pile_{l}"]).any(1).sum())
        bank_meet("qkv", x16, w)
        both("rest", water(x16).sum() + (ones(pile).sum(1) * gain_ones(-LS, EXP2)).sum() + NP * ones(b).sum())
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2) + b
        if l in (0, 6): LIGHTS[f"qkv_{l}"] = x16.astype(np.int32)
        # the arrival, text by text
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C, l) for t in range(30)])
        # proj
        w, we, b = wplate(f"proj.{l}"); o16, eo = to16(o, -ES); pile = product(o16, w)
        chk("proj piles vs plain_refs (rows)", (pile != REF[f"proj_pile_{l}"]).any(1).sum())
        bank_meet("proj", o16, w); att = rshift(pile, -(we[None, :] + eo[:, None] + ES)) + b
        both("rest", NP * ones(b).sum() + ones(att).sum())
        Ra = Rin + att; chk("streams after the arrival vs plain_refs (rows)", (Ra != REF[f"stream_a_{l}"]).any(1).sum())
        if l in (0, 6): LIGHTS[f"proj_{l}"] = o16.astype(np.int32)
        # fc
        w, we, b = wplate(f"fc.{l}"); x16, ex, LS = gauge_log(Ra); pile = product(x16, w)
        chk("fc piles vs plain_refs (rows)", (pile != REF[f"fc_pile_{l}"]).any(1).sum())
        bank_meet("fc", x16, w); both("rest", water(x16).sum())
        H = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2) + b; sh = share_of(H)
        both("fc", (ones(pile).sum(1) * gain_ones(-LS, EXP2)).sum() + NP * ones(b).sum() + (ones(H) * ones(sh)).sum())
        G = rshift(H * sh, 16)
        if l in (0, 6): LIGHTS[f"fc_{l}"] = x16.astype(np.int32)
        # mproj
        w, we, b = wplate(f"mproj.{l}"); g16, eg = to16(G, -ES); pile = product(g16, w)
        chk("mproj piles vs plain_refs (rows)", (pile != REF[f"mproj_pile_{l}"]).any(1).sum())
        bank_meet("mproj", g16, w); mo = rshift(pile, -(we[None, :] + eg[:, None] + ES)) + b
        both("rest", NP * ones(b).sum() + ones(mo).sum())
        chk("streams after the layer vs plain_refs (rows)", (Ra + mo != REF[f"stream_{l}"]).any(1).sum())
        if l in (0, 6): LIGHTS[f"mproj_{l}"] = g16.astype(np.int32)
        log(f"  layer {l:2d}: piles and streams checked ({time.time() - t0:.0f}s, {rss()} MB)")
    np.savez(f"{CG}/audit_gpt2_lights.npz", **LIGHTS)
    # the top
    x16, ex, LS = gauge_log(REF["stream_11"].astype(np.int64))
    chk("light at the top vs plain_refs (positions)", (x16 != REF["x16"]).any(1).sum() + (ex != REF["ex"]).sum() + (LS != REF["LS"]).sum())
    both("rest", water(x16).sum())
    Wh = np.ascontiguousarray(member(NPZ, "head.w")); He = np.asarray(member(NPZ, "head.e")); Hb = np.asarray(member(NPZ, "head.b")).astype(np.int64)
    assert (He == He[0]).all(), "the word table has more than one power of two"
    pph, nph = np.zeros(768, np.int64), np.zeros(768, np.int64)
    for j0 in range(0, Wh.shape[1], 8192):
        blk = np.asarray(Wh[:, j0:j0 + 8192]).astype(np.int64); pph += ones(blk).sum(1); nph += naf(blk).sum(1)
    Fp = (int(He[0]) + ex + ES) * 65536 - LS; ge = gain_ones(Fp, EXP2); bias1 = int(ones(Hb).sum())
    top_div = np.zeros(NP, np.int64); sha_same = 0
    for p0 in range(0, NP, 48):
        PL = product(x16[p0:p0 + 48], Wh); top_div[p0:p0 + 48] = ones(PL).sum(1) * ge[p0:p0 + 48]
        S = power(PL, Fp[p0:p0 + 48, None], EXP2) + Hb[None, :]
        for i in range(len(S)): sha_same += hashlib.sha256(np.ascontiguousarray(S[i], np.int64).tobytes()).digest() == bytes(REF["score_sha"][p0 + i])
    C["plain|top"] = int((ones(x16) @ pph).sum() + top_div.sum() + NP * bias1)
    C["sd|top"] = int((naf(x16) @ nph).sum() + ones(x16).sum() + top_div.sum() + NP * bias1)
    C["platesd|top"] = int((ones(x16) @ nph).sum() + top_div.sum() + NP * bias1)
    chk("score rows' sha256 vs plain_refs (positions differing)", NP - sha_same)
    # against the delivered plain
    J = json.load(open(f"{HERE}/groups2/machine/gpt2_refs.json")); T = np.load(f"{HERE}/groups2/machine/gpt2_top_refs.npz")
    dl = {p_: 0 for p_ in ("qkv", "proj", "fc", "mproj", "arrival", "rest")}
    for k, v in J["counts"].items():
        w_, L_, p_, it = k.split("|", 3)
        if w_ == "plain": dl[p_] += v
    dl["top"] = int(T["plain"].sum())
    log("")
    log(f"EXACT: " + ", ".join(f"{k} {v}" for k, v in BAD.items()))
    log("")
    log(f"PLAIN, PART BY PART (a token): {'part':8s} {'this rebuild':>16s} {'delivered':>16s} same")
    parts = ("qkv", "proj", "fc", "mproj", "arrival", "top", "rest")
    for p_ in parts:
        log(f"    {p_:8s} {C['plain|' + p_] / NP:16,.0f} {dl[p_] / NP:16,.0f} {C['plain|' + p_] == dl[p_]}")
    P0 = sum(C["plain|" + p_] for p_ in parts); D0 = sum(dl.values())
    log(f"    WHOLE    {P0 / NP:16,.0f} {D0 / NP:16,.0f} {P0 == D0}   (sums over {NP} positions: {P0:,} and {D0:,})")
    log("")
    log("THE SAME PAIRS IN SIGNED DIGITS (light written once, its 1s; lit digit x lit digit), a token:")
    SD0 = sum(C["sd|" + p_] for p_ in parts)
    for p_ in parts:
        extra = f"   plate-only signed digits {C['platesd|' + p_] / NP:14,.0f} ({pct(C['platesd|' + p_], C['plain|' + p_])})" if f"platesd|{p_}" in C else ""
        log(f"    {p_:8s} {C['sd|' + p_] / NP:16,.0f} = {pct(C['sd|' + p_], C['plain|' + p_])} of its plain{extra}")
    banks_p = sum(C["plain|" + p_] for p_ in ("qkv", "proj", "fc", "mproj")); banks_s = sum(C["sd|" + p_] for p_ in ("qkv", "proj", "fc", "mproj"))
    log(f"    banks    {banks_s / NP:16,.0f} = {pct(banks_s, banks_p)} of the banks' plain")
    log(f"    WHOLE    {SD0 / NP:16,.0f} = {pct(SD0, P0)} of plain")
    MG = 711431068 * NP
    log(f"  base2_groups' delivered 711,431,068 a token: {pct(MG, P0)} of plain; {pct(MG, SD0)} of the same pairs in signed digits")
    json.dump(dict(counts=C, npos=NP, bad=BAD, delivered_plain=dl, plain_total=P0, sd_total=SD0), open(f"{CG}/audit_gpt2_plain.json", "w"), indent=0)
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

if __name__ == "__main__":
    main()
