"""CHECKER'S OWN COUNT of the whole GPT-2 group machine BY PART, on refs texts 5 and 12 (43 positions), from base2.py's
fresh run (gpt2_ref.npz: the light at every bank, the piles, QKV, the streams, the light at the top) and the plates in
base2_gpt2.npz. Every bank's group work by mycount.py (the checker's own code), its pile checked against base2's pile;
layer 0 as base2_groups' rules fixed it (qkv.0: stored rows + the carry in signed digits; proj.0: the whole light, rule
3; fc.0: stored rows + the rest of the light; position 0's stream read); fc in the whole form (rule 4: the knife kept at
none) with base2's own divide, bias and bend; the arrival and the rest by base2_fringes2's counting rules, written again
here from base2's own attention steps; the top: the plain read, the ready piles and the light written (own code), and
(ii) recounted by topg_own.py if present. Set beside the builder's counts of the same two texts (gpt2_machine_check.json,
aggregated as base2_groups' phase_report aggregates them).
    python3 count_gpt2.py -> count_gpt2.log / .json
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"
sys.path.insert(0, OUT)
from mycount import slab_work, ones, sdc, rule, NPL
src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
for a_, b_ in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"), ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert a_ in src; src = src.replace(a_, b_)
B = {}; cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = argv; os.chdir(cwd)
Mz, ES, SC, ARR, EXP2, PHI = B["M"], B["ES"], B["SC"], B["ARR"], B["EXP2"], B["PHI"]
gauge_log, power, rshift, to16, race, log2fix = (B[k] for k in ("gauge_log", "power", "rshift", "to16", "race", "log2fix"))
LOGF = open(f"{OUT}/count_gpt2.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def plate(name): return np.asarray(Mz[name + ".w"]).astype(np.int64), np.asarray(Mz[name + ".e"]).astype(np.int64), np.asarray(Mz[name + ".b"]).astype(np.int64)
def Eones(LS): return ones(EXP2[np.bitwise_and(-np.asarray(LS, np.int64), 0xFFFF)])
def water(x16):
    S1, S2 = x16.sum(1), (x16 * x16).sum(1); return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
def share_of(H):
    idx = rshift(H, ES - 12) + 32768; return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]; G = 12
REF = np.load(f"{OUT}/gpt2_ref.npz"); TEXTS_ALL = [r["ids"] + r["ref"] for r in json.load(open(f"{HERE}/refs_many.json"))]
GW = ("light", "table", "pwrite", "xsum", "xoff")
mine = {p: 0 for p in ("qkv", "proj", "fc", "mproj", "arrival", "rest", "top")}; plain = dict(mine); items = {}
def add(d, p, v): d[p] += int(np.sum(v))
def addi(k, v): items[k] = items.get(k, 0) + int(np.sum(v))
def group(x, kind, l, w, ref_pile, masks=None, who=None):
    pile, c, ch = slab_work(x, w, [(kind, 0, w.shape[1])], G, [1, SLABS.index(kind), l, G], masks=masks)
    assert np.array_equal(pile, ref_pile), f"{kind}.{l}: the checker's group pile differs from base2's"
    tot = sum(int(c[k].sum()) for k in GW) + int(c[f"reads|{kind}"].sum()) + int(c[f"sdreads|{kind}"].sum())
    for k in GW: addi(f"{kind}|{k}", c[k])
    addi(f"{kind}|reads", c[f"reads|{kind}"]); addi(f"{kind}|sdreads", c[f"sdreads|{kind}"])
    return tot, int(c[f"plain|{kind}"].sum()), ch[kind]

def arrival(QKV, first):
    """base2's attention step for step (its own functions), counted by base2_fringes2's rules, written here"""
    P = len(QKV); out = np.zeros((P, 768), np.int64); fr = pl = 0
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        qk = q16 @ k16.T; sc = rshift(qk, -(eq[:, None] + ek[None, :] - 3 + SC))
        pl += np.tril(ones(q16) @ ones(k16).T).sum(); fr += np.tril(sdc(q16) @ sdc(k16).T).sum(1)[first:].sum()
        fr += ones(q16[first:]).sum() + ones(k16).sum() + ones(v).sum()
        ov, nv = ones(v).sum(1), sdc(v).sum(1)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row; a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            Lp = log2fix([a.sum()])[0]; num = a @ v[:p+1]; out[p, h*64:(h+1)*64] = power(num, -Lp)
            dv = ones(num).sum() * Eones([Lp])[0]
            pl += ones(a).sum() + ones(a) @ ov[:p+1] + dv
            if p >= first: fr += ones(a).sum() + sdc(a) @ nv[:p+1] + dv
    return out, int(fr), int(pl)

t0 = time.time(); TS = [5, 12]; masks_seen = {}
for t in TS:
    ids = np.asarray(TEXTS_ALL[t], np.int64); P = len(ids); pos = np.arange(P); r1 = pos[1:]
    WTE, WPE = np.asarray(Mz["wte"]).astype(np.int64), np.asarray(Mz["wpe"]).astype(np.int64)
    Ar, Br = WTE[ids], WPE[pos]; R = Ar + Br
    add(mine, "rest", ones(Br)); add(plain, "rest", ones(Br))                                    # entry: word + position
    g = lambda k, l: REF[f"t{t}_{k}_{l}"].astype(np.int64)
    for l in range(12):
        # ---- qkv ----
        w, we, b = plate(f"qkv.{l}"); x16, ex, LS = gauge_log(R); assert np.array_equal(x16, g("qkv_light", l))
        pile = g("qkv_pile", l); wrow = ones(w).sum(1)
        add(plain, "qkv", ones(x16) @ wrow)
        for d in (mine, plain): add(d, "rest", water(x16)); add(d, "rest", ones(pile).sum(1) * Eones(LS)); add(d, "rest", P * ones(b).sum())
        if l == 0:
            s = ex + ES; assert set(np.unique(s)) <= {1, 2, 4} and (s[pos == 0] == 4).all()
            m0 = np.flatnonzero(s == 4); mc = np.flatnonzero(np.isin(s, [1, 2]))
            add(mine, "qkv", ones(pile[m0])); addi("qkv.0|stored rows (position 0)", ones(pile[m0]))
            a_ = rshift(Ar[mc], s[mc][:, None]); b_ = rshift(Br[mc], s[mc][:, None]); carry = x16[mc] - a_ - b_
            assert np.abs(carry).max() <= 1
            t1, t2 = a_ @ w, b_ @ w
            tot, _, _ = group(carry, "qkv", 0, w, pile[mc] - t1 - t2, masks={"qkv": 0})
            add(mine, "qkv", ones(t1) + ones(t2)); add(mine, "qkv", ones(a_) + ones(b_)); add(mine, "qkv", tot)
            addi("qkv.0|stored rows", ones(t1) + ones(t2)); addi("qkv.0|split", ones(a_) + ones(b_)); addi("qkv.0|carry", tot)
        else:
            tot, _, m = group(x16, "qkv", l, w, pile); add(mine, "qkv", tot); masks_seen[f"qkv.{l}"] = m
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        assert np.array_equal(QKV, g("QKV", l))
        o, fr, pl = arrival(QKV, 1 if l == 0 else 0); assert np.array_equal(o, g("arrival", l))
        add(mine, "arrival", fr); add(plain, "arrival", pl)
        # ---- proj ----
        w, we, b = plate(f"proj.{l}"); o16, eo = to16(o, -ES); pile = g("proj_pile", l)
        add(plain, "proj", ones(o16) @ ones(w).sum(1))
        att = rshift(pile, -(we[None, :] + eo[:, None] + ES)) + b
        rows = r1 if l == 0 else pos
        tot, _, m = group(o16[rows], "proj", l, w, pile[rows]); add(mine, "proj", tot); masks_seen[f"proj.{l}"] = m
        add(mine, "rest", len(rows) * ones(b).sum()); add(mine, "rest", ones(att[rows]))
        add(plain, "rest", P * ones(b).sum()); add(plain, "rest", ones(att))
        Ra = R + att; assert np.array_equal(Ra, g("stream_a", l))
        # ---- fc ----
        w, we, b = plate(f"fc.{l}"); x16, ex, LS = gauge_log(Ra); assert np.array_equal(x16, g("fc_light", l)); pile = g("fc_pile", l)
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]; f = F & 0xFFFF; assert (f == f[:, :1]).all()
        e = EXP2[f[:, 0]].astype(np.int64); H = power(pile, F) + b
        fcfix = ones(pile).sum(1) * ones(e) + ones(b).sum() + (ones(H) * ones(share_of(H))).sum(1)      # divide, bias, bend (a position)
        add(plain, "fc", ones(x16) @ ones(w).sum(1)); add(plain, "fc", fcfix); add(plain, "rest", water(x16))
        add(mine, "rest", water(x16[rows]))
        if l == 0:
            s1 = (ex + ES)[r1]; assert set(np.unique(s1)) <= {5, 6}
            BP = plate("proj.0")[2]
            a_ = rshift(WTE[ids[r1]], s1[:, None]); b_ = rshift(WPE[r1] + BP[None, :], s1[:, None]); rest = x16[r1] - a_ - b_
            FT, FP = a_ @ w, b_ @ w; c0 = FT + FP
            tot, _, m = group(rest, "fc", 0, w, pile[r1] - c0); masks_seen["fc.0"] = m
            add(mine, "fc", ones(c0)); add(mine, "fc", ones(a_)); add(mine, "fc", ones(b_)); add(mine, "fc", tot); add(mine, "fc", fcfix[r1])
            addi("fc.0|stored rows (ones of the sum, as counted)", ones(c0)); addi("fc.0|stored rows (each row's ones)", ones(FT) + ones(FP))
        else:
            tot, _, m = group(x16, "fc", l, w, pile); add(mine, "fc", tot); add(mine, "fc", fcfix); masks_seen[f"fc.{l}"] = m
        out16 = g("fc_out16", l); sh = g("fc_sh", l)
        # ---- mproj ----
        w, we, b = plate(f"mproj.{l}"); pile = g("mproj_pile", l)
        add(plain, "mproj", ones(out16) @ ones(w).sum(1))
        tot, _, m = group(out16[rows], "mproj", l, w, pile[rows]); add(mine, "mproj", tot); masks_seen[f"mproj.{l}"] = m
        mo = rshift(pile, -(we[None, :] + (sh - ES)[:, None] + ES)) + b
        add(mine, "rest", len(rows) * ones(b).sum()); add(mine, "rest", ones(mo[rows])); add(plain, "rest", P * ones(b).sum()); add(plain, "rest", ones(mo))
        R = Ra + mo; assert np.array_equal(R, g("stream", l))
        if l == 0: add(mine, "rest", ones(R[pos == 0]))                                            # position 0's stream read
    x16, ex, LS = gauge_log(R); assert np.array_equal(x16, REF[f"t{t}_top_x16"]) and np.array_equal(LS, REF[f"t{t}_top_LS"])
    for d in (mine, plain): add(d, "rest", water(x16))
    log(f"text {t}: {P} positions ({time.time() - t0:.0f}s)")
log("masks (up/hole places) the checker's rule chose: " + json.dumps({k: [b for b in range(16) if (v >> b) & 1 == 0] for k, v in sorted(masks_seen.items())}))
json.dump(dict(mine=mine, plain=plain, items=items), open(f"{OUT}/count_gpt2_banks.json", "w"))
log("banks, arrival, rest (mine): " + json.dumps(mine)); log("plain: " + json.dumps(plain)); log("items: " + json.dumps(items))
