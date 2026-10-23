"""QuantumGranite, all base 2, checked against fp32 Granite-3.1-1B-A400M's own greedy words. Every value is a whole
number. It is qwen/qwen_base2.py with three changes:
  - no per-head gauges (Granite has none); 64-wide heads; head h reads key/value head h // 2
  - the router: a small plate (1024 x 32) after the arrival's gauge. Its 32 piles are raced at full size, and the 8
    largest experts are lit (ties go to the lower number, as in the race). The others are not read at all.
  - the lit experts' writes are mixed by the arrival's own table: share e^-(lead - pile) among the 8, then divided by
    the shares' total through log2 and 2^x, exactly as the arrival mixes its values.
Switches:  words=16  texts=30  from=0   show=1 (print the lit experts of the last position, layer by layer)
    python3 granite_base2.py words=16 texts=30
"""
import sys, json, numpy as np
A = dict(a.split("=") for a in sys.argv[1:])
WORDS, TEXTS, FROM, SHOW = int(A.get("words", 16)), int(A.get("texts", 30)), int(A.get("from", 0)), int(A.get("show", 0))
ES, SC, L, D, HD, NH, NE, TOPK, I = 16, 10, 24, 1024, 64, 16, 32, 8, 512
def ld(name): return np.load(f"granite_base2/{name}.npy", mmap_mode="r")
LOG2, EXP2, ARR, SIG, COS, SIN = (np.asarray(ld(n)) for n in ("log2", "exp2", "arrival", "sig", "cos", "sin"))
EPS40 = int(ld("eps40")); EMBED = ld("embed")
HEAD = (np.asarray(ld("head.w")).astype(np.int64), np.asarray(ld("head.e")))
def plate(name): return np.asarray(ld(name + ".w")).astype(np.int64), np.asarray(ld(name + ".e"))
GUW, GUE, DNW, DNE = ([ld(f"{k}.{l}.{s}") for l in range(L)] for k, s in (("gu", "w"), ("gu", "e"), ("down", "w"), ("down", "e")))
def expert_plates(l, x): return (np.asarray(GUW[l][x]).astype(np.int64), np.asarray(GUE[l][x])), (np.asarray(DNW[l][x]).astype(np.int64), np.asarray(DNE[l][x]))
read = {"experts": 0}

def rshift(X, s):
    s = np.asarray(s, dtype=np.int64); up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)
def to16(X):                                                   # each row on 16 bits, its own power of two (from the OR of its sizes)
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)]); sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), sh - ES
def log2fix(v):
    v = int(v); t = v.bit_length() - 1; m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF; return t * 65536 + int(LOG2[m])
def power(X, F):
    F = np.asarray(F, dtype=np.int64); I_, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF); return rshift(X * EXP2[f], 16 - I_)
def gauge(x16, ex, n):                                         # log2 of the root mean square, no root and no divide: halve the log
    V = (x16 * x16).sum(1); half = int(round(np.log2(n) * 65536)) // 2
    return np.array([(log2fix(V[p] + ((EPS40 * n) >> (40 + 2 * int(ex[p])) if 40 + 2 * int(ex[p]) >= 0 else (EPS40 * n) << -(40 + 2 * int(ex[p])))) >> 1) - half + int(ex[p]) * 65536 for p in range(len(V))])
def gauged_slab(R, wp):
    w, we = wp; x16, ex = to16(R); pile = x16 @ w
    return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - gauge(x16, ex, D)[:, None])
def plain_slab(X, wp):
    w, we = wp; x16, ex = to16(X); return rshift(x16 @ w, -(we[None, :] + ex[:, None] + ES))
def turn(X, pos):                                              # pair (i, i+32) turns by its position's angle
    a, b = X[:, :HD // 2], X[:, HD // 2:]; c, s = COS[pos], SIN[pos]
    return np.concatenate([rshift(a * c - b * s, 30), rshift(b * c + a * s, 30)], 1)
def race(vals):
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63); alive = np.ones(len(u), bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])
def share_mix(lead_row, Y):                                    # e^-(lead - pile) shares, then divide by their total
    d = lead_row.max() - lead_row; a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
    S = a[None, :] @ Y; t = max(0, int(np.abs(S).max()).bit_length() - 46)   # Granite's experts write values in the thousands:
    return power(rshift(S, t), t * 65536 - log2fix(a.sum()))[0]                 # keep 46 bits before the 2^x table, the rest in the power
def attention(QKV, pos):
    P = QKV.shape[0]; out = np.zeros((P, NH * HD), np.int64)
    K = [turn(QKV[:, 1024 + j * HD:1024 + (j + 1) * HD], pos) for j in range(8)]
    V = [QKV[:, 1536 + j * HD:1536 + (j + 1) * HD] for j in range(8)]
    for h in range(NH):
        q = turn(QKV[:, h * HD:(h + 1) * HD], pos); k, v = K[h // 2], V[h // 2]
        q16, eq = to16(q); k16, ek = to16(k); sc = rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] + SC))
        for p in range(P): out[p, h * HD:(h + 1) * HD] = share_mix(sc[p, :p + 1], v[:p + 1])
    return out
def bend(G, U):                                                # the gated bend: g * sigma(g) * u
    idx = rshift(G, ES - 12) + 65536; share = np.where(idx < 0, 0, np.where(idx > 131071, 65536, SIG[np.clip(idx, 0, 131071)]))
    return rshift(rshift(G * share, 16) * U, ES)
def experts(R, l):
    P = R.shape[0]; rp = gauged_slab(R, plate(f"router.{l}")); lit = np.zeros((P, TOPK), np.int64)
    for p in range(P):                                         # the 8 largest router piles, by the race
        left = rp[p].copy(); taken = np.zeros(NE, bool)
        for s in range(TOPK):
            x = race(np.where(taken, np.iinfo(np.int64).min, left)); lit[p, s] = x; taken[x] = True
    Y = np.zeros((P, TOPK, D), np.int64)
    for x in np.unique(lit):
        rows, slot = np.nonzero(lit == x); gu, dn = expert_plates(l, x); read["experts"] += 1
        GU = gauged_slab(R[rows], gu); Y[rows, slot] = plain_slab(bend(GU[:, :I], GU[:, I:]), dn)
    out = np.stack([share_mix(rshift(rp[p, lit[p]], ES - SC), Y[p]) for p in range(P)])
    return out, lit
def forward(ids):
    R = np.asarray(EMBED[ids]).astype(np.int64); pos = np.arange(len(ids)); lits = []
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, plate(f"qkv.{l}")), pos), plate(f"o.{l}"))
        out, lit = experts(R, l); R = R + out; lits.append(lit)
    return race(gauged_slab(R[-1:], HEAD)[0]), lits
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-base"); refs = json.load(open("refs_granite.json"))
    print(f"QuantumGranite, all base 2: {WORDS} words, texts {FROM}..{FROM + TEXTS - 1}")
    same = 0
    for r in refs[FROM:FROM + TEXTS]:
        got = []
        for _ in range(WORDS):
            w, lits = forward(r["ids"] + got); got.append(w)
        want = r["ref"][:WORDS]; same += got == want
        print(f"  {r['prompt']!r:42s} {'IDENTICAL' if got == want else 'DIFFERENT'}  {tok.decode(got)!r}" + ("" if got == want else f"   model: {tok.decode(want)!r}"), flush=True)
        if SHOW: print("   lit experts at the last position, layer by layer:", [sorted(x[-1].tolist()) for x in lits])
    print(f"IDENTICAL {same} of {len(refs[FROM:FROM + TEXTS])}; expert plates read {read['experts']:,}")
