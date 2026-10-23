"""The plain work of the whole-number Granite machine, in the unit of every earlier count (qwen06b, qwen_fringes2):
photon meetings, where the photons of a whole number are its 1s.
  - a plate: on each line, the light's 1s times the 1s of every weight on that line, summed over the lines
  - the arrival: q's 1s against k's 1s (scores), and each share's 1s against its value row's 1s (values)
  - the experts' mix: each of the 8 shares' 1s against its expert's write row's 1s
  - the top read: the last light's 1s against every word's weights' 1s
Water levels, turns, bends and divides are left out, as qwen_fringes2 leaves them out.

Two machines on the same light, at every position of the 30 refs texts (prompt + the first 15 reference words):
  LIT   the machine as it runs: each layer reads its router and the 8 lit experts, and leaves 24 dark
  ALL   the same light run through all 32 experts. Counted only: their writes are not used, and the stream is LIT's.
Check: at every position, the race winner here must equal granite_base2's own word (and fp32's where the texts agree).
    python3 count_granite.py
"""
import sys, json, time, numpy as np
sys.argv = sys.argv[:1]
import granite_base2 as g
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
L, NE, I, ES, SC = g.L, g.NE, g.I, g.ES, g.SC
def wrow(w): return ones(w).sum(1)                                  # each line: the 1s of all its weights
t0 = time.time()
ROW = {}
for l in range(L):
    for k in ("qkv", "o", "router"): ROW[f"{k}.{l}"] = wrow(g.ld(f"{k}.{l}.w"))
    ROW[f"gu.{l}"] = np.stack([wrow(g.GUW[l][x]) for x in range(NE)]); ROW[f"down.{l}"] = np.stack([wrow(g.DNW[l][x]) for x in range(NE)])
ROW["head"] = wrow(g.HEAD[0])
print(f"line counts ready ({time.time() - t0:.0f}s)", flush=True)

C = {}
def add(k, v): C[k] = C.get(k, 0) + int(v)
def light(X): return g.to16(X)[0]
def count_plate(name, X, row): add(name, (ones(light(X)) @ row).sum())
def attention_counted(QKV, pos, l):
    P = QKV.shape[0]; out = np.zeros((P, g.NH * g.HD), np.int64)
    K = [g.turn(QKV[:, 1024 + j * 64:1024 + (j + 1) * 64], pos) for j in range(8)]
    V = [QKV[:, 1536 + j * 64:1536 + (j + 1) * 64] for j in range(8)]
    for h in range(g.NH):
        q = g.turn(QKV[:, h * 64:(h + 1) * 64], pos); k, v = K[h // 2], V[h // 2]
        q16, eq = g.to16(q); k16, ek = g.to16(k); sc = g.rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] + SC))
        add("arrival", np.tril(ones(q16) @ ones(k16).T).sum()); ov = ones(v).sum(1)
        for p in range(P):
            row = sc[p, :p + 1]; d = row.max() - row; a = np.where(d < len(g.ARR), g.ARR[np.minimum(d, len(g.ARR) - 1)], 0)
            add("arrival", ones(a) @ ov[:p + 1]); out[p, h * 64:(h + 1) * 64] = g.share_mix(row, v[:p + 1])
    return out
def experts_counted(R, l):
    P = R.shape[0]; rp = g.gauged_slab(R, g.plate(f"router.{l}")); count_plate("router", R, ROW[f"router.{l}"])
    lit = np.zeros((P, g.TOPK), np.int64)
    for p in range(P):
        taken = np.zeros(NE, bool)
        for s in range(g.TOPK):
            x = g.race(np.where(taken, np.iinfo(np.int64).min, rp[p])); lit[p, s] = x; taken[x] = True
    Y = np.zeros((P, g.TOPK, g.D), np.int64); xl = ones(light(R))
    for x in range(NE):
        gu, dn = g.expert_plates(l, x)
        B = g.bend(*np.split(g.gauged_slab(R, gu), 2, axis=1))      # every position through expert x (count for ALL)
        per = xl @ ROW[f"gu.{l}"][x] + ones(light(B)) @ ROW[f"down.{l}"][x]
        add("experts ALL", per.sum())
        rows, slot = np.nonzero(lit == x)
        if len(rows):
            add("experts LIT", per[rows].sum()); Y[rows, slot] = g.plain_slab(B[rows], dn)
    out = np.zeros((P, g.D), np.int64)
    for p in range(P):
        lead = g.rshift(rp[p, lit[p]], ES - SC); d = lead.max() - lead
        a = np.where(d < len(g.ARR), g.ARR[np.minimum(d, len(g.ARR) - 1)], 0); add("experts' mix", ones(a) @ ones(Y[p]).sum(1))
        out[p] = g.share_mix(lead, Y[p])
    return out, lit
refs = json.load(open("refs_granite.json")); npos = 0; wins = []; lits_all = []
for t, r in enumerate(refs):
    seq = r["ids"] + r["ref"][:15]; P = len(seq); pos = np.arange(P); npos += P
    R = np.asarray(g.EMBED[seq]).astype(np.int64); lits = []
    for l in range(L):
        count_plate("qkv", R, ROW[f"qkv.{l}"]); QKV = g.gauged_slab(R, g.plate(f"qkv.{l}"))
        A_ = attention_counted(QKV, pos, l); count_plate("o", A_, ROW[f"o.{l}"]); R = R + g.plain_slab(A_, g.plate(f"o.{l}"))
        out, lit = experts_counted(R, l); R = R + out; lits.append(lit)
    count_plate("top read", R, ROW["head"])
    sc = g.gauged_slab(R, g.HEAD); wins.append([g.race(sc[p]) for p in range(P)]); lits_all.append(np.stack(lits))
    print(f"text {t:2d}: {P} positions ({time.time() - t0:.0f}s)", flush=True)
np.savez("count_lits.npz", *lits_all)
json.dump(dict(npos=npos, counts=C, wins=wins), open("count_granite.json", "w"))
lit = sum(v for k, v in C.items() if k != "experts ALL"); allx = lit - C["experts LIT"] + C["experts ALL"]
print(f"\nPLAIN WORK A TOKEN (mean over {npos} positions), photon meetings:")
for k in ("qkv", "arrival", "o", "router", "experts LIT", "experts' mix", "top read"):
    print(f"  {k:14s} {C[k] / npos:16,.0f}   {100 * C[k] / lit:6.2f}% of LIT")
print(f"  {'LIT whole':14s} {lit / npos:16,.0f}")
print(f"  {'experts ALL':14s} {C['experts ALL'] / npos:16,.0f}   (the 32 experts on the same light)")
print(f"  {'ALL whole':14s} {allx / npos:16,.0f}   LIT is {100 * lit / allx:.2f}% of ALL; the 8 lit experts are {100 * C['experts LIT'] / C['experts ALL']:.2f}% of all 32")
