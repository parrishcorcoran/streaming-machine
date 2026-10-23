"""JUST ADD THE ONES. Every value becomes its photons only: sign x (number of 1s). Every weight the same. A meeting adds
+1 or -1 for every photon pair, nothing else (no place, no brightness). Two looks:
  1. at plates qkv.6, fc.6, mproj.6 with the machine's real light: do the ones-only piles agree with the true piles?
  2. the whole machine with every bank met ones-only (one fixed scale per bank, set from text 0), judged on texts 1-29."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); import judge5 as J
POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):
    A = np.asarray(A).astype(np.int64); m = np.abs(A); n = np.zeros(A.shape, np.int64)
    while m.any(): n += POP[m & 0xFFFF]; m = m >> 16
    return np.sign(A) * ((1 << n) - 1)
plain = J.product; BANKS = {id(J.W[k][0]): k for k in J.W if k != "head"}; CW, CAP, SCALE, MODE = {}, {}, {}, ["plain"]
def prod(x16, w16):
    k = BANKS.get(id(w16))
    if k is None: return plain(x16, w16)
    if k not in CW: CW[k] = ones(np.rint(w16).astype(np.int64)).astype(np.float64)
    c = ones(x16).astype(np.float64) @ CW[k]
    if MODE[0] == "plain":
        t = plain(x16, w16); CAP.setdefault(k, []).append((t, c)); return t
    return np.rint(c * SCALE[k]).astype(np.int64)
J.product = prod
refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
def run(ids):
    P = len(ids); R = J.M["wte"][ids] + J.M["wpe"][:P]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = J.W["head"]; x16, ex, LS = J.gauge_log(R)
    return J.power(plain(x16, w), (we[None, :] + ex[:, None] + J.ES) * 65536 - LS[:, None]) + b
# 1. the plates, real light, all 30 texts
for t in range(30): run(TEXTS[t])
print("1. ones-only piles against the true piles (real light, 30 texts, all positions):")
for k in ("qkv.6", "fc.6", "mproj.6"):
    T = np.concatenate([a for a, _ in CAP[k]]).astype(np.float64); C = np.concatenate([b for _, b in CAP[k]])
    same = float((np.sign(T) == np.sign(C)).mean()); r = np.corrcoef(T.ravel(), C.ravel())[0, 1]
    print(f"   {k:8s} same sign {100 * same:5.1f}%   (coin toss 50%)   line-up of sizes r = {r:+.3f}")
# 2. the whole machine ones-only, scale per bank from text 0
for k in CAP:
    T = np.abs(CAP[k][0][0]).astype(np.float64); C = np.abs(CAP[k][0][1])
    SCALE[k] = float(np.median(T[C > 0] / C[C > 0]))
MODE[0] = "ones"
right = in5 = 0
for t in range(1, 30):
    ids = TEXTS[t]; n0 = len(refs[t]["ids"]); S = run(ids)
    for p in range(n0 - 1, len(ids) - 1):
        top5 = np.argsort(-S[p])[:5]; right += int(top5[0] == ids[p + 1]); in5 += int(ids[p + 1] in top5)
n = sum(len(TEXTS[t]) - len(refs[t]["ids"]) for t in range(1, 30))
print(f"\n2. the whole machine, every bank met ones-only: right next word {right}/{n}, in the top 5 {in5}/{n}   (plain machine: {n}/{n} same words)")
