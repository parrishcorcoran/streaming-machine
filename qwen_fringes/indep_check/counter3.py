"""Checker's own counter of the work, on 3 of the drawn texts (4, 7, 12; 54 positions), from the checker's own reference
light (ref10.py: qwen_base2.py's own steps). Nothing imported from qwen_fringes.py or qwen_fringes/check.py; the counting
code is written here from the definitions in qwen_fringes.py's docstring. qwen_base2.py's source is executed only for its
attention steps (to recover q16, k16, the shares and the values, checked against the reference attention output's digest)
and its power/race for the read's last step.

Every product is formed directly (no walk): the pours are ones(|x w|) over every lit line and neuron; their signed digits
by Prodinger's form (popcount(h ^ (n + h)), h = n >> 1), not by n ^ 3n; the line-up's gaps from np.sort; W1's writing
as ones(|x v|) over each line's DISTINCT weight values (a different route to "each new running product once").
Output: per position, every count the builder's arrays hold, compared with the delivered machine's counts on the same
texts (machine10_counts.npz, machine10_top.npz), then the shares by part on these 54 positions.
"""
import sys, json, time, ctypes, subprocess, resource, hashlib, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; NPZ = f"{QD}/qwen_base2.npz"; OUT = f"{HERE}/qwen_fringes/indep_check"
T10 = [3, 4, 5, 7, 8, 9, 12, 16, 18, 22]; T3 = [4, 7, 12]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def pc(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)
def nf(a):
    n = np.abs(np.asarray(a, np.int64)); h = n >> 1; return np.bitwise_count(h ^ (n + h)).astype(np.int64)
def pc1(v): return bin(abs(int(v))).count("1")
def dg(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
t0 = time.time()
so = f"{OUT}/cnt.dylib"; subprocess.check_call(["cc", "-O3", "-shared", "-o", so, f"{OUT}/cnt.c"])
C = ctypes.CDLL(so); P_, I_ = ctypes.c_void_p, ctypes.c_int64
C.plate_counts.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, P_]; C.band_parts.argtypes = [I_, I_, P_, P_, P_, P_, I_, P_, P_, P_]
MM = np.load(NPZ)
src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a, b in [('M = np.load("qwen_base2.npz")', 'M = MM'), ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = {}')]:
    assert src.count(a) == 1; src = src.replace(a, b)
G = dict(MM=MM); sys.argv = ["x"]; exec(src, G)
to16, gauge, turn, head_gauge, race, power, rshift, log2fix = (G[k] for k in ("to16", "gauge", "turn", "head_gauge", "race", "power", "rshift", "log2fix"))
ARR, EXP2, ES, SC, HD, L = G["ARR"], G["EXP2"], G["ES"], G["SC"], G["HD"], G["L"]
LZ = np.load(f"{OUT}/ref10_lights.npz"); RF = json.load(open(f"{OUT}/ref10.json"))
meta = json.load(open(f"{HERE}/qwen06b/light/meta.json")); lens = [len(meta["texts"][t]) for t in T3]; NP = sum(lens)
off = np.concatenate([[0], np.cumsum(lens)])
def light(name): return np.concatenate([LZ[f"{t}__{name}"].astype(np.int64) for t in T3])

# ---------------- the plates ----------------
KINDS = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
         "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
KIDX = dict(q=0, k=1, v=2, o=3, gate=4, up=5, down=6)
mine = {}
def put(k, v): mine[k] = np.asarray(v, np.int64)
zg = {}
for l in range(L):
    for slab, kinds in KINDS.items():
        X = light(f"{slab}.{l}"); W16 = np.asarray(MM[f"{slab}.{l}.w"])
        put(f"{slab}.{l}|writing", pc(X).sum(1)); put(f"{slab}.{l}|writing3", pc(3 * X).sum(1))
        for kind, c0, c1 in kinds:
            w = np.ascontiguousarray(W16[:, c0:c1]); wl = w.astype(np.int64); Ln, N = w.shape
            ws = np.sort(wl, axis=1); gp = np.diff(ws, axis=1); GN = nf(ws[:, 0]) + nf(gp).sum(1); zg[f"{kind}.{l}"] = int((gp == 0).sum())
            rng = np.random.default_rng([2026, l, KIDX[kind]]); osh = rng.permuted(np.tile(np.arange(N), (Ln, 1)), axis=1)
            wsh = np.take_along_axis(wl, osh, 1); GNS = nf(wsh[:, 0]) + nf(np.diff(wsh, axis=1)).sum(1)
            keep = np.concatenate([np.ones((Ln, 1), bool), gp != 0], 1); DV = np.ascontiguousarray(ws[keep]); DO = np.concatenate([[0], np.cumsum(keep.sum(1))]).astype(np.int64)
            Xc = np.ascontiguousarray(X); p1, ps, w1 = (np.zeros(NP, np.int64) for _ in range(3))
            C.plate_counts(NP, Ln, N, Xc.ctypes.data, w.ctypes.data, DV.ctypes.data, DO.ctypes.data, p1.ctypes.data, ps.ctypes.data, w1.ctypes.data)
            k = f"{kind}.{l}|"
            put(k + "plain", pc(X) @ pc(wl).sum(1)); put(k + "signed", nf(X) @ nf(wl).sum(1))
            put(k + "gaps", nf(X) @ GN); put(k + "walk3", nf(3 * X) @ GN); put(k + "gaps_shuf", nf(X) @ GNS); put(k + "walk3_shuf", nf(3 * X) @ GNS)
            put(k + "pours", p1); put(k + "pours_sd", ps); put(k + "write1", w1)
    if l % 7 == 6: print(f"  plates: layer {l} counted ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)

# ---------------- the arrival ----------------
arr_bad = 0
for l in range(L):
    acc = {k: np.zeros(NP, np.int64) for k in ("plain_scores", "sd_scores", "write_q", "write_k", "plain_values", "sd_values", "write_v")}
    for ti, t in enumerate(T3):
        QKV = LZ[f"{t}__qkv_out.{l}"]; P = len(QKV); pos = np.arange(P); r0 = off[ti]; out = np.zeros((P, 16 * HD), np.int64)
        qg, qge, kg, kge = MM[f"qg.{l}"], int(MM[f"qge.{l}"]), MM[f"kg.{l}"], int(MM[f"kge.{l}"])
        K = [turn(head_gauge(QKV[:, 2048 + j * HD:2048 + (j + 1) * HD], kg, kge), pos) for j in range(8)]
        V = [QKV[:, 3072 + j * HD:3072 + (j + 1) * HD] for j in range(8)]
        for h in range(16):
            q = turn(head_gauge(QKV[:, h * HD:(h + 1) * HD], qg, qge), pos); k, v = K[h // 2], V[h // 2]
            q16, eq = to16(q); k16, ek = to16(k); sc = rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] + SC))
            oq, ok_, nq, nk = pc(q16).sum(1), None, None, None
            # scores: every pair j <= p, every dimension: ones(q) x ones(k) (plain) / signed digits
            pq, pk, sq, sk = pc(q16), pc(k16), nf(q16), nf(k16)
            for p in range(P):
                acc["plain_scores"][r0 + p] += int((pq[p][None, :] * pk[:p + 1]).sum()); acc["sd_scores"][r0 + p] += int((sq[p][None, :] * sk[:p + 1]).sum())
            acc["write_q"][r0:r0 + P] += pq.sum(1)
            vo, vn = pc(v).sum(1), nf(v).sum(1)
            if h % 2 == 0: acc["write_k"][r0:r0 + P] += pk.sum(1); acc["write_v"][r0:r0 + P] += vo
            for p in range(P):
                row = sc[p, :p + 1]; d = row[race(row)] - row; a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
                out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1], -log2fix(a.sum()))[0]
                acc["plain_values"][r0 + p] += int(pc(a) @ vo[:p + 1]); acc["sd_values"][r0 + p] += int(nf(a) @ vn[:p + 1])
        arr_bad += dg(out) != RF["dig"][f"arrival|{l}|{t}"]
    for k, v in acc.items(): put(f"arr.{l}|{k}", v)
print(f"  arrival counted; recovered attention outputs differing from the reference's: {arr_bad} of {L * len(T3)} ({time.time() - t0:.0f}s)", flush=True)

# ---------------- the band read ----------------
T = np.ascontiguousarray(MM["head.w"]); HE = MM["head.e"].astype(np.int64); EMIN = int(HE.min()); KW = HE - EMIN; NW = T.shape[1]
GNT = np.zeros(T.shape[0], np.int64); PPH = np.zeros(T.shape[0], np.int64)
for i0 in range(0, T.shape[0], 32):
    tb = T[i0:i0 + 32].astype(np.int64); ts = np.sort(tb, axis=1); GNT[i0:i0 + 32] = nf(ts[:, 0]) + nf(np.diff(ts, axis=1)).sum(1); PPH[i0:i0 + 32] = pc(tb).sum(1)
PB = np.array([26, 22, 18, 14, 10, 6, 2, 0], np.int64); MASK = np.array([15] * 7 + [3], np.int64); NB = 8
TOP = {k: [] for k in ("walk", "writing", "end", "plain_meet", "plain_div", "pours", "bound", "surv", "top5", "win")}
pile_bad = 0; lab = np.arange(NW)
for t in T3:
    X, EX, LS = LZ[f"top__{t}__x16"], LZ[f"top__{t}__ex"], LZ[f"top__{t}__LS"]
    for p in range(len(X)):
        x = np.ascontiguousarray(X[p].astype(np.int64)); F = int((EMIN + int(EX[p]) + ES) * 65536 - int(LS[p]))
        pile = np.zeros(NW, np.int64); DN = np.zeros((NW, NB), np.int32); HP = np.zeros((NW, NB), np.int32)
        C.band_parts(len(x), NW, x.ctypes.data, T.ctypes.data, PB.ctypes.data, MASK.ctypes.data, NB, pile.ctypes.data, DN.ctypes.data, HP.ctypes.data)
        pile_bad += int(((DN.astype(np.int64) << PB[None, :]).sum(1) != pile).sum())
        n = int((x != 0).sum()); e = int(EXP2[F & 0xFFFF]); ge = pc1(e); s = 16 - (F >> 16)
        delta = max(1, -(-(1 << s) // e)) if s >= 0 else 1; mu = 2 + delta
        alive = np.ones(NW, bool); V = np.zeros(NW, np.int64); pours = []; bound = 0; surv = []
        for b in range(NB):
            pours.append(int(HP[alive, b].sum())); V[alive] += DN[alive, b].astype(np.int64) << PB[b] << KW[alive]
            if b == NB - 1: break
            R = n * ((1 << int(PB[b])) - 1); Rw = R << KW
            L5 = np.sort((V - Rw)[alive])[-5]
            bound += pc1(n) + sum(min(5, int((alive & (KW == k)).sum())) * pc1(R) + pc1(mu) + pc1((R << k) + mu) for k in range(4) if (alive & (KW == k)).any())
            alive &= ~(V + Rw <= L5 - mu); surv.append(int(alive.sum()))
        assert np.array_equal(V[alive], pile[alive] << KW[alive])
        sc = power(V[alive], F); Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
        TOP["top5"].append(lab[np.lexsort((lab, -Sp))[:5]].tolist()); TOP["win"].append(race(Sp))
        TOP["walk"].append(int(nf(x) @ GNT)); TOP["writing"].append(int(pc(x).sum())); TOP["end"].append(int(pc(pile[alive]).sum()) * ge)
        TOP["plain_meet"].append(int(pc(x) @ PPH)); TOP["plain_div"].append(int(pc(pile).sum()) * ge)
        TOP["pours"].append(pours); TOP["bound"].append(bound); TOP["surv"].append(surv)
    print(f"  band read: text {t} done ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
print(f"  band parts summed vs the pile formed directly: {pile_bad} words differ")
np.savez(f"{OUT}/counter3.npz", **{k.replace("|", "__"): v for k, v in mine.items()}, **{f"TOP__{k}": np.array(v) for k, v in TOP.items()})
json.dump(dict(zg=zg), open(f"{OUT}/counter3.json", "w"))
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
