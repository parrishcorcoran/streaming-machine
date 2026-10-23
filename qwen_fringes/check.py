"""A SEPARATE RECOUNT of qwen_fringes.py's counts (no code shared with it or with qwen06b/: its own products, its own
signed-digit count, its own line-up, its own set-aside rule). Written by the same session after the runs, so it is a
second code path, not an independent checker in the full sense.

  A. PLATES: 14 plates (2 layers drawn at random for each of the 7 kinds) x 24 positions drawn at random. From capture's
     light (qwen06b/light) and the stored plate, every product r = x w on every line and neuron is formed directly (no
     walk): the pours ones(|r|), the pours in signed digits (the digits counted by the carry loop below, not by the
     3v trick), the plain meetings, the gap meetings and the 3x walk's meetings from a line-up made here (np.sort), and
     the light's writing. Each must equal qwen_fringes/machine_counts.npz at that position.
  B. THE BAND READ at 12 positions drawn at random: every exposure formed directly, chunk by chunk; per word and band the
     band's part and photons; then topband's rule written out over ALL words still in (each word's high against the
     5th-best low, every word's bound at its own shift), no per-class shortcut. The words still in after every band, the
     photons poured into them, the walk and the plain meetings, and the top 5 must equal qwen_fringes/top_machine.npz.
    python3 qwen_fringes/check.py
"""
import json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen_fringes"; CAP = f"{HERE}/qwen06b/light"
t0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def pop(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)
def sd_count(v):
    """lit digits of the non-adjacent signed-digit form, by the carry loop: while a: if odd, take digit d = 2 - (a mod 4)
    (+1 or -1), a -= d; a >>= 1"""
    a = np.abs(np.asarray(v, np.int64)).copy(); n = np.zeros(a.shape, np.int64)
    while a.any():
        odd = (a & 1) == 1; d = np.where(odd, 2 - (a & 3), 0); n += odd; a = (a - d) >> 1
    return n
M = np.load(f"{HERE}/qwen/qwen_base2.npz"); C = np.load(f"{OUT}/machine_counts.npz"); T5 = np.load(f"{OUT}/top_machine.npz")
rng = np.random.default_rng(31); POS = np.sort(rng.choice(613, 24, replace=False))
KS = [("q", "qkv", 0, 2048), ("k", "qkv", 2048, 3072), ("v", "qkv", 3072, 4096), ("o", "o", 0, 1024),
      ("gate", "gu", 0, 3072), ("up", "gu", 3072, 6144), ("down", "down", 0, 1024)]
diff = []; n_cmp = 0
print(f"A. PLATES at positions {POS.tolist()}", flush=True)
for kind, slab, c0, c1 in KS:
    for l in np.sort(rng.choice(28, 2, replace=False)):
        light = np.load(f"{CAP}/{slab}_light.npy", mmap_mode="r"); X = np.asarray(light[l][POS]).astype(np.int64)
        w = M[f"{slab}.{l}.w"][:, c0:c1].astype(np.int64)
        ws = np.sort(w, axis=1); glit = sd_count(ws[:, 0]) + sd_count(np.diff(ws, axis=1)).sum(1)
        for j, p in enumerate(POS):
            x = X[j]; r = x[:, None] * w
            mine = dict(pours=int(pop(r).sum()), pours_sd=int(sd_count(r).sum()), plain=int(pop(x) @ pop(w).sum(1)),
                        gaps=int(sd_count(x) @ glit), walk3=int(sd_count(3 * x) @ glit))
            for k, v in mine.items():
                n_cmp += 1
                if int(C[f"{kind}.{l}__{k}"][p]) != v: diff.append((kind, int(l), int(p), k, v, int(C[f"{kind}.{l}__{k}"][p])))
            if kind in ("q", "o", "gate", "down"):
                n_cmp += 1
                if int(C[f"{slab}.{l}__writing"][p]) != int(pop(x).sum()): diff.append((slab, int(l), int(p), "writing"))
        print(f"  {kind}.{l} done ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
print(f"  plate counts compared: {n_cmp}; differing: {len(diff)} {diff[:5]}")

print("B. THE BAND READ", flush=True)
TL = np.load(f"{CAP}/top_light.npy").astype(np.int64); TE = np.load(f"{CAP}/top_ex.npy"); TLS = np.load(f"{CAP}/top_LS.npy")
Z = np.load(f"{OUT}/machine_top_light.npz"); assert np.array_equal(Z["x16"].astype(np.int64), TL) and np.array_equal(Z["ex"], TE) and np.array_equal(Z["LS"], TLS)
T = np.asarray(M["head.w"]); HE = M["head.e"].astype(np.int64); KW = HE - HE.min(); NW = T.shape[1]; EXP2 = M["exp2"]
PB = [26, 22, 18, 14, 10, 6, 2, 0]; MASK = [15] * 7 + [3]
glit_t = np.zeros(T.shape[0], np.int64); pph = np.zeros(T.shape[0], np.int64)
for i0 in range(0, T.shape[0], 32):                # line by line chunks (under 3 GB)
    Ts = np.sort(T[i0:i0 + 32].astype(np.int64), axis=1)
    glit_t[i0:i0 + 32] = sd_count(Ts[:, 0]) + sd_count(np.diff(Ts, axis=1)).sum(1); pph[i0:i0 + 32] = pop(T[i0:i0 + 32]).sum(1); del Ts
def scale(A, F):                                   # the read's last step: A x e, shifted by s with the rounding half
    e = int(EXP2[F & 0xFFFF]); s = 16 - (F >> 16); y = A * e
    return (y + (1 << (s - 1))) >> s if s > 0 else y << (-s)
bad = []; PP = np.sort(rng.choice(613, 12, replace=False))
for p in PP:
    x = TL[p]; n = int((x != 0).sum()); F = int((HE.min() + TE[p] + 16) * 65536 - TLS[p])
    e = int(EXP2[F & 0xFFFF]); s = 16 - (F >> 16); delta = max(1, -(-(1 << s) // e)) if s >= 0 else 1; mu = 2 + delta
    DV = np.zeros((NW, 8), np.int64); HP = np.zeros((NW, 8), np.int64)
    for c in range(0, NW, 8192):
        r = x[:, None] * T[:, c:c + 8192].astype(np.int64); a = np.abs(r); sg = np.sign(r)
        for b in range(8):
            nib = (a >> PB[b]) & MASK[b]; DV[c:c + 8192, b] = (sg * nib).sum(0); HP[c:c + 8192, b] = pop(nib).sum(0)
    pile = (DV << np.array(PB)[None, :]).sum(1); A = pile << KW
    alive = np.ones(NW, bool); V = np.zeros(NW, np.int64); surv = []; pours = []
    for b in range(8):
        pours.append(int(HP[alive, b].sum())); V[alive] += DV[alive, b] << PB[b] << KW[alive]
        if b == 7: break
        Rw = (n * ((1 << PB[b]) - 1)) << KW; low = np.where(alive, V - Rw, np.iinfo(np.int64).min)
        L5 = np.sort(low[alive])[-5]; alive &= ~(V + Rw <= L5 - mu); surv.append(int(alive.sum()))
    sc = scale(V[alive], F); words = np.flatnonzero(alive); order = np.lexsort((words, -sc))[:5]; t5 = words[order]
    mine = dict(surv=surv, pours=pours, walk=int(sd_count(x) @ glit_t), plain_meet=int(pop(x) @ pph), top5=t5.tolist())
    theirs = dict(surv=T5["surv"][p].tolist(), pours=T5["pours"][p].tolist(), walk=int(T5["walk"][p]), plain_meet=int(T5["plain_meet"][p]), top5=T5["top5"][p].tolist())
    d = [k for k in mine if mine[k] != theirs[k]]; bad += [(int(p), k) for k in d]
    print(f"  position {p:3d}: words still in {surv}; pours {sum(pours):,}; {'equal' if not d else 'DIFFER ' + str(d)} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
print(f"  band read positions compared: {len(PP)} (5 items each); differing: {len(bad)} {bad}")
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
