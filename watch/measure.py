"""How the 1s of the stream move, measured line by line and then summed (any line order gives the same sums).

Every count is built from each line's own trajectory (its value as it enters, then after each of the 24 writes) and
added up over lines, positions and writes. Whole numbers only: places, photons (1s of |value|, sign apart), shifts,
compares. Run on one trajectory file; it saves every count to watch/results_<which>.json.
    python3 watch/measure.py real              (watch/traj.npz, lines in the machine's own order)
    python3 watch/measure.py lines_shuffled    (watch/traj_lines_shuffled.npz, the machine run in a random line order)

Measurements (each for the real pairing and for the controls):
 M1  each line's top-photon place, before and after a write: how many places it moves (up +, down -).
 M2  the coin pusher (and how often the write points the same way as the stream it lands on, sign against sign): places that change when the write lands (|before| XOR |after|), places changed where the write
     has no photon (carry and borrow runs), how far above the write's top photon the change reaches, sign flips;
     the water level (the next read keeps 16 places from the top place any line reaches at that position) and the
     photons under it before and after the write; how far the water rises.
 M3  is the write the line's own value moved, +-x x 2^k (any k), plus a tip with 8 |tip| < |write|?
     At most one k can satisfy that for a given line (two shifts a factor 2 apart cannot both be within 1/8), so
     there is no winner to pick: the relation holds at its one k or it does not. Tip photons against write photons.
 M4  one power-of-two gain +-2^k shared by all 768 lines of a position (per write), tried for every k and both
     signs, and "no gain" (the tip is then the whole write): fewest tip photons. Also one gain per line, the same
     for every position; and one gain per value. THIS IS A SEARCH (it keeps the fewest photons over 127 choices);
     its controls get the very same search, so the search's own reach is in the control.
 M5  M1, M3, M4 (and changed places) between the same snapshot's light at consecutive positions (pulse p-1 -> p).

Controls (all random draws are made by line NAME with fixed seeds, so they too come out the same in any line order):
 b1  another token: each position's write meets the stream of another position (a random derangement of the 648
     positions, whole row, every line keeps its own name, so every line's values keep their distribution).
 b2  another line: each line's write meets the stream of another line at the same position (a random cycle over
     line names, a new one for every position).
 c   write order: each line's 24 writes added in a random order of their own (Abelian: same start, same end).
"""
import sys, json, time, numpy as np
ROOT = "/Users/abundancemachine/streaming-machine/"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
FILE = {"real": "watch/traj.npz", "lines_shuffled": "watch/traj_lines_shuffled.npz"}[WHICH]
z = np.load(ROOT + FILE)
nt = len([k for k in z.files if k.startswith("R")])
Xs = [z[f"R{i}"].astype(np.int64) for i in range(nt)]
names = z["lines"] if "lines" in z.files else np.arange(768)            # column j holds line names[j]
inv = np.argsort(names)                                                    # line name -> column
S = np.concatenate(Xs, axis=1)                                             # (25 snapshots, 648 positions, 768)
pos = np.concatenate([np.arange(x.shape[1]) for x in Xs]); N = S.shape[1]
Wr = S[1:] - S[:-1]                                                        # the 24 writes
K = np.arange(-31, 32)

# ---- random draws, by line name, fixed seeds (independent of the file's line order) ----
rng = np.random.default_rng(20261004)
def derange(n):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any(): return p
def name_cycles(rows):                                                     # a random cycle over line names, per row
    o = rng.permuted(np.tile(np.arange(768), (rows, 1)), axis=1)
    part = np.empty_like(o); part[np.arange(rows)[:, None], o] = np.roll(o, -1, axis=1); return part
pi1 = derange(N)                                                           # b1: another position's stream
sig = name_cycles(N)                                                       # b2: partner line name, per position
order = rng.permuted(np.tile(np.arange(24, dtype=np.int8)[:, None, None], (1, N, 768)), axis=0)   # c: write order, by name
tgt = np.flatnonzero(pos >= 1); src = tgt - 1                              # M5 pulse pairs (p-1 -> p) inside a text
pi5 = derange(len(tgt)); sig5 = name_cycles(len(tgt))
col_b2 = inv[sig[:, names]]                                                # column of the partner line, per (row, column)
col_b5 = inv[sig5[:, names]]
ord_c = order[:, :, names].astype(np.int64)

def pc(v): return np.bitwise_count(v).astype(np.int64)                    # photons of a value >= 0
def bitlen(v):
    v = v | (v >> 1); v = v | (v >> 2); v = v | (v >> 4); v = v | (v >> 8); v = v | (v >> 16); v = v | (v >> 32)
    return pc(v)
def top(v): return bitlen(np.abs(v)) - 1                                   # -1 when no photon
def shift(m, k): return (m << k) if k >= 0 else (m >> -k)
def hist(v, lo, hi):                                                       # counts for lo..hi, tails folded into ends
    return np.bincount(np.clip(v, lo, hi) - lo, minlength=hi - lo + 1).astype(np.int64)
def water(B):                                                              # per position: lowest kept place (to16)
    return np.maximum(bitlen(np.bitwise_or.reduce(np.abs(B), axis=1)) - 15, 0)

def m1(A, B):
    d = top(B) - top(A)
    return {"move": hist(d.ravel(), -8, 8)}

def m2(A, W):
    B = A + W; aA, aB, aW = np.abs(A), np.abs(B), np.abs(W); ch = aA ^ aB
    shA, shB = water(A), water(B); mA = (np.int64(1) << shA)[:, None] - 1; mB = (np.int64(1) << shB)[:, None] - 1
    both = (ch > 0) & (aW > 0)
    return {"changed": pc(ch).sum(), "own": pc(aW).sum(), "beyond": pc(ch & ~aW).sum(),
            "beyond_hist": hist(pc(ch & ~aW).ravel(), 0, 12), "cancelled": pc(aW & ~ch).sum(),
            "reach": hist((top(ch) - top(aW))[both], -8, 8), "flips": int(np.count_nonzero(((A > 0) & (B < 0)) | ((A < 0) & (B > 0)))),
            "values": A.size, "write_zero": int(np.count_nonzero(W == 0)),
            "phA": pc(aA).sum(), "phA_under": pc(aA & mA).sum(), "phB": pc(aB).sum(), "phB_under": pc(aB & mB).sum(),
            "write_under": pc(aW & mB).sum(), "changed_under": pc(ch & mB).sum(), "rise": hist(shB - shA, -4, 8),
            "gap": hist((top(A) - top(W))[aW > 0], -8, 24),
            "same_sign": int(np.count_nonzero((A * np.sign(W)) > 0)), "both_lit": int(np.count_nonzero((A != 0) & (W != 0)))}

def m3(X, Y):
    aX, aY = np.abs(X), np.abs(Y); valid = aY > 0
    nh = np.zeros(aY.shape, np.int64); tip = np.zeros(aY.shape, np.int64); kk = np.zeros(aY.shape, np.int64)
    for k in K:
        dif = np.abs(aY - shift(aX, int(k))); h = (dif * 8 < aY) & valid
        nh += h; tip = np.where(h, dif, tip); kk = np.where(h, k, kk)
    h = nh > 0; pY = pc(aY)
    return {"valid": int(valid.sum()), "holds": int(h.sum()), "multi": int((nh > 1).sum()),
            "tip_ph_held": pc(tip[h]).sum(), "y_ph_held": pY[h].sum(), "y_ph": pY.sum(),
            "k": hist(kk[h], -16, 16)}

def m4(X, Y):
    aX, sX = np.abs(X), np.sign(X); py = pc(np.abs(Y))
    el, row, line = py.copy(), py.sum(1), py.sum(0); row0, line0 = row.copy(), line.copy()
    for k in K:
        v = sX * shift(aX, int(k))
        for s in (1, -1):
            t = pc(np.abs(Y - s * v)); el = np.minimum(el, t); row = np.minimum(row, t.sum(1)); line = np.minimum(line, t.sum(0))
    return {"y_ph": py.sum(), "row": row.sum(), "line": line.sum(), "el": el.sum(),
            "row_nogain": int((row == row0).sum()), "rows": len(row), "line_nogain": int((line == line0).sum()), "lines": len(line)}

def add(acc, key, d):
    for n, v in d.items():
        v = np.asarray(v, dtype=np.int64); k = f"{key}.{n}"
        acc[k] = acc[k] + v if k in acc else v

res = {}
t0 = time.time()
# writes: real pairing, b1, b2, c   (per write t and in total)
At = S[0].copy(); Ct = S[0].copy()
for t in range(24):
    A, W = S[t], Wr[t]
    var = {"real": A, "b1": A[pi1], "b2": np.take_along_axis(A, col_b2, axis=1)}
    Wc = np.take_along_axis(Wr, ord_c[t][None], axis=0)[0]                  # c: this step's write, by its own order
    for v, Av in var.items():
        for key, f in (("m1", lambda: m1(Av, Av + W)), ("m2", lambda: m2(Av, W)), ("m3", lambda: m3(Av, W)), ("m4", lambda: m4(Av, W))):
            d = f(); add(res, f"w.{v}.{key}", d); add(res, f"w.{v}.t{t:02d}.{key}", d)
    for key, f in (("m1", lambda: m1(Ct, Ct + Wc)), ("m2", lambda: m2(Ct, Wc)), ("m3", lambda: m3(Ct, Wc)), ("m4", lambda: m4(Ct, Wc))):
        add(res, f"w.c.{key}", f())
    Ct = Ct + Wc
    print(f"  write {t:2d} done, {time.time() - t0:.0f}s", flush=True)
res["abelian.c_end_differs"] = np.int64(np.count_nonzero(Ct != S[24]))   # same writes, own order: same end?
res["abelian.c_end_values"] = np.int64(Ct.size)

# pulse to pulse: snapshot t at position p-1 -> p
for t in range(25):
    X, Y = S[t][src], S[t][tgt]
    var = {"real": X, "b1": X[pi5], "b2": np.take_along_axis(X, col_b5, axis=1)}
    for v, Xv in var.items():
        d1 = m1(Xv, Y); d1["changed"] = pc(np.abs(Xv) ^ np.abs(Y)).sum(); d1["y_ph"] = pc(np.abs(Y)).sum(); d1["diff_ph"] = pc(np.abs(Y - Xv)).sum()
        d1["same_sign"] = int(np.count_nonzero((Xv * np.sign(Y)) > 0)); d1["both_lit"] = int(np.count_nonzero((Xv != 0) & (Y != 0)))
        for key, d in (("m1", d1), ("m3", m3(Xv, Y)), ("m4", m4(Xv, Y))):
            add(res, f"p.{v}.{key}", d); add(res, f"p.{v}.t{t:02d}.{key}", d)
    print(f"  snapshot {t:2d} done, {time.time() - t0:.0f}s", flush=True)

json.dump({k: (v.tolist() if v.ndim else int(v)) for k, v in res.items()}, open(ROOT + f"watch/results_{WHICH}.json", "w"))
print(f"{WHICH}: {len(res)} counts saved, {time.time() - t0:.0f}s")
