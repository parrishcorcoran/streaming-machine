"""Eyes level with the water: the stream as every bank reads it, line by line, write by write.

EYE LEVEL, per position and snapshot (as the bank reads it, base2.py to16 + the folded layer norm):
  centred   the position's whole-number mean across the 768 lines taken off: m = floor((S1 + 384) / 768)
  scaled    put on 16 places by one power of two for the position, exactly as to16 does: sh = max(top - 15, 0)
            where top is the bit length of the OR of the |centred| values, then rshift (add half, shift)
  per line  HEIGHT = place of its top photon in that value (0..15, -1 = dark), DIRECTION = its sign.
TOP-DOWN is the same with the raw stream (no centring, no scaling); its height moves must equal report M1.

Every count is per line, then summed over lines (no line meets another line except through the position's own sum
and OR, which do not care about order). Whole numbers only; shares are counts over counts. Run once, it does the
real file and the lines-shuffled file and checks that every count is identical.

Controls (all random draws made by line NAME with fixed seeds, so the lines-shuffled file must give the same counts):
  b   each line's 24 writes added in a random order of its own (per position, per line): same values, same start,
      same end (Abelian); its eye level is recomputed from the reordered light.
  c   E3: another random position (any text, not p and not p-1) in place of p-1.
  c2  E3 along positions: each line's positions inside its text taken in a random order of its own, eye level
      recomputed per position.
    python3 watch/eye_waves.py
"""
import json, time, numpy as np
ROOT = "/Users/abundancemachine/streaming-machine/"
FILES = {"real": "watch/traj.npz", "lines_shuffled": "watch/traj_lines_shuffled.npz"}


def pc(v): return np.bitwise_count(v).astype(np.int64)                    # photons of a value >= 0
def bitlen(v):
    v = v | (v >> 1); v = v | (v >> 2); v = v | (v >> 4); v = v | (v >> 8); v = v | (v >> 16); v = v | (v >> 32)
    return pc(v)
def height(V): return (bitlen(np.abs(V)) - 1).astype(np.int16)            # -1 = dark
def rshift(X, s):                                                          # base2.rshift for s >= 0: add half, shift
    half = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    return np.right_shift(X + half, s)
def hist(v, lo, hi): return np.bincount(np.clip(np.asarray(v).ravel(), lo, hi) - lo, minlength=hi - lo + 1).astype(np.int64)
def cnt(m): return np.int64(np.count_nonzero(m))
def add(acc, key, d):
    for n, v in d.items():
        v = np.asarray(v, dtype=np.int64); k = f"{key}.{n}"
        acc[k] = acc[k] + v if k in acc else v


def eye_one(X):                                                            # X (M, 768) raw light, one snapshot
    S1 = X.sum(1); C = X - np.floor_divide(S1 + 384, 768)[:, None]
    sh = np.maximum(bitlen(np.bitwise_or.reduce(np.abs(C), axis=1)) - 15, 0)
    return rshift(C, sh[:, None]), sh

def view(S, which):                                                        # S (L, M, 768) -> light, heights, signs, water
    V = np.empty_like(S); sh = np.empty(S.shape[:2], np.int64)
    for t in range(len(S)):
        if which == "eye": V[t], sh[t] = eye_one(S[t])
        else: V[t] = S[t]; sh[t] = np.maximum(bitlen(np.bitwise_or.reduce(np.abs(S[t]), axis=1)) - 15, 0)
    h = np.stack([height(V[t]) for t in range(len(V))])
    return V, h, np.sign(V).astype(np.int8), sh


def steps(A, B, hA, hB, refA):
    """A -> B, rows = groups (positions or pairs), columns = lines. Thinness, height moves, flips, swells, restoring."""
    D = B - A; dh = (hB - hA).astype(np.int64); sA, sB = np.sign(A), np.sign(B)
    up, dn = (dh > 0).sum(1).astype(np.int64), (dh < 0).sum(1).astype(np.int64); mov = up + dn
    rel = np.where(hA < 0, -1, np.clip(hA.astype(np.int64) - refA[:, None], -1, 15))   # height above the water
    ch = D != 0; lit = hA >= 0
    return {"values": A.size, "ph_light": pc(np.abs(A)).sum(), "ph_change": pc(np.abs(D)).sum(), "ph_after": pc(np.abs(B)).sum(),
            "unchanged": cnt(~ch), "move": hist(dh, -8, 8),
            "dark_before": cnt(hA < 0), "dark_after": cnt(hB < 0),
            "lit_both": cnt((sA != 0) & (sB != 0)), "flips": cnt(sA * sB < 0),
            "gap": hist((height(D) - hA)[ch & lit], -24, 8),
            "groups": len(A), "agree": (up * (up - 1) // 2 + dn * (dn - 1) // 2).sum(), "disagree": (up * dn).sum(),
            "swell34": cnt((mov > 0) & (4 * np.maximum(up, dn) >= 3 * mov)), "rise16": hist((16 * up // np.maximum(mov, 1))[mov > 0], 0, 16),
            "restore": np.bincount(((rel + 1) * 3 + np.sign(dh) + 1).ravel(), minlength=17 * 3).astype(np.int64)}


def pairs(tag, m):                                                         # m: (steps, ...) moves in {-1, 0, +1}
    a, b = m[:-1], m[1:]; p = a * b
    last = np.zeros(m.shape[1:], np.int8); r2 = c2 = np.int64(0)
    for k in range(len(m)):
        cur = m[k]; both = (cur != 0) & (last != 0)
        r2 += cnt(both & (cur != last)); c2 += cnt(both & (cur == last)); last = np.where(cur != 0, cur, last)
    return {f"{tag}_rev": cnt(p < 0), f"{tag}_cont": cnt(p > 0), f"{tag}_move_stay": cnt((a != 0) & (b == 0)),
            f"{tag}_stay_move": cnt((a == 0) & (b != 0)), f"{tag}_stay_stay": cnt((a == 0) & (b == 0)),
            f"{tag}_rev_skip": r2, f"{tag}_cont_skip": c2, f"{tag}_moves": cnt(m != 0), f"{tag}_steps": m.size}

def bob(h, sg, V):                                                         # sequences along axis 0
    out = pairs("h", np.sign(np.diff(h, axis=0)).astype(np.int8))
    out.update(pairs("v", np.sign(np.diff(V, axis=0)).astype(np.int8)))
    f = sg[:-1] * sg[1:]; fa, fb = f[:-1], f[1:]                           # -1 flip, +1 keep, 0 dark on either side
    out.update({"d_flip_flip": cnt((fa < 0) & (fb < 0)), "d_flip_keep": cnt((fa < 0) & (fb > 0)),
                "d_keep_flip": cnt((fa > 0) & (fb < 0)), "d_keep_keep": cnt((fa > 0) & (fb > 0)),
                "d_flips_per_line": hist((f < 0).sum(0), 0, 12)})
    return out


def run(which):
    t0 = time.time()
    z = np.load(ROOT + FILES[which]); nt = len([k for k in z.files if k.startswith("R")])
    Xs = [z[f"R{i}"].astype(np.int64) for i in range(nt)]
    names = z["lines"] if "lines" in z.files else np.arange(768)            # column j holds line names[j]
    S = np.concatenate(Xs, axis=1); lens = [x.shape[1] for x in Xs]; offs = np.cumsum([0] + lens[:-1]); del Xs
    pos = np.concatenate([np.arange(n) for n in lens]); N = S.shape[1]

    rng = np.random.default_rng(20261004)                                  # draws by line name / position, fixed order
    order = rng.permuted(np.tile(np.arange(24, dtype=np.int64)[:, None, None], (1, N, 768)), axis=0)[:, :, names]
    tgt = np.flatnonzero(pos >= 1); src = tgt - 1
    q = rng.integers(0, N, len(tgt))
    while True:
        bad = (q == tgt) | (q == src)
        if not bad.any(): break
        q[bad] = rng.integers(0, N, int(bad.sum()))
    perms = [rng.permuted(np.tile(np.arange(n)[:, None], (1, 768)), axis=0)[:, names] for n in lens]

    res = {}
    Wr = S[1:] - S[:-1]
    Sb = np.empty_like(S); Sb[0] = S[0]
    for t in range(24): Sb[t + 1] = Sb[t] + np.take_along_axis(Wr, order[t][None], axis=0)[0]
    res["abelian.b_end_differs"] = cnt(Sb[24] != S[24]); res["abelian.b_end_values"] = np.int64(S[24].size)
    del Wr, order

    for vw in ("eye", "top"):
        for var, X in (("real", S), ("b", Sb)):
            V, h, sg, sh = view(X, vw); key = f"w.{vw}.{var}"; rf = sh if vw == "top" else 0 * sh   # eye heights are on the water already
            for t in range(24):
                d = steps(V[t], V[t + 1], h[t], h[t + 1], rf[t]); add(res, key, d); add(res, f"{key}.t{t:02d}", d)
                still = sh[t] == sh[t + 1]
                add(res, key + ".still", steps(V[t][still], V[t + 1][still], h[t][still], h[t + 1][still], rf[t][still]))
                add(res, key + ".water", {"rise": hist(sh[t + 1] - sh[t], -4, 8)})
                if vw == "eye":
                    add(res, f"{key}.t{t:02d}.water", {"moved": cnt(~still)})
            add(res, key + ".bob", bob(h, sg, V))
            res[key + ".heights"] = hist(h, -1, 31)
            if var == "real":                                              # E3: the surface between pulses
                for t in range(25):
                    for pv, sv in (("real", src), ("c", q)):
                        d = steps(V[t][sv], V[t][tgt], h[t][sv], h[t][tgt], rf[t][sv]); add(res, f"p.{vw}.{pv}", d); add(res, f"p.{vw}.{pv}.t{t:02d}", d)
                for o, n in zip(offs, lens):
                    tr = lambda a: a[:, o:o + n].transpose(1, 0, 2)
                    add(res, f"p.{vw}.real.bob", bob(tr(h), tr(sg), tr(V)))
            del V, h, sg
            print(f"  {which}: writes {vw} {var} done, {time.time() - t0:.0f}s", flush=True)
        for i, (o, n) in enumerate(zip(offs, lens)):                       # c2: each line's positions in its own order
            Sp = np.take_along_axis(S[:, o:o + n], perms[i][None], axis=1)
            V, h, sg, sh = view(Sp, vw); rf = sh if vw == "top" else 0 * sh
            for t in range(25):
                add(res, f"p.{vw}.c2", steps(V[t][:-1], V[t][1:], h[t][:-1], h[t][1:], rf[t][:-1]))
            add(res, f"p.{vw}.c2.bob", bob(h.transpose(1, 0, 2), sg.transpose(1, 0, 2), V.transpose(1, 0, 2)))
        print(f"  {which}: positions {vw} c2 done, {time.time() - t0:.0f}s", flush=True)
    del S, Sb
    return res


R = run("real"); Q = run("lines_shuffled")
diff = [k for k in R if not np.array_equal(R[k], Q.get(k))]
J = {k: (v.tolist() if v.ndim else int(v)) for k, v in R.items()}
json.dump(J, open(ROOT + "watch/eye_results.json", "w"))
json.dump({"keys": len(R), "identical": len(R) - len(diff), "differ": diff}, open(ROOT + "watch/eye_check.json", "w"))
print(f"order-free check: {len(R) - len(diff)} of {len(R)} counts identical between traj.npz and traj_lines_shuffled.npz")
