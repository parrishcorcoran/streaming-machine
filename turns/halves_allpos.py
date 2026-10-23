"""STEP 2 AT EVERY POSITION OF 10 TEXTS, in the 16-place style: the lead at each position split at each layer's newest
bends into the straight halves' part and the symmetric halves' part (see halves.py for the picture and the exact split).

For each of the first 10 refs texts (prompt + its reference continuation, 19-24 tokens), at every position t: the machine
run on the tokens up to t (its own forward, recorded: the same numbers as position t of the whole text, nothing
after t is ever read), the lead = its first word's score minus its second's at t, and a random word's lead (control).
The lead's row is sent back down every step in the 16-place style (film_lead.rev16_from's steps, line for line, with the
constants left out: only the light is needed for the split; checked against rev16_from's own light on the first 3
positions of text 0). At each layer, with Gbo the light at the newest bend outputs:
     straight = sum_n Gbo[t, n] x H[t, n] / 2,   symmetric = sum_n Gbo[t, n] x e[t, n],   e = Bout - H/2,
exact given the 16-place light; straight + symmetric == the part through the newest bends (checked). Shuffle control:
the symmetric halves' values shuffled among the layer's neurons (VALUES CHANGED, labelled).

PREDICTION (written before running; from halves.py's results on the 5-token texts): at every position as at the 5th:
the gross straight share 0.53-0.57 at every layer; the two signed parts of opposite sign in over 90% of layer-positions,
each about 2 leads in median size while their sum is about 0.3 leads; |sym| > |straight| in 40-45% (no half holds the
lead); the shuffle moves the symmetric part by about 0.8 of itself.
    python3 turns/halves_allpos.py I0 I1     -> turns/allpos_I0_I1.npz (texts I0..I1-1), log on stdout
    python3 turns/halves_allpos.py report    -> the tables over all parts
"""
import sys, time, glob; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; checked exact
OUTD = "/Users/abundancemachine/streaming-machine/turns"

def med(x): return f"{np.nanmedian(x):8.3f} [{np.nanpercentile(x, 10):7.3f},{np.nanpercentile(x, 90):7.3f}]"
if sys.argv[1:2] == ["report"]:
    rows = np.concatenate([np.load(p)["rows"] for p in sorted(glob.glob(f"{OUTD}/allpos_*.npz"))])
    # rows: text, t, layer, which (0 lead, 1 random lead), lead, straight, sym, part, gross str, gross sym, shuffled sym
    for w_, nm in ((0, "THE LEAD (first - 2nd word at each position)"), (1, "CONTROL: A RANDOM WORD'S LEAD")):
        R = rows[rows[:, 3] == w_]; npos = len(np.unique(R[:, 0] * 100 + R[:, 1]))
        print(f"\n{nm}: {npos} positions of {len(np.unique(R[:, 0]))} texts, 16-place style; per layer, median [10th, 90th]")
        print("  layer | straight / lead            | symmetric / lead           | part / lead                | gross straight share"
              "       | |str| / (|str| + |sym|)    | opposite      | both > |part|  | SHUFFLED sym / lead (values changed)")
        for l in range(12):
            x = R[R[:, 2] == l]; ld = x[:, 4]
            print(f"  {l:5d} | {med(x[:, 5] / ld)} | {med(x[:, 6] / ld)} | {med(x[:, 7] / ld)} | {med(x[:, 8] / (x[:, 8] + x[:, 9]))}"
                  f" | {med(np.abs(x[:, 5]) / (np.abs(x[:, 5]) + np.abs(x[:, 6])))} | {int((np.sign(x[:, 5]) != np.sign(x[:, 6])).sum()):4d}/{len(x)}"
                  f" | {int(((np.abs(x[:, 5]) > np.abs(x[:, 7])) & (np.abs(x[:, 6]) > np.abs(x[:, 7]))).sum()):4d}/{len(x)} | {med(x[:, 10] / ld)}")
        ld = R[:, 4]
        print(f"  all {len(R)} layer-positions: gross straight share > 1/2 at {int((R[:, 8] > R[:, 9]).sum())}; opposite signs at"
              f" {int((np.sign(R[:, 5]) != np.sign(R[:, 6])).sum())}; |sym| > |straight| at {int((np.abs(R[:, 6]) > np.abs(R[:, 5])).sum())};"
              f" median |straight| / lead {np.median(np.abs(R[:, 5]) / ld):.3f}, |sym| / lead {np.median(np.abs(R[:, 6]) / ld):.3f},"
              f" |part| / lead {np.median(np.abs(R[:, 7]) / ld):.3f}; |sym - shuffled| / |sym| median {np.median(np.abs(R[:, 6] - R[:, 10]) / np.abs(R[:, 6])):.3f}")
        for lo, hi in ((0, 5), (5, 12), (12, 30)):
            x = R[(R[:, 1] >= lo) & (R[:, 1] < hi)]
            if len(x): print(f"    positions {lo:2d}-{hi - 1:2d}: |str|/(|str|+|sym|) median {np.median(np.abs(x[:, 5]) / (np.abs(x[:, 5]) + np.abs(x[:, 6]))):.3f},"
                             f" opposite {int((np.sign(x[:, 5]) != np.sign(x[:, 6])).sum())}/{len(x)}, |part| / (|str| + |sym|) median"
                             f" {np.median(np.abs(x[:, 7]) / (np.abs(x[:, 5]) + np.abs(x[:, 6]))):.3f}")
    sys.exit()

from film_lead import *
from machine import _hcols
import resource
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1 << 20)
I0, I1 = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()

def light16(rec, Ntop, etop, cap):
    """film_lead.rev16_from's light, line for line, without the constants; cap('bo', l, Gbo=(N, e)) at every layer"""
    P, t = rec["P"], rec["t"]; tp = rec["top"]; nw = Ntop.shape[0]
    Nk, ek = keep32_obj(Ntop, etop)
    N = np.zeros((nw, P, 768), np.int64); e = np.zeros((nw, P), np.int64); N[:, t, :] = Nk; e[:, t] = ek
    e = e.copy(); e[:, t] -= tp["sh"]
    G = (N, e)
    for l in range(11, -1, -1):
        Ly = rec["layers"][l]
        mp = Ly["mproj"]; w = W[mp["name"]][0]
        Gp = dim16(*G, np.ones((P, 768), np.int64), -mp["s"])
        Gx = (plate16(Gp[0], w.T), Gp[1])
        Gx = (Gx[0], Gx[1] - mp["sh"][None, :])
        cap("bo", l, Gbo=Gx)
        bd = Ly["bend"]; Gh = dim16(*Gx, bd["share"], np.full((P, 3072), -16, np.int64))
        fc = Ly["fc"]; w = W[fc["name"]][0]
        Gq = dim16(*Gh, EXP2[fc["F"] & 0xFFFF], (fc["F"] >> 16) - 16)
        Gq = (plate16(Gq[0], w.T), Gq[1])
        Gq = (Gq[0], Gq[1] - fc["sh"][None, :])
        G = add16(G, Gq)
        pj = Ly["proj"]; w = W[pj["name"]][0]
        Gp = dim16(*G, np.ones((P, 768), np.int64), -pj["s"])
        Ga = (plate16(Gp[0], w.T), Gp[1])
        Ga = (Ga[0], Ga[1] - pj["sh"][None, :])
        ar = Ly["arr"]; Fa = _hcols(-ar["L"])
        Gn = keep(*Ga, 16); Gn = dim16(*Gn, EXP2[Fa & 0xFFFF], (Fa >> 16) - 16)
        Nn, en = Gn
        Vn = np.zeros((nw, P, 768), np.int64); ev = np.zeros((nw, P), np.int64)
        for j in range(P):
            lit = (np.abs(Nn[:, j:, :]).max(axis=2) > 0)
            ej = np.where(lit, en[:, j:], np.iinfo(np.int64).min // 4)
            T = np.maximum(np.where(lit, en[:, j:], np.iinfo(np.int64).max // 4).min(axis=1), ej.max(axis=1) - 14)
            T = np.where(lit.any(axis=1), T, 0)
            acc = np.zeros((nw, 768), np.int64)
            for h in range(12):
                cs = slice(h * 64, (h + 1) * 64)
                for p in range(j, P):
                    a = int(ar["A"][h, p, j])
                    if a: acc[:, cs] += rshift(Nn[:, p, cs], (T - en[:, p])[:, None]) * a
            Vn[:, j, :] = acc; ev[:, j] = T
        Gv = keep(Vn, ev, 16)
        q = Ly["qkv"]; w = W[q["name"]][0]
        Gr = dim16(*Gv, EXP2[q["F"][:, 1536:] & 0xFFFF], (q["F"][:, 1536:] >> 16) - 16)
        Gr = (plate16(Gr[0], np.ascontiguousarray(w[:, 1536:].T)), Gr[1])
        Gr = (Gr[0], Gr[1] - q["sh"][None, :])
        G = add16(G, Gr)
    return G

def F2(n, e): return float(n) * 2.0 ** float(e)   # for the table only (the sums themselves are exact Python whole numbers)
rows = []; ok_copy = 0; n_copy = 0; ok_sum = 0; n_sum = 0
for i in range(I0, I1):
    ids = LONG[i]
    for t in range(len(ids)):
        rec = record(ids[:t + 1]); S = rec["top"]["scores"]
        order = np.argsort(-S, kind="stable"); a, b = int(order[0]), int(order[1]); assert a == race(S)
        rg = np.random.default_rng(54321 + 100 * i + t); r = int(rg.integers(0, len(S)))
        while r in order[:5]: r = int(rg.integers(0, len(S)))
        leads = [int(S[a]) - int(S[b]), int(S[a]) - int(S[r])]
        N1, e1, c1 = top16(rec, [a, b], lead=True); N2, e2, c2 = top16(rec, [a, r], lead=True)
        Ntop, etop = np.concatenate([N1, N2]), np.concatenate([e1, e2])
        caps = {}
        def cap(stage, l, **kw): caps[(stage, l)] = kw
        light16(rec, Ntop, etop, cap)
        if i == 0 and t < 3:                           # the light equals rev16_from's own (constants and all)
            res = residues(rec); kap = kappas(rec, res); caps2 = {}
            def cap2(stage, l, **kw): caps2[(stage, l)] = kw
            rev16_from(rec, Ntop, etop, c1 + c2, kap, res, cap2)
            for l in range(12):
                n_copy += 1; ok_copy += bool((caps[("bo", l)]["Gbo"][0] == caps2[("bo", l)]["Gbo"][0]).all()
                                              and (caps[("bo", l)]["Gbo"][1] == caps2[("bo", l)]["Gbo"][1]).all())
        P = rec["P"]
        for l in range(12):
            Ly = rec["layers"][l]; H, Bout = Ly["bend"]["H"][t], Ly["bend"]["out"][t]; Y2 = 2 * Bout - H
            Nb, eb = caps[("bo", l)]["Gbo"]; prm = np.random.default_rng(1000 * i + 31 * t + l).permutation(3072)
            for w_ in range(2):
                g = Nb[w_, t].astype(object); ex = int(eb[w_, t])
                sn, yn, pn = g * H.astype(object), g * Y2.astype(object), g * Bout.astype(object)
                Ss, Ys, Ps = int(sn.sum()), int(yn.sum()), int(pn.sum()); n_sum += 1; ok_sum += (Ss + Ys == 2 * Ps)
                gs, gy = sum(abs(int(v)) for v in sn), sum(abs(int(v)) for v in yn)
                Ysh = int((g * Y2[prm].astype(object)).sum())
                rows.append([i, t, l, w_, leads[w_], F2(Ss, ex - 1), F2(Ys, ex - 1), F2(Ps, ex), F2(gs, ex - 1), F2(gy, ex - 1), F2(Ysh, ex - 1)])
    print(f"  text {i:2d} ({len(ids)} positions) done; light = rev16_from's {ok_copy}/{n_copy}; straight + sym == 2 x part {ok_sum}/{n_sum}"
          f"  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
np.savez(f"{OUTD}/allpos_{I0:02d}_{I1:02d}.npz", rows=np.array(rows))
print(f"saved {OUTD}/allpos_{I0:02d}_{I1:02d}.npz ({len(rows)} rows)  ({time.time() - t0:.0f}s, max {rss()} MB)")
