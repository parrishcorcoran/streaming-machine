"""THE LONGER LOOK: every position of the first 10 refs texts' ids+ref (20-24 tokens). At every read position t the answer
is the machine's own word there (the race over its whole-number scores, the text's own tokens going in).

At each t: word w's row sent back down every step with every dimming of this run held (reverse16, 16 places):
  - identity: E.R_w + c_w against the machine's score, for the top 5 and 20 random words; top 5 equal?
  - the same word read one position earlier (t-1): how alike R_w is (same sign share, photons of the difference /
    photons of R, over positions 0..t-1); control: a random word read at t-1.
  - the answer against a random word, same read: how much of the pattern belongs to the text and not to the word.
  - across texts: A's first t+1 words run through B's dimmings at its first t+1 positions (B = the next of the 10
    texts, every dimming held from B, the machine's own arithmetic): where A's answer at t falls among 50,257 words.
    Coincidence control: A's answer in B's own scores at t.
  - the fringes of the final meeting (E met with R_answer, up/down form) against two shuffles and the coin toss.

PREDICTIONS (written before running):
  his:    predetermined; the same word's reversed pattern carries over, and grows more settled as the text grows.
  Claude: the same text's pattern is shared between consecutive reads (about 3/4 of signs, as at 5 tokens) whatever
          the word; between texts it stays at coin toss at every length, and the answer through another text's
          dimmings stays near the coincidence rate at every length.
    python3 reverse/longer.py           -> reverse/longer.npz, report (reverse/longer.log)
"""
import sys, time, math, bisect; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; all checked exact
from machine import *
from fractions import Fraction

def retop(rec, t):
    R = rec["R12"]; x16, ex, LS = gauge_log(R[t:t + 1]); pile = product(x16, HEADW)
    F = (HEADE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]; scores = power(pile, F)[0] + HEADB
    r = dict(rec); r["t"] = t; r["top"] = dict(x16=x16[0], sh=int(ex[0] + ES), LS=LS, Rt=R[t], scores=scores); return r

def topres(r): return dict(x=res_shift(r["top"]["x16"], r["top"]["Rt"], r["top"]["sh"]))

def held_slice(rec, t):
    """B's dimmings at its first t+1 positions (the arrival never looks ahead, so they are B's run cut at t)"""
    L = []
    for Ly in rec["layers"]:
        L.append(dict(qkv=dict(LS=Ly["qkv"]["LS"][:t + 1]), fc=dict(LS=Ly["fc"]["LS"][:t + 1]),
                      bend=dict(share=Ly["bend"]["share"][:t + 1]),
                      arr=dict(A=Ly["arr"]["A"][:, :t + 1, :t + 1], L=Ly["arr"]["L"][:, :t + 1])))
    return dict(layers=L, top=dict(LS=retop(rec, t)["top"]["LS"]))

def photons(A): return int(np.bitwise_count(np.abs(A).astype(np.uint64)).sum())
def alike(Nx, ex, Ny, ey, pos):
    same = tot = px = pd = 0
    for p in pos:
        sx, sy = np.sign(Nx[p]), np.sign(Ny[p]); same += int((sx == sy).sum()); tot += sx.size
        T = max(int(ex[p]) + int(bitlen(np.bitwise_or.reduce(np.abs(Nx[p])))), int(ey[p]) + int(bitlen(np.bitwise_or.reduce(np.abs(Ny[p]))))) - 15
        x16 = rshift(Nx[p], T - int(ex[p])); y16 = rshift(Ny[p], T - int(ey[p]))
        px += photons(x16); pd += photons(x16 - y16)
    return same / tot, pd / max(px, 1)

def bits(V, n):
    U = V.astype(np.int64) + (1 << (n - 1)); assert U.min() >= 0 and U.max() < (1 << n)
    return ((U[None, :] >> np.arange(n)[:, None]) & 1).astype(np.float64)
def fringe(Ep, Rp):
    nE = int(np.abs(Ep).max()).bit_length() + 1; nR = int(np.abs(Rp).max()).bit_length() + 1
    bE, bR = bits(Ep, nE), bits(Rp, nR); N = Ep.size
    d = (bE @ (1 - bR).T + (1 - bE) @ bR.T).astype(np.int64)
    S = sum((N - 2 * int(d[a, b])) << (a + b) for a in range(nE) for b in range(nR))
    ok = (S - 2 * int(Ep.sum()) - 2 * int(Rp.sum()) - N) == 4 * int((Ep.astype(object) * Rp.astype(object)).sum())
    return d, ok
COIN = Fraction(sum(math.comb(768, d) * abs(768 - 2 * d) for d in range(769)), (1 << 768) * 768)
def rank_of(V, i): return int((V > V[i]).sum())

NTX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
t0 = time.time(); rng0 = np.random.default_rng(4102026)
rows = []
recs = {}
def get(i):
    if i not in recs:
        ids = LONG[i]; rec = record(ids); res = residues(rec); kap = kappas(rec, res)
        S = machine_scores(ids); assert (rec["top"]["scores"] == S).all() and race(S) == forward(ids)
        tm = len(ids) // 2; assert (retop(rec, tm)["top"]["scores"] == machine_scores(ids, at=tm)).all()
        recs[i] = (rec, res, kap)
        for k in list(recs):
            if k not in (i, (i + 1) % NTX, 0): del recs[k]
    return recs[i]

for i in range(NTX):
    ids = LONG[i]; P = len(ids); B = (i + 1) % NTX
    rec, res, kap = get(i); recB = get(B)[0]; PB = recB["P"]
    prev = None
    for t in range(1, P):
        rt = retop(rec, t); St = rt["top"]["scores"]; rs_t = (res[0], topres(rt)); kp_t = (kap[0], rs_t[1])
        wt = race(St); top5 = [int(x) for x in np.argsort(-St, kind="stable")[:5]]; assert top5[0] == wt
        rnd = []
        while len(rnd) < 20:
            x = int(rng0.integers(50257))
            if x not in top5 and x not in rnd: rnd.append(x)
        SET = top5 + rnd; u = rnd[0]
        (N, e), c = reverse16(rt, SET, kp_t, rs_t)
        m = meetE(rec["R0"], N, e); rev = [frac(a + b) for a, b in zip(m, c)]
        gaps = [abs(float(r - int(St[w]))) for r, w in zip(rev, SET)]
        o5 = [SET[k] for k in sorted(range(len(SET)), key=lambda k: -rev[k])[:5]]
        row = dict(text=i, t=t, gap_max=max(gaps), gap_med=float(np.median(gaps)), same_order=o5 == top5, same_set=set(o5) == set(top5),
                   ans_vs_u=alike(N[0], e[0], N[5], e[5], range(t + 1)))
        r1 = retop(rec, t - 1); rs1 = (res[0], topres(r1)); kp1 = (kap[0], rs1[1])
        (N1, e1), _ = reverse16(r1, [wt, u], kp1, rs1)
        row["same_word_prev"] = alike(N[0], e[0], N1[0], e1[0], range(t))
        row["ctrl_prev"] = alike(N[0], e[0], N1[1], e1[1], range(t))
        if prev is not None: row["ans_vs_prev_ans"] = alike(N[0], e[0], prev[0], prev[1], range(t))
        prev = (N[0], e[0])
        if PB > t:
            fz = record(ids[:t + 1], held=held_slice(recB, t))["top"]["scores"]
            SBt = retop(recB, t)["top"]["scores"]
            row["rank_frozen"] = rank_of(fz, wt); row["rank_coinc"] = rank_of(SBt, wt)
            row["rank_frozen_2to5"] = [rank_of(fz, w) for w in top5[1:]]
        fr = [0, 0]; fs = [0, 0]; cr = [0, 0]; cs = [0, 0]; ok = True; rngs = np.random.default_rng(1000 * i + t)
        for p in range(t + 1):
            R16 = rshift(N[0, p], int(max(int(bitlen(np.bitwise_or.reduce(np.abs(N[0, p])))) - 15, 0)))
            for key, Rp in (("real", R16), ("shuf", rngs.permutation(R16))):
                d, okp = fringe(rec["R0"][p], Rp); ok &= okp; f = np.abs(768 - 2 * d)
                tgt, ctg = (fr, cr) if key == "real" else (fs, cs)
                tgt[0] += int(f.sum()); tgt[1] += f.size * 768; ctg[0] += int(f[-2:, -2:].sum()); ctg[1] += 4 * 768
        row.update(fr_real=fr[0] / fr[1], fr_shuf=fs[0] / fs[1], corner_real=cr[0] / cr[1], corner_shuf=cs[0] / cs[1], fr_ok=ok)
        rows.append(row)
    print(f"  text {i} ({P} tokens) {''.join(word(x) for x in ids[:8])!r}...  done ({time.time() - t0:.0f}s)", flush=True)

np.savez("/Users/abundancemachine/streaming-machine/reverse/longer.npz", rows=np.array(rows, dtype=object))
pc = lambda x: f"{100 * x:.1f}%"
print(f"\nLONGER LOOK: {len(rows)} read positions over {NTX} texts")
print(f"  every fringe meeting rebuilt exactly from its counts: {all(r['fr_ok'] for r in rows)}")
print(f"  identity at 16 places: largest gap per read median {np.median([r['gap_max'] for r in rows]):.0f} units, max {max(r['gap_max'] for r in rows):.0f};"
      f" top 5 equal: same order {sum(r['same_order'] for r in rows)}/{len(rows)}, same set {sum(r['same_set'] for r in rows)}/{len(rows)}")
bins = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 23)]
print("  by read position t:      reads | same word, read t vs t-1      | control: random word at t-1  | answer vs random word, same read"
      " | answer at t vs answer at t-1 | A's words, B's dimmings: answer top 5, median rank | coincidence: top 5, median rank"
      " | fringe real / shuffle | top corner real / shuffle")
for lo, hi in bins:
    rr = [r for r in rows if lo <= r["t"] <= hi]
    if not rr: continue
    f = lambda k, j: np.mean([r[k][j] for r in rr if k in r])
    fz = [r["rank_frozen"] for r in rr if "rank_frozen" in r]; cz = [r["rank_coinc"] for r in rr if "rank_coinc" in r]
    print(f"   t {lo:2d}-{hi:2d}            {len(rr):4d}  | sign {pc(f('same_word_prev', 0))} photons {pc(f('same_word_prev', 1))} "
          f"| sign {pc(f('ctrl_prev', 0))} photons {pc(f('ctrl_prev', 1))} | sign {pc(f('ans_vs_u', 0))} photons {pc(f('ans_vs_u', 1))} "
          f"| sign {pc(f('ans_vs_prev_ans', 0)) if any('ans_vs_prev_ans' in r for r in rr) else '-'} "
          f"| {sum(1 for x in fz if x < 5)}/{len(fz)}, {np.median(fz) if fz else float('nan'):.0f} | {sum(1 for x in cz if x < 5)}/{len(cz)}, "
          f"{np.median(cz) if cz else float('nan'):.0f} | {pc(np.mean([r['fr_real'] for r in rr]))} / {pc(np.mean([r['fr_shuf'] for r in rr]))} "
          f"| {pc(np.mean([r['corner_real'] for r in rr]))} / {pc(np.mean([r['corner_shuf'] for r in rr]))}")
f25 = [x for r in rows if "rank_frozen_2to5" in r for x in r["rank_frozen_2to5"]]
print(f"  frequency-matched control: A's own 2nd..5th words through B's dimmings in the top 5: {pc(np.mean([x < 5 for x in f25]))}")
print(f"  coin-toss fringe share: {pc(float(COIN))}   ({time.time() - t0:.0f}s)")
