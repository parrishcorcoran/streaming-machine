"""THE TIP COUNT, on the GPU (torch, MPS), whole numbers only. Shared by every script in tips_blank/.

The up/down form of banks.py (qwen06b/banks.py `updown`) and the flip (FINDINGS section 2), unchanged: every place of
every value is a photon up (+1) or a hole (-1); offset 2^(n-1), D = 2v + 1, n = 16 places (17 for a light row holding a
value at exactly 2^15). For every meeting (position p, neuron j, light place a, plate place b) over the N lines:
    d = lines where light place a and plate place b disagree,   tip T = N - 2d  (the up meetings minus the down meetings),
    paired off = N - |T|  (the meetings that pair off: the non-tips).
The tip identity (FINDINGS 1.2 / 32): pile = 1/4 [ sum over (a, b) of 2^(a+b) T - token stack - neuron stack + N ]
(token stack 2 sum x + N, neuron stack 2 sum w + N). It is checked at every position of every plate: the pile from the
tips must equal the direct pile, digit for digit. The plate is written in nb = 16 places (the machine's own form); a
plate whose values need more (the tiny models' pushes-only plates reach 16 places of size) is written in 17 (nb=17).
The identity holds for any number of places, since sum over places of 2^a (up or hole) = 2v + 1 whatever n is.

Returned per plate (all whole numbers):
  FR[a, b]  sum over (p, j) of |T|                  -- the tips' size (the non-tips are N x CNT - FR)
  CNT[a, b] the number of (p, j) meetings at (a, b)
  SG[a, b]  sum over (p, j) of T                    -- the tips' net
  DS[k]     sum over (p, j) of |sum over a+b=k of T| -- the size of diagonal k's signed part of the pile (units 2^k)
  FRW[a, b] (only with rw, colw) sum over (p, j) of rw_p colw_j |T|: the tips' size on the real scale, where rw_p and
            colw_j are the position's and the neuron's powers of two over the smallest (whole numbers)
  bad       positions whose pile from the tips differs from the direct pile (must be 0)
Exactness of the arithmetic: every plane product is a sum of N terms of +-1, |T| <= N <= 3072 < 2^24, so float32 holds it
exactly; a row's sum of |T| over one neuron chunk stays below 2^22 (asserted); every larger sum and every pile is int64.
"""
import numpy as np, torch

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def _planes(v, n):
    """(n, R, C) float32 on the device: place a of U = v + 2^(n-1), written as an up (+1) or a hole (-1)"""
    U = torch.from_numpy(np.ascontiguousarray(v + (1 << (n - 1)))).to(DEV)
    assert int(U.min()) >= 0 and int(U.max()) < (1 << n)
    return torch.stack([((U >> a) & 1) for a in range(n)]).to(torch.float32).mul_(2).sub_(1)

def tipcount(x, w, nb=16, pc=128, cw=None, check=True, rw=None, colw=None):
    x = np.asarray(x, np.int64)                                         # the plate stays in its own type; each chunk -> int64
    P, N = x.shape; N2, M = w.shape; assert N == N2 and N <= 3072
    assert np.abs(x).max(initial=0) <= 1 << 15
    if cw is None: cw = max(256, min(M, (1 << 21) // N))
    assert cw * N <= 1 << 22                                             # a row's sum of |T| over a chunk stays below 2^24
    big = np.abs(x).max(1) >= 1 << 15                                   # rows needing a 17th light place
    groups = [(np.flatnonzero(~big)[i:i + pc], 16) for i in range(0, int((~big).sum()), pc)]
    groups += [(np.flatnonzero(big)[i:i + pc], 17) for i in range(0, int(big.sum()), pc)]
    K = 17 + nb - 1
    FR = torch.zeros((17, nb), dtype=torch.int64, device=DEV); SG = torch.zeros((17, nb), dtype=torch.int64, device=DEV)
    FRW = torch.zeros((17, nb), dtype=torch.int64, device=DEV)
    DS = torch.zeros(K, dtype=torch.int64, device=DEV); CNT = np.zeros((17, nb), np.int64); bad = np.zeros(P, bool)
    LP = [(rows, n, _planes(x[rows], n)) for rows, n in groups]
    weighted = rw is not None
    if weighted:
        rw = np.asarray(rw, np.int64); colw = np.asarray(colw, np.int64); assert rw.min() >= 1 and colw.min() >= 1
        assert float(rw.sum()) * float(colw.max()) * N < 2 ** 62
        RW = [torch.from_numpy(rw[rows]).to(DEV) for rows, _, _ in LP]
    ts = 2 * x.sum(1) + N                                                # the token stack
    shifts = torch.arange(K, device=DEV, dtype=torch.int64)
    for c0 in range(0, M, cw):
        wc = np.asarray(w[:, c0:c0 + cw]).astype(np.int64); m = wc.shape[1]
        assert np.abs(wc).max(initial=0) < 1 << (nb - 1)
        Wp = _planes(wc, nb)                                             # (nb, N, m), +-1
        ws = 2 * wc.sum(0) + N                                           # the neuron stack
        if weighted: CW = torch.from_numpy(colw[c0:c0 + cw]).to(DEV)
        if check: dp = np.asarray(x, np.float64) @ np.asarray(wc, np.float64); assert np.abs(dp).max(initial=0) < 2 ** 52
        for gi, (rows, n, Xp) in enumerate(LP):
            r = len(rows); Xs = Xp.reshape(n * r, N)
            D = torch.zeros((K, r, m), dtype=torch.float32, device=DEV)   # |sum of a diagonal's tips| <= 17 N < 2^24: exact
            for b in range(nb):
                T = (Xs @ Wp[b]).reshape(n, r, m)                        # ups minus downs: the tip T = N - 2d, exact
                A = T.abs()
                FR[:n, b] += A.sum(2).to(torch.int64).sum(1)             # per row below 2^22 (asserted above): exact
                SG[:n, b] += T.sum(2).to(torch.int64).sum(1)
                if weighted: FRW[:n, b] += ((A.to(torch.int64) * CW).sum(2) * RW[gi]).sum(1)
                D[b:b + n] += T
                CNT[:n, b] += r * m
            Di = D.to(torch.int64); del D
            DS += Di.abs().sum(dim=(1, 2))
            if check:
                S = (Di << shifts[:, None, None]).sum(0).cpu().numpy()
                V = S - ts[rows][:, None] - ws[None, :] + N
                bad[rows] |= (V % 4 != 0).any(1) | ((V // 4) != dp[rows].astype(np.int64)).any(1)
            del Di
        del Wp
    if DEV.type == "mps": torch.mps.synchronize()
    out = dict(FR=FR.cpu().numpy(), CNT=CNT, SG=SG.cpu().numpy(), DS=DS.cpu().numpy(), N=N, P=P, M=M, nb=nb,
               bad=int(bad.sum()) if check else None)
    if weighted:
        MRW = [int(sum(int(rw[rows].sum()) for rows, n, _ in LP if a < n)) for a in range(17)]   # light weight with place a
        out.update(FRW=FRW.cpu().numpy(), MRW=MRW, CWSUM=int(colw.sum()))
    return out

def diag(R):
    """per diagonal k = a + b: the tips' size, the meetings, the paired-off share, and the signed part's size"""
    FR, CNT, N = np.asarray(R["FR"]), np.asarray(R["CNT"]), R["N"]
    na, nb = FR.shape; K = na + nb - 1; tip = np.zeros(K); meet = np.zeros(K)
    for a in range(na):
        for b in range(nb): tip[a + b] += FR[a, b]; meet[a + b] += N * CNT[a, b]
    used = meet > 0
    out = dict(tip=tip, meet=meet, paired=np.where(used, 1 - tip / np.where(used, meet, 1), np.nan),
               ds=np.asarray(R["DS"], float), used=used)
    if "FRW" in R:
        FRW = np.asarray(R["FRW"], float); tw = np.zeros(K); mw = np.zeros(K)
        for a in range(na):
            for b in range(nb):
                tw[a + b] += FRW[a, b]
                mw[a + b] += N * float(R["MRW"][a]) * float(R["CWSUM"])
        out.update(tip_real=tw, meet_real=mw)                            # in units of 2^(k + smallest powers)
    return out

def weighted_paired(R, top=None):
    """the paired-off share over all diagonals with 2^k weights (top: only the top `top` used diagonals)"""
    d = diag(R); ks = np.flatnonzero(d["used"])
    if top is not None: ks = ks[-top:]
    w = np.exp2(ks.astype(float))
    return 1 - float((w * d["tip"][ks]).sum() / (w * d["meet"][ks]).sum())

def coin(N):
    """the coin-toss tip size per meeting as a share of N, exact: E|N - 2d| / N for d ~ Binomial(N, 1/2)"""
    from math import comb
    if N % 2 == 0: return comb(N, N // 2) / 2 ** N
    return comb(N - 1, (N - 1) // 2) / 2 ** (N - 1)

def shuffled_light(x, rng):
    """each position's light lines in their own random order (same values, the pairing with the plate broken)"""
    xs = rng.permuted(np.asarray(x), axis=1); assert (np.sort(xs, 1) == np.sort(np.asarray(x), 1)).all(); return xs

def tojson(R): return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
