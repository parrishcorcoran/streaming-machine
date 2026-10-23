"""THE WORD TABLE OF QWEN3-0.6B (1,024 lines x 151,936 words; the last water level's gain folded in; one power of two per
word, four powers in all, -16..-13): items 1, 2, 3 and 5 of the task at the table, every position (613), exact checks.

  phase walk:    WAY 2 at the table: each line's 151,936 weights lined up across the words once; the walk and its pours
                 at every position, its piles checked equal to the direct piles. ORDER-FREE: every 8th position walked
                 DOWN with the lines in a random order. SHUFFLE CONTROLS: the gaps' lit digits with each line's words in a
                 random order (count only); the light shuffled across its lines (walked, piles checked against its own
                 direct product). Counts: plain photon meetings (the plain read), signed digits, gap meetings, pours,
                 the light written in signed digits. WEIGHT-ONLY: lit digits per weight, dark weights, way 2's gap lit
                 digits and zero gaps, way 1's (each word's 1,024 weights lined up down the lines), and the gaps
                 against words a line: each line's words drawn at random, 128 ... 151,936 (5,120 and 17,408 included).
  phase updown:  THE UP/DOWN FORM at the table (as banks.py): the straight-through share of every meeting, the piles
                 from the counts checked equal to the direct piles; CONTROL: the light shuffled across its lines.

PREDICTIONS (written before any run of this file)
  The task's: the walk at the word table nearly free.
  The session's:
  - Exact: walk piles = direct piles 613/613, the walk down 613/613 of those walked, the control's walk equal to its own
    direct product; up/down piles = direct piles 613/613.
  - Walk per pair at the table 0.02-0.15 (GPT-2's 0.283 at 50,257 words): three times the words, and each line's weights
    are 8-place numbers times one gain, so 85-97% of the gaps are zero (ties); GPT-2's table had 92.7% zero steps.
  - Pours per pair 7-10 (GPT-2 8.57); way 2 at the table 30-45% of the plain read's meetings (GPT-2 38.07%).
  - Signed digits at the table 57-62% (GPT-2's 60.75%).
  - Up/down: fringe share within 0.97-1.15 x the coin-toss value at 1,024 lines (2.493%); the control within 0.1 point.
    python3 qwen06b/table.py walk | updown
"""
import os, sys, json, time, resource, warnings, numpy as np, torch
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen06b"
sys.path.insert(0, OUT)
import importlib.util
_spec = importlib.util.spec_from_file_location("banks", f"{OUT}/banks.py"); BK = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BK)
from capture import member
ones, naf, Lineup, subsample_gaps = BK.ones, BK.naf, BK.Lineup, BK.subsample_gaps
class TableLineup(Lineup):
    """banks.Lineup's record and walk, laid 16 lines at a time (the table's lines are 151,936 words long)"""
    def __init__(self, w):
        w = np.asarray(w); L, N = w.shape; self.L, self.N = L, N
        self.NR = np.empty((L, N), np.uint32); self.G = np.zeros((L, N), np.uint16)
        self.LOW = np.empty(L, np.int64); self.TOP = np.empty(L, np.int64); self.GN = np.empty(L, np.int64); self.ZG = np.empty(L, np.int64)
        for i0 in range(0, L, 16):
            wb = w[i0:i0 + 16].astype(np.int64); o = np.argsort(wb, axis=1, kind="stable")
            ws = np.take_along_axis(wb, o, 1); g = np.diff(ws, axis=1)
            assert (g >= 0).all() and g.max(initial=0) < 1 << 16
            back = np.empty_like(wb); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
            assert np.array_equal(back, wb), "the record does not give the table back"
            self.NR[i0:i0 + 16] = o; self.G[i0:i0 + 16, 1:] = g; self.LOW[i0:i0 + 16] = ws[:, 0]; self.TOP[i0:i0 + 16] = ws[:, -1]
            self.GN[i0:i0 + 16] = naf(ws[:, 0]) + naf(g).sum(1); self.ZG[i0:i0 + 16] = (g == 0).sum(1)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
torch.set_num_threads(10)

def direct_chunks(x, T, cw=16384):
    """the direct piles of the read, whole numbers (float64, exact: below 2^47)"""
    xf = np.asarray(x, np.float64); out = np.empty((len(x), T.shape[1]), np.int64)
    for c in range(0, T.shape[1], cw):
        p = xf @ T[:, c:c + cw].astype(np.float64); assert np.abs(p).max(initial=0) < 2 ** 52; out[:, c:c + cw] = p.astype(np.int64)
    return out

def updown_torch(x, T, cw=4096, pc=96):
    """banks.updown at the table, in torch (float32 plane counts, exact: every count <= 1,024): the fringe sum per (a, b)
    and the meetings; the piles from the counts checked against the direct piles here. Returns FR, meetings, rows differing."""
    x = np.asarray(x, np.int64); P, N = x.shape; NW = T.shape[1]
    assert np.abs(x).max() < 1 << 15                                     # 16 places at the top (capture.log: largest 32,7xx)
    FR = torch.zeros((16, 16), dtype=torch.float64); meet = 0; bad = np.zeros(P, bool)
    pow2 = (2.0 ** torch.arange(16, dtype=torch.float64))[None, :]
    Xps, bases, rows_l = [], [], []
    for r0 in range(0, P, pc):
        U = torch.from_numpy(x[r0:r0 + pc] + (1 << 15)); Xp = torch.stack([((U >> a) & 1) for a in range(16)]).to(torch.float32)
        popX = Xp.sum(2); Xps.append(Xp.reshape(16 * len(U), N)); bases.append((N - 2 * popX).reshape(-1, 1)); rows_l.append((r0, r0 + len(U)))
    for c in range(0, NW, cw):
        Tc = np.asarray(T[:, c:c + cw]).astype(np.int64); n_c = Tc.shape[1]
        Uw = torch.from_numpy(Tc + (1 << 15)); Wp = torch.empty((16, N, n_c), dtype=torch.float32)
        for b in range(16): Wp[b] = (Uw >> b) & 1
        popW = Wp.sum(1)
        ws_ = torch.from_numpy(2 * Tc.sum(0) + N)
        dp = direct_chunks(x, Tc)
        for (r0, r1), Xs, base in zip(rows_l, Xps, bases):
            nr = r1 - r0; S = torch.zeros((nr, n_c), dtype=torch.float64)
            for b in range(16):
                Tm = Xs @ Wp[b]; Tm.mul_(4).add_(base).sub_(2 * popW[b][None, :])
                FR[:, b] += Tm.abs().reshape(16, nr, n_c).sum(dim=(1, 2), dtype=torch.float64)
                S += (pow2 @ Tm.to(torch.float64).reshape(16, -1)).reshape(nr, n_c) * float(1 << b)
            assert S.abs().max().item() < 2 ** 53
            V = S.to(torch.int64) - torch.from_numpy(2 * x[r0:r1].sum(1) + N)[:, None] - ws_[None, :] + N
            assert bool((V % 4 == 0).all()); bad[r0:r1] |= ((V // 4).numpy() != dp[r0:r1]).any(1)
            meet += 16 * 16 * nr * n_c
        del Wp, Uw, dp
    return FR.numpy(), meet, int(bad.sum())

if __name__ == "__main__":
    phase = sys.argv[1]; t0 = time.time()
    T = np.ascontiguousarray(member("head.w")); NW = T.shape[1]                     # (1024, 151936) int16, C order
    X = np.load(f"{OUT}/light/top_light.npy").astype(np.int64); P = len(X)
    print(f"THE WORD TABLE, phase {phase}: {T.shape} , {P} positions ({rss()} MB)", flush=True)
    rng = np.random.default_rng([2026, 99])
    xs = rng.permuted(X, axis=1); assert (np.sort(xs, 1) == np.sort(X, 1)).all()      # CONTROL: light shuffled across lines
    r = {}
    if phase == "walk":
        PPH = np.zeros(1024, np.int64); NROW = np.zeros(1024, np.int64)
        for c in range(0, NW, 8192): PPH += ones(T[:, c:c + 8192]).sum(1); NROW += naf(T[:, c:c + 8192]).sum(1)
        r.update(w_ones=int(PPH.sum()), w_naf=int(NROW.sum()), w_dark=int((T == 0).sum()), w_count=int(T.size))
        lu = TableLineup(T); r.update(w_gap_lit=int(lu.GN.sum()), w_zero_gaps=int(lu.ZG.sum()))
        print(f"  the line-up laid, read back exact ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
        GNS = np.empty(1024, np.int64)
        for i in range(1024):
            ws = T[i, rng.permutation(NW)].astype(np.int64); GNS[i] = naf(ws[0]) + naf(np.diff(ws)).sum()
        r["w_gap_lit_shuffled"] = int(GNS.sum())
        nx, ox = naf(X), ones(X); nxs, oxs = naf(xs), ones(xs)
        r.update(pairs=int(((X != 0).sum(1) * NW).sum()), plain=int((ox @ PPH).sum()), signed=int((nx @ NROW).sum()), writing=int(ox.sum()),
                 gaps=int((nx @ lu.GN).sum()), gaps_shuffled=int((nx @ GNS).sum()),
                 ctl_plain=int((oxs @ PPH).sum()), ctl_signed=int((nxs @ NROW).sum()), ctl_gaps=int((nxs @ lu.GN).sum()))
        pours = ctl_pours = 0; bad = bad_c = bad_o = 0; opn = 0; pe = True; n_of = 0
        for b0 in range(0, P, 32):
            xb = X[b0:b0 + 32]; dp = direct_chunks(xb, T)
            pw, p1, _ = lu.walk(xb); bad += int((pw != dp).any(1).sum()); pours += int(p1.sum()); del pw
            if b0 % 256 == 0:                                                        # order-free: 32 positions of every 256
                pw2, p2, o2 = lu.walk(xb, line_order=rng.permutation(1024), down=True)
                bad_o += int((pw2 != dp).any(1).sum()); opn += o2; pe &= bool(np.array_equal(p1, p2)); n_of += len(xb); del pw2
            del dp
            xsb = xs[b0:b0 + 32]; dps = direct_chunks(xsb, T)
            pws, p3, _ = lu.walk(xsb); bad_c += int((pws != dps).any(1).sum()); ctl_pours += int(p3.sum()); del pws, dps
            if b0 % 128 == 0: print(f"    walked {b0 + len(xb)} positions ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
        r.update(pours=pours, ctl_pours=ctl_pours, walk_piles_differ=bad, ctl_walk_piles_differ=bad_c,
                 orderfree_positions=n_of, orderfree_walk_piles_differ=bad_o, orderfree_open=opn, orderfree_pours_equal=pe)
        del lu
        wc_lit = wc_zero = 0
        for c in range(0, NW, 8192):                                                 # way 1: each word's weights down its lines
            wc = np.sort(T[:, c:c + 8192].astype(np.int64), 0); g1 = np.diff(wc, axis=0)
            wc_lit += int(naf(wc[0]).sum() + naf(g1).sum()); wc_zero += int((g1 == 0).sum())
        r.update(w1_gap_lit=wc_lit, w1_zero_gaps=wc_zero)
        r["subsample"] = subsample_gaps(T, [128, 256, 512, 1024, 2048, 3072, 5120, 17408, 50257, 151936], rng)
        print(f"  walk {bad} differ, down {bad_o}/{n_of}, control {bad_c}; per pair: plain {r['plain'] / r['pairs']:.2f} walk {r['gaps'] / r['pairs']:.4f} "
              f"pour {pours / r['pairs']:.2f}; way 2 {(r['gaps'] + pours + r['writing']) / r['plain'] * 100:.2f}% of the plain read; signed {r['signed'] / r['plain'] * 100:.2f}%")
    else:
        FR, meet, bad = updown_torch(X, T); print(f"  up/down real: piles differing {bad} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
        FRc, meetc, badc = updown_torch(xs, T)
        r.update(N=1024, ud_fringe_ab=FR.tolist(), ud_fringe=float(FR.sum()), ud_meetings=meet, ud_piles_differ=bad,
                 ctl_ud_fringe_ab=FRc.tolist(), ctl_ud_fringe=float(FRc.sum()), ctl_ud_meetings=meetc, ctl_ud_piles_differ=badc)
        print(f"  fringe share {FR.sum() / (1024 * meet) * 100:.4f}%  control {FRc.sum() / (1024 * meetc) * 100:.4f}%  (control piles differing {badc})")
    json.dump(r, open(f"{OUT}/table_{phase}.json", "w"))
    print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
