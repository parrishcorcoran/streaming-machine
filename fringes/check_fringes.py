"""CHECKER (own code). The fringe map from the lights the plain machine formed at each bank (30 refs texts x 5 positions,
saved by check_flip.py), n = 16 places, offset 2^15, every place a photon (+1) or a hole (-1).
For each meeting (position x neuron x light place a x plate place b): 2 tip = sum over lines of D_a D_b = N - 2 d.
Real, and two controls: each position's light lines in their own random order (own seeds), plate untouched.
Also the order-free check on two banks.
    python3 fringes/check_fringes.py [bank names ...]     (default: all 48 banks and the top read)
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
OUT = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
L = np.load(f"{OUT}/lights.npz"); M = np.load("base2_gpt2.npz")
names = sys.argv[1:] or [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
n, off = 16, 1 << 15
SEEDS = (424242, 777)

def pm_planes(u):                                    # (16*P, N) float32 +1/-1, row a*P + p
    return (2.0 * ((u[None] >> np.arange(n)[:, None, None]) & 1) - 1).astype(np.float32).reshape(n * u.shape[0], -1)

def tips(X, w):                                      # histogram of |tip| per (a, b) and sum |tip| per (a, b)
    P, N = X.shape; H = N // 2 + 1
    assert np.abs(X).max() < off and np.abs(w.astype(np.int64)).max() <= off - 1 + 1
    Xp = pm_planes(X.astype(np.int64) + off)
    uw = (w.astype(np.int32) + off)
    assert uw.min() >= 0 and uw.max() < 2 ** n
    hist = np.zeros((n, n, H), np.int64); s = np.zeros((n, n), np.int64)
    for b in range(n):
        Wb = (2 * ((uw >> b) & 1) - 1).astype(np.float32)
        T = (Xp @ Wb).reshape(n, P, -1)              # N - 2d, exact (|.| <= 3072)
        A = (np.abs(T).astype(np.int64)) // 2        # |tip|
        assert ((np.abs(T).astype(np.int64)) % 2 == 0).all()
        hist[:, b] = np.bincount((A + np.arange(n)[:, None, None] * H).ravel(), minlength=n * H).reshape(n, H)
        s[:, b] = A.reshape(n, -1).sum(1)
    return hist, s

def pct(h, q):                                       # smallest v with at least q of the mass at or below it
    c = np.cumsum(h, -1); return (c >= q * c[..., -1:]).argmax(-1)

res = {}; t0 = time.time()
for nm in names:
    X = L[nm]; w = M[nm + ".w"]; P, N = X.shape; K = w.shape[1]
    hr, sr = tips(X, w)
    ctl = [tips(np.random.default_rng(sd).permuted(X, axis=1), w) for sd in SEEDS]
    cnt = hr[0, 0].sum()
    med = pct(hr, .5); medc = [pct(h, .5) for h, _ in ctl]
    allr, allc = hr.sum((0, 1)), ctl[0][0].sum((0, 1))
    share = lambda s: float(2 * s.sum() / (n * n * cnt * N))
    above1 = med >= medc[0] + 1; aboveB = (med >= medc[0] + 1) & (med >= medc[1] + 1)
    belowB = (med <= medc[0] - 1) & (med <= medc[1] - 1)
    pw = (2.0 ** (np.arange(n)[:, None] + np.arange(n)[None, :]))
    mag = lambda s, g: float((pw * s)[g].sum() / (pw * s).sum())
    la, pb = np.nonzero(aboveB)
    r = dict(N=int(N), K=int(K), P=int(P), meetings=int(n * n * cnt),
             all_real=[int(pct(allr, q)) for q in (.5, .9, .99)], all_ctl=[int(pct(allc, q)) for q in (.5, .9, .99)],
             share_real=round(100 * share(sr), 3), share_c1=round(100 * share(ctl[0][1]), 3), share_c2=round(100 * share(ctl[1][1]), 3),
             ctl_typical=int(np.median(medc[0])), ctl_range=[int(medc[0].min()), int(medc[0].max())],
             ctl_pairs_disagree=int((medc[0] != medc[1]).sum()),
             above_c1=int(above1.sum()), above_both=int(aboveB.sum()), below_both=int(belowB.sum()),
             above_light=[int(la.min()), int(la.max())] if len(la) else None, above_plate=[int(pb.min()), int(pb.max())] if len(pb) else None,
             top=dict(L15=[int(v) for v in med[15, ::-1][:6]], L15c=[int(v) for v in medc[0][15, ::-1][:6]], L14P15=int(med[14, 15]), L14P15c=int(medc[0][14, 15])),
             mean_ratio_1515=round(float(sr[15, 15] / ctl[0][1][15, 15]), 3),
             mean_ratio_elsewhere=[round(float(v), 3) for v in np.percentile((sr / ctl[0][1])[~(above1 | belowB)], [1, 50, 99])] if (~(above1 | belowB)).any() else None,
             mag_above_real=round(100 * mag(sr, above1), 2), mag_above_c1=round(100 * mag(ctl[0][1], above1), 2),
             med_real=med.tolist(), med_c1=medc[0].tolist(), med_c2=medc[1].tolist())
    res[nm] = r
    print(f"{nm:9s} N={N} |tip| med/p90/p99 real {r['all_real']} ctl {r['all_ctl']}  share {r['share_real']:.2f}% "
          f"(ctl {r['share_c1']:.2f}/{r['share_c2']:.2f})  above c1 {r['above_c1']} both {r['above_both']} below both {r['below_both']} "
          f"light {r['above_light']} plate {r['above_plate']}  ctl {r['ctl_typical']} {r['ctl_range']} ctl-ctl differ {r['ctl_pairs_disagree']}  "
          f"L15xP15.. {r['top']['L15']} vs {r['top']['L15c']}  ratio1515 {r['mean_ratio_1515']}  mag {r['mag_above_real']} ({r['mag_above_c1']})  "
          f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(f"{OUT}/check_fringes.json", "w"))

# order-free: the same random order of the lines on both sides
for nm in ("fc.6", "mproj.3"):
    if nm not in L.files: continue
    X = L[nm].astype(np.int64); w = M[nm + ".w"].astype(np.int64); N = X.shape[1]; P, K = X.shape[0], w.shape[1]
    o = np.random.default_rng(99).permutation(N)
    def bits(V): return (((V[None] + off) >> np.arange(n)[:, None, None]) & 1).astype(np.float32)   # (16, rows, cols)
    Xs = [bits(X), bits(X[:, o])]; Ws = [bits(w), bits(w[o])]
    ch_both = ch_light = 0; Sx = [np.zeros((P, K), np.int64) for _ in range(2)]
    for b in range(n):
        d = []
        for xi, wi in ((0, 0), (1, 1), (1, 0)):
            O = (Xs[xi].reshape(-1, N) @ Ws[wi][b]).reshape(n, P, K)
            d.append((Xs[xi].sum(-1, dtype=np.float64)[:, :, None] + Ws[wi][b].sum(0, dtype=np.float64)[None, None, :] - 2 * O).astype(np.int64))
        ch_both += int((d[1] != d[0]).sum()); ch_light += int((d[2] != d[0]).sum())
        for k in range(2): Sx[k] += sum(((N - 2 * d[k][a]) << (a + b)) for a in range(n))
    pile = [(S - (2 * X.sum(1) + N)[:, None] - (2 * w.sum(0) + N)[None, :] + N) // 4 for S in Sx]
    p0 = (X.astype(np.float64) @ w.astype(np.float64)).astype(np.int64); tot = n * n * P * K
    print(f"order-free {nm}: meetings {tot:,}, counts changed with both reordered {ch_both}, piles changed {int((pile[1] != pile[0]).sum())}, "
          f"piles != base2 {int((pile[1] != p0).sum())}; light only reordered: counts changed {ch_light:,} ({100 * ch_light / tot:.1f}%)", flush=True)
print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB, {time.time() - t0:.0f}s")
