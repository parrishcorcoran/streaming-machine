"""THE CROSS-CHECK OF gates/gates.py, BY ANOTHER ROUTE.  A MEASUREMENT ONLY; NOTHING IN THE MACHINE CHANGES.

A second script (the same session; not an independent checker) rebuilds every total and every array digest of
gates/gates_<light>.json by another route:
  - lens 1: the fc pile met here in int64 whole-number arithmetic (numpy's integer product, no float container at all);
    for the real light it must equal the saved run's fc pile (recording/C_ref/plain_refs.npz).
  - H: the machine's own water level (base2's gauge_log and power: there is only one) applied to that pile, plus fc's bias.
  - the share: its own index, ((H + 8) >> 4) + 32768, its own clipping (0 below the table, 65536 above).
  - THE GLOW READ STRAIGHT FROM THE TABLE'S COMPLEMENT, never from out:
        g = -((H x (share - 65536 [H > 0]) + 2^15) >> 16)
    on the open side that is what the edge holds back, H x (65536 - share); on the dark side the leak, H x share.
    Then out = o - g. Its own to16 (the top 1 of the OR of the magnitudes, rounded shift), out16 met with mproj in int64,
    must equal the saved run's mproj pile, and the stream rebuilt with its own shift must equal the saved run's stream
    after the layer (real light).
  - lens 2: o, g and out met with mproj in int64 (every |X|.|w| bounded below 2^61 first), each group's part likewise,
    its own groups (whole-number thresholds), its own column scaling, its own photon count (a bit loop).
  - the mirrored glow g(-H) by the same complement formula.
  - the shuffles regenerated from the same seeds (they are the control's definition).
Every per-layer total is compared with gates_<light>.json key by key, and the arrays' sha256 with its digests.
    python3 gates/xchk_gates.py [light=real|shuffled|shuffled447]   -> gates/xchk_gates_<light>.log
"""
import sys, os, time, json, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; GD = f"{HERE}/gates"; os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:]); LIGHT = ARG.get("light", "real")
LOG = open(f"{GD}/xchk_gates_{LIGHT}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, PHI, gauge_log, power = J["M"], J["PHI"].astype(np.int64), J["gauge_log"], J["power"]
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); P, N, L = 648, 3072, 768
MAIN = json.load(open(f"{GD}/gates_{LIGHT}.json"))

def imeet(X, w):                                       # whole-number product in int64, bounded first
    assert float((np.abs(X).astype(np.float64) @ np.abs(w).astype(np.float64)).max(initial=0)) < 2.0 ** 61
    return np.matmul(X, w)
def pop(A):                                            # photons: a bit loop
    A = np.abs(A); n = np.zeros(A.shape, np.int64); b = 0
    while (A >> b).any(): n += (A >> b) & 1; b += 1
    return n
def rsh(X, s):                                         # X x 2^-s rounded half up; s may be <= 0 (then a left shift)
    s = np.broadcast_to(np.asarray(s, np.int64), X.shape)
    pos = np.maximum(s, 0); neg = np.maximum(-s, 0)
    half = np.where(pos > 0, np.int64(1) << np.maximum(pos - 1, 0), 0)
    return np.where(s > 0, (X + half) >> pos, X << neg)
def glow(H):                                           # straight from the complement of the table
    idx = ((H + 8) >> 4) + 32768
    sh_ = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    return -((H * (sh_ - 65536 * (H > 0)) + 32768) >> 16)
def group(H):
    g = np.full(H.shape, 4, np.int8)
    g[100 * H < 17432576] = 3; g[H < 0] = 2; g[H < -131072] = 1; g[H < -262144] = 0
    return g
def scaled(A, k):
    assert int(np.abs(A).max(initial=0)) < 1 << 53
    return sum(int(v) << int(s) for v, s in zip(np.abs(A).sum(0), k))   # per column int64 sums, then whole
def sha(A): return hashlib.sha256(np.ascontiguousarray(A, np.int64).tobytes()).hexdigest()

BAD = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
MIS = []
log(f"cross-check of gates_{LIGHT}.json by another route")
for l in MAIN["layers"]:
    R = REF[f"stream_a_{l}"].astype(np.int64)
    if LIGHT == "shuffled":
        rng = np.random.default_rng(2026 + l); R = np.stack([row[rng.permutation(L)] for row in R])
    elif LIGHT == "shuffled447":
        rng = np.random.default_rng(4470 + l); keep = np.r_[0:447, 448:L]; R2 = R.copy()
        for p in range(P): R2[p, keep] = R[p, rng.permutation(keep)]
        R = R2
    w1 = M[f"fc.{l}.w"].astype(np.int64); e1, b1 = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    w2 = M[f"mproj.{l}.w"].astype(np.int64); e2, b2 = M[f"mproj.{l}.e"], M[f"mproj.{l}.b"]
    x16, ex, LS = gauge_log(R)
    pile = imeet(x16, w1)
    if LIGHT == "real": bad("fc_pile_vs_saved_run", (pile != REF[f"fc_pile_{l}"]).sum())
    H = power(pile, (e1[None, :] + ex[:, None] + 16) * 65536 - LS[:, None]) + b1
    o = np.where(H > 0, H, 0); g = glow(H); out = o - g
    bad("g_negative", (g < 0).sum())
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(out), axis=1)])
    sh = np.maximum(top - 15, 0); out16 = rsh(out, sh[:, None])
    pile2 = imeet(out16, w2)
    mo = rsh(pile2, -(e2[None, :] + sh[:, None])) + b2
    if LIGHT == "real":
        bad("mproj_pile_vs_saved_run", (pile2 != REF[f"mproj_pile_{l}"]).sum())
        bad("stream_vs_saved_run", (R + mo != REF[f"stream_{l}"]).sum())
    WO, WG, WX = imeet(o, w2), imeet(g, w2), imeet(out, w2)
    bad("write_identity", (WO - WG != WX).sum())
    GR = group(H); k2 = (e2 - e2.min()).astype(np.int64)
    gm = glow(-H)
    o16 = rsh(o, sh[:, None]); g16 = o16 - out16; bad("g16_negative", (g16 < 0).sum())
    rp = pop(w2).sum(1)[None, :]; dark, brt = H < 0, H > 0
    r = {"edges": int(H.size), "dark": int(dark.sum()), "zero": int((H == 0).sum()), "bright": int(brt.sum()),
         "g0": int((g == 0).sum()), "g0_dark": int(((g == 0) & dark).sum()), "g0_bright": int(((g == 0) & brt).sum()),
         "sum_o": int(o.sum()), "sum_g": int(g.sum()), "sum_out": int(np.abs(out).sum()),
         "sum_g_dark": int(g[dark].sum()), "sum_g_bright": int(g[brt].sum()), "sum_out_bright": int(out[brt].sum()),
         "glow_mirror_equal": int((gm == g).sum()), "glow_mirror_maxdiff": int(np.abs(gm - g).max()),
         "we2_min": int(e2.min()), "W_open": scaled(WO, k2), "W_glow": scaled(WG, k2), "W_out": scaled(WX, k2),
         "lines": int(WX.size), "lines_glow_gt_open": int((np.abs(WG) > np.abs(WO)).sum()),
         "lines_glow_gt_out": int((np.abs(WG) > np.abs(WX)).sum()), "lines_open_gt_out": int((np.abs(WO) > np.abs(WX)).sum()),
         "lines_out_zero": int((WX == 0).sum())}
    ex_c = [int(v) for v in (WX << k2[None, :]).ravel()]
    ma_c = [int(v) << int(-e2.min()) for v in (mo - b2[None, :]).ravel()]
    r["W_machine_minus_exact"] = sum(abs(a - b) for a, b in zip(ma_c, ex_c)); del ex_c, ma_c
    r["groups"] = []
    for k in range(5):
        m = GR == k
        r["groups"].append({"edges": int(m.sum()), "g0": int((m & (g == 0)).sum()), "sum_o": int(o[m].sum()),
                            "sum_g": int(g[m].sum()), "sum_out": int(np.abs(out[m]).sum()),
                            "W_open": scaled(imeet(np.where(m, o, 0), w2), k2),
                            "W_glow": scaled(imeet(np.where(m, g, 0), w2), k2),
                            "meet_plain": int((pop(np.where(m, out16, 0)) * rp).sum())})
    r["meet_plain"] = int((pop(out16) * rp).sum()); r["meet_open"] = int((pop(o16) * rp).sum())
    r["meet_glow"] = int((pop(g16) * rp).sum()); r["meet_plain_dark"] = int((pop(np.where(dark, out16, 0)) * rp).sum())
    r["adds_join"] = P * L
    ref = MAIN["per_layer"][str(l)]
    for kk in ref:
        if ref[kk] != r.get(kk): MIS.append((l, kk, ref[kk], r.get(kk)))
    dg = {"H": sha(H), "out": sha(out), "g": sha(g), "W_open": sha(WO), "W_glow": sha(WG), "W_out": sha(WX),
          "g16": sha(g16), "sh": sha(sh)}
    nd = sum(MAIN["digests"][str(l)][kk] == dg[kk] for kk in dg)
    bad("digests_differ", len(dg) - nd)
    log(f"  layer {l:2d}: {len(ref) - sum(1 for m in MIS if m[0] == l)} of {len(ref)} totals and {nd} of {len(dg)} "
        f"digests equal ({time.time() - t0:.0f}s, {mb()} MB)")
log(); log("CHECKS (every count must be 0): " + ", ".join(f"{k} {v}" for k, v in BAD.items()))
log(f"totals that differ from gates_{LIGHT}.json: {len(MIS)}")
for m in MIS[:40]: log(f"  layer {m[0]} {m[1]}: main {m[2]}  here {m[3]}")
log(f"VERDICT: {'every total and digest rebuilt' if not MIS and all(v == 0 for v in BAD.values()) else 'DIFFERENCES'}"
    f" ({time.time() - t0:.0f}s, max memory {mb()} MB)")
