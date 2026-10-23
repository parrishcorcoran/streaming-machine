"""INDEPENDENT CHECK OF knife/settle.py ON 3 TEXTS (the first three refs texts, ids + ref, 62 positions).

1. base2.py's own forward (every bank, the machine unchanged) on the 3 texts: the stream entering each fc.l and each
   fc pile must equal recording/C_ref/plain_refs.npz, which settle.py read its light from.
2. The bounds rebuilt with other arithmetic: the counted part as one product of the masked light with the plate (light
   schedule), or of each light place with the plate's places at or above the stage's diagonal (diagonal schedule),
   never from pieces; the plain and signed bounds pair by pair, never from tables. At every stage the exact pile and H
   lie inside.
3. SIDE and SHARE: the first settled stage of every edge of the 3 texts (12 layers x 2 schedules x 2 bounds), rebuilt,
   against knife/stages_real.npz.
4. OUTPUT: on sampled edges per layer, schedule and bound (half among those settle.py says settled before the full
   pile), at the stage settle.py says it settled, every H in the interval and every sh in settle.py's range give one
   out16 (base2's edge evaluated at every H in the table, straight above it, dark below it), and it is the machine's
   own; at the stage before, they do not give one. settle.py's sh range holds the machine's sh at every stage.
5. The work rule rebuilt for these 3 texts by direct place masks: meetings up to each edge's output stage.
    python3 knife/check3.py   -> knife/check3.log
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
LOG = open(f"{HERE}/knife/check3.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES, PHI = J["M"], J["W"], J["ES"], J["PHI"]
rshift, power, edge, to16, gauge_log, gauged_bank, product = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "gauged_bank", "product"))
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
TEXTS = [r["ids"] + r["ref"] for r in json.load(open(f"{HERE}/refs_many.json"))[:3]]
P3 = sum(len(t) for t in TEXTS); assert P3 == int(REF["npos"][:3].sum())
N, PL, PW, ND = 3072, 15, 15, 29
BAD = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)

# ---- 1. the machine's own forward ----
RIN = {l: [] for l in range(12)}
for ids in TEXTS:
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + J["plain_bank"](J["attention"](gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        RIN[l].append(R.copy())
        R = R + J["plain_bank"](edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
for k in [k for k in W if not k.startswith("fc.")]: del W[k]
for l in range(12):
    RIN[l] = np.concatenate(RIN[l])
    bad("stream_vs_saved_run", (RIN[l] != REF[f"stream_a_{l}"][:P3]).sum())
log(f"1. base2's own forward on 3 texts ({P3} positions): the stream entering fc.l differs from the saved run at "
    f"{BAD['stream_vs_saved_run']} values over 12 layers ({int(time.time() - t0)}s)")

ST = np.load(f"{HERE}/knife/stages_real.npz"); SR = np.load(f"{HERE}/knife/settle_real.npz"); SHR = SR["shr"]
assert list(SR["layers"]) == list(range(12))
HA, HB = -(1 << 19) - 64, (1 << 19) + 64
GRID = np.arange(HA, HB + 1, dtype=np.int64); OUTG = edge(GRID)                     # base2's edge at every H
probe = np.int64(1) << np.arange(20, 46)
assert (edge(-probe) == 0).all() and (edge(np.arange(HA - 5000, HA)) == 0).all()      # dark below the table
assert np.array_equal(edge(probe), probe) and np.array_equal(edge(np.arange(HB + 1, HB + 5000)), np.arange(HB + 1, HB + 5000))
def values(lo, hi, s):                                 # min and max of out16 over every H in [lo, hi]
    vs = []
    if lo < HA: vs += [0, 0]
    if hi >= HA and lo <= HB:
        r = rshift(OUTG[max(lo, HA) - HA: min(hi, HB) - HA + 1], s); vs += [int(r.min()), int(r.max())]
    if hi > HB:
        vs += [int(rshift(np.int64(max(lo, HB + 1)), s)), int(rshift(np.int64(hi), s))]
    return min(vs), max(vs)
def brute(lo, hi, slo, shi):                           # one out16 for every H and every sh?
    mn, mx = zip(*[values(int(lo), int(hi), s) for s in range(int(slo), int(shi) + 1)])
    return min(mn) == max(mx), min(mn)
def share_entry(H):
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))

SCHED = [("light", [2, 4, 6, 8, 10, 12, 15]), ("diag", list(range(2, 29, 2)) + [29])]
rng = np.random.default_rng(31)
WORK3 = {}; FULL3 = 0; NCHK = 0
for l in range(12):
    R = RIN[l]; x16, ex, LS = gauge_log(R); w16 = M[f"fc.{l}.w"]; we, bias = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    Hx = gauged_bank(R, f"fc.{l}"); o16, eo = to16(edge(Hx), -ES); shx = eo + ES
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    pile = product(x16, W[f"fc.{l}"][0])
    ax, sx = np.abs(x16), np.sign(x16); aw = np.abs(w16.astype(np.int64)); sw = np.sign(w16).astype(np.int64)
    nA = [((ax >> a) & 1).sum(1) for a in range(PL)]; nP = [(((ax >> a) & 1) * (sx > 0)).sum(1) for a in range(PL)]
    nM = [(((ax >> a) & 1) * (sx < 0)).sum(1) for a in range(PL)]
    mA = [((aw >> b) & 1).sum(0) for b in range(PW)]; mP = [(((aw >> b) & 1) * (sw > 0)).sum(0) for b in range(PW)]
    mM = [(((aw >> b) & 1) * (sw < 0)).sum(0) for b in range(PW)]
    popw = np.bitwise_count(aw).astype(np.float64)
    for si, (sname, ks) in enumerate(SCHED):
        NS = len(ks)
        mine = {(bi, ci): np.full((P3, N), -1, np.int8) for bi in range(2) for ci in range(2)}
        saved_out = [ST[f"L{l}_{sname}_{b}_output"][:P3] for b in ("plain", "signed")]
        samp = []
        for bi in range(2):
            early = np.flatnonzero(saved_out[bi].ravel() < NS - 1); allx = np.arange(P3 * N)
            samp.append(np.concatenate([rng.choice(early, min(50, len(early)), replace=False) if len(early) else np.array([], int),
                                        rng.choice(allx, 50, replace=False)]))
        IV = {}                                          # (bi, s) -> sampled intervals
        meet = []
        for s, k in enumerate(ks):
            c = np.zeros((P3, N), np.int64); mt = np.zeros((P3, N), np.int64)
            if sname == "light":
                a0 = PL - k; xm = sx * ((ax >> a0) << a0)
                c = (xm.astype(np.float64) @ w16.astype(np.float64)).astype(np.int64)
                mt = (np.bitwise_count((ax >> a0) << a0).astype(np.float64) @ popw).astype(np.int64)
                U = [(a, b) for a in range(PL) for b in range(PW) if a < a0]
            else:
                d0 = ND - k
                for a in range(PL):
                    sh_ = max(d0 - a, 0)
                    if sh_ >= PW: continue
                    wm = (aw >> sh_) << sh_; xa = (sx * ((ax >> a) & 1)).astype(np.float64)
                    c += (xa @ (sw * wm).astype(np.float64)).astype(np.int64) << a
                    mt += (((ax >> a) & 1).astype(np.float64) @ np.bitwise_count(wm).astype(np.float64)).astype(np.int64)
                U = [(a, b) for a in range(PL) for b in range(PW) if a + b < d0]
            meet.append(mt)
            Rb = np.zeros((P3, N), np.int64); up = np.zeros((P3, N), np.int64); dn = np.zeros((P3, N), np.int64)
            for a, b in U:                               # pair by pair
                Rb += np.minimum(nA[a][:, None], mA[b][None, :]) << (a + b)
                up += (np.minimum(nP[a][:, None], mP[b][None, :]) + np.minimum(nM[a][:, None], mM[b][None, :])) << (a + b)
                dn += (np.minimum(nP[a][:, None], mM[b][None, :]) + np.minimum(nM[a][:, None], mP[b][None, :])) << (a + b)
            for bi, (lo, hi) in enumerate(((c - Rb, c + Rb), (c - dn, c + up))):
                bad("pile_outside", ((pile < lo) | (pile > hi)).sum())
                Hlo, Hhi = power(lo, F) + bias, power(hi, F) + bias
                bad("H_outside", ((Hx < Hlo) | (Hx > Hhi)).sum())
                m = mine[(bi, 0)]; m[(m < 0) & (np.sign(Hlo) == np.sign(Hhi))] = s
                m = mine[(bi, 1)]; m[(m < 0) & (share_entry(Hlo) == share_entry(Hhi))] = s
                IV[(bi, s)] = (Hlo.ravel()[samp[bi]], Hhi.ravel()[samp[bi]])
                sr = SHR[l, si, bi, s, :P3].astype(np.int64)
                bad("sh_outside_saved_range", ((shx < sr[:, 0]) | (shx > sr[:, 1])).sum())
        for bi, bn in enumerate(("plain", "signed")):
            for ci, cn in enumerate(("side", "share")):
                bad(f"{cn}_stage_differs", (mine[(bi, ci)] != ST[f"L{l}_{sname}_{bn}_{cn}"][:P3]).sum())
            so = saved_out[bi].ravel()[samp[bi]].astype(np.int64)
            for q, e in enumerate(samp[bi]):
                p = e // N; s = int(so[q])
                lo, hi = IV[(bi, s)][0][q], IV[(bi, s)][1][q]; r = SHR[l, si, bi, s, p]
                ok, v = brute(lo, hi, r[0], r[1]); NCHK += 1
                bad("output_not_settled_at_saved_stage", not ok); bad("output_value_wrong", ok and v != int(o16.ravel()[e]))
                if s > 0:
                    lo, hi = IV[(bi, s - 1)][0][q], IV[(bi, s - 1)][1][q]; r = SHR[l, si, bi, s - 1, p]
                    bad("output_settled_before_saved_stage", brute(lo, hi, r[0], r[1])[0])
            so_all = saved_out[bi].astype(np.int64)
            w3 = sum(int(meet[s][so_all == s].sum()) for s in range(NS))
            WORK3[(sname, bn)] = WORK3.get((sname, bn), 0) + w3
        full = int(meet[-1].sum())
        if si == 0: FULL3 += full
    log(f"  layer {l} done ({int(time.time() - t0)}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB); "
        + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))

log(f"\n2-4. every stage, both schedules, both bounds, 12 layers, {P3} positions x {N} edges:")
for k, v in sorted(BAD.items()): log(f"    {k}: {v}")
log(f"    output checked by brute force on {NCHK} sampled edges (at the saved stage, and the stage before)")
log(f"5. the work rule on these 3 texts (meetings up to each edge's output stage, against {FULL3:,}):")
for (sname, bn), v in sorted(WORK3.items()):
    q = (1000 * v + FULL3 // 2) // FULL3; log(f"    {sname} schedule, {bn} bound: {v:,} = {q // 10}.{q % 10}%")
log(f"ALL CHECKS PASS: {sum(BAD.values()) == 0}   ({int(time.time() - t0)}s, max memory "
    f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
