"""Independent recount (a separate checker, 2026-10-06) of the session's zero-sum counts on four 70m plates.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

This script shares no code with zerosum_count.py or zerosum_checker.py. It reads Pythia-70m's own fp16 files
(foundation0/models/pythia-70m-step0 and -step143000) with a hand-written safetensors reader (numpy only, no torch),
in Pythia's own [out][in] layout, and works in float64 (every fp16 value and every difference of two is exact there).
c (the never-built rows' surviving share) is found again here from the fp16 input word table: rows whose finished row
is a scaled copy of its blank row; c = the median slope of those rows.

Plates: mproj.3 (MLP-out, a write plate), qkv.4 (query/key/value, read), head (output word table, read),
embed (input word table, write).
  write line = one input of the write plate across the stream's 512 lanes (mproj: a COLUMN of [512][2048];
               embed: one word's row of [50304][512])
  read neuron = one output over its 512 lines (qkv: a ROW of [1536][512]; head: a ROW of [50304][512])
  z = net / sqrt(sum of squares); RMS over the plate's lines; a fair coin gives 1.

Controls (every one keeps the values):
  S1  item 1: each finished weight paired with a blank weight from elsewhere in the same plate
  S2  the change values scrambled within the plate, then the line z again
  S3  (extra) D with each finished line paired with ANOTHER line's blank (same plate): if D's line evenness is just
      the finished line's evenness plus the blank's own coin-toss leftover, S3 gives the same RMS as the real D
Order-free: lanes and lines permuted together give the same multiset of line nets.
Grid check: the npz's whole numbers x 2^e against the fp16 values (how many differ: the 16-place grid's rounding).

PREDICTIONS (written before the first run), from the session's zerosum_70m.log:
  mproj.3: 10->12 5.79% (scr 5.45), |D|<|W0| 28.4%, |D|<|W| 43.1%, med |D|/|W0| 1.95, med |D|/|W| 1.12, sign kept
           52.2%; line RMS z D 0.515, D' 0.163, W0 0.983, scrambles about 1; plate z D +0.11, D' +0.55;
           his count D: 524,493 up / 523,937 down; up 3,218,479,590 / down 3,217,609,355 grid units.
  qkv.4:   10->12 4.48% (4.14), |D|<|W0| 32.7%, |D|<|W| 34.7%, meds 1.48 / 1.25, sign kept 52.5%;
           neuron RMS z D 1.989, D' 3.189; plate z D -1.43, D' -1.57.
  head:    10->12 0.88% (0.90), |D|<|W0| 5.4%, |D|<|W| 49.4%, meds 5.74 / 1.00, sign kept 50.0%;
           neuron RMS z D 0.331, D' 0.233; cross D' 211.1; plate z D -2.43, D' -2.16;
           his count D: 12,802,620 up / 12,952,580 down (count z -29.5).
  embed:   10->12 6.40% (6.32), |D|<|W0| 34.4%, |D|<|W| 36.8%, meds 1.48 / 1.34; line RMS z D 0.693, D' 0.215,
           cross D' 14.90; plate z D +2.19, D' +2.77.
  Every share within 0.3 points, medians within 0.02, RMS z within 3% (fp16 against the 16-place grid, no rounding
  of c x W0 here). Checker's own prediction for S3: the real D slightly MORE even than S3 on the write plates
  (the finished plates keep 9-28% of the blank's shape), within 10%.
    python3 zerosum/recheck_indep_70m.py
"""
import json, struct, os, math, time, resource
import numpy as np

T0 = time.time()
MD = "/Users/abundancemachine/streaming-machine/foundation0/models"
NPZ = "/Users/abundancemachine/streaming-machine/foundation0/base2_pythia-70m.npz"
rng = np.random.default_rng(77001)
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30

class ST:
    def __init__(self, path):
        self.f = open(path, "rb"); n = struct.unpack("<Q", self.f.read(8))[0]; self.h = json.loads(self.f.read(n)); self.base = 8 + n
    def get(self, name):
        m = self.h[name]; assert m["dtype"] == "F16"; a, b = m["data_offsets"]
        self.f.seek(self.base + a); buf = self.f.read(b - a)
        return np.frombuffer(buf, dtype="<f2").reshape(m["shape"]).astype(np.float64)

S0 = ST(f"{MD}/pythia-70m-step0/model.safetensors"); S1 = ST(f"{MD}/pythia-70m-step143000/model.safetensors")
P = "gpt_neox.layers"
PLATES = [  # name, hf key, kind, line axis in [out][in]: the axis SUMMED is the lanes
    ("mproj.3", f"{P}.3.mlp.dense_4h_to_h.weight", "w"),   # [512 out lanes][2048 in]: a write line is a column
    ("qkv.4", f"{P}.4.attention.query_key_value.weight", "r"),  # [1536][512]: a read neuron is a row
    ("head", "embed_out.weight", "r"),                    # [50304][512]: a word's neuron is a row
    ("embed", "gpt_neox.embed_in.weight", "e"),           # [50304][512]: a write line is a word's row
]

def lines(X, kind):
    """(net, sum of squares) per line (write) or neuron (read), and per cross unit"""
    if kind == "w": ax, cx = 0, 1          # write line = column: sum over rows (the 512 lanes)
    else: ax, cx = 1, 0                    # read neuron / embed word row = row: sum over the 512 lanes
    return (X.sum(ax), (X * X).sum(ax)), (X.sum(cx), (X * X).sum(cx))

def zrms(net, sq):
    ok = sq > 0; z = net[ok] / np.sqrt(sq[ok])
    return float(np.sqrt(np.mean(z * z))), float((np.abs(z) < 0.5).mean()), float((np.abs(z) > 3).mean())

def finest_power(*arrs):
    o = 0
    for a in arrs:
        q = np.abs(a) * 2.0 ** 24; assert np.all(q == np.round(q)); o |= int(np.bitwise_or.reduce(q.astype(np.int64).ravel()))
    tz = (o & -o).bit_length() - 1
    return -24 + tz

# ---- c from the fp16 input word table: rows whose finished row is a scaled copy of the blank row ----
E0, E1 = S0.get("gpt_neox.embed_in.weight"), S1.get("gpt_neox.embed_in.weight")
slope = (E0 * E1).sum(1) / (E0 * E0).sum(1)
resid = np.sqrt(((E1 - slope[:, None] * E0) ** 2).sum(1) / (E1 * E1).sum(1))
copies = np.flatnonzero(resid < 0.05)
c = float(np.median(slope[copies]))
c_sess = 656960 / 2 ** 30
print(f"c: {len(copies)} rows are scaled copies of their blank (residual < 5%); median slope c = {c:.7f} "
      f"(spread {slope[copies].std():.2e}); the session's c = 656960/2^30 = {c_sess:.7f}", flush=True)
del E0, E1

zf = np.load(NPZ)
OUT = {}
for name, key, kind in PLATES:
    W0, W = S0.get(key), S1.get(key)
    # grid check against the npz (npz is [in][out] for plates, [words][hidden] for embed; one power per neuron)
    w_npz, s_npz, e_npz = zf[name + ".w"].astype(np.float64), zf[name + ".s"].astype(np.float64), zf[name + ".e"]
    if kind == "e": Wg, W0g = w_npz * 2.0 ** e_npz[:, None], s_npz * 2.0 ** e_npz[:, None]
    else: Wg, W0g = (w_npz * 2.0 ** e_npz[None, :]).T, (s_npz * 2.0 ** e_npz[None, :]).T
    grid_diff = int((Wg != W).sum() + (W0g != W0).sum()); grid_maxrel = float(np.max(np.abs(Wg - W)) / np.abs(W).max())
    pmin_npz = int(e_npz.min()); del w_npz, s_npz, Wg, W0g
    pmin_fp16 = finest_power(W, W0)
    D = W - W0
    # ---- item 1, real pairing and S1 ----
    def item1(W, W0):
        n = W.size; k = dict(within20=0, d_lt_blank=0, d_lt_fin=0, sign_kept=0)
        r0 = np.empty(n, np.float32); rW = np.empty(n, np.float32); rb = max(1, (1 << 20) // W.shape[1])
        for a in range(0, W.shape[0], rb):
            w, w0 = W[a:a + rb], W0[a:a + rb]; aD, a0, aW = np.abs(w - w0), np.abs(w0), np.abs(w)
            k["within20"] += int((5 * aD <= a0).sum()); k["d_lt_blank"] += int((aD < a0).sum()); k["d_lt_fin"] += int((aD < aW).sum())
            k["sign_kept"] += int((w * w0 > 0).sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                r0[a * W.shape[1]:a * W.shape[1] + aD.size] = (aD / a0).ravel(); rW[a * W.shape[1]:a * W.shape[1] + aD.size] = (aD / aW).ravel()
        out = {kk: v / n for kk, v in k.items()}
        out.update(med_blank=float(np.nanmedian(r0)), med_fin=float(np.nanmedian(rW)))
        return out
    i1 = item1(W, W0); W0p = rng.permutation(W0.ravel()).reshape(W0.shape); i1s = item1(W, W0p); del W0p
    # ---- item 2: lines, cross, plate; D and D' = W - c W0 (no rounding) ----
    Dp = W - c * W0
    rec = dict(item1=i1, item1_scr=i1s, grid_diff=grid_diff, grid_maxrel=grid_maxrel, pmin_npz=pmin_npz, pmin_fp16=pmin_fp16)
    for lab, X in (("D", D), ("Dp", Dp), ("W0", W0), ("W", W)):
        (un, uq), (cn, cq) = lines(X, kind)
        rec[lab] = dict(units=zrms(un, uq), cross=zrms(cn, cq)[0], plate_z=float(X.sum() / math.sqrt((X * X).sum())))
    for lab, X in (("D", D), ("Dp", Dp)):   # S2: scramble within the plate
        Y = rng.permutation(X.ravel()).reshape(X.shape); (un, uq), _ = lines(Y, kind); rec[lab]["scr"] = zrms(un, uq)[0]; del Y
    # S3: each finished line against another line's blank (lines permuted on the blank only)
    if kind == "w": W0x = W0[:, rng.permutation(W0.shape[1])]
    else: W0x = W0[rng.permutation(W0.shape[0])]
    (un, uq), _ = lines(W - W0x, kind); rec["D_otherblank"] = zrms(un, uq)[0]; del W0x
    # order-free: lanes and lines permuted together, same multiset of line nets (exact: grid integers)
    q = np.rint(D * 2.0 ** (-pmin_fp16)); assert np.array_equal(q * 2.0 ** pmin_fp16, D)
    pr, pc = rng.permutation(D.shape[0]), rng.permutation(D.shape[1])
    (u1, _), _ = lines(q, kind); (u2, _), _ = lines(q[pr][:, pc], kind)
    rec["order_free"] = bool(np.array_equal(np.sort(u1), np.sort(u2)))
    # his count, D: how many went up and down, how much (grid units at the npz plate scale and at the fp16 finest)
    pos, neg = D > 0, D < 0
    su, sd = float(D[pos].sum()), float(-D[neg].sum())
    rec["count"] = dict(n_up=int(pos.sum()), n_down=int(neg.sum()), n_zero=int((D == 0).sum()),
                        up_units_npz=su / 2.0 ** pmin_npz, down_units_npz=sd / 2.0 ** pmin_npz,
                        count_z=(int(pos.sum()) - int(neg.sum())) / math.sqrt(int(pos.sum()) + int(neg.sum())))
    OUT[name] = rec
    r = rec
    print(f"\n{name} ({kind}) {W.shape}: grid check: {grid_diff} values of W, W0 differ from the npz x 2^e (largest {grid_maxrel:.1e} of the plate's top); "
          f"plate finest power npz 2^{pmin_npz}, fp16 2^{pmin_fp16}; order-free {r['order_free']}", flush=True)
    print(f"  item 1: 10->12 {100*i1['within20']:.2f}% (scr {100*i1s['within20']:.2f}) |D|<|W0| {100*i1['d_lt_blank']:.1f}% (scr {100*i1s['d_lt_blank']:.1f}) "
          f"|D|<|W| {100*i1['d_lt_fin']:.1f}% med|D|/|W0| {i1['med_blank']:.2f} med|D|/|W| {i1['med_fin']:.2f} sign kept {100*i1['sign_kept']:.1f}% (scr {100*i1s['sign_kept']:.1f})")
    print(f"  units RMS z: D {r['D']['units'][0]:.3f} (scr {r['D']['scr']:.3f}; other line's blank {r['D_otherblank']:.3f})  "
          f"D' {r['Dp']['units'][0]:.3f} (scr {r['Dp']['scr']:.3f}; under half {100*r['Dp']['units'][1]:.0f}%, beyond 3 {100*r['Dp']['units'][2]:.2f}%)  "
          f"W {r['W']['units'][0]:.3f}  W0 {r['W0']['units'][0]:.3f}")
    print(f"  cross RMS z D' {r['Dp']['cross']:.2f}; plate z D {r['D']['plate_z']:+.2f} D' {r['Dp']['plate_z']:+.2f}")
    cc = r["count"]
    print(f"  his count D: up {cc['n_up']:,} down {cc['n_down']:,} zero {cc['n_zero']:,} (count z {cc['count_z']:+.1f}); "
          f"up {cc['up_units_npz']:,.1f} down {cc['down_units_npz']:,.1f} units at 2^{pmin_npz}  [{time.time()-T0:.0f}s {gb():.2f}GB]", flush=True)
    del W, W0, D, Dp, q
json.dump(dict(c=c, c_rows=int(len(copies)), plates=OUT), open("/Users/abundancemachine/streaming-machine/zerosum/recheck_indep_70m.json", "w"), indent=1)
print(f"\ndone {time.time()-T0:.0f}s peak {gb():.2f} GB")
