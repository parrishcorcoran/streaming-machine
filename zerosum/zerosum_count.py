"""His sentence, counted on Pythia's plates (14m, 70m, 160m).  2026-10-06.

HIS WORDS (2026-10-06), the only words quoted as his:
  "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2 /
   A 10 that becomes a 8 is -2"

WHAT IS COUNTED (the session's set-up, not his words)
  The change of each weight from its blank start, on the shared whole-number grid of
  ~/streaming-machine/foundation0/base2_pythia-<size>.npz (start .s and finished .w are whole numbers on one power
  of two per neuron, so the change is exact):
    D  = W - W0                          his "10 becomes 12 is 2", the whole blank taken away
    D' = W - round(c x W0)               only the share c of the blank that is still there taken away; c is the
                                         never-built rows' surviving share (foundation0/foundation_rows.log,
                                         foundation_check.json), one whole number at 2^-30 per size and step.
                                         round = half away from zero (the same for + and -, so it cannot tip a balance)
  Every plate is put on one scale (each neuron's whole numbers shifted up to the plate's finest power: exact).

  Write lines (what goes INTO the stream): attention-out (proj) and MLP-out (mproj) one input line across the
  stream's lanes; the input word table (embed) one word's row across the lanes.  Read neurons (what reads FROM the
  stream after the water level): query/key/value (qkv) and MLP-in (fc) one neuron over its lines (the lanes); the
  output word table (head) one word over the lanes.  Cross axis, for contrast only: one stream lane over all write
  lines, or over all read neurons.

  part=12 size=<s>   items 1 and 2 at the finished step:
    1. size of a typical change against its blank weight and its finished weight: share |D| < |W0|; the 10 -> 12
       picture, share 5|D| <= |W0| (change within 20% of the blank's size); share |D| < |W|; share of weights that
       kept their sign; medians of |D|/|W0| and |D|/|W|.  Shuffle control: each finished weight paired with a blank
       weight drawn from elsewhere in the same plate (pairing broken, every value kept).
    2. zero-sum: ups against downs of the changes
       - per plate: how many went up and down, how much went up and down; net in coin-toss widths
         z = net / sqrt(sum of squares) (a fair coin: |z| about 1)
       - at every place, in sign and size (an UP is a 1 in the size of a positive change) and in signed digits (+1/-1
         digits, no two non-zero side by side): ups, downs, z = (ups - downs)/sqrt(ups + downs); sum over places of
         2^p (ups - downs) = the plate's net is asserted
       - per write line over the stream's lanes and per read neuron over its lines: z of each; the root mean square
         over the plate's lines (a fair coin: 1), the median |z|, the share beyond 3; and per (line, place) cell
       - controls: the changes scrambled within the plate (every value kept, pairing to lines broken); a fresh bell
         draw of each neuron's own spread on the same grid.  Whole-plate counts hold every value, so the scramble
         cannot move them (asserted, the order-free check); for those the fresh draw is the only reference.
       - order-free: lanes and lines permuted together give the same multiset of line nets (asserted).
       - write biases (zero at the start, so the change is the bias): z over the lanes.
  part=3             item 3, 70m, between saved steps 0 -> 1000 -> 10000 -> 50000 -> 143000:
       the change over each span, raw (W(t2) - W(t1)) and pushes-only (W(t2) - round(c2/c1 x W(t1)): the shave of
       the earlier plate over the span taken out), per write line over the lanes and per read neuron over its
       lines, against the coin-toss width; scramble and fresh draw; the shave alone; the gains and offsets.

------------------------------------------------------------------------------------------------------------------
SESSION'S READING (the session's, not his): every read of the stream goes through a water level that pours off
the stream's level (its average over the lanes) first, so any level a write adds never reaches anything; every
push the building sent back into the stream was therefore level-free (what goes up in one lane goes down in the
others), so for writes it IS a zero-sum game across the stream's lanes; it is not exact in the finished weights
because each weight's push was rescaled by that weight's own run of pushes.

SESSION'S PREDICTION (as given to this run, written before it): a typical change is about as large as the finished
weight (the 10 -> 12 picture holds for under 25% of weights); per plate and per read neuron the changes are
zero-sum only to the coin toss or worse; per write line over the stream's lanes they are zero-sum well beyond
chance (under half the coin toss) at every span.

THIS RUN'S DETAILED PREDICTIONS (the session's, written before the first run; from foundation0's known numbers:
finished-vs-blank agreement 0.02-0.36 on layer plates and 0.004-0.03 on word tables, rows grew x1.04-7.8 at 70m
and x1.2-5 at 160m, the 14m blank larger than its finished plate in most neurons, finished write rows' even part
0.13-0.28 of their own coin toss):
 P1 size of the change, D at 143000:
   a. 10 -> 12 picture (5|D| <= |W0|): 3-12% of weights on every layer plate, 3-7% on the word tables, every size;
      the shuffle 2-7%; the real share at most about twice the shuffle (layer plates) and within 1 point of it
      (word tables).
   b. |D| < |W0|: 14m 35-60%, 70m 25-50%, 160m 10-35%.     c. |D| < |W|: 30-50%; median |D|/|W| 1.0-1.6.
   d. median |D|/|W0|: 14m 0.9-1.6, 70m 1.2-2.5, 160m 1.8-5.
   e. D' differs from W by c x W0 (c <= 0.0118): every D' share within 1 point of W's own against W0.
 P2 zero-sum at 143000:
   a. write lines, D': RMS z 0.1-0.35 on every write plate of every size (input word table, attention-out,
      MLP-out); scramble 0.9-1.1 (more only where the whole plate leans); fresh 0.97-1.03.
   b. write lines, D (the whole blank taken away too): the blank's own coin-toss level rides in D, so RMS z about
      1/sqrt(1 + g^2) with g the row growth: 14m 0.7-1.0, 70m 0.35-0.8 (word table about 0.65), 160m layer plates
      0.2-0.5, 160m word table about 0.6.  "Under half" holds for D only where the plate grew past about x2.
   c. read neurons, D': at or above the coin toss on 70m qkv (0.8-3.5) and 160m qkv layers 6-11 (0.7-1.4); below
      it on 160m MLP-in (0.16-1.0) and the 70m output word table (about 0.23).  So "coin toss or worse" for read
      neurons fails on some plates.
   d. per plate: D' net within 3 widths on write plates; beyond 3 on the output word table and the MLP-in plates
      that lean as finished plates.  Places: sign and size beyond 3 at most half-full places of a leaning plate;
      signed digits at the coin toss except the top live places of those plates.
   e. cross axis (one lane over all write lines / read neurons): RMS z above 2 on MLP-out, MLP-in, word tables.
   f. write biases: |z| under 0.5 at every layer, every size.
 P3 spans, 70m:
   a. write lines, pushes-only: RMS z under 0.35 at every span.
   b. write lines, raw: 0->1000 under 0.35; 1000->10000 0.3-0.6 (59% of the step-1000 plate, which still holds 0.97
      of the blank's level, is shaved in that span); 10000->50000 0.2-0.5; 50000->143000 0.15-0.4.
   c. read neurons, pushes-only: 0->1000 under 0.35 (at the start every gain is 1 and every offset 0, so the light a
      read neuron meets right after the water level is itself level-free: the session's reading carried to reads);
      later spans rising toward or past the coin toss on qkv as the gains spread and the offsets grow; MLP-in and
      the output word table in between.  This contradicts the given prediction for read neurons at the first span.
   d. the shave alone carries the earlier plate's level at about the coin toss of its own size (RMS z about 1 per
      line at 0->1000 and 1000->10000), so whatever raw spans lose against pushes-only is the shave.
   e. scramble about 1, fresh about 1, every span.

    python3 zerosum_count.py part=12 size=14m|70m|160m      python3 zerosum_count.py part=3
"""
import os, sys, json, math, time, resource
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
PART, SIZE, SEED = A.get("part", "12"), A.get("size", "70m"), int(A.get("seed", "1"))
HERE = os.path.dirname(os.path.abspath(__file__))
F0 = os.path.join(os.path.dirname(HERE), "foundation0")
CHK = json.load(open(os.path.join(F0, "foundation_check.json")))
I64 = np.int64


def c30(size, step):
    """the never-built rows' surviving share, a whole number at 2^-30 (as foundation_rows.py writes it)"""
    if step == 0: return 1 << 30
    return int(round(CHK[f"{size}@{step}"]["embed"]["c_med"] * 2 ** 30))


def shaved(v, num, den):
    """round(v x num / den), half away from zero, exact whole numbers (v on its own neuron's grid)"""
    a = np.abs(v.astype(I64)); r = (2 * a * num + den) // (2 * den)
    return np.where(v < 0, -r, r)


def plates(L):
    """(name, kind, power axis) in the npz's [in][out] layout. kind 'w' = write: lines are rows, lanes are columns;
    'r' = read: neurons are columns, their lines (lanes) are rows. power axis 0: one power per column; 1: per row."""
    out = []
    for l in range(L):
        out += [(f"qkv.{l}", "r", 0), (f"proj.{l}", "w", 0), (f"fc.{l}", "r", 0), (f"mproj.{l}", "w", 0)]
    return out + [("head", "r", 0), ("embed", "w", 1)]


def shift_of(e, pa, shape):
    sh = (e - e.min()).astype(I64)
    return sh[None, :] if pa == 0 else sh[:, None]


def rowblock(C): return max(1, (1 << 20) // C)


def sums(X):
    """row and column nets and sums of squares, exact int64, in row blocks"""
    R, C = X.shape; rb = rowblock(C)
    rn, rq = np.zeros(R, I64), np.zeros(R, I64); cn, cq = np.zeros(C, I64), np.zeros(C, I64)
    for a in range(0, R, rb):
        x = X[a:a + rb].astype(I64); x2 = x * x
        rn[a:a + rb] = x.sum(1); rq[a:a + rb] = x2.sum(1); cn += x.sum(0); cq += x2.sum(0)
    return rn, rq, cn, cq


def zstats(net, sq):
    ok = sq > 0
    z = net[ok] / np.sqrt(sq[ok].astype(np.float64))
    return dict(n=int(ok.sum()), rms=float(np.sqrt(np.mean(z * z))) if z.size else None,
                med=float(np.median(np.abs(z))) if z.size else None, over3=float((np.abs(z) > 3).mean()) if z.size else None,
                under_half=float((np.abs(z) < 0.5).mean()) if z.size else None), z


def masks(x, form):
    a = np.abs(x).astype(np.uint32)
    if form == "ss":
        zero = np.zeros_like(a)
        return np.where(x > 0, a, zero), np.where(x < 0, a, zero)
    xh = a >> 1; x3 = a + xh; c = xh ^ x3; pp = x3 & c; mm = xh & c
    assert np.array_equal(pp.astype(I64) - mm.astype(I64), a.astype(I64))
    return np.where(x > 0, pp, mm), np.where(x > 0, mm, pp)


def places(X, unit_axis_rows):
    """per place, both forms: plate ups/downs, and per-unit cell d/t (units = rows if unit_axis_rows else columns)"""
    R, C = X.shape; rb = rowblock(C)
    top = int(np.abs(X).max()); nP = {"ss": max(1, top.bit_length()), "sd": top.bit_length() + 1}
    out = {}
    for form in ("ss", "sd"):
        n = nP[form]; nu = R if unit_axis_rows else C
        out[form] = dict(up=np.zeros(n, I64), dn=np.zeros(n, I64), ud=np.zeros((n, nu), I64), ut=np.zeros((n, nu), I64))
    for a in range(0, R, rb):
        x = X[a:a + rb]
        for form in ("ss", "sd"):
            P, M = masks(x, form); o = out[form]
            assert not np.any(P & M)
            for p in range(nP[form]):
                bu = ((P >> p) & 1).astype(np.int8); bd = ((M >> p) & 1).astype(np.int8)
                d = bu - bd; t = bu + bd
                o["up"][p] += int(bu.sum(dtype=I64)); o["dn"][p] += int(bd.sum(dtype=I64))
                if unit_axis_rows:
                    o["ud"][p, a:a + rb] = d.sum(1, dtype=I64); o["ut"][p, a:a + rb] = t.sum(1, dtype=I64)
                else:
                    o["ud"][p] += d.sum(0, dtype=I64); o["ut"][p] += t.sum(0, dtype=I64)
            del P, M
    return out


def place_summary(pc, net):
    res = {}
    for form, o in pc.items():
        n = len(o["up"]); w = np.array([1 << p for p in range(n)], dtype=object)
        assert int((w * (o["up"] - o["dn"]).astype(object)).sum()) == net, "sum of 2^p (ups - downs) is not the net"
        t = o["up"] + o["dn"]; live = t > 0
        z = np.where(live, (o["up"] - o["dn"]) / np.sqrt(np.maximum(t, 1)), np.nan)
        cl = o["ut"] > 0; cz = o["ud"][cl] / np.sqrt(o["ut"][cl])
        res[form] = dict(ups=o["up"].tolist(), downs=o["dn"].tolist(),
                         z=[None if np.isnan(v) else round(float(v), 3) for v in z],
                         live=int(live.sum()), over2=int((np.abs(z[live]) > 2).sum()), over3=int((np.abs(z[live]) > 3).sum()),
                         maxz=float(np.nanmax(np.abs(z))) if live.any() else None,
                         cell_rms=float(np.sqrt(np.mean(cz * cz))) if cz.size else None,
                         cell_over3=float((np.abs(cz) > 3).mean()) if cz.size else None)
    return res


def plate_counts(X, rn, rq):
    """his ups and downs over the whole plate: how many and how much"""
    R, C = X.shape; rb = rowblock(C); nu = nd = 0; su = sd_ = 0
    for a in range(0, R, rb):
        x = X[a:a + rb].astype(I64)
        nu += int((x > 0).sum()); nd += int((x < 0).sum()); su += int(x[x > 0].sum()); sd_ += int(-x[x < 0].sum())
    net = int(rn.sum()); sq = int(rq.astype(object).sum())
    assert su - sd_ == net
    return dict(n_up=nu, n_down=nd, n_zero=X.size - nu - nd, size_up=su, size_down=sd_, net=net, sq=sq,
                net_z=net / math.sqrt(sq) if sq else 0.0, count_z=(nu - nd) / math.sqrt(nu + nd) if nu + nd else 0.0)


def fresh_like(X, sh, pa, rng):
    """a fresh bell draw with each neuron's own spread, on that neuron's own grid, put on the plate's scale"""
    R, C = X.shape; _, rq, _, cq = sums(X)
    shv = sh.reshape(-1).astype(np.float64)
    if pa == 0: rms_own = np.sqrt(cq / R) / 2.0 ** shv
    else: rms_own = np.sqrt(rq / C) / 2.0 ** shv
    F = np.empty_like(X); rb = rowblock(C)
    for a in range(0, R, rb):
        b = min(R, a + rb)
        g = rng.standard_normal((b - a, C)).astype(np.float64)
        s = rms_own[None, :] if pa == 0 else rms_own[a:b, None]
        shb = sh if pa == 0 else sh[a:b]
        F[a:b] = (np.rint(g * s).astype(I64) << shb).astype(X.dtype)
    return F


def unit_and_cross(X, kind):
    rn, rq, cn, cq = sums(X)
    if kind == "w": u, cr = (rn, rq), (cn, cq)
    else: u, cr = (cn, cq), (rn, rq)
    return u, cr, rn, rq


def analyse(X, kind, sh, pa, rng, qkv_parts=None, with_places=True):
    """item 2 on one change array X (plate scale): the plate, the places, the units, the controls (one copy at a time)"""
    out = {}
    for lab in ("real", "scramble", "fresh"):
        if lab == "real": Y = X
        elif lab == "scramble": Y = rng.permutation(X.reshape(-1)).reshape(X.shape)
        else: Y = fresh_like(X, sh, pa, rng)
        u, cr, rn, rq = unit_and_cross(Y, kind)
        us, uz = zstats(*u); cs, _ = zstats(*cr)
        rec = dict(units=us, cross=cs, plate=plate_counts(Y, rn, rq))
        if qkv_parts is not None:
            rec["qkv_parts"] = {nm: zstats(u[0][m], u[1][m])[0] for nm, m in qkv_parts.items()}
        if with_places:
            pc = places(Y, kind == "w"); rec["places"] = place_summary(pc, rec["plate"]["net"]); del pc
        out[lab] = rec
        del Y
    if with_places:
        for form in ("ss", "sd"):   # order-free: the scramble holds every value, so every place count is the same
            assert out["real"]["places"][form]["ups"] == out["scramble"]["places"][form]["ups"]
            assert out["real"]["places"][form]["downs"] == out["scramble"]["places"][form]["downs"]
    # order-free: lanes and lines permuted together, the multiset of line nets is the same whole numbers
    pr, pcol = rng.permutation(X.shape[0]), rng.permutation(X.shape[1])
    u1 = unit_and_cross(X, kind)[0][0]
    Y = X[pr]; Y = Y[:, pcol]; u2 = unit_and_cross(Y, kind)[0][0]; del Y
    assert np.array_equal(np.sort(u1), np.sort(u2)); out["order_free"] = True
    return out


def size_stats(Wc, W0c, mode, cc, den, sh):
    """item 1: exact whole-number comparisons on the plate's scale, in row blocks; the change is built block by block:
    mode 'D' W - W0, mode 'Dp' W - round(c W0) on the grid of the neuron the blank weight now sits in"""
    R, C = Wc.shape; rb = rowblock(C); n = Wc.size
    k = dict(d_lt_blank=0, within20=0, d_lt_fin=0, fin_lt_blank=0, sign_kept=0, unchanged=0)
    s2 = dict(d=0.0, blank=0.0, fin=0.0)
    r0 = np.empty(n, np.float32); rW = np.empty(n, np.float32)
    shf = np.broadcast_to(sh, Wc.shape)
    for a in range(0, R, rb):
        w, w0 = Wc[a:a + rb].astype(I64), W0c[a:a + rb].astype(I64)
        if mode == "D": d = w - w0
        else:
            q = den << shf[a:a + rb]; r = (2 * np.abs(w0) * cc + q) // (2 * q)
            d = w - (np.where(w0 < 0, -r, r) << shf[a:a + rb])
        aD, a0, aW = np.abs(d), np.abs(w0), np.abs(w)
        k["d_lt_blank"] += int((aD < a0).sum()); k["within20"] += int((5 * aD <= a0).sum()); k["d_lt_fin"] += int((aD < aW).sum())
        k["fin_lt_blank"] += int((aW < a0).sum()); k["unchanged"] += int((aD == 0).sum()); k["sign_kept"] += int(((w * w0) > 0).sum())
        s2["d"] += float((aD.astype(np.float64) ** 2).sum()); s2["blank"] += float((a0.astype(np.float64) ** 2).sum()); s2["fin"] += float((aW.astype(np.float64) ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r0[a * C:a * C + aD.size] = (aD / a0).reshape(-1); rW[a * C:a * C + aD.size] = (aD / aW).reshape(-1)
    res = {kk: v / n for kk, v in k.items()}
    res.update(n=n, med_d_over_blank=float(np.nanmedian(r0)), med_d_over_fin=float(np.nanmedian(rW)), zero_over_zero=int(np.isnan(r0).sum()),
               rms_d=math.sqrt(s2["d"] / n), rms_blank=math.sqrt(s2["blank"] / n), rms_fin=math.sqrt(s2["fin"] / n))
    del r0, rW
    return res


def shaved_to_grid(vc, num, den, sh):
    """round(vc x num/den) onto the grid of the neuron it now sits in (vc on the plate's scale), half away from zero"""
    R, C = vc.shape; rb = rowblock(C); out = np.empty(vc.shape, np.int32)
    for a in range(0, R, rb):
        v = vc[a:a + rb].astype(I64); s = np.broadcast_to(sh, vc.shape)[a:a + rb]
        q = den << s; r = (2 * np.abs(v) * num + q) // (2 * q)
        out[a:a + rb] = (np.where(v < 0, -r, r) << s).astype(np.int32)
    return out


def maxrss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def part12():
    z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz"))
    L = int(z["L"]); cc = c30(SIZE, 143000); den = 1 << 30
    res = dict(size=SIZE, c30=cc, plates={}, biases={})
    logf = open(os.path.join(HERE, f"zerosum_{SIZE}.log"), "w")
    def say(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    say(f"pythia-{SIZE}: c = {cc}/2^30 = {cc / 2 ** 30:.6g}; D = W - W0, D' = W - round(c W0)")
    t0 = time.time()
    for pi, (name, kind, pa) in enumerate(plates(L)):
        rng = np.random.default_rng(SEED * 1_000_003 + pi + {"14m": 0, "70m": 10_000, "160m": 20_000}[SIZE])
        W, W0, e = z[name + ".w"], z[name + ".s"], z[name + ".e"]
        sh = shift_of(e, pa, W.shape)
        Wc = (W.astype(I64) << sh).astype(np.int32); W0c = (W0.astype(I64) << sh).astype(np.int32)
        rb, shf = rowblock(W.shape[1]), np.broadcast_to(sh, W.shape)
        for a in range(0, W.shape[0], rb):   # the grid-placed rounding is D''s own rounding on the true pairing
            assert np.array_equal(shaved(W0[a:a + rb], cc, den) << shf[a:a + rb], shaved_to_grid(W0c[a:a + rb], cc, den, shf[a:a + rb]))
        shape = W.shape; del W, W0
        rec = dict(kind=kind, shape=list(shape), shift_max=int(sh.max()))
        # item 1, real pairing and the scramble (finished weight against a blank weight from elsewhere in the plate)
        rec["size"] = dict(D=size_stats(Wc, W0c, "D", cc, den, sh), Dp=size_stats(Wc, W0c, "Dp", cc, den, sh))
        W0p = rng.permutation(W0c.reshape(-1)).reshape(W0c.shape)
        rec["size"].update(D_scramble=size_stats(Wc, W0p, "D", cc, den, sh), Dp_scramble=size_stats(Wc, W0p, "Dp", cc, den, sh))
        del W0p
        qp = None
        if name.startswith("qkv"):
            hs = int(z["hs"]); col = np.arange(shape[1]); part = (col % (3 * hs)) // hs
            qp = {"q": part == 0, "k": part == 1, "v": part == 2}
        X = Wc - W0c; rec["D"] = analyse(X, kind, sh, pa, rng, qp); del X
        X = Wc - shaved_to_grid(W0c, cc, den, sh); rec["Dp"] = analyse(X, kind, sh, pa, rng, qp); del X
        rec["W"] = analyse(Wc, kind, sh, pa, rng, qp, with_places=False)   # the finished plate itself, for reference
        rec["W0"] = analyse(W0c, kind, sh, pa, rng, qp, with_places=False)  # the blank itself
        res["plates"][name] = rec
        s1, s1s = rec["size"]["D"], rec["size"]["D_scramble"]
        u = {k: rec[k]["real"]["units"]["rms"] for k in ("D", "Dp", "W", "W0")}
        say(f"{name:9s} {'write' if kind == 'w' else 'read '} {str(shape):14s} | 10->12 {100 * s1['within20']:5.2f}% (scr {100 * s1s['within20']:5.2f}) "
            f"|D|<|W0| {100 * s1['d_lt_blank']:5.1f}% (scr {100 * s1s['d_lt_blank']:5.1f}) |D|<|W| {100 * s1['d_lt_fin']:5.1f}% "
            f"med|D|/|W0| {s1['med_d_over_blank']:.2f} med|D|/|W| {s1['med_d_over_fin']:.2f} sign kept {100 * s1['sign_kept']:.1f}% "
            f"| {'line' if kind == 'w' else 'neuron'} rms z: D {u['D']:.3f} D' {u['Dp']:.3f} W {u['W']:.3f} W0 {u['W0']:.3f} "
            f"(D scr {rec['D']['scramble']['units']['rms']:.3f} fresh {rec['D']['fresh']['units']['rms']:.3f}; D' scr {rec['Dp']['scramble']['units']['rms']:.3f} "
            f"fresh {rec['Dp']['fresh']['units']['rms']:.3f}) | cross rms D' {rec['Dp']['real']['cross']['rms']:.2f} "
            f"| plate z D {rec['D']['real']['plate']['net_z']:+.2f} D' {rec['Dp']['real']['plate']['net_z']:+.2f} "
            f"(fresh {rec['Dp']['fresh']['plate']['net_z']:+.2f}) | {time.time() - t0:.0f}s {maxrss_gb():.2f}GB")
        del Wc, W0c
    for l in range(L):   # write biases: zero at the start, so the change is the bias (attention-out and MLP-out, nearly the same vector)
        b, b2, bs = z[f"proj.{l}.b"].astype(I64), z[f"mproj.{l}.b"].astype(I64), z[f"proj.{l}.bs"]
        assert not bs.any() and not z[f"mproj.{l}.bs"].any()
        def zb(v): return float(v.sum() / math.sqrt(float((v * v).sum()))) if (v * v).sum() else 0.0
        res["biases"][l] = dict(z=zb(b), z_mlp=zb(b2), max_diff=int(np.abs(b - b2).max()), max_abs=int(np.abs(b).max()), lanes=int(b.size))
    say("write biases z over the lanes (attention-out / MLP-out): " + " ".join(f"L{l} {v['z']:+.3f}/{v['z_mlp']:+.3f}" for l, v in res["biases"].items()))
    res["peak_gb"] = maxrss_gb()
    json.dump(res, open(os.path.join(HERE, f"zerosum_{SIZE}.json"), "w"))
    say(f"wrote zerosum_{SIZE}.json; peak {res['peak_gb']:.2f} GB; {time.time() - t0:.0f}s")


def part3():
    z = np.load(os.path.join(F0, "base2_pythia-70m.npz")); zs = np.load(os.path.join(F0, "base2_pythia-70m-steps.npz"))
    L = int(z["L"]); STEPS = (0, 1000, 10000, 50000, 143000)
    C = {t: c30("70m", t) for t in STEPS}
    SPANS = [(0, 1000), (1000, 10000), (10000, 50000), (50000, 143000), (0, 10000), (0, 50000), (0, 143000)]
    logf = open(os.path.join(HERE, "zerosum_spans70.log"), "w")
    def say(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    def get(name, t):
        if t == 0: return z[name + ".s"].astype(np.int32)
        if t == 143000: return z[name + ".w"].astype(np.int32)
        assert np.array_equal(zs[name + ".e"], z[name + ".e"])
        return zs[f"{name}.step{t}"]
    res = dict(c30=C, plates={}, biases={}, norms={})
    say("pythia-70m spans; c30 " + str(C))
    t0 = time.time()
    for pi, (name, kind, pa) in enumerate(plates(L)):
        rng = np.random.default_rng(SEED * 7_000_003 + pi)
        e = z[name + ".e"]; Wt = {t: get(name, t) for t in STEPS}
        sh = shift_of(e, pa, Wt[0].shape).astype(np.int32); rec = dict(kind=kind, spans={})
        R0, C0 = Wt[0].shape; rb = rowblock(C0)
        def shaved32(v, num, den):
            out = np.empty(v.shape, np.int32)
            for a0 in range(0, R0, rb): out[a0:a0 + rb] = shaved(v[a0:a0 + rb], num, den).astype(np.int32)
            return out
        for t1, t2 in SPANS:
            kept = shaved32(Wt[t1], C[t2], C[t1])
            raw = (Wt[t2] - Wt[t1]) << sh
            push = (Wt[t2] - kept) << sh
            shave = (Wt[t1] - kept) << sh                      # what the shave took off the earlier plate
            del kept
            r = {}
            for lab, X in (("raw", raw), ("push", push)):
                u, cr, rn, rq = unit_and_cross(X, kind); us, _ = zstats(*u); cs, _ = zstats(*cr)
                net, sq = int(rn.sum()), int(rq.astype(object).sum())
                S = rng.permutation(X.reshape(-1)).reshape(X.shape); uS = unit_and_cross(S, kind)[0]; del S
                Fd = fresh_like(X, sh, pa, rng); uF = unit_and_cross(Fd, kind)[0]; del Fd
                r[lab] = dict(units=us, cross=cs, scramble=zstats(*uS)[0], fresh=zstats(*uF)[0],
                              plate_z=net / math.sqrt(sq) if sq else 0.0, rms=float(np.sqrt(sq / X.size)))
            u = unit_and_cross(shave, kind)[0]; r["shave"] = dict(units=zstats(*u)[0])
            _, _, rn1, rq1 = unit_and_cross(Wt[t1] << sh, kind)
            r["rms_before"] = float(np.sqrt(int(rq1.astype(object).sum()) / raw.size))
            rec["spans"][f"{t1}->{t2}"] = r
            del raw, push, shave
        u1 = {}
        for t in STEPS:   # the plate itself at each step
            u = unit_and_cross(Wt[t] << sh, kind)[0]; u1[t] = zstats(*u)[0]["rms"]
        rec["plate_rms_z"] = u1
        res["plates"][name] = rec
        say(f"{name:9s} {'write' if kind == 'w' else 'read '} " + " | ".join(
            f"{k}: raw {v['raw']['units']['rms']:.3f} push {v['push']['units']['rms']:.3f} (scr {v['push']['scramble']['rms']:.2f} fr {v['push']['fresh']['rms']:.2f}) shave {v['shave']['units']['rms']:.2f}"
            for k, v in rec["spans"].items() if not k.startswith("0->") or k == "0->1000") + f" | {time.time() - t0:.0f}s {maxrss_gb():.2f}GB")
        del Wt
    for l in range(L):
        bt = {0: z[f"proj.{l}.bs"].astype(I64), 143000: z[f"proj.{l}.b"].astype(I64)}
        for t in (1000, 10000, 50000): bt[t] = zs[f"proj.{l}.b.step{t}"].astype(I64)
        def zb(v): return float(v.sum() / math.sqrt(float((v * v).sum()))) if (v * v).sum() else 0.0
        res["biases"][l] = {**{f"at {t}": zb(bt[t]) for t in STEPS if t}, **{f"{t1}->{t2}": zb(bt[t2] - bt[t1]) for t1, t2 in SPANS[:4]}}
    say("write biases z: " + "; ".join(f"L{l} " + " ".join(f"{k} {v:+.2f}" for k, v in d.items()) for l, d in res["biases"].items()))
    for nm in [f"ln1.{l}" for l in range(L)] + [f"ln2.{l}" for l in range(L)] + ["lnf"]:
        d = {}
        for t in STEPS:
            if t == 0: g, o = z[nm + ".gs"], z[nm + ".bs"]
            elif t == 143000: g, o = z[nm + ".g"], z[nm + ".b"]
            else: g, o = zs[f"{nm}.g.step{t}"], zs[f"{nm}.b.step{t}"]
            g = g.astype(np.float64); o = o.astype(np.float64) / 2 ** 16
            d[t] = dict(gain_spread=float(g.std() / g.mean()), offset_rms=float(np.sqrt(np.mean(o * o))), offset_sum=float(o.sum()))
        res["norms"][nm] = d
    for t in STEPS:
        say(f"gains at {t:6d}: spread/mean ln1 {np.mean([res['norms'][f'ln1.{l}'][t]['gain_spread'] for l in range(L)]):.3f} "
            f"ln2 {np.mean([res['norms'][f'ln2.{l}'][t]['gain_spread'] for l in range(L)]):.3f} lnf {res['norms']['lnf'][t]['gain_spread']:.3f}; "
            f"offsets rms ln1 {np.mean([res['norms'][f'ln1.{l}'][t]['offset_rms'] for l in range(L)]):.3f} "
            f"ln2 {np.mean([res['norms'][f'ln2.{l}'][t]['offset_rms'] for l in range(L)]):.3f} lnf {res['norms']['lnf'][t]['offset_rms']:.3f}")
    res["peak_gb"] = maxrss_gb()
    json.dump(res, open(os.path.join(HERE, "zerosum_spans70.json"), "w"), default=str)
    say(f"wrote zerosum_spans70.json; peak {res['peak_gb']:.2f} GB; {time.time() - t0:.0f}s")


if PART == "12": part12()
else: part3()
