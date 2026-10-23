"""groups2/gtop.py -- the top read of the whole machines (base2_groups.py, qwen_groups.py): groups2/topg.py's (ii) in the
per-place form, run on a machine's own light at the top, with two counts topg.py leaves out: the ready piles written in
signed digits (the 1s of every half-table entry the word table reads, once a position) and the light written in signed
digits (its 1s, as the band read counts it). topg.py's own functions do the read, its set-aside and its checks."""
import sys, time, numpy as np
from gbank import ones
def rss():
    import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def top_module(model, g, order):
    a_ = sys.argv; sys.argv = ["topg.py", f"model={model}", f"g={g}", "forms=pp", f"run={'order' if order else 'main'}"]
    import topg as TG
    sys.argv = a_; return TG
def half_writing(X, glines, goff, readh):
    """the ready piles written in signed digits: the 1s of every half-table entry the table reads, per position"""
    X = np.asarray(X, np.int64); P = len(X); out = np.zeros(P, np.int64)
    for k in range(len(goff) - 1):
        L = glines[goff[k]:goff[k + 1]]; gk = len(L); xl = X[:, L]
        v = (xl[:, gk - 1] - xl[:, :gk - 1].sum(1))[:, None]                            # last line up, every other a hole
        for t in range(gk - 1): v = np.concatenate([v, v + 2 * xl[:, t:t + 1]], 1)
        out += (ones(v) * readh[k][None, :v.shape[1]]).sum(1)
    return out
def top_readh(TG, P_, glines, goff, ppmask):
    G = len(goff) - 1; gmax = int(np.diff(goff).max()); rh = np.zeros((G, 1 << (gmax - 1)), np.uint8)
    ups = [b for b in range(16) if (ppmask >> b) & 1]
    for k in range(G):
        gk = int(goff[k + 1] - goff[k]); top = 1 << (gk - 1); full = (1 << gk) - 1
        p = P_["pp"][k][:, ups].astype(np.int64).ravel(); key = np.where(p & top, p, p ^ full) - top
        rh[k, np.unique(key)] = 1
    return rh

def run_top(TG, X, EX, LS, order, log, batch=10):
    """topg.py's main loop for the per-place form on the given light; returns its records and the added counts"""
    NP_ = len(X); seed = 1
    glines, goff = TG.make_groups(seed, reorder=[4242, TG.N, TG.G_] if order else None)
    G = len(goff) - 1; wide = int(np.diff(goff).max()) > 16
    perm = np.random.default_rng([4343, TG.M]).permutation(TG.M) if order else None
    Lb = TG.lit_by_place(); U = G * TG.M; ppmask = sum(1 << b for b in range(16) if not Lb[b] < U)
    hi = ppmask.bit_length() - 1; TG.KCORR.update(pp=ppmask - 2 * (1 << hi))
    P_ = TG.build(glines, goff, ["pp"], ppmask, wide, perm=perm)
    readh = top_readh(TG, P_, glines, goff, ppmask)
    ch, cf, nh, nf = TG.table_coeffs(glines, goff); bnd = {"pp": TG.bound_per_line("pp", ppmask)}
    Wt = TG.WT if perm is None else (np.ascontiguousarray if TG.WT.flags["C_CONTIGUOUS"] else np.asfortranarray)(np.asarray(TG.WT)[:, perm])
    lab = np.arange(TG.M) if perm is None else perm; inv = np.argsort(lab)
    ctx = dict(bias=TG.BIAS[lab] if TG.MODEL == "gpt2" else None, bias1=TG.BIAS_ONES[lab] if TG.MODEL == "gpt2" else None, kw=TG.KW[lab],
               ch=ch, cf=cf, bnd=bnd, fmt=P_["fmt"], lab=lab, inv=inv)
    pph = np.zeros(TG.N, np.int64)
    for j0 in range(0, TG.M, 8192): pph += ones(np.asarray(Wt[:, j0:j0 + 8192])).sum(1)
    ctx["pph"] = pph
    Z = lambda *s: np.zeros(s, np.int64)
    rec = dict(plain_meet=Z(NP_), plain_div=Z(NP_), plain_stand=Z(NP_), X=Z(NP_), S1=Z(NP_), mu=Z(NP_), ptop5=Z(NP_, 5), pwin=Z(NP_), sha=[""] * NP_)
    rec["pp"] = dict(table=Z(NP_), i_reads=Z(NP_), i_light=Z(NP_), i_total=Z(NP_), i_byplace=Z(NP_, TG.NPMAX), i_adds=Z(NP_, TG.NPMAX),
                     i_top5=Z(NP_, 5), i_win=Z(NP_), digest=[""] * NP_, ii_reads=Z(NP_), ii_light=Z(NP_), ii_bound=Z(NP_), ii_cmp=Z(NP_),
                     ii_end=Z(NP_), ii_stand=Z(NP_), ii_total=Z(NP_), ii_surv=np.full((NP_, TG.NPMAX), -1, np.int64),
                     ii_byplace=Z(NP_, TG.NPMAX), ii_top5=Z(NP_, 5), ii_win=Z(NP_), ii_last=Z(NP_))
    bad = {"pp": {}}; pwrite = Z(NP_); lwrite = Z(NP_); tl = time.time()
    for c0 in range(0, NP_, batch):
        Xb = np.asarray(X[c0:c0 + batch], np.int64); Bn = len(Xb)
        A, H, NA = TG.run_kernel(Xb, glines, goff, P_, ppmask, wide)
        PILE = TG.direct(Xb, Wt)
        for p in range(Bn):
            ctx["A"] = {"pp": A["pp"][p]}; ctx["H"] = {"pp": H["pp"][p]}; ctx["NA"] = NA[p]
            TG.position(c0 + p, Xb[p], int(EX[c0 + p]), int(LS[c0 + p]), PILE[p], ctx, bad, rec)
        pwrite[c0:c0 + Bn] = half_writing(Xb, glines, goff, readh); lwrite[c0:c0 + Bn] = ones(Xb).sum(1)
        del A, H, NA, PILE
        if (c0 // batch) % 20 == 0 or c0 + batch >= NP_: log(f"    {min(NP_, c0 + batch)}/{NP_} positions ({time.time() - tl:.0f}s, {rss()} MB)")
    return rec, bad, pwrite, lwrite, ppmask, lab

