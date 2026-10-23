"""DIFFUSE look (not a judge -- the analysis of v8's negative): how much of each bank's pile error, as the calibration
light reads it, stands still (the same on every position) and how much moves. For the plain v12 A=6 development and
for v4's carry: per bank, standing share = sum_j (mean_p err_pj)^2 / sum_j mean_p err_pj^2, and the carry's error
power over plain's. Whole-number errors from the machine's own arithmetic; the shares are printed as ratios."""
from diffuse_dark import *
import table_bits as _tb
from diffuse_dark import _swap
wins = windows()
def dev_all(develop, rule="v12", A=6):
    out = {}
    for n in NAMES:
        q = plate(n); t, off = terms(n, q, rule); s = steps(t, off, A); out[n] = shift_back(develop(q, s), s)
    return out
def errs(Wdev):
    orig = {n: _tb.W[n] for n in NAMES}; S1 = {}; S2 = {}
    R = [_tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)] for ids in wins]
    for l in range(12):
        for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
            o = [_tb.gauged_bank(r, bank) for r in R]
            _swap(bank, Wdev[bank]); d = [_tb.gauged_bank(r, bank) for r in R]; _tb.W[bank] = orig[bank]
            E = np.concatenate([dk - ok for dk, ok in zip(d, o)]); S1[bank] = E.sum(0); S2[bank] = (E * E).sum(0)
            xo = [nxt(ok) for ok in o]; po = [_tb.plain_bank(x, pb) for x in xo]
            _swap(pb, Wdev[pb]); pd = [_tb.plain_bank(x, pb) for x in xo]; _tb.W[pb] = orig[pb]
            E = np.concatenate([dk - ok for dk, ok in zip(pd, po)]); S1[pb] = E.sum(0); S2[pb] = (E * E).sum(0)
            R = [r + x for r, x in zip(R, po)]
    return S1, S2, len(wins) * 5
plain = errs(dev_all(develop_plain))
carry = errs(dev_all(lambda q, s: develop_carry(q, s, axis=0, kind=1, unit=True)))
carry_raw = errs(dev_all(lambda q, s: develop_carry(q, s, axis=0, kind=1, unit=False)))
print("bank      standing share (plain)   carry/plain error power (steps)   raw carry/plain")
tot = {}
for n in NAMES:
    S1, S2, N = plain[0][n], plain[1][n], plain[2]
    st = float((S1.astype(float) ** 2).sum() / N / S2.sum())
    rc = float(carry[1][n].sum() / S2.sum()); rr = float(carry_raw[1][n].sum() / S2.sum())
    k = n.split(".")[0]; tot.setdefault(k, []).append((st, rc, rr))
    if n.endswith(".0") or n.endswith(".5") or n.endswith(".11"): print(f"{n:9s} {st:8.4f}                 {rc:6.2f}                            {rr:6.2f}")
for k, v in tot.items():
    v = np.array(v); print(f"{k:6s} mean over 12: standing {v[:,0].mean():.4f}  carry/plain {v[:,1].mean():.2f}  raw carry/plain {v[:,2].mean():.2f}")
