"""THE 12 FIXED TURNS: the straight half of every bend, folded with its two plates.

HIS PICTURE (his words): "The edge is the same backwards and forwards. That's how we can collapse it. We could even make 49
little sections and just record 49 little turns. And have a tiny little turning model." And: "That tiny collapsed model is
the fringe of the original token AND every token previously in the sentence! And it's still 1.2 mb".

Restated in his picture: the bend is a Gaussian knife edge that passes exactly as much light backwards as forwards
(Babinet: the share table and its mirror add to the whole, PHI[32768+k] + PHI[32768-k] = 65536). So every bend splits
exactly into two halves: bend(H) = H/2 + e(H). The straight half H/2 passes like half-silvered glass at every neuron, the
same for every token. The symmetric half e(H) = bend(H) - H/2 is the part that sees only the size of the pile (it is the
same for H and -H up to the table's one-step offset). With the water level between the fc plate and the bend held, the
straight halves of one layer's 3,072 neurons fold the fc plate and the mproj plate into ONE fixed 768 x 768 turn:
        T_l = (1/2) x (fc plate, each column at its own power of two) x (mproj plate, each column at its power of two)
and per position p:
        (straight half of every bend) met with mproj  ==  g_p x (the light R_p met with T_l)
                                                         + g_p x (the light's 16-place rounding met with T_l)
                                                         + (1/2) x (fc's out-rounding + fc's bias) met with mproj
with g_p the water level (the gauge, one number per position: EXP2[f_p] x 2^(floor(-LS_p / 2^16) - 16)).

WHAT THIS SCRIPT DOES (measurement plus recording; nothing in the machine changes; base2.py is executed from its own
source through reverse/machine.py and never edited):
  0. the knife edge both ways, rechecked: PHI[32768+k] + PHI[32768-k] == 65536 for every k, PHI[32768] == 32768,
     PHI[0] == 0; bend(H) - bend(-H) - H over the real piles of the 30 refs texts and over random H.
  1. records T_l for every layer from base2_gpt2.npz as exact whole numbers: T_l[i, j] = TN_l[i, j] x 2^(m0_l + we_m[j] - 1)
     (m0_l = the smallest power of two among fc.l's columns, we_m[j] = mproj.l's column j power), and also on ONE power
     of two per layer; the bias part (1/2) b_fc met with mproj, exact. Saved to turns/turns.npz.
  2. checks the identity above exactly, as whole numbers on a common power of two, at every position, every line and
     every layer of texts 0, 1, 2 (first 5 tokens of refs_many.json).

GATING (holographicplate CLAUDE.md, answered before writing): (1) it builds a turn from two plates and the bend's straight
half, and measures an identity; (2) every read is on whole numbers (exact rationals over powers of two); (3) exact,
digit for digit; (4) nothing discarded; (5) nothing fitted, no word-count comparison; (6) the split of the bend into its
straight and symmetric halves is not in FINDINGS.md or NOT_TO_RERUN.md.

PREDICTION (written before running): every check exact by construction (36/36 layer-texts, every entry). TN fits in 64-bit
whole numbers (fc and mproj weights are 16 places, 3,072 neurons: about 30 + 12 places plus the spread of fc's column
powers). The 12 turns are 12 x 589,824 = 7,077,888 numbers, the size of ONE layer's four plates in count, but each
number needs about 45-50 places, so about 40 MB as written (against 1.2 MB for the print, which also folds the exposure).
    python3 turns/turns.py      -> turns/turns.npz, turns/turns.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; checked exact
from film_lead import *
import resource
OUTD = "/Users/abundancemachine/streaming-machine/turns"
t0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1 << 20)

# ---------------------------------------------------------------- exact helpers
def limbs16(X):
    """whole numbers (int64 or object) -> list of float64 arrays L_k, |L_k| < 2^16, X = sum_k L_k 2^(16k)"""
    X = np.asarray(X)
    if X.dtype == object:
        sg = np.vectorize(lambda v: (v > 0) - (v < 0), otypes=[np.int64])(X)
        A = np.vectorize(lambda v: abs(int(v)), otypes=[object])(X); out = []
        while any(int(v) for v in A.ravel()):
            out.append((np.vectorize(lambda v: int(v) & 0xFFFF, otypes=[np.int64])(A) * sg).astype(np.float64))
            A = np.vectorize(lambda v: int(v) >> 16, otypes=[object])(A)
        return out or [np.zeros(X.shape)]
    X = X.astype(np.int64); sg = np.sign(X); A = np.abs(X); out = []
    while A.any(): out.append(((A & 0xFFFF) * sg).astype(np.float64)); A = A >> 16
    return out or [np.zeros(X.shape)]

def exact_mm(A, B):
    """A @ B for whole-number matrices, exact (object result), by 16-place limbs: each float product sum < 2^53"""
    LA, LB = limbs16(A), limbs16(B); n = np.asarray(A).shape[-1]; assert n <= (1 << 20)
    out = None
    for a, La in enumerate(LA):
        for b, Lb in enumerate(LB):
            S = La @ Lb; assert np.abs(S).max() < 2.0 ** 53
            S = S.astype(np.int64).astype(object) * (1 << (16 * (a + b)))
            out = S if out is None else out + S
    return out

def obits(X): return max(abs(int(v)).bit_length() for v in np.asarray(X).ravel())

# ---------------------------------------------------------------- 0. the knife edge both ways
k = np.arange(1, 32768)
mirror = bool(np.all(PHI[32768 + k].astype(np.int64) + PHI[32768 - k].astype(np.int64) == 65536))
print(f"0. the knife edge: PHI[32768+k] + PHI[32768-k] == 65536 for all {len(k):,} k: {mirror};  PHI[32768] = {int(PHI[32768])};"
      f"  PHI[0] = {int(PHI[0])} (mirror of the open end 65536);  PHI[65535] = {int(PHI[65535])}")
def bend_int(H):                                       # base2.py's edge, its own lines (EDGE = int)
    return edge(np.asarray(H, np.int64))
rngH = np.random.default_rng(7)
Hr = rngH.integers(-(1 << 19), 1 << 19, 1_000_000)
d = bend_int(Hr) - bend_int(-Hr) - Hr
print(f"   bend(H) - bend(-H) - H over 1,000,000 random H in [-2^19, 2^19): zero for {100 * (d == 0).mean():.2f}%, range"
      f" [{d.min()}, {d.max()}];  where H = 8 mod 16 (the read index rshift(H, 4) lands one step off its mirror):"
      f" {100 * ((Hr % 16) == 8).mean():.2f}% of H, nonzero only there: {bool(((d != 0) <= ((Hr % 16) == 8)).all())}", flush=True)

# ---------------------------------------------------------------- 1. the 12 fixed turns
TN = np.zeros((12, 768, 768), np.int64); M0 = np.zeros(12, np.int64); WEM = np.zeros((12, 768), np.int64)
TB_N = np.zeros((12, 768), dtype=object); TB_E = np.zeros((12, 768), np.int64)
TN1_bits = []; spread = []
for l in range(12):
    wf, wef, bf = W[f"fc.{l}"]; wm, wem, bm = W[f"mproj.{l}"]
    wf = wf.astype(np.int64); wm = wm.astype(np.int64); m0 = int(wef.min()); spread.append(int(wef.max() - m0))
    acc = np.zeros((768, 768), dtype=object); acc[:] = 0
    for g in np.unique(wef):                           # neurons on the same power of two: one exact float product each
        sel = np.flatnonzero(wef == g)
        S = wf[:, sel].astype(np.float64) @ wm[sel, :].astype(np.float64); assert np.abs(S).max() < 2.0 ** 53
        acc = acc + S.astype(np.int64).astype(object) * (1 << int(g - m0))
    assert obits(acc) <= 62
    TN[l] = acc.astype(np.int64); M0[l] = m0; WEM[l] = wem
    # one power of two for the whole turn: 2^(m0 + min we_m - 1)
    TN1_bits.append(obits(acc * np.array([1 << int(x - wem.min()) for x in wem], dtype=object)[None, :]))
    # the bias part (1/2) b_fc met with mproj: (b_fc @ wm)[j] x 2^(we_m[j] - 1)
    TB_N[l] = exact_mm(bf[None, :].astype(np.int64), wm)[0]; TB_E[l] = wem - 1
bits = [int(np.abs(TN[l]).max()).bit_length() for l in range(12)]
print(f"1. the 12 fixed turns T_l = TN_l[i, j] x 2^(m0_l + we_m[j] - 1), TN_l whole numbers (int64):")
for l in range(12):
    print(f"   layer {l:2d}: m0 = {int(M0[l])}, fc column powers spread {spread[l]} places, mproj column powers"
          f" {int(WEM[l].min())}..{int(WEM[l].max())};  |TN| up to {bits[l]} places + sign; on ONE power of two"
          f" 2^({int(M0[l] + WEM[l].min() - 1)}): {TN1_bits[l]} places + sign;  nonzero entries {int((TN[l] != 0).sum()):,}")
nb = sum(768 * 768 * (b + 1) for b in bits)
print(f"   12 turns: {12 * 768 * 768:,} whole numbers; written at each layer's own width (places + sign):"
      f" {nb / 8 / 1e6:.1f} MB; at 64 bits: {12 * 768 * 768 * 8 / 1e6:.1f} MB.  One layer's fc + mproj plates:"
      f" {2 * 768 * 3072:,} weights x 16 bits = {2 * 768 * 3072 * 2 / 1e6:.1f} MB (12 layers {12 * 2 * 768 * 3072 * 2 / 1e6:.1f} MB)", flush=True)
np.savez(f"{OUTD}/turns.npz", TN=TN, m0=M0, we_m=WEM,
         bias_num=np.array([[int(v) for v in TB_N[l]] for l in range(12)], dtype=np.int64)
         if obits(TB_N) <= 62 else TB_N, bias_exp=TB_E,
         note=np.array("T_l[i,j] = TN[l,i,j] * 2^(m0[l] + we_m[l,j] - 1): maps the stream (units of 2^-16) to the stream; "
                       "applied as g_p * (R_p @ T_l) with g_p the water level between fc and the bend. "
                       "bias part (1/2) b_fc @ mproj = bias_num[l,j] * 2^bias_exp[l,j]."))
print(f"   saved {OUTD}/turns.npz  (bias part fits int64: {obits(TB_N) <= 62})  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)

# ---------------------------------------------------------------- 2. the identity, exact, on 3 texts
print("2. (straight half of every bend) met with mproj == g x (light met with T_l) + g x (light's rounding met with T_l)"
      " + (1/2)(fc out-rounding + fc bias) met with mproj   [exact, whole numbers on a common power of two]")
ok_all = True; n_ent = 0; n_ok = 0
for i in range(3):
    rec = record(TEXTS[i]); P = rec["P"]; res = residues(rec)
    for l in range(12):
        Ly = rec["layers"][l]; fc = Ly["fc"]; wm, wem, bm = W[f"mproj.{l}"]; wm = wm.astype(np.int64)
        wf, wef, bf = W[f"fc.{l}"]
        H, R, x16, sh, F, pile = fc["out"], fc["R"], fc["x16"], fc["sh"], fc["F"], fc["pile"]
        assert (Ly["bend"]["H"] == H).all()
        LS = fc["LS"]; f = (-LS) & 0xFFFF; kq = (-LS) >> 16
        assert ((F & 0xFFFF) == f[:, None]).all() and ((F >> 16) == wef[None, :] + sh[:, None] + kq[:, None]).all()
        # left side: (H/2) met with mproj: (H @ wm)[p, j] x 2^(we_m[j] - 1)
        LHS = exact_mm(H, wm); LHS_e = np.broadcast_to(wem - 1, (P, 768))
        # light met with T_l, with the gauge: EXP2[f_p] x (R_p @ TN)[j] x 2^(kq_p - 16 + m0 + we_m[j] - 1)
        RT = exact_mm(R, TN[l]) * O(EXP2[f])[:, None]; RT_e = (kq - 16 + M0[l])[:, None] + (wem - 1)[None, :]
        rho = (x16 << sh[:, None]) - R                                   # the light's 16-place rounding, in stream units
        assert (np.abs(rho) <= (1 << sh[:, None]) // 2).all()
        QT = exact_mm(rho, TN[l]) * O(EXP2[f])[:, None]
        # fc's out-rounding, (numerator, exponent) per (p, n), brought to one power per position, met with mproj / 2
        on, oe = res[0][l]["fc_out"]; oe = np.broadcast_to(np.asarray(oe), on.shape); emin = oe.min(axis=1)
        onP = on * np.vectorize(lambda z: 1 << int(z), otypes=[object])(oe - emin[:, None])
        OR = exact_mm(onP, wm); OR_e = emin[:, None] + (wem - 1)[None, :]
        BI = np.broadcast_to(TB_N[l], (P, 768)); BI_e = np.broadcast_to(TB_E[l], (P, 768))
        terms = [(LHS, LHS_e), (RT, RT_e), (QT, RT_e), (OR, OR_e), (BI, BI_e)]
        E0 = min(int(np.min(e)) for _, e in terms)
        def at0(n, e): return n * np.vectorize(lambda z: 1 << int(z), otypes=[object])(np.asarray(e) - E0)
        L0, A0, B0, C0, D0 = [at0(n, e) for n, e in terms]
        eq = (L0 == A0 + B0 + C0 + D0)
        n_ent += eq.size; n_ok += int(eq.sum()); ok_all &= bool(eq.all())
        if i == 0 and l in (0, 5, 11):
            p, j = P - 1, 447
            print(f"   text 0 layer {l:2d}, position {p}, line {j}, every term over 2^{-E0}:\n"
                  f"      straight halves met with mproj       {L0[p, j]}\n"
                  f"      g x light met with T_l               {A0[p, j]}\n"
                  f"      g x light's rounding met with T_l    {B0[p, j]}\n"
                  f"      (1/2) fc out-rounding met with mproj {C0[p, j]}\n"
                  f"      (1/2) fc bias met with mproj         {D0[p, j]}\n"
                  f"      sum of the four - left side          {A0[p, j] + B0[p, j] + C0[p, j] + D0[p, j] - L0[p, j]}")
    print(f"   text {i} {''.join(word(x) for x in rec['ids'])!r}: all {12 * P * 768:,} entries equal: "
          f"{n_ok == n_ent}  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
print(f"   identity exact at {n_ok:,} of {n_ent:,} entries (3 texts x 12 layers x 5 positions x 768 lines): "
      f"{'EXACT' if ok_all else 'NOT EXACT'}")
print(f"({time.time() - t0:.0f}s, max {rss()} MB)")
