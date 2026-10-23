"""THE TWO HALVES OF EVERY BEND, CARRIED TO THE ANSWER: the exact split of the lead (and of the score) into what came
through the straight halves (the fixed turns) and what came through the symmetric halves.

HIS PICTURE (his words): "The edge is the same backwards and forwards. That's how we can collapse it. We could even make 49
little sections and just record 49 little turns. And have a tiny little turning model." And: "That tiny collapsed model is
the fringe of the original token AND every token previously in the sentence! And it's still 1.2 mb".
Restated: every bend output is exactly bend(H) = H/2 + e(H). The straight half H/2 is half-silvered glass at every neuron,
the same for every token; with the water level held it folds fc and mproj into one fixed turn T_l per layer (turns.py,
exact at 138,240 of 138,240 entries). The symmetric half e(H) = bend(H) - H/2 sees only the size of the pile.

WHAT IS MEASURED (measurement only; base2.py is executed from its own source through reverse/machine.py; no file is
edited; every number below is an exact rational, whole numbers over powers of two):
  STEP 2, layer by layer (reverse/film_split.py's split, one level finer). With every dimming held, the lead (answer minus
     2nd word) sent back down every step stands at layer l's bend outputs as the light Gbo (film_lead.py). The part of the
     lead through layer l's newest bends is sum_n Gbo[t, n] x Bout[t, n]; with Bout = H/2 + e it splits exactly into
        (a) straight:  sum_n Gbo[t, n] x H[t, n] / 2        (b) symmetric:  sum_n Gbo[t, n] x e[t, n]
     checked: (a) + (b) == the part, per neuron and in sum, and part + the earlier positions' bends + the stream + every
     constant so far == the lead, digit for digit. Also the split of the write of each layer into the stream:
        mproj's output = (H/2) met with mproj + e met with mproj + mproj's two roundings + its bias  (exact, every line).
  STEP 3, all layers at once. Send the lead back with every bend replaced by its straight half (the light goes back
     through 1/2 at every neuron, every position: the 12 fixed turns, each with its water level) and every symmetric half
     held at the run's own value, met with the light at its bend output on the way. Then exactly
        lead = (the tokens' entry numbers met with the fixed machine's reverse light)       [fixed turns + arrivals + stream]
             + sum over layers and positions of (symmetric half met with the light there)   [the symmetric halves]
             + every bias and rounding (fc's bias through the fixed turns written apart)
     and, cutting the bend path altogether ("cut"), the entry numbers' part through the stream and the arrivals alone,
     so (fixed machine part - cut part) = what the entry light carries through at least one fixed turn.
     Rows: the lead (answer - 2nd), a random word's lead (answer - random word), and the answer's score itself.
  STEP 4, controls: the random word's lead (row 2 above); the symmetric halves' values shuffled among each layer's
     neurons (each position apart; same values, pairing broken) and met with the same light. THE SHUFFLE CHANGES VALUES:
     the shuffled rows do not add up to the lead and are labelled so. Order-free: every sum of whole-number parts is
     also taken in a random order (identical by construction).
  The 16-place style (reverse16's steps, film_lead.rev16_from) gives step 2 on all 30 texts as well; its gap to the
  exact split is printed.

GATING (holographicplate CLAUDE.md, answered before writing): (1) it measures how the lead is carried by the two halves
of the bend (an exposure split into a fixed half and a symmetric half) and by the fixed turns; (2) every read is on
whole numbers; (3) exact, digit for digit (the shuffle control excepted, labelled); (4) nothing discarded; (5) nothing
fitted, no word-count comparison; (6) not in FINDINGS.md or NOT_TO_RERUN.md (section 10 splits by neuron, not by half).

PREDICTIONS (written before running):
  The task's (the orchestrating script's): the straight halves carry a large part of each write's size, but the lead --
    what separates the answer from the runner-up -- sits mostly in the symmetric halves at most layers, because the
    gates carry the text (page shuffle, newest film). The fixed turns alone carry under a third of the lead.
  Claude's (this session), from the relation itself: per neuron |e(H)| = |H| x |Phi(H) - 1/2| <= |H|/2 (up to the table's
    half-unit roundings), with near-equality for fully open or shut neurons and e ~ 0.4 H^2 near the edge. So
    (i) in gross (sum over neurons of |part|), the straight halves carry MORE than half at every layer of every text,
        55-70% typically;
    (ii) for a shut neuron (H << 0) the two halves cancel (H/2 + |H|/2 = 0), so the two signed parts of the lead are
        each larger than their sum and of opposite sign at most layers; neither half "holds" the lead alone;
    (iii) the write: both parts of mproj's output larger than the write; the symmetric part at least as large as the
        straight part, because every e >= 0 adds along the out-plate's shared lines (line 447 etc.);
    (iv) all layers at once: the fixed machine's part of the lead and the symmetric halves' part are each several leads
        (median |.| >= 2 leads) and opposite in sign; signed shares scattered, of both signs across texts;
    (v) shuffle: the shuffled symmetric part differs from the real one by about the real one's own size (pairing matters);
        the random word's lead (mostly the common-word floor) puts a larger signed share on the fixed machine than the
        runner-up lead does.
    python3 turns/halves.py I0 I1   -> turns/halves_part_I0_I1.npz (texts I0..I1-1); turns/halves_report.py prints the tables
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; checked exact
from film_lead import *
from machine import _addc, _hcols
from fractions import Fraction
import resource
OUTD = "/Users/abundancemachine/streaming-machine/turns"
t0 = time.time()
NT = len(TEXTS)
I0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # this run's texts: I0 .. I1 - 1 (all done in unrounded whole numbers)
I1 = int(sys.argv[2]) if len(sys.argv) > 2 else NT
NX = NT
DIGITS = (0, 1, 2)                                      # texts printed digit for digit
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1 << 20)
def F2(n, e): return Fraction(int(n)) * Fraction(2) ** int(e)
BK = ("top", "mproj", "bend", "fc_bias", "fc_round", "arrival")

# ---------------------------------------------------------------- the reverse, with the bend step chosen
def rev_modes(rec, Ntop, k0, c0, res, mode, on_bend=None, seed=0):
    """film_lead.rev_exact_from's steps (machine.py's reverse_exact), line for line, with the bend step chosen:
       held      the bend's share as the run had it (identical to rev_exact_from: checked);
       straight  the light goes back through the straight half only (x 1/2 at every neuron and position); the
                 symmetric half e = Bout - H/2 (the bend's rounding inside it) is met with the light at the bend output;
       cut       nothing goes back through the bend; the whole bend output is met with the light there.
    Returns G (light at the entry numbers), C (bucket -> per-row Fractions), INJ (per layer 0..11: per-row Fractions;
    'all' every position, 'new' the newest position, 'shuf' (straight only) the symmetric halves shuffled among neurons)."""
    lres, tres = res; P, t = rec["P"], rec["t"]; tp = rec["top"]; nw = Ntop.shape[0]
    C = {k: [Fraction(0)] * nw for k in BK}; INJ = dict(all=[None] * 12, new=[None] * 12, shuf=[None] * 12)
    N = np.zeros((nw, P, 768), dtype=object); N[:] = 0; N[:, t, :] = Ntop
    G = BackExact(N, k0)
    xn, xe = tres["x"]; z = np.zeros((P, 768), dtype=object); z[:] = 0; z[t] = xn
    ze = np.zeros((P, 768), np.int64); ze[t] = xe
    C["top"] = _addc(list(c0), G.meet(z, ze))
    G = BackExact(G.N, G.e - tp["sh"])
    for l in range(11, -1, -1):
        Ly, rs = rec["layers"][l], lres[l]
        mp = Ly["mproj"]; w, we, b = W[mp["name"]]
        C["mproj"] = _addc(C["mproj"], G.meet(O(b)[None, :] * np.ones((P, 1), object) * 1, np.zeros((P, 768), np.int64)))
        C["mproj"] = _addc(C["mproj"], G.meet(*rs["mproj_out"]))
        Gp = G.dim(np.ones((P, 768), np.int64), -mp["s"])
        Gx = Gp.plate(w.T)
        C["mproj"] = _addc(C["mproj"], Gx.meet(*rs["mproj_x"]))
        Gx = Gx.dim(np.ones((P, 3072), np.int64), -mp["sh"][:, None] * np.ones((1, 3072), np.int64))
        bd = Ly["bend"]; H, Bout = bd["H"], bd["out"]
        if on_bend: on_bend(l, Gx, G, C)
        if mode == "held":
            C["bend"] = _addc(C["bend"], Gx.meet(*rs["bend_out"]))
            Gh = Gx.dim(bd["share"], np.full((P, 3072), -16, np.int64))
        elif mode == "straight":
            Y2 = 2 * Bout - H                                              # 2 e, whole numbers
            INJ["all"][l] = Gx.meet(Y2, -1)
            Yn = np.zeros_like(Y2); Yn[t] = Y2[t]; INJ["new"][l] = Gx.meet(Yn, -1)
            rg = np.random.default_rng(seed * 100 + l)
            Ys = np.stack([Y2[p, rg.permutation(3072)] for p in range(P)]); INJ["shuf"][l] = Gx.meet(Ys, -1)
            Gh = Gx.dim(np.ones((P, 3072), np.int64), np.full((P, 3072), -1, np.int64))
        else:
            INJ["all"][l] = Gx.meet(Bout, 0)
            Bn = np.zeros_like(Bout); Bn[t] = Bout[t]; INJ["new"][l] = Gx.meet(Bn, 0)
            Gh = None
        if Gh is not None:
            fc = Ly["fc"]; w, we, b = W[fc["name"]]
            C["fc_bias"] = _addc(C["fc_bias"], Gh.meet(O(b)[None, :] * np.ones((P, 1), object), np.zeros((P, 3072), np.int64)))
            C["fc_round"] = _addc(C["fc_round"], Gh.meet(*rs["fc_out"]))
            I_, f_ = fc["F"] >> 16, fc["F"] & 0xFFFF
            Gq = Gh.dim(EXP2[f_], I_ - 16).plate(w.T)
            C["fc_round"] = _addc(C["fc_round"], Gq.meet(*rs["fc_x"]))
            Gq = Gq.dim(np.ones((P, 768), np.int64), -fc["sh"][:, None] * np.ones((1, 768), np.int64))
            G = G + Gq
        pj = Ly["proj"]; w, we, b = W[pj["name"]]
        C["arrival"] = _addc(C["arrival"], G.meet(O(b)[None, :] * np.ones((P, 1), object), np.zeros((P, 768), np.int64)))
        C["arrival"] = _addc(C["arrival"], G.meet(*rs["proj_out"]))
        Ga = G.dim(np.ones((P, 768), np.int64), -pj["s"]).plate(w.T)
        C["arrival"] = _addc(C["arrival"], Ga.meet(*rs["proj_x"]))
        Ga = Ga.dim(np.ones((P, 768), np.int64), -pj["sh"][:, None] * np.ones((1, 768), np.int64))
        ar = Ly["arr"]
        C["arrival"] = _addc(C["arrival"], Ga.meet(*rs["arr_out"]))
        Fa = _hcols(-ar["L"]); Gn = Ga.dim(EXP2[Fa & 0xFFFF], (Fa >> 16) - 16)
        Vn = np.zeros_like(Gn.N); Vn[:] = 0
        for h in range(12):
            cs = slice(h * 64, (h + 1) * 64)
            for j in range(P):
                acc = 0
                for p in range(j, P): acc = acc + Gn.N[:, p, cs] * int(ar["A"][h, p, j])
                Vn[:, j, cs] = acc
        Gv = BackExact(Vn, Gn.e)
        q = Ly["qkv"]; w, we, b = W[q["name"]]
        bq = O(b[1536:])[None, :] * np.ones((P, 1), object)
        C["arrival"] = _addc(C["arrival"], Gv.meet(bq, np.zeros((P, 768), np.int64)))
        on, oe = rs["qkv_out"]; C["arrival"] = _addc(C["arrival"], Gv.meet(on[:, 1536:], np.asarray(oe)[:, 1536:]))
        I_, f_ = q["F"][:, 1536:] >> 16, q["F"][:, 1536:] & 0xFFFF
        Gr = Gv.dim(EXP2[f_], I_ - 16).plate(np.ascontiguousarray(w[:, 1536:].T))
        C["arrival"] = _addc(C["arrival"], Gr.meet(*rs["qkv_x"]))
        Gr = Gr.dim(np.ones((P, 768), np.int64), -q["sh"][:, None] * np.ones((1, 768), np.int64))
        G = G + Gr
    return G, C, INJ

def csum(C, w): return sum((C[k][w] for k in BK), Fraction(0))

import machine as _machine
_plate_whole = _machine.BackExact.plate
def _plate_by_word(s, wT):
    """machine.py's BackExact.plate, run on one word's row at a time (the same exact whole numbers; peak memory a third)"""
    return _machine.BackExact(np.concatenate([_plate_whole(_machine.BackExact(s.N[k:k + 1], s.e), wT).N
                                              for k in range(s.N.shape[0])], axis=0), s.e)
_machine.BackExact.plate = _plate_by_word

# ---------------------------------------------------------------- the write of each layer, split (forward, exact)
def limbs16(X):
    """whole numbers (int64) -> list of float64 arrays L_k, |L_k| < 2^16, X = sum_k L_k 2^(16k)"""
    X = np.asarray(X).astype(np.int64); sg = np.sign(X); A = np.abs(X); out = []
    while A.any(): out.append(((A & 0xFFFF) * sg).astype(np.float64)); A = A >> 16
    return out or [np.zeros(X.shape)]

def exact_mm(A, B):
    """A @ B for int64 whole-number matrices, exact (object result), by 16-place limbs: each float sum < 2^53"""
    out = None
    for a, La in enumerate(limbs16(A)):
        for b_, Lb in enumerate(limbs16(B)):
            S_ = La @ Lb; assert np.abs(S_).max() < 2.0 ** 53
            S_ = S_.astype(np.int64).astype(object) * (1 << (16 * (a + b_)))
            out = S_ if out is None else out + S_
    return out

def write_split(rec, res):
    """per layer, per position: sizes of (H/2) met with mproj and e met with mproj (stream units), and the exact check
    S + E + mproj's in-rounding part + out-rounding + bias == mproj's output, every line"""
    P = rec["P"]; out = np.zeros((12, P, 6)); ok = True
    for l in range(12):
        Ly = rec["layers"][l]; bd, mp = Ly["bend"], Ly["mproj"]; wm, wem, bm = W[mp["name"]]
        H, Bout = bd["H"], bd["out"]; Y2 = 2 * Bout - H
        wmi = wm.astype(np.int64)
        SH = exact_mm(H, wmi); SY = exact_mm(Y2, wmi)                    # x 2^(we_m[j] - 1)
        rx = exact_mm((mp["x16"] << mp["sh"][:, None]) - Bout, wmi)      # x 2^we_m[j]
        on, oe = res[0][l]["mproj_out"]; oe = np.broadcast_to(np.asarray(oe), on.shape)
        for p in range(P):
            for j in range(768):
                lhs = F2(mp["out"][p, j] - bm[j], 0)
                rhs = F2(SH[p, j] + SY[p, j], int(wem[j]) - 1) + F2(rx[p, j], int(wem[j])) + F2(on[p, j], oe[p, j])
                if lhs != rhs: ok = False
            s = np.array([float(F2(SH[p, j], int(wem[j]) - 1)) for j in range(768)])
            y = np.array([float(F2(SY[p, j], int(wem[j]) - 1)) for j in range(768)])
            out[l, p] = [np.abs(s).sum(), np.abs(y).sum(), np.abs(s + y).sum(), np.sqrt((s * s).sum()), np.sqrt((y * y).sum()),
                         np.sqrt(((s + y) ** 2).sum())]
    return out, ok

# ---------------------------------------------------------------- the run
NL = 3                                                   # rows: lead (answer - 2nd), random word's lead, the answer's score
OUT = dict(ans=np.zeros(NT, np.int64), second=np.zeros(NT, np.int64), rnd=np.zeros(NT, np.int64), target=np.zeros((NT, NL)),
           # step 2 (exact, held): per text, layer, row (leads only): straight, sym, part, gross straight, gross sym,
           #   shuffled sym, shuffled straight, older positions' straight, older positions' sym
           s2=np.full((NT, 12, 2, 9), np.nan), s2_16=np.full((NT, 12, 2, 3), np.nan),
           # step 3 (exact): per row: fixed machine's entry part, symmetric halves newest, older, constants (6 buckets),
           #   cut: entry part, whole bend outputs newest, older, constants; shuffled symmetric (all positions)
           s3=np.full((NT, NL, 14), np.nan), s3_pos=np.full((NT, NL, 5), np.nan), s3_lay=np.full((NT, NL, 12), np.nan),
           write=np.zeros((NT, 12, 5, 6)), eneg=np.zeros(NT, np.int64), ebig=np.zeros(NT, np.int64))
checks = dict(plate=0, copy=0, copy_n=0, neuron=0, neuron_n=0, rest=0, rest_n=0, s3=0, s3_n=0, cut=0, cut_n=0, write=0, order=0,
              order_n=0, s16=0, s16_n=0)
for i in range(I0, I1):
    ids = TEXTS[i]; rec = record(ids); S = machine_scores(ids)
    assert (rec["top"]["scores"] == S).all() and race(S) == forward(ids)
    order = np.argsort(-S, kind="stable"); a, b = int(order[0]), int(order[1]); assert a == race(S)
    rg = np.random.default_rng(12345 + i); r = int(rg.integers(0, len(S)))
    while r in order[:5]: r = int(rg.integers(0, len(S)))
    P, t = rec["P"], rec["t"]; res = residues(rec)
    targets = [int(S[a]) - int(S[b]), int(S[a]) - int(S[r]), int(S[a])]
    OUT["ans"][i], OUT["second"][i], OUT["rnd"][i], OUT["target"][i] = a, b, r, targets
    # the symmetric half: e >= 0 and |e| <= |H|/2, per neuron, every layer and position (whole numbers: 2e = 2 Bout - H)
    for l in range(12):
        H, Bo = rec["layers"][l]["bend"]["H"], rec["layers"][l]["bend"]["out"]; Y2 = 2 * Bo - H
        OUT["eneg"][i] += int((Y2 < 0).sum()); OUT["ebig"][i] += int((np.abs(Y2) > np.abs(H) + 1).sum())
    wr, okw = write_split(rec, res); OUT["write"][i] = wr; checks["write"] += okw
    line = (f"  text {i:2d} {''.join(word(x) for x in ids)!r:40s} answer {word(a)!r:12s} 2nd {word(b)!r:10s} random {word(r)!r:14s}"
            f" lead {targets[0]:7d}  random lead {targets[1]:8d}")
    if i < NX:
        Ntop3, k0, c3 = top_exact(rec, [a, b, r])
        Ntop = np.stack([Ntop3[0] - Ntop3[1], Ntop3[0] - Ntop3[2], Ntop3[0]]); c0 = [c3[0] - c3[1], c3[0] - c3[2], c3[0]]
        # ---- step 2: held, split at each layer's newest bends
        def on_bend(l, Gx, G, C):
            E2 = Gx.e; Ly = rec["layers"][l]; H, Bout = Ly["bend"]["H"], Ly["bend"]["out"]
            if i == I0 and l == 11:
                wfT = W[f"fc.{l}"][0].T; A1, A2 = _plate_whole(Gx, wfT), _plate_by_word(Gx, wfT)
                checks["plate"] = int(A1.e == A2.e and all(int(x) == int(y) for x, y in zip(A1.N.ravel(), A2.N.ravel())))
            Hn, Bn, Yn = O(H[t]), O(Bout[t]), O(2 * Bout[t] - H[t])
            Bz = Bout.copy(); Bz[t] = 0; Hz = H.copy(); Hz[t] = 0; Yz = 2 * Bout - H; Yz[t] = 0
            earlier = Gx.meet(Bz, 0); mid = G.meet(Ly["Rmid"], np.zeros((P, 768), np.int64))
            so, yo = Gx.meet(Hz, -1), Gx.meet(Yz, -1)
            prm = np.random.default_rng(1000 * i + l).permutation(3072); po = np.random.default_rng(7 + l).permutation(3072)
            for w_ in range(2):
                gt = Gx.N[w_, t, :]
                sn, yn, pn = gt * Hn, gt * Yn, gt * Bn
                checks["neuron"] += bool(((sn + yn) == 2 * pn).all()); checks["neuron_n"] += 1
                Ss, Ys, Ps = int(sn.sum()), int(yn.sum()), int(pn.sum())
                checks["order"] += (int(sn[po].sum()) == Ss and int(yn[po].sum()) == Ys); checks["order_n"] += 1
                rest = earlier[w_] + mid[w_] + csum(C, w_)
                ex = (F2(Ps, E2) + rest == targets[w_]) and (Ss + Ys == 2 * Ps)
                checks["rest"] += ex; checks["rest_n"] += 1
                gs = sum(abs(int(v)) for v in sn); gy = sum(abs(int(v)) for v in yn)
                ys_, ss_ = int((gt * Yn[prm]).sum()), int((gt * Hn[prm]).sum())
                OUT["s2"][i, l, w_] = [float(F2(Ss, E2 - 1)), float(F2(Ys, E2 - 1)), float(F2(Ps, E2)), float(F2(gs, E2 - 1)),
                                       float(F2(gy, E2 - 1)), float(F2(ys_, E2 - 1)), float(F2(ss_, E2 - 1)), float(so[w_]), float(yo[w_])]
                if i in DIGITS and w_ == 0 and l != 11:
                    D = 1 - E2; restN = rest * (1 << D); assert restN.denominator == 1
                    print(f"     text {i} layer {l:2d}: over 2^{D}: straight {Ss.bit_length()} places, symmetric {Ys.bit_length()} places;"
                          f" straight + symmetric == 2 x part: {Ss + Ys == 2 * Ps}; + earlier + stream + constants == lead x 2^{D}:"
                          f" {Ss + Ys + int(restN) == targets[0] << D}", flush=True)
                if i in DIGITS and w_ == 0 and l == 11:
                    D = 1 - E2; restN = rest * (1 << D); assert restN.denominator == 1
                    print(f"     text {i} layer {l:2d}, the lead split at the newest bends, every term over 2^{D}:\n"
                          f"        straight halves  sum_n Gbo x H/2   = {Ss}\n"
                          f"        symmetric halves sum_n Gbo x e     = {Ys}\n"
                          f"        their sum (= the part, x2 places) = {Ss + Ys}   (2 x part = {2 * Ps}: {Ss + Ys == 2 * Ps})\n"
                          f"        + earlier positions + stream + constants so far = {int(restN)}\n"
                          f"        = {Ss + Ys + int(restN)}\n"
                          f"        lead x 2^{D}                     = {targets[0] << D}   equal: {Ss + Ys + int(restN) == targets[0] << D}", flush=True)
        Gh_, Ch_, _ = rev_modes(rec, Ntop, k0, c0, res, "held", on_bend)
        if i in DIGITS:                                  # the copy against film_lead's own reverse (machine.py's steps)
            Gf, cf = rev_exact_from(rec, Ntop, k0, c0, res)
            same = Gf.e == Gh_.e and all(int(x) == int(y) for x, y in zip(Gf.N.ravel(), Gh_.N.ravel())) and \
                all(cf[w_] == csum(Ch_, w_) for w_ in range(NL))
            checks["copy"] += same; checks["copy_n"] += 1
        # ---- step 3: straight and cut
        Gs_, Cs_, Is_ = rev_modes(rec, Ntop, k0, c0, res, "straight", seed=i)
        Gc_, Cc_, Ic_ = rev_modes(rec, Ntop, k0, c0, res, "cut")
        As, Ac = meet_exact(rec["R0"], Gs_), meet_exact(rec["R0"], Gc_)
        for w_ in range(NL):
            Bnew = sum((Is_["new"][l][w_] for l in range(12)), Fraction(0)); Ball = sum((Is_["all"][l][w_] for l in range(12)), Fraction(0))
            ok3 = As[w_] + Ball + csum(Cs_, w_) == targets[w_]; checks["s3"] += ok3; checks["s3_n"] += 1
            Cnew = sum((Ic_["new"][l][w_] for l in range(12)), Fraction(0)); Call = sum((Ic_["all"][l][w_] for l in range(12)), Fraction(0))
            okc = Ac[w_] + Call + csum(Cc_, w_) == targets[w_]; checks["cut"] += okc; checks["cut_n"] += 1
            Bsh = sum((Is_["shuf"][l][w_] for l in range(12)), Fraction(0))
            OUT["s3"][i, w_] = [float(As[w_]), float(Bnew), float(Ball - Bnew)] + [float(Cs_[k][w_]) for k in BK] + \
                               [float(Ac[w_]), float(Cnew), float(Call - Cnew), float(csum(Cc_, w_)), float(Bsh)]
            OUT["s3_lay"][i, w_] = [float(Is_["all"][l][w_]) for l in range(12)]
            pe = (Gs_.N[w_] * O(rec["R0"])).sum(axis=1)                  # per position: entry numbers met with the fixed machine's light
            OUT["s3_pos"][i, w_, :P] = [float(F2(int(v), Gs_.e)) for v in pe]
            if i in DIGITS and w_ == 0:
                print(f"     text {i} ALL LAYERS AT ONCE (lead {targets[0]}): fixed machine's entry part {float(As[0]):.1f}"
                      f" + symmetric halves {float(Ball):.1f} (newest {float(Bnew):.1f}) + constants {float(csum(Cs_, 0)):.1f}"
                      f" (fc bias through the turns {float(Cs_['fc_bias'][0]):.1f}) = {float(As[0] + Ball + csum(Cs_, 0)):.1f}:"
                      f" {'EXACT' if ok3 else 'DIFFERENT'};  cut: {float(Ac[0]):.1f} + {float(Call):.1f} + {float(csum(Cc_, 0)):.1f}:"
                      f" {'EXACT' if okc else 'DIFFERENT'}", flush=True)
                tot = As[0] + Ball + csum(Cs_, 0); den = tot.denominator
                print(f"        in whole numbers over {den.bit_length() - 1 if den > 1 else 0} places: fixed {As[0] * den}"
                      f" + symmetric {Ball * den} + constants {csum(Cs_, 0) * den} = {tot * den} = lead x {den}")
        line += f" | step 2 exact {checks['rest'] == checks['rest_n']} | step 3 exact {checks['s3'] == checks['s3_n']}, cut {checks['cut'] == checks['cut_n']}"
    # ---- step 2 in the 16-place style (reverse16's steps)
    kap = kappas(rec, res)
    N1, e1, c1 = top16(rec, [a, b], lead=True); N2, e2, c2 = top16(rec, [a, r], lead=True)
    caps = {}
    def cap(stage, l, **kw): caps[(stage, l)] = kw
    rev16_from(rec, np.concatenate([N1, N2]), np.concatenate([e1, e2]), c1 + c2, kap, res, cap)
    for l in range(12):
        Ly = rec["layers"][l]; H, Bout = Ly["bend"]["H"], Ly["bend"]["out"]
        Nb, eb = caps[("bo", l)]["Gbo"]
        for w_ in range(2):
            g = O(Nb[w_, t]); Ss, Ys, Ps = int((g * O(H[t])).sum()), int((g * O(2 * Bout[t] - H[t])).sum()), int((g * O(Bout[t])).sum())
            checks["s16"] += (Ss + Ys == 2 * Ps); checks["s16_n"] += 1
            OUT["s2_16"][i, l, w_] = [float(F2(Ss, eb[w_, t] - 1)), float(F2(Ys, eb[w_, t] - 1)), float(F2(Ps, eb[w_, t]))]
    print(line + f" | write split exact {okw} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)

OUT["checks"] = np.array([checks[k] for k in sorted(checks)], np.int64); OUT["check_names"] = np.array(sorted(checks))
OUT["done"] = np.zeros(NT, bool); OUT["done"][I0:I1] = True
np.savez(f"{OUTD}/halves_part_{I0:02d}_{I1:02d}.npz", **OUT)
print("\nCHECKS (texts %d-%d):" % (I0, I1 - 1), {k: int(v) for k, v in checks.items()})
print(f"  patched plate (one word at a time) == machine.py's plate: {checks['plate']};  per-neuron (a)+(b) == 2 x part:"
      f" {checks['neuron']}/{checks['neuron_n']};  part + earlier + stream + constants == lead, digit for digit:"
      f" {checks['rest']}/{checks['rest_n']};  copy == film_lead.rev_exact_from: {checks['copy']}/{checks['copy_n']}")
print(f"  all layers at once exact: {checks['s3']}/{checks['s3_n']} (straight), {checks['cut']}/{checks['cut_n']} (cut);  write split"
      f" exact: {checks['write']}/{I1 - I0};  order-free sums: {checks['order']}/{checks['order_n']};  16-place (a)+(b) == 2 x part:"
      f" {checks['s16']}/{checks['s16_n']}")
print(f"  the symmetric half: 2e < 0 at {int(OUT['eneg'].sum())} of {(I1 - I0) * 12 * 5 * 3072:,} neurons x positions;"
      f" |2e| > |H| + 1 at {int(OUT['ebig'].sum())}")
print(f"({time.time() - t0:.0f}s, max {rss()} MB)")
