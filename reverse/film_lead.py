"""THE LEAD'S REVERSE, shared parts for the film_ scripts. Nothing in machine.py or base2.py is changed; this file only
reuses them.

The lead = the answer's score minus the 2nd word's score (the machine's own whole numbers at the last position).
At the top, each word's row carries its own dimming (EXP2[f_w] x 2^(I_w - 16)); the lead's row is the answer's dimmed
row minus the 2nd word's dimmed row, one row of whole numbers over one power of two. It is sent back down every step
exactly as machine.py's reverse_exact and reverse16 send a word's row (their steps are repeated here line for line, only
the start row differs, and the light is written down at each layer's bend output on the way). The copies are checked
against machine.py's own reverse_exact / reverse16 started from single words: same whole numbers, same constants.

At layer l, after the out-bank of the bend's path (mproj) has been reversed, the light stands at the bend outputs of
every position: Gbo[p, n]. With every dimming held, the lead is then exactly
      lead = sum over p, n of Gbo[p, n] x Bout_l[p, n]  +  G . Rmid_l  +  c_now
where Rmid_l is the stream after layer l's arrival, G the light on the stream above layer l, and c_now every bias and
rounding met so far (the top's standing amounts and rounding, every step above layer l, and mproj.l's bias, out
rounding and in rounding). The newest column's part is the p = t term:
      part_l = sum over n of part_n,   part_n = Gbo[t, n] x Bout_l[t, n].
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
from fractions import Fraction
from machine import *
from machine import _addc, _addi, _hcols

def stream_after(rec, l): return rec["layers"][l + 1]["R"] if l < 11 else rec["R12"]

# ---------------------------------------------------------------- top rows
def top_exact(rec, words, lead=False):
    """(Ntop object (nw, 768), k0, c list of Fractions): the top's dimmed rows as reverse_exact builds them; with
    lead=True a single row, words[0]'s minus words[1]'s, and c the difference of their constants"""
    tp = rec["top"]; words = np.asarray(words)
    rn, re_ = top_residue(rec, words)
    c = [Fraction(int(b)) + Fraction(int(n)) * Fraction(2) ** int(e) for b, n, e in zip(HEADB[words], rn, re_)]
    F = (HEADE[words] + tp["sh"]) * 65536 - tp["LS"][0]; I, f = F >> 16, F & 0xFFFF
    k0 = int((I - 16).min())
    rows = [O(HEADW[:, w].astype(np.int64)) * int(EXP2[f[i]]) * (1 << int(I[i] - 16 - k0)) for i, w in enumerate(words)]
    if lead: return np.stack([rows[0] - rows[1]]), k0, [c[0] - c[1]]
    return np.stack(rows), k0, c

def top16(rec, words, lead=False):
    """(Ntop object (nw, 768) unkept whole numbers, etop (nw,), c whole numbers at 2^EFIX) as reverse16 builds them;
    with lead=True one row: words[0]'s minus words[1]'s on their common power of two"""
    tp = rec["top"]; words = np.asarray(words)
    rn, re_ = top_residue(rec, words)
    c = [(int(b) << -EFIX) + (int(n) << (int(x) - EFIX)) for b, n, x in zip(HEADB[words], rn, re_)]
    F = (HEADE[words] + tp["sh"]) * 65536 - tp["LS"][0]; I, f = F >> 16, F & 0xFFFF
    rows = [O(HEADW[:, w].astype(np.int64)) * int(EXP2[f[i]]) for i, w in enumerate(words)]; ew = [int(x) - 16 for x in I]
    if lead:
        T = min(ew); row = rows[0] * (1 << (ew[0] - T)) - rows[1] * (1 << (ew[1] - T))
        return np.stack([row]), np.array([T], np.int64), [c[0] - c[1]]
    return np.stack(rows), np.array(ew, np.int64), c

def keep32_obj(Ntop, etop):
    """machine.py's keep(N, e, 32) on rows of Python whole numbers: top 1 of the OR of the magnitudes, shift so 31 places
    remain, rounded with the machine's rshift (add half, floor shift)"""
    out = np.zeros(Ntop.shape, np.int64); e = etop.copy()
    for i in range(Ntop.shape[0]):
        top = max(abs(int(x)).bit_length() for x in Ntop[i]); sh = max(top - 31, 0)
        out[i] = [((int(x) + (1 << (sh - 1))) >> sh) if sh > 0 else int(x) for x in Ntop[i]]; e[i] += sh
    return out, e

# ---------------------------------------------------------------- exact (reverse_exact's steps)
def rev_exact_from(rec, Ntop, k0, c, res, cap=None):
    lres, tres = res; P, t = rec["P"], rec["t"]; tp = rec["top"]; nw = Ntop.shape[0]
    N = np.zeros((nw, P, 768), dtype=object); N[:] = 0; N[:, t, :] = Ntop
    G = BackExact(N, k0)
    xn, xe = tres["x"]; z = np.zeros((P, 768), dtype=object); z[:] = 0; z[t] = xn
    ze = np.zeros((P, 768), np.int64); ze[t] = xe
    c = [a + b for a, b in zip(c, G.meet(z, ze))]
    G = BackExact(G.N, G.e - tp["sh"])
    for l in range(11, -1, -1):
        Ly, rs = rec["layers"][l], lres[l]
        if cap: cap("in", l, G=G, c=list(c))
        mp = Ly["mproj"]; w, we, b = W[mp["name"]]
        c = _addc(c, G.meet(O(b)[None, :] * np.ones((P, 1), object) * 1, np.zeros((P, 768), np.int64)))
        c = _addc(c, G.meet(*rs["mproj_out"]))
        Gp = G.dim(np.ones((P, 768), np.int64), -mp["s"])
        Gx = Gp.plate(w.T)
        c = _addc(c, Gx.meet(*rs["mproj_x"]))
        Gx = Gx.dim(np.ones((P, 3072), np.int64), -mp["sh"][:, None] * np.ones((1, 3072), np.int64))
        if cap: cap("bo", l, Gbo=Gx, G=G, c=list(c))
        bd = Ly["bend"]
        c = _addc(c, Gx.meet(*rs["bend_out"]))
        Gh = Gx.dim(bd["share"], np.full((P, 3072), -16, np.int64))
        fc = Ly["fc"]; w, we, b = W[fc["name"]]
        c = _addc(c, Gh.meet(O(b)[None, :] * np.ones((P, 1), object), np.zeros((P, 3072), np.int64)))
        c = _addc(c, Gh.meet(*rs["fc_out"]))
        I_, f_ = fc["F"] >> 16, fc["F"] & 0xFFFF
        Gq = Gh.dim(EXP2[f_], I_ - 16).plate(w.T)
        c = _addc(c, Gq.meet(*rs["fc_x"]))
        Gq = Gq.dim(np.ones((P, 768), np.int64), -fc["sh"][:, None] * np.ones((1, 768), np.int64))
        G = G + Gq
        pj = Ly["proj"]; w, we, b = W[pj["name"]]
        c = _addc(c, G.meet(O(b)[None, :] * np.ones((P, 1), object), np.zeros((P, 768), np.int64)))
        c = _addc(c, G.meet(*rs["proj_out"]))
        Ga = G.dim(np.ones((P, 768), np.int64), -pj["s"]).plate(w.T)
        c = _addc(c, Ga.meet(*rs["proj_x"]))
        Ga = Ga.dim(np.ones((P, 768), np.int64), -pj["sh"][:, None] * np.ones((1, 768), np.int64))
        ar = Ly["arr"]
        c = _addc(c, Ga.meet(*rs["arr_out"]))
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
        c = _addc(c, Gv.meet(bq, np.zeros((P, 768), np.int64)))
        on, oe = rs["qkv_out"]; c = _addc(c, Gv.meet(on[:, 1536:], np.asarray(oe)[:, 1536:]))
        I_, f_ = q["F"][:, 1536:] >> 16, q["F"][:, 1536:] & 0xFFFF
        Gr = Gv.dim(EXP2[f_], I_ - 16).plate(np.ascontiguousarray(w[:, 1536:].T))
        c = _addc(c, Gr.meet(*rs["qkv_x"]))
        Gr = Gr.dim(np.ones((P, 768), np.int64), -q["sh"][:, None] * np.ones((1, 768), np.int64))
        G = G + Gr
    return G, c

# ---------------------------------------------------------------- 16 places (reverse16's steps)
def rev16_from(rec, Ntop, etop, c, kap, res, cap=None):
    lk, tres = kap[0], kap[1]; P, t = rec["P"], rec["t"]; tp = rec["top"]; nw = Ntop.shape[0]
    Nk, ek = keep32_obj(Ntop, etop)
    N = np.zeros((nw, P, 768), np.int64); e = np.zeros((nw, P), np.int64); N[:, t, :] = Nk; e[:, t] = ek
    xn, xe = tres["x"]; z = np.zeros((P, 768), dtype=object); z[:] = 0; z[t] = xn
    ze = np.zeros((P, 768), np.int64); ze[t] = xe
    KN, Ke = kappa32(z, ze); c = [a + b for a, b in zip(c, meet16(N, e, KN, Ke))]
    e = e.copy(); e[:, t] -= tp["sh"]
    G = (N, e)
    for l in range(11, -1, -1):
        Ly, kk = rec["layers"][l], lk[l]
        if cap: cap("in", l, G=G, c=list(c))
        mp = Ly["mproj"]; w = W[mp["name"]][0]
        c = _addi(c, meet16(*G, *kk["mproj"]))
        Gp = dim16(*G, np.ones((P, 768), np.int64), -mp["s"])
        Gx = (plate16(Gp[0], w.T), Gp[1])
        c = _addi(c, meet16(*Gx, *kk["mproj_x"]))
        Gx = (Gx[0], Gx[1] - mp["sh"][None, :])
        if cap: cap("bo", l, Gbo=Gx, Gp=Gp, G=G, c=list(c))
        c = _addi(c, meet16(*Gx, *kk["bend"]))
        bd = Ly["bend"]; Gh = dim16(*Gx, bd["share"], np.full((P, 3072), -16, np.int64))
        fc = Ly["fc"]; w = W[fc["name"]][0]
        c = _addi(c, meet16(*Gh, *kk["fc"]))
        Gq = dim16(*Gh, EXP2[fc["F"] & 0xFFFF], (fc["F"] >> 16) - 16)
        Gq = (plate16(Gq[0], w.T), Gq[1])
        c = _addi(c, meet16(*Gq, *kk["fc_x"]))
        Gq = (Gq[0], Gq[1] - fc["sh"][None, :])
        G = add16(G, Gq)
        pj = Ly["proj"]; w = W[pj["name"]][0]
        c = _addi(c, meet16(*G, *kk["proj"]))
        Gp = dim16(*G, np.ones((P, 768), np.int64), -pj["s"])
        Ga = (plate16(Gp[0], w.T), Gp[1])
        c = _addi(c, meet16(*Ga, *kk["proj_x"]))
        Ga = (Ga[0], Ga[1] - pj["sh"][None, :])
        c = _addi(c, meet16(*Ga, *kk["arr"]))
        ar = Ly["arr"]; Fa = _hcols(-ar["L"])
        Gn = keep(*Ga, 16); Gn = dim16(*Gn, EXP2[Fa & 0xFFFF], (Fa >> 16) - 16)
        Nn, en = Gn
        Vn = np.zeros((nw, P, 768), np.int64); ev = np.zeros((nw, P), np.int64)
        for j in range(P):
            lit = (np.abs(Nn[:, j:, :]).max(axis=2) > 0)
            ej = np.where(lit, en[:, j:], np.iinfo(np.int64).min // 4)
            T = np.maximum(np.where(lit, en[:, j:], np.iinfo(np.int64).max // 4).min(axis=1), ej.max(axis=1) - 14)
            T = np.where(lit.any(axis=1), T, 0)
            acc = np.zeros((nw, 768), np.int64)
            for h in range(12):
                cs = slice(h * 64, (h + 1) * 64)
                for p in range(j, P):
                    a = int(ar["A"][h, p, j])
                    if a: acc[:, cs] += rshift(Nn[:, p, cs], (T - en[:, p])[:, None]) * a
            Vn[:, j, :] = acc; ev[:, j] = T
        Gv = keep(Vn, ev, 16)
        q = Ly["qkv"]; w = W[q["name"]][0]
        c = _addi(c, meet16(*Gv, *kk["qkv"]))
        Gr = dim16(*Gv, EXP2[q["F"][:, 1536:] & 0xFFFF], (q["F"][:, 1536:] >> 16) - 16)
        Gr = (plate16(Gr[0], np.ascontiguousarray(w[:, 1536:].T)), Gr[1])
        c = _addi(c, meet16(*Gr, *kk["qkv_x"]))
        Gr = (Gr[0], Gr[1] - q["sh"][None, :])
        G = add16(G, Gr)
    return G, c

# ---------------------------------------------------------------- exact meetings in Python whole numbers
def dot_int(a, b):
    """sum of a x b over all entries, exact Python whole number (a, b int64 or object arrays of one shape)"""
    return int((np.asarray(a).astype(object) * np.asarray(b).astype(object)).sum())

def meet16_frac(N, e, X):
    """(N x 2^e) met with X, N (P, n) int64, e (P,), X (P, n) whole numbers: exact Fraction"""
    return sum((Fraction(dot_int(N[p], X[p])) * Fraction(2) ** int(e[p]) for p in range(N.shape[0])), Fraction(0))

def token_alone(rec):
    """the token-alone column (swap_alone.py's construction): position t's arrival over only itself at every layer,
    every other dimming set by the light; returns that run's record"""
    t = rec["t"]; L24 = int(log2fix([1 << 24])[0]); arr = []
    for Ly in rec["layers"]:
        Aa, La = Ly["arr"]["A"].copy(), Ly["arr"]["L"].copy()
        Aa[:, t, :] = 0; Aa[:, t, t] = 1 << 24; La[:, t] = L24
        arr.append(dict(arr=dict(A=Aa, L=La)))
    return record(rec["ids"], held=dict(layers=arr))
