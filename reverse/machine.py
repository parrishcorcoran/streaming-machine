"""THE REVERSE, shared parts. Every step of the base-2 machine run forward with every dimming it got written down,
then a word's row at the top sent back down every step to the embedding numbers.

The machine is base2.py, executed from its own source and never changed. Units are the machine's own: the stream,
the biases and the scores are whole numbers (one unit = 2^-16).

Forward, recorded (record): at every step the dimming the light got, held:
  water level   x16 = R x 2^-sh (to16, the machine's rounding), pile = x16 met with the plate (centering folded in the
                plate), out = pile x EXP2[f_p] x 2^(I_pn - 16) + bias.  The dimming per position is 1/size of the light
                (EXP2[f_p] x 2^floor(-LS_p / 65536)), per neuron the plate's own power of two.
  arrival       out_p = (sum over j <= p of a_pj x v_j) x EXP2[f'_p] x 2^(I'_p - 16): the share s_pj of position j's
                value; a_pj from the machine's arrival table, the total's log negated.  Only the value is on the path.
  plain bank    out = (x16 met with the plate) x 2^(we_n + sh_p) + bias.
  bend          out = pile x share / 2^16, the share read from the bend's table by where the pile arrives.
  top           score_w = (x16 met with word w's row) x EXP2[f] x 2^(I_w - 16) + the word's standing amount.
Every rounding the machine made is written down as an exact rational (whole number over a power of two): with every
dimming held, score_w = (embedding numbers) met with R_w + c_w, exactly, where c_w gathers every bias and every
rounding, each met with the reverse light at its own step.

Reverse (BackExact: Python whole numbers, one power of two per array, no rounding at all;
         Back16: the machine's own style, see Back16's notes).
"""
import sys, os, json, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv; os.chdir(_cwd)

refs = json.load(open(f"{HERE}/refs_many.json"))
TEXTS = [(r["ids"] + r["ref"])[:5] for r in refs]
LONG = [r["ids"] + r["ref"] for r in refs]
WTE = M["wte"]; WPE = M["wpe"]
HEADW, HEADE, HEADB = W["head"]                        # (768, 50257) float64 holding int16; per-word power; standing amount

_vocab = json.load(open(os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/"
                                           "607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json")))
_bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_cs = _bs[:]; _n = 0
for _b in range(256):
    if _b not in _bs: _bs.append(_b); _cs.append(256 + _n); _n += 1
_BYTE = {chr(c): b for b, c in zip(_bs, _cs)}
_INV = {v: k for k, v in _vocab.items()}
def word(i): return bytes(_BYTE[c] for c in _INV[int(i)]).decode("utf-8", "replace")

# ---------------------------------------------------------------- the untouched machine's scores
def machine_scores(ids, at=None):
    """base2.py's forward, its own lines, returning the whole-number score of every word at position `at` (last if None)"""
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]
    t = P - 1 if at is None else at
    x16, ex, LS = gauge_log(R[t:t + 1])
    return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b

# ---------------------------------------------------------------- exact residues (rational: numerator, exponent)
def O(a): return np.asarray(a).astype(object)
def res_shift(out, X, s):
    """out = rshift(X, s) (X x 2^-s rounded); residue out - X x 2^-s as (numerator, exponent), exact"""
    s = np.broadcast_to(np.asarray(s, np.int64), np.shape(out)); sp = np.maximum(s, 0)
    num = O(out) * O(np.left_shift(np.int64(1), sp)) - O(X) * O(np.left_shift(np.int64(1), np.maximum(-s, 0)))
    # when s <= 0 the shift is exact: numerator out - X x 2^-s must be 0
    return num, -sp

def res_power(out, X, F):
    """out = power(X, F) = rshift(X x EXP2[f], 16 - I); residue against X x EXP2[f] x 2^(I - 16), exact"""
    F = np.asarray(F, np.int64); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    s = 16 - I; sp = np.maximum(s, 0)
    num = O(out) * O(np.left_shift(np.int64(1), sp)) - O(X) * O(EXP2[f]) * O(np.left_shift(np.int64(1), np.maximum(-s, 0)))
    return num, -sp

def check_half(num, e):                                # |residue| <= 1/2: the machine's rounding is honest (nothing wrapped)
    e = np.broadcast_to(e, num.shape)
    for n_, x in zip(num.ravel(), e.ravel()):
        if abs(Fraction(int(n_)) * Fraction(2) ** int(x)) > Fraction(1, 2): return False
    return True

# ---------------------------------------------------------------- forward, recorded
def record(ids, at=None, held=None, check=True):
    """Run the machine on ids, recording every dimming. at: read position (last if None).
    held: another run's record -- then every dimming (water level sizes, bend shares, arrival shares) is taken from it
    instead of from this light (the machine's own arithmetic otherwise), 'B's dimmings with A's words'."""
    P = len(ids); t = P - 1 if at is None else at
    R = WTE[ids] + WPE[:P]; rec = dict(ids=list(ids), P=P, t=t, R0=R.copy(), layers=[])
    def gauged(R, name, hk):
        w, we, b = W[name]
        x16, ex, LS = gauge_log(R)
        if hk is not None: LS = hk["LS"]
        pile = product(x16, w); F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
        out = power(pile, F) + b
        return out, dict(name=name, R=R, x16=x16, sh=ex + ES, LS=LS, pile=pile, F=F, out=out)
    def plain(X, name):
        w, we, b = W[name]; x16, ex = to16(X, -ES); pile = product(x16, w); s = -(we[None, :] + ex[:, None] + ES)
        out = rshift(pile, s) + b
        return out, dict(name=name, X=X, x16=x16, sh=ex + ES, pile=pile, s=s, out=out)
    def bend(H, hk):
        if hk is None:
            idx = rshift(H, ES - 12) + 32768
            share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
        else: share = hk["share"]
        out = rshift(H * share, 16)
        return out, dict(H=H, share=share, out=out)
    def arrival(QKV, hk):
        out = np.zeros((P, 768), np.int64); A = np.zeros((12, P, P), np.int64); L = np.zeros((12, P), np.int64)
        NUM = np.zeros((P, 768), np.int64)
        for h in range(12):
            q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
            if hk is None:
                q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
                sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
            for p in range(P):
                if hk is None:
                    row = sc[p, :p+1]; d = row[race(row)] - row
                    a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
                    Lp = log2fix([a.sum()])[0]
                else: a = hk["A"][h, p, :p+1]; Lp = hk["L"][h, p]
                num = product(a[None, :], v[:p+1].astype(np.float64))[0]
                out[p, h*64:(h+1)*64] = power(num, -Lp)
                A[h, p, :p+1] = a; L[h, p] = Lp; NUM[p, h*64:(h+1)*64] = num
        return out, dict(A=A, L=L, num=NUM, out=out)
    for l in range(12):
        H = held["layers"][l] if held is not None else {}
        lay = dict(R=R)
        Q, lay["qkv"] = gauged(R, f"qkv.{l}", H.get("qkv"))
        Aout, lay["arr"] = arrival(Q, H.get("arr"))
        Pout, lay["proj"] = plain(Aout, f"proj.{l}")
        R = R + Pout; lay["Rmid"] = R
        Hh, lay["fc"] = gauged(R, f"fc.{l}", H.get("fc"))
        Bout, lay["bend"] = bend(Hh, H.get("bend"))
        Mout, lay["mproj"] = plain(Bout, f"mproj.{l}")
        R = R + Mout
        rec["layers"].append(lay)
    rec["R12"] = R
    x16, ex, LS = gauge_log(R[t:t + 1])
    if held is not None and "top" in held: LS = held["top"]["LS"]
    pile = product(x16, HEADW)
    F = (HEADE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    scores = power(pile, F)[0] + HEADB
    rec["top"] = dict(x16=x16[0], sh=int(ex[0] + ES), LS=LS, Rt=R[t], scores=scores)
    return rec

def residues(rec):
    """every rounding of the recorded run as an exact rational (numerator object array, exponent array), and the check
    that each is at most half a unit (no wrap-around anywhere)"""
    out = []
    for l, L_ in enumerate(rec["layers"]):
        q, a, pj, fc, bd, mp = L_["qkv"], L_["arr"], L_["proj"], L_["fc"], L_["bend"], L_["mproj"]
        d = {}
        b = W[q["name"]][2]
        d["qkv_out"] = res_power(q["out"] - b, q["pile"], q["F"])
        d["qkv_x"] = res_shift(q["x16"], q["R"], q["sh"][:, None])
        P = rec["P"]; Fa = np.repeat(-a["L"].T, 64, axis=1)       # (P, 768): head h's total for columns h*64..
        d["arr_out"] = res_power(a["out"], a["num"], Fa)
        d["proj_out"] = res_shift(pj["out"] - W[pj["name"]][2], pj["pile"], pj["s"])
        d["proj_x"] = res_shift(pj["x16"], pj["X"], pj["sh"][:, None])
        d["fc_out"] = res_power(fc["out"] - W[fc["name"]][2], fc["pile"], fc["F"])
        d["fc_x"] = res_shift(fc["x16"], fc["R"], fc["sh"][:, None])
        d["bend_out"] = res_shift(bd["out"], bd["H"] * bd["share"], 16)
        d["mproj_out"] = res_shift(mp["out"] - W[mp["name"]][2], mp["pile"], mp["s"])
        d["mproj_x"] = res_shift(mp["x16"], mp["X"], mp["sh"][:, None])
        out.append(d)
    tp = rec["top"]
    top = dict(x=res_shift(tp["x16"], tp["Rt"], tp["sh"]))
    return out, top

def top_residue(rec, words):
    """score_w - standing amount - pile_w x dimming, exact, for the given words"""
    tp = rec["top"]; words = np.asarray(words)
    pile = (O(tp["x16"])[None, :] * O(HEADW[:, words].T.astype(np.int64))).sum(1)
    F = (HEADE[words] + tp["sh"]) * 65536 - tp["LS"][0]
    return res_power(tp["scores"][words] - HEADB[words], pile, F)

# ---------------------------------------------------------------- reverse, exact (no rounding anywhere)
LB = 24
def _limbs(N):
    """object array of whole numbers -> (K, *shape) float64 signed limbs of 24 places: N = sum_k limb_k 2^(24k)"""
    flat = N.ravel(); mags = [abs(int(x)) for x in flat]
    nb = max(1, max(m.bit_length() for m in mags)); K = -(-nb // LB)
    buf = b"".join(m.to_bytes(3 * K, "little") for m in mags)
    u = np.frombuffer(buf, np.uint8).reshape(len(mags), K, 3).astype(np.int64)
    Lm = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16)
    sg = np.fromiter(((x > 0) - (x < 0) for x in flat), np.int64, len(flat))
    return np.ascontiguousarray((Lm * sg[:, None]).T).reshape((K,) + N.shape).astype(np.float64)

def _unlimb(Lf):
    """(K, ...) float64 holding exact whole numbers -> object array sum_k L_k 2^(24k)"""
    Li = Lf.astype(np.int64); assert (Li.astype(np.float64) == Lf).all()
    acc = Li[-1].astype(object)
    for k in range(Li.shape[0] - 2, -1, -1): acc = (acc << LB) + Li[k].astype(object)
    return acc

class BackExact:
    """the reverse light, exactly: N (object whole numbers, shape (words, positions, lines)) x 2^e (one power of two)"""
    def __init__(s, N, e): s.N, s.e = N, int(e)
    def dim(s, fi, fe):                                # times fi x 2^fe, elementwise (broadcast over words)
        fe = np.asarray(fe, np.int64); k0 = int(fe.min())
        fac = O(fi) * O(np.left_shift(np.int64(1), fe - k0)) if (fe - k0).max() < 62 else O(fi) * np.vectorize(lambda z: 1 << int(z), otypes=[object])(fe - k0)
        return BackExact(s.N * fac[None], s.e + k0)
    def plate(s, wT):                                  # met with a plate reversed: (.., n_out) @ (n_out, n_in), exact
        Lf = _limbs(s.N); sh = Lf.shape
        out = (Lf.reshape(-1, sh[-1]) @ wT).reshape(sh[:-1] + (wT.shape[1],))
        assert np.abs(out).max() < 2.0 ** 53
        return BackExact(_unlimb(out), s.e)
    def __add__(s, o):
        e = min(s.e, o.e); return BackExact((s.N << (s.e - e)) + (o.N << (o.e - e)), e)
    def meet(s, num, ex):                              # sum over positions, lines of N x 2^e x num x 2^ex, per word: Fractions
        ex = np.broadcast_to(np.asarray(ex, np.int64), np.shape(num)); k0 = int(ex.min())
        z = O(num) * np.vectorize(lambda q: 1 << int(q), otypes=[object])(ex - k0)
        tot = (s.N * z[None]).reshape(s.N.shape[0], -1).sum(1)
        return [Fraction(int(t)) * Fraction(2) ** (s.e + k0) for t in tot]
    def bits(s): return max(abs(int(x)).bit_length() for x in s.N.ravel())

def _hcols(x):                                         # (12, P) per head -> (P, 768) per column
    return np.repeat(np.asarray(x).T, 64, axis=1)

def reverse_exact(rec, words, res=None):
    """send each word's row back down every step. Returns (R_w as BackExact (words, P, 768), c_w Fractions)."""
    if res is None: res = residues(rec)
    lres, tres = res
    words = np.asarray(words); nw = len(words); P, t = rec["P"], rec["t"]; tp = rec["top"]
    # the top: score_w = pile_w x EXP2[f] x 2^(I_w - 16) + standing amount + rounding
    rn, re_ = top_residue(rec, words)
    c = [Fraction(int(b)) + Fraction(int(n)) * Fraction(2) ** int(e) for b, n, e in zip(HEADB[words], rn, re_)]
    F = (HEADE[words] + tp["sh"]) * 65536 - tp["LS"][0]; I, f = F >> 16, F & 0xFFFF
    N = np.zeros((nw, P, 768), dtype=object); N[:] = 0
    k0 = int((I - 16).min())
    for i, w in enumerate(words):
        N[i, t, :] = O(HEADW[:, w].astype(np.int64)) * int(EXP2[f[i]]) * (1 << int(I[i] - 16 - k0))
    G = BackExact(N, k0)                               # the reverse light at the top's x16
    xn, xe = tres["x"]; z = np.zeros((P, 768), dtype=object); z[:] = 0; z[t] = xn
    ze = np.zeros((P, 768), np.int64); ze[t] = xe
    c = [a + b for a, b in zip(c, G.meet(z, ze))]
    G = BackExact(G.N, G.e - tp["sh"])                 # x16 = R x 2^-sh (only position t is lit)
    for l in range(11, -1, -1):
        Ly, rs = rec["layers"][l], lres[l]
        # ---- the bend's path: mproj (plain), bend, fc (water level)
        mp = Ly["mproj"]; w, we, b = W[mp["name"]]
        c = _addc(c, G.meet(O(b)[None, :] * np.ones((P, 1), object) * 1, np.zeros((P, 768), np.int64)))
        c = _addc(c, G.meet(*rs["mproj_out"]))
        Gp = G.dim(np.ones((P, 768), np.int64), -mp["s"])               # pile -> out: x 2^(we + sh)
        Gx = Gp.plate(w.T)                                               # (P, 3072)
        c = _addc(c, Gx.meet(*rs["mproj_x"]))
        Gx = Gx.dim(np.ones((P, 3072), np.int64), -mp["sh"][:, None] * np.ones((1, 3072), np.int64))
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
        G = G + Gq                                                       # the add reverses to a copy
        # ---- the arrival's path: proj (plain), arrival, qkv's value columns (water level)
        pj = Ly["proj"]; w, we, b = W[pj["name"]]
        c = _addc(c, G.meet(O(b)[None, :] * np.ones((P, 1), object), np.zeros((P, 768), np.int64)))
        c = _addc(c, G.meet(*rs["proj_out"]))
        Ga = G.dim(np.ones((P, 768), np.int64), -pj["s"]).plate(w.T)
        c = _addc(c, Ga.meet(*rs["proj_x"]))
        Ga = Ga.dim(np.ones((P, 768), np.int64), -pj["sh"][:, None] * np.ones((1, 768), np.int64))
        ar = Ly["arr"]
        c = _addc(c, Ga.meet(*rs["arr_out"]))
        Fa = _hcols(-ar["L"]); Gn = Ga.dim(EXP2[Fa & 0xFFFF], (Fa >> 16) - 16)   # times the total's dimming
        Vn = np.zeros_like(Gn.N); Vn[:] = 0
        for h in range(12):
            cs = slice(h * 64, (h + 1) * 64)
            for j in range(P):
                acc = 0
                for p in range(j, P): acc = acc + Gn.N[:, p, cs] * int(ar["A"][h, p, j])
                Vn[:, j, cs] = acc
        Gv = BackExact(Vn, Gn.e)                                         # the share s_pj sent back from p to j
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

def _addc(c, d): return [a + b for a, b in zip(c, d)]

def meet_exact(R0, G):
    """the embedding numbers met with R_w, exactly, per word (Fractions)"""
    tot = (G.N * O(R0)[None]).reshape(G.N.shape[0], -1).sum(1)
    return [Fraction(int(x)) * Fraction(2) ** G.e for x in tot]

# ---------------------------------------------------------------- reverse, the machine's own style (16 places)
"""Back16: the reverse light as whole numbers N (int64) with one power of two per word per position, e (words, P):
   value = N x 2^e.  Kept the way the machine keeps its own light:
     - the reverse stream (between layers): 32 places per word-position (the machine's stream is 32 bits);
     - whatever next meets a plate or a dimming: 16 places per word-position (the machine's to16: top 1 of the OR of the
       magnitudes, shift so 15 places remain, rounded to nearest with the machine's own rshift);
     - a plate meeting itself is exact (16 x 16 places x up to 3,072 lines < 2^53), its output kept whole until the
       next keep;
     - a dimming that differs only by a power of two across neurons is applied exactly, then kept to 16 places;
     - the arrival's sum over positions is done on a common power of two per word (exact when the positions' powers lie
       within 14 places of each other, rounded below that), then kept to 16 places;
     - every bias + rounding (the constant) is kept on 32 places per position (the machine's bias width) and met with
       the reverse light exactly; the top word's standing amount and its rounding are exact."""
def bitlen(m):
    m = np.asarray(m, np.int64); out = np.zeros(m.shape, np.int64); x = m.copy()
    for b in (32, 16, 8, 4, 2, 1):
        big = x >= (np.int64(1) << np.int64(b)); out += big * b; x = np.where(big, x >> b, x)
    return out + (x > 0)

def keep(N, e, places):
    top = bitlen(np.bitwise_or.reduce(np.abs(N), axis=-1)); sh = np.maximum(top - (places - 1), 0)
    return rshift(N, sh[..., None]), e + sh

def to_exp(N, e, T):
    """N x 2^e written at power T (per row): left shift exact where e > T, rounded right shift where e < T"""
    return rshift(N, (T - e)[..., None])

def add16(A, B):
    (Na, ea), (Nb, eb) = A, B
    ta = ea + bitlen(np.bitwise_or.reduce(np.abs(Na), axis=-1)); tb = eb + bitlen(np.bitwise_or.reduce(np.abs(Nb), axis=-1))
    T = np.maximum(np.minimum(ea, eb), np.maximum(ta, tb) - 61)
    return keep(to_exp(Na, ea, T) + to_exp(Nb, eb, T), T, 32)

def plate16(N, wT):
    nw, P, n = N.shape
    out = N.reshape(-1, n).astype(np.float64) @ wT
    assert np.abs(out).max() < 2.0 ** 53
    return out.astype(np.int64).reshape(nw, P, wT.shape[1])

def dim16(N, e, fi, fe):
    """times fi x 2^fe (fi, fe: (P, n)); the power differing over neurons applied exactly; then 16 places"""
    fe = np.asarray(fe, np.int64); k0 = fe.min(axis=1)                         # per position
    fac = np.asarray(fi, np.int64) << (fe - k0[:, None])
    N = N * fac[None]
    return keep(N, e + k0[None, :], 16)

def kappa32(num, ex, bias=None):
    """exact constant (bias + rounding numerator x 2^ex) on 32 places per position: (KN int64 (P, n), Ke (P,))"""
    num = np.asarray(num, dtype=object); ex = np.broadcast_to(np.asarray(ex, np.int64), num.shape)
    k0 = ex.min(axis=1)
    V = num * np.vectorize(lambda q: 1 << int(q), otypes=[object])(ex - k0[:, None])
    if bias is not None: V = V + O(bias)[None, :] * np.vectorize(lambda q: 1 << int(q), otypes=[object])(-k0)[:, None]
    KN = np.zeros(V.shape, np.int64); Ke = np.zeros(V.shape[0], np.int64)
    for p in range(V.shape[0]):
        top = max(abs(int(x)).bit_length() for x in V[p]); sh = max(top - 31, 0)
        row = [((int(x) + (1 << (sh - 1))) >> sh) if sh > 0 else int(x) for x in V[p]]
        KN[p] = row; Ke[p] = k0[p] + sh
    return KN, Ke

EFIX = -400
def meet16(N, e, KN, Ke):
    """per word: sum over positions, lines of (N x 2^e) x (KN x 2^Ke), exact, as a whole number at 2^EFIX"""
    nw, P, n = N.shape
    parts_k = [(KN & 0xFFFF, 0), (KN >> 16, 16)]
    parts_g = [(N & 0xFFFFF, 0), ((N >> 20) & 0xFFFFF, 20), (N >> 40, 40)]
    acc = [0] * nw
    for p in range(P):
        for kp, ks in parts_k:
            kf = kp[p].astype(np.float64)
            for gp, gs in parts_g:
                S = gp[:, p, :].astype(np.float64) @ kf
                assert np.abs(S).max() < 2.0 ** 53
                Si = S.astype(np.int64)
                for w in range(nw):
                    if Si[w]:
                        x = int(e[w, p]) + int(Ke[p]) + ks + gs - EFIX; assert x >= 0
                        acc[w] += int(Si[w]) << x
    return acc

def meetE(R0, N, e):
    """the embedding numbers met with R_w (32 places), exact, per word as a whole number at 2^EFIX"""
    nw, P, n = N.shape; acc = [0] * nw
    for gp, gs in [(N & 0xFFFF, 0), (N >> 16, 16)]:
        S = np.einsum("wpn,pn->wp", gp.astype(np.float64), R0.astype(np.float64))
        assert np.abs(S).max() < 2.0 ** 53
        Si = S.astype(np.int64)
        for w in range(nw):
            for p in range(P):
                if Si[w, p]:
                    x = int(e[w, p]) + gs - EFIX; assert x >= 0
                    acc[w] += int(Si[w, p]) << x
    return acc

def kappas(rec, res=None, part="both"):
    """every step's constant (bias + the machine's rounding there) on 32 places, once per recorded run.
    part: "both", "bias" (the biases alone) or "round" (the roundings alone)"""
    if res is None: res = residues(rec)
    lres, tres = res; out = []
    if part != "both":
        lres = [{k: ((np.zeros(n.shape, dtype=object) + 0) if part == "bias" else n, e) for k, (n, e) in d.items()} for d in lres]
        tres = dict(x=((np.zeros(tres["x"][0].shape, dtype=object) + 0) if part == "bias" else tres["x"][0], tres["x"][1]))
    _W = W if part != "round" else {k: (v[0], v[1], np.zeros_like(v[2])) for k, v in W.items()}
    for l in range(12):
        rs = lres[l]; d = {}
        d["mproj"] = kappa32(*rs["mproj_out"], bias=_W[f"mproj.{l}"][2]); d["mproj_x"] = kappa32(*rs["mproj_x"])
        d["bend"] = kappa32(*rs["bend_out"])
        d["fc"] = kappa32(*rs["fc_out"], bias=_W[f"fc.{l}"][2]); d["fc_x"] = kappa32(*rs["fc_x"])
        d["proj"] = kappa32(*rs["proj_out"], bias=_W[f"proj.{l}"][2]); d["proj_x"] = kappa32(*rs["proj_x"])
        d["arr"] = kappa32(*rs["arr_out"])
        on, oe = rs["qkv_out"]
        d["qkv"] = kappa32(on[:, 1536:], np.asarray(oe)[:, 1536:], bias=_W[f"qkv.{l}"][2][1536:]); d["qkv_x"] = kappa32(*rs["qkv_x"])
        out.append(d)
    return out, tres, part

def reverse16(rec, words, kap=None, res=None):
    """the reverse in the machine's own style. Returns (N, e) of R_w (words, P, 768) on 32 places and c_w as
    whole numbers at 2^EFIX (exact meetings of the kept light with the kept constant)."""
    if res is None: res = residues(rec)
    if kap is None: kap = kappas(rec, res)
    lk, tres = kap[0], kap[1]; part = kap[2] if len(kap) > 2 else "both"
    words = np.asarray(words); nw = len(words); P, t = rec["P"], rec["t"]; tp = rec["top"]
    rn, re_ = top_residue(rec, words)
    c = [(int(b) << -EFIX) * (part != "round") + (int(n) << (int(x) - EFIX)) * (part != "bias") for b, n, x in zip(HEADB[words], rn, re_)]
    F = (HEADE[words] + tp["sh"]) * 65536 - tp["LS"][0]; I, f = F >> 16, F & 0xFFFF
    N = np.zeros((nw, P, 768), np.int64); e = np.zeros((nw, P), np.int64)
    N[:, t, :] = HEADW[:, words].T.astype(np.int64) * EXP2[f][:, None]; e[:, t] = I - 16
    N, e = keep(N, e, 32)
    xn, xe = tres["x"]; z = np.zeros((P, 768), dtype=object); z[:] = 0; z[t] = xn
    ze = np.zeros((P, 768), np.int64); ze[t] = xe
    KN, Ke = kappa32(z, ze); c = [a + b for a, b in zip(c, meet16(N, e, KN, Ke))]
    e = e.copy(); e[:, t] -= tp["sh"]
    G = (N, e)
    for l in range(11, -1, -1):
        Ly, kk = rec["layers"][l], lk[l]
        # the bend's path
        mp = Ly["mproj"]; w = W[mp["name"]][0]
        c = _addi(c, meet16(*G, *kk["mproj"]))
        Gp = dim16(*G, np.ones((P, 768), np.int64), -mp["s"])
        Gx = (plate16(Gp[0], w.T), Gp[1])
        c = _addi(c, meet16(*Gx, *kk["mproj_x"]))
        Gx = (Gx[0], Gx[1] - mp["sh"][None, :])
        c = _addi(c, meet16(*Gx, *kk["bend"]))
        bd = Ly["bend"]; Gh = dim16(*Gx, bd["share"], np.full((P, 3072), -16, np.int64))
        fc = Ly["fc"]; w = W[fc["name"]][0]
        c = _addi(c, meet16(*Gh, *kk["fc"]))
        Gq = dim16(*Gh, EXP2[fc["F"] & 0xFFFF], (fc["F"] >> 16) - 16)
        Gq = (plate16(Gq[0], w.T), Gq[1])
        c = _addi(c, meet16(*Gq, *kk["fc_x"]))
        Gq = (Gq[0], Gq[1] - fc["sh"][None, :])
        G = add16(G, Gq)
        # the arrival's path
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
        topn = en + bitlen(np.bitwise_or.reduce(np.abs(Nn), axis=-1))
        Vn = np.zeros((nw, P, 768), np.int64); ev = np.zeros((nw, P), np.int64)
        for j in range(P):
            lit = (np.abs(Nn[:, j:, :]).max(axis=2) > 0)                     # (nw, P-j)
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

def _addi(c, d): return [a + b for a, b in zip(c, d)]
def frac(x): return Fraction(x, 1 << -EFIX)
