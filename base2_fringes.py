"""BASE2_FRINGES: ONE EXACT MACHINE WITH EVERY EXACT CUT FOUND SO FAR, ALL AT ONCE, AT EVERY LAYER.

His goal: "keep all the resolution, decrease the compute". Every bend, water level and arrival is base2.py's own, and
nothing is rounded, dropped or averaged. Only the excess is cut, in four exact cuts:

 1. LAYER 0 FROM STORAGE (base2_layer0.py's tables: entry/ and layer0/, read here by this file's own readers).
    - qkv.0: the word's row and the position's row are stored with their meetings (tips); only the -1/0/+1 carry
      meets the plate. Position 0 reads its whole qkv pile (ZP0).
    - position 0: the whole stream after layer 0 is read (Z0). No arrival, proj.0, fc.0 or mproj.0 there.
    - proj.0 (positions 1 on): the token's own value at o's shift is stored with its tip (OWN); only o16 - own meets.
    - fc.0 (positions 1 on): the word's and position's parts at x1's shift are stored with their tips (FT, FP); only
      the rest (the arrival write's part plus a carry) meets the plate.
 2. SIGNED DIGITS on every plate's weights and on every light that meets a plate: up / flat / down, no two lit side by
    side (recording/C: the 48 plates as up and down sheets, read here from those files). A pile is the sum over place
    pairs (a of the light, b of the plate) of 2^(a+b) x [(up.up + down.down) - (up.down + down.up)], each a count over
    the lines. Meetings happen only between lit pixels. (The line-up is not used: its running sums cost more than
    they save, combined/ found.)
 3. KNIFE-EDGE STOPS at every fc plate (lens 1). Each edge's pile is counted by diagonals (a + b) from the top, in
    signed digits (31 diagonals, places 0..15 on both sides), on knife/settle.py's grid: the top 2, 4, ..., 30, then
    all. After each stage the uncounted part is bounded exactly by settle.py's plain bound,
        |rest| <= sum over uncounted (a, b) of 2^(a+b) min(n_a, m_b)
    with n_a the light's lit lines at place a, and m_b taken at the plate's largest lit count at place b over its
    neurons (stored once with the plate). That makes the bound ONE number per position and stage, shared by all
    3,072 edges: no edge reads a bound table. Each edge stops at the stage where its bend OUTPUT, the 16 places
    mproj meets after its water level (out16 and the shift sh), is provably fixed (settle.py's test: the same out16
    over the whole interval and over every sh the position's intervals allow; settled edges are frozen and the sh
    range is bounded from the frozen intervals). Each edge then meets mproj once (order is free).
    THE EDGE'S OWN SCALE. To read an interval on the edge, the pile must be carried to the edge's scale: H =
    rshift(pile x e, s) + bias, where e (the water level's gain, a table entry) is the same for every edge of a
    position and s is each neuron's own shift. The machine carries Y = (counted pile) x e + (bias << s) and moves it
    by (newly counted part) x e at each stage; then H_lo, H_hi = rshift(Y -/+ (bound x e), s), adds and shifts only.
    WHERE THE BOUND CAN BITE (a relation fixed in advance, no search): no edge is carried to its own scale before
    its interval is narrower than the knife edge itself, one unit of x (2^16 in H), for every neuron of the
    position: (shared bound) x e <= 2^(15 + the smallest s). From that stage on every edge of the position is
    carried and tested at every stage.
 4. THE TOP READ CARRIED DOWN THE FINE DIAGONALS (collapse_top/fine_staged_top.py's grid and bound, plain photons:
    sign and places 0..14, 29 diagonals, the top 2, 4, ..., 28, then all), with:
    - the set-aside test made at the pile's own scale, so no word's bound is divided at any stage: each word's
      standing amount is written at pile scale once per position, beta_w = power(bias_w, -F) (the machine's own
      divide: the negated power), and poured into the word's counted pile; each stage the 5th-best low is found at
      pile scale and lowered once by mu = 2 gamma + delta; a word is set aside when its high is at or below it.
      gamma (whole number, once per position) covers the rounding of beta_w; delta = ceil(2^s / e) makes one pile-
      scale step worth at least one score step. So a word set aside is surely below five others.
    - each word's bound carried down from stage to stage: the newly counted pairs' part (Delta R, from a small table
      of the light) is taken off its high and its bound, instead of rebuilding the whole bound.
    - the 5th-best low read from the words in order of their highs, only as far as needed.
    The words still in after the top 28 are read to the end exactly. Exact top 5 and the race winner.

WORK, in the unit of all earlier results (combined/machine.py's and recording/C_machine.py's rules; plain is
3,666,596,259 a token on the 648 positions of the 30 refs texts):
  - a meeting is, on each line, the light's photons x the plate's photons, for the pairs met. In signed digits the
    photons are the lit places. The top read is in plain photons (1s), as fine_staged_top.py counts it.
  - an add counts the photons it adds; a divide (a product by a gain) counts the number's photons x the gain's.
  - table reads and compares are stated as counts, not photons.
  Every part of the plain machine is counted again on the same light here, so the plain total is rebuilt in the same
  run (it must be 3,666,596,259 on the refs).

THE CHECK PATH (never counted, and never used by the machine): the full fc pile is rebuilt from its diagonals (the
pieces past each stop are simulated only to check); every interval must hold the true H; every settled output must
equal base2's own; every top-read bound must hold the true pile and every set-aside word must be below the true
5th-best score; position 0's layer 0 is run with base2.py's own steps beside the stored read.

The machine reads only recording/C (49 tables in signed digits and rest.npz), entry/ and layer0/. It never opens
base2_gpt2.npz (asserted on the source).

PREDICTIONS (written before anything here ran):
  The task's: about 50-55% of plain with every add counted, every word identical.
  The session's:
  - Exact: every pile (qkv, proj, mproj), every fc output that mproj meets (out16 and its shift), and the stream
    after every half layer identical to base2.py on all 30 refs texts and all 300 judge windows; judge5 6000/6000
    and 6000/6000; the top read's top 5 and race winner identical at all 6,648 read positions (648 refs + 6,000
    judged); every bound held; order-free identical on 3 texts.
  - Layer 0 from storage: layer 0's four banks in signed digits about 127M a token without storage, about 88M with
    it (about -39M, 1.1% of the whole plain machine).
  - Signed digits at the 45 other banks: 58-59% of their plain meetings, as recording/C measured.
  - Knife stops: settle.py's rule recounted in signed digits (per-edge bound, every stage): fc work 90-93% (plain
    digits gave 91.8%). With the shared bound and the gate: within 1-2 points of that. The gate opens at the top
    12-16 diagonals for most positions. The carrying to the edge's scale costs about 500-900 photons an edge, so the
    knife cut nets between a small loss and +0.6% of the whole plain machine; it is the smallest of the four cuts.
  - Top read: meetings 17-21% of the plain top read on all 648 positions; the bound's own work with every add
    counted 2-5%; together 20-25%; survivors at the deciding stages within a few percent of section 29's (the
    margins mu only set aside a little later).
  - Whole machine: about 1,750-1,850M a token, 47-51% of plain, most likely about 48%.
"""
import os, sys, json, time, resource, hashlib, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"
CDIR, ENT, L0D = f"{HERE}/recording/C", f"{HERE}/entry", f"{HERE}/layer0"
REC_PLACES, PL = 17, 16                                       # the record's places; place 16 is dark everywhere (checked)

# ---------- base2.py's own steps, its numbers from recording/C/rest.npz, no plate read by its loader ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{CDIR}/rest.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _a in _src, _a; _src = _src.replace(_a, _b)
assert "base2_gpt2" not in _src
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert B["W"] == {}
M = B["M"]; ES, SC, ARR, EXP2, PHI = B["ES"], B["SC"], B["ARR"], B["EXP2"], B["PHI"]
rshift, to16, product, race, log2fix, power, gauge_log, edge = (B[k] for k in
    ("rshift", "to16", "product", "race", "log2fix", "power", "gauge_log", "edge"))
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
BIAS = {n: M[n + ".b"] for n in BANKS}; HEAD_B = M["head.b"]; HEAD_E = M["head.e"]
MAN = json.load(open(f"{CDIR}/manifest.json"))
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); FIRST, LAST = 3, 23
PLAIN_TOTAL = 3666596259
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6       # MB
def per_token(total, n): return (total + n // 2) // n
def pct2(a, b): return f"{(10000 * a + b // 2) // b / 100:.2f}%" if b else "-"
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()

# ---------- photons ----------
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def signed_digits(v):                                         # (up, down) place masks; no two lit side by side
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def E_ones(LS): return ones(EXP2[np.bitwise_and(-np.asarray(LS, dtype=np.int64), 0xFFFF)])   # the gain's photons
def water(x16):                                               # the gauge's sums: S1 adds, S2 squares, 768 S2 - S1^2
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
class Count(dict):                                            # key "who|layer|part|item"; who: plain / fr / side / ctl
    def add(self, k, v): self[k] = self.get(k, 0) + int(np.sum(v))

# ---------- the record: the 48 plates and the top table in signed digits (recording/C) ----------
def unpack_order(buf, lines, neurons, nb):
    n = lines * neurons; out = np.empty(n, np.int64); sh = np.arange(nb - 1, -1, -1, dtype=np.int64); step = 1 << 20
    for i in range(0, n, step):
        m = min(step, n - i); b0 = i * nb // 8
        bits = np.unpackbits(buf[b0:b0 + (m * nb + 7) // 8])[:m * nb].reshape(m, nb).astype(np.int64)
        out[i:i + m] = (bits << sh[None, :]).sum(1)
    return out.reshape(lines, neurons)
def read_files(name):
    r = MAN["plates"][name]; L, N = r["lines"], r["neurons"]
    up = np.fromfile(f"{CDIR}/{r['files']['up']['file']}", np.uint8).reshape(REC_PLACES, L // 8, N)
    dn = np.fromfile(f"{CDIR}/{r['files']['down']['file']}", np.uint8).reshape(REC_PLACES, L // 8, N)
    o = unpack_order(np.fromfile(f"{CDIR}/{r['files']['order']['file']}", np.uint8), L, N, r["order_bits"])
    e = np.fromfile(f"{CDIR}/{r['files']['power']['file']}", np.dtype(r["power_dtype"]))
    return up, dn, o, e
def sheets_on_lines(up, dn, inv, p, cols=slice(None)):        # place p's two sheets, each bit back on its own line
    u = np.take_along_axis(np.unpackbits(up[p][:, cols], axis=0), inv, 0)
    d = np.take_along_axis(np.unpackbits(dn[p][:, cols], axis=0), inv, 0)
    return u, d
def table_values(name):                                       # a table's whole numbers, rebuilt from its sheets
    up, dn, o, e = read_files(name); L, N = o.shape; out = np.empty((L, N), np.int16)
    for c0 in range(0, N, 8192):
        sl = slice(c0, c0 + 8192); w = np.zeros((L, min(8192, N - c0)), np.int64); inv = np.argsort(o[:, sl], axis=0)
        for p in range(REC_PLACES):
            u, d = sheets_on_lines(up, dn, inv, p, sl); w += (u.astype(np.int64) - d.astype(np.int64)) << p
        out[:, sl] = w
    return out, e

class Plate:
    """a bank's plate laid from its files: PS[b] = [UP sheet of place b | DOWN sheet of place b] on the lines, 0/1.
    Stored with it (once, never per token): lit places per line (nrow), plain 1s per line (wrow, for the plain count),
    lit lines per neuron and place (mnb) and the largest of those over the neurons (mbar, the shared bound's m_b).
    line_perm / neuron_perm: ORDER-FREE check (lines, with the light's lines; neurons, with their powers)."""
    def __init__(self, name, line_perm=None, neuron_perm=None):
        up, dn, o, e = read_files(name); L, N = o.shape; inv = np.argsort(o, axis=0)
        self.name, self.L, self.N = name, L, N
        self.PS = np.empty((PL, L, 2 * N), np.float32)
        nrow = np.zeros(L, np.int64); mnb = np.zeros((PL, N), np.int64); w = np.zeros((L, N), np.int64)
        for p in range(REC_PLACES):
            u, d = sheets_on_lines(up, dn, inv, p)
            if neuron_perm is not None: u, d = u[:, neuron_perm], d[:, neuron_perm]
            if p >= PL: assert not u.any() and not d.any(); continue
            self.PS[p, :, :N] = u; self.PS[p, :, N:] = d
            lit = u | d; nrow += lit.sum(1, dtype=np.int64); mnb[p] = lit.sum(0, dtype=np.int64)
            w += (u.astype(np.int64) - d.astype(np.int64)) << p
        self.e = e if neuron_perm is None else e[neuron_perm]
        self.nrow, self.wrow, self.mnb, self.mbar = nrow, ones(w).sum(1), mnb, mnb.max(1)
        self.w = w.astype(np.int16) if np.abs(w).max() < 1 << 15 else w
        self.perm = line_perm
        if line_perm is not None:
            self.PS = np.ascontiguousarray(self.PS[:, line_perm, :]); self.nrow = nrow[line_perm]
            self.wrow = self.wrow[line_perm]; self.w = self.w[line_perm]
        del w

def light_places(x):                                          # places the light's signed digits need (16 from layer 1 on)
    up, dn = signed_digits(x); lit = up | dn
    return max(PL, int(np.bitwise_or.reduce(lit.ravel(), initial=0)).bit_length())
def light_sheets(x, LPL=None):                                # the light written in signed digits: (2, LPL, P, L) 0/1
    up, dn = signed_digits(x); lit = up | dn
    LPL = light_places(x) if LPL is None else LPL
    assert not (up & dn).any() and not (lit & (lit >> 1)).any() and not (lit >> LPL).any()
    LSH = np.empty((2, LPL, len(x), x.shape[1]), np.float32)
    for a in range(LPL): LSH[0, a] = (up >> a) & 1; LSH[1, a] = (dn >> a) & 1
    return LSH, lit

def meet(pl, x, chunk=256, pair_rng=None):
    """the pile at a plain bank: sum over place pairs (a, b) of 2^(a+b) [(uu + dd) - (ud + du)], counted over lines.
    pair_rng (ORDER-FREE check): the plate's places met in a random order, the light's places added in a random order."""
    x = np.asarray(x, np.int64); P, N = len(x), pl.N
    if pl.perm is not None: x = x[:, pl.perm]
    out = np.empty((P, N), np.int64)
    for c0 in range(0, P, chunk):
        xs = x[c0:c0 + chunk]; Pc = len(xs); LPL = light_places(xs); LSH, _ = light_sheets(xs, LPL)
        LSH = LSH.reshape(2 * LPL * Pc, pl.L)
        acc = np.zeros((Pc, N), np.int64)
        bs = range(PL) if pair_rng is None else pair_rng.permutation(PL)
        for b in bs:
            Rb = (LSH @ pl.PS[b]).reshape(2, LPL, Pc, 2 * N)
            S = ((Rb[0, :, :, :N] + Rb[1, :, :, N:]) - (Rb[0, :, :, N:] + Rb[1, :, :, :N])).astype(np.int64)
            for a in (range(LPL) if pair_rng is None else pair_rng.permutation(LPL)): acc += S[a] << (a + int(b))
            del Rb, S
        out[c0:c0 + Pc] = acc
    assert np.abs(out).max(initial=0) < 2 ** 62
    return out

# ---------- the knife edge, read from the machine itself (knife/settle.py's tables, unchanged) ----------
def share_of(H):                                              # base2.edge's own two lines: the table entry the edge reads
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
assert (np.diff(PHI) >= 0).all() and PHI.min() >= 0 and PHI.max() <= 65536
HA, HB = -(1 << 19) - 64, (1 << 19) + 64
GRID = np.arange(HA, HB + 1, dtype=np.int64); OUTT = edge(GRID).astype(np.int64)
assert np.array_equal(rshift(GRID * share_of(GRID), 16), OUTT) and OUTT[0] == 0 and OUTT[-1] == HB
_pos = GRID >= 0; assert (OUTT[_pos] >= 0).all() and (np.diff(OUTT[_pos]) >= 0).all()
NEGMAX = int(np.abs(OUTT[~_pos]).max())
_probe = np.concatenate([np.arange(HA - 4096, HA), -(np.int64(1) << np.arange(20, 44)),
                         np.arange(HB + 1, HB + 4096), np.int64(1) << np.arange(20, 44)])
assert np.array_equal(edge(_probe), np.where(_probe < HA, 0, _probe))
SMT = 21
CP = np.zeros((SMT + 1, len(GRID)), np.int32)
for _s in range(SMT + 1):
    _f = rshift(OUTT, _s); CP[_s, 1:] = np.cumsum(_f[1:] != _f[:-1])
assert not rshift(OUTT, SMT).any(); del _f
def out_ext(H): return np.where(H < HA, 0, np.where(H > HB, H, OUTT[np.clip(H - HA, 0, len(GRID) - 1)]))
def constant(lo, hi, s):                                      # out16 = rshift(out, s) the same for every H in [lo, hi]?
    v = rshift(out_ext(lo), s); ok = v == rshift(out_ext(hi), s)
    sc = np.minimum(s, SMT); g0 = np.clip(lo, HA, HB) - HA; g1 = np.clip(hi, HA, HB) - HA
    ok &= (hi < HA) | (lo > HB) | (CP[sc, g1] == CP[sc, g0])
    ok &= (hi <= HB) | (rshift(np.maximum(lo, HB), s) == rshift(hi, s))
    return ok, v
def bitlen(v): return np.array([int(x).bit_length() for x in v], np.int64)
def sh_range(lo, hi):                                         # (P, N) intervals of H -> the water level's shift, lowest, highest
    U = np.maximum(np.where(hi >= 0, out_ext(np.maximum(hi, 0)), 0), np.where(lo < 0, NEGMAX, 0)).max(1)
    L = np.where(lo >= 0, out_ext(np.maximum(lo, 0)), 0).max(1)
    return np.maximum(bitlen(L) - 15, 0), np.maximum(bitlen(U) - 15, 0)
def out_settled(lo, hi, slo, shi):                            # flat: out16 the same over [lo, hi] and every sh in [slo, shi]
    ok, v = constant(lo, hi, slo)
    for d in range(1, int((shi - slo).max(initial=0)) + 1):
        m = ok & (slo + d <= shi)
        if not m.any(): break
        o2, v2 = constant(lo[m], hi[m], slo[m] + d); ok[m] = o2 & (v2 == v[m])
    return ok, v

def grid(LPL):                                                # the diagonals for a light of LPL places and a plate of PL
    NDG = LPL + PL - 1                                        # 31 diagonals (0..30) from layer 1 on
    KD = list(range(2, NDG, 2)) + [NDG]                       # the top 2, 4, ..., then all
    return NDG, KD, np.arange(LPL)[:, None] + np.arange(PL)[None, :]
NDG, KD, DIAG = grid(PL); NSK = len(KD)
H45 = 1 << 45
def _clip45(X): return np.clip(X, -H45, H45)                 # only for the measurement rows (see knife_measure)

def knife(pl, x, c0, F, bias, C=None, lay="", plain_x=None, chk=None, chunk=192, pair_rng=None, measure=None,
          stop_groups=None):
    """lens 1 with each knife edge stopped as soon as its output is fixed. x: the light that meets the plate (P, L);
    c0: the stored part of the pile (P, N) or None; F: the gain's power per position and neuron (P, N); bias (N,).
    Returns out16 (P, N), the shift sh (P,), each edge's stop stage (P, N), and the full pile (CHECK ONLY).
    plain_x: the plain machine's light at this plate, for the plain count (layer 0: x1, not the rest).
    measure: a dict to fill with the measurement rows (settle.py's rule and the shared bound at every stage)."""
    x = np.asarray(x, np.int64); P, N, L = len(x), pl.N, pl.L
    if pl.perm is not None: x = x[:, pl.perm]
    f = F & 0xFFFF; assert (f == f[:, :1]).all()                # one gain table entry per position, all edges
    e = EXP2[f[:, 0]].astype(np.int64); s = 16 - (F >> 16); assert s.min() >= 1
    smin = s.min(1)
    Bsh = np.left_shift(np.asarray(bias, np.int64)[None, :], s)
    assert np.abs(Bsh).max() < 2 ** 60
    OUT16 = np.empty((P, N), np.int64); SH = np.empty(P, np.int64); STOP = np.empty((P, N), np.int8)
    GATE = np.empty(P, np.int8); PILE = np.empty((P, N), np.int64)
    mbar = pl.mbar
    LPL = light_places(x); NDG, KD, DIAG = grid(LPL)              # the same grid for every position of this plate
    for r0 in range(0, P, chunk):
        r = slice(r0, min(P, r0 + chunk)); xs = x[r]; Pc = len(xs)
        LSH, lit = light_sheets(xs, LPL); LSH = LSH.reshape(2 * LPL * Pc, L)
        nA = np.stack([((lit >> a) & 1).sum(1) for a in range(LPL)], 1)          # (Pc, LPL) the light's lit lines
        D = np.zeros((NDG, Pc, N), np.int16); MD = np.zeros((NDG, Pc, N), np.int16)
        for b in (range(PL) if pair_rng is None else pair_rng.permutation(PL)):
            Rb = (LSH @ pl.PS[b]).reshape(2, LPL, Pc, 2 * N)
            uu, ud, du, dd = Rb[0, :, :, :N], Rb[0, :, :, N:], Rb[1, :, :, :N], Rb[1, :, :, N:]
            Sg = ((uu + dd) - (ud + du)).astype(np.int16); Mt = ((uu + dd) + (ud + du)).astype(np.int16)
            for a in (range(LPL) if pair_rng is None else pair_rng.permutation(LPL)):
                D[a + int(b)] += Sg[a]; MD[a + int(b)] += Mt[a]
            del Rb, uu, ud, du, dd, Sg, Mt
        del LSH
        c0r = np.zeros((Pc, N), np.int64) if c0 is None else np.asarray(c0[r], np.int64)
        pile = c0r.copy()                                                         # CHECK ONLY: the whole pile
        for d in range(NDG): pile += D[d].astype(np.int64) << d
        PILE[r] = pile
        Fr, sr, er, Br, br = F[r], s[r], e[r], Bsh[r], bias
        if C is not None:
            C.add(f"fr|{lay}|fc|light written in signed digits", ones(xs).sum())
            C.add(f"side|{lay}|fc|meetings, signed digits, no stops", naf(xs) @ pl.nrow)
        # ---- the shared bound, from the light's lit lines and the plate's largest lit counts ----
        TAB = (np.int64(1) << DIAG)[None] * np.minimum(nA[:, :, None], mbar[None, None, :])   # (Pc, PL, PL)
        OTAB = ones(np.minimum(nA[:, :, None], mbar[None, None, :]))
        if C is not None: C.add(f"fr|{lay}|fc|knife: shared bound built from the light", OTAB.sum())
        thr = np.array([(1 << (15 + int(smin[r0 + i]))) // int(er[i]) for i in range(Pc)], np.int64)
        c = c0r.copy(); cm = np.zeros((Pc, N), np.int64)
        opened = np.zeros(Pc, bool); Y = np.zeros((Pc, N), np.int64)
        st = np.full((Pc, N), -1, np.int8); FLO = np.zeros((Pc, N), np.int64); FHI = np.zeros((Pc, N), np.int64)
        V = np.zeros((Pc, N), np.int64); gate = np.full(Pc, -1, np.int8)
        prev = 0
        for si, k in enumerate(KD):
            newd = [d for d in range(NDG) if NDG - k <= d < NDG - prev]; prev = k
            dc = np.zeros((Pc, N), np.int64)
            for d in newd: dc += D[d].astype(np.int64) << d; cm += MD[d]
            c += dc
            newpairs = (DIAG >= NDG - k) & (DIAG < NDG - (KD[si - 1] if si else 0))
            Rbar = (TAB * (DIAG < NDG - k)[None]).sum((1, 2))
            if C is not None: C.add(f"fr|{lay}|fc|knife: shared bound built from the light", (OTAB * newpairs[None]).sum())
            uns = st < 0
            newly = ~opened & (Rbar <= thr)
            if newly.any():                                                       # carried to the edge's scale
                m = newly[:, None] & uns
                Y[newly] = c[newly] * er[newly][:, None] + Br[newly]
                if C is not None:
                    C.add(f"fr|{lay}|fc|knife: pile carried to the edge's scale (gain x pile)",
                          (ones(c) * ones(er)[:, None])[m].sum())
                    C.add(f"fr|{lay}|fc|biases", ones(br)[None, :].repeat(Pc, 0)[m].sum())
            old = opened & ~newly
            if old.any():
                m = old[:, None] & uns
                assert int(np.abs(dc[m]).max(initial=0)) < (1 << 62) // (1 << 17); inc = dc * er[:, None]
                Y = np.where(m, Y + inc, Y)
                if C is not None:
                    C.add(f"fr|{lay}|fc|knife: newly counted part carried (gain x part, and its add)",
                          (ones(dc) * ones(er)[:, None] + ones(inc))[m].sum())
            gate[newly] = si; opened |= newly
            if not opened.any(): continue
            op = np.flatnonzero(opened)
            RE = Rbar[op] * er[op]; assert np.abs(RE).max(initial=0) < 2 ** 61
            Yo = Y[op]; so_ = sr[op]
            Hlo = rshift(Yo - RE[:, None], so_); Hhi = rshift(Yo + RE[:, None], so_)
            if C is not None:
                uo = uns[op]
                C.add(f"fr|{lay}|fc|knife: bound carried to the edge's scale (gain x bound)", (ones(Rbar[op]) * ones(er[op])).sum())
                C.add(f"fr|{lay}|fc|knife: interval ends (pile -/+ bound at the edge's scale)", (2 * ones(RE)[:, None] * uo).sum())
            sto = st[op]; fr = sto >= 0
            LOc = np.where(fr, FLO[op], Hlo); HIc = np.where(fr, FHI[op], Hhi)
            slo, shi = sh_range(LOc, HIc)
            if chk is not None:
                Ht = power(pile[op], Fr[op]) + br[None, :]
                chk("knife: true H inside the interval", int(((Ht < Hlo) | (Ht > Hhi))[~fr].sum()))
            idx = np.flatnonzero(~fr.ravel())
            if C is not None:
                C.add(f"reads|{lay}|fc|knife: table reads (out at both ends, change counts, sh range)",
                      len(idx) * 2 + 2 * ((shi - slo + 1)[:, None] * ~fr).sum() + 2 * fr.size)
                C.add(f"cmp|{lay}|fc|knife: compares", len(idx))
            okc, v = out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo, N)[idx], np.repeat(shi, N)[idx])
            new = idx[okc]
            if len(new):
                pr, nn = op[new // N], new % N
                st[pr, nn] = si; FLO[pr, nn] = Hlo.ravel()[new]; FHI[pr, nn] = Hhi.ravel()[new]; V[pr, nn] = v[okc]
                if C is not None:
                    hl = Hlo.ravel()[new]
                    C.add(f"fr|{lay}|fc|meetings, signed digits, edges stopped", cm[pr, nn].sum())
                    C.add(f"fr|{lay}|fc|bends (at the interval's low end)", (ones(hl) * ones(share_of(hl))).sum())
        assert (st >= 0).all(), "an edge not settled at the full pile"
        sto, sth = sh_range(FLO, FHI); assert np.array_equal(sto, sth), "the water level's shift not fixed"
        OUT16[r], SH[r], STOP[r], GATE[r] = V, sto, st, gate
        if measure is not None: knife_measure(measure, D, MD, c0r, nA, pl, Fr, br, pile, LPL)
        del D, MD
    if C is not None:
        px = x if plain_x is None else np.asarray(plain_x, np.int64)
        if plain_x is not None and pl.perm is not None: px = px[:, pl.perm]
        C.add(f"plain|{lay}|fc|meetings, plain", ones(px) @ pl.wrow)
        Ht = power(PILE, F) + bias[None, :]
        C.add(f"plain|{lay}|fc|divides", ones(PILE).sum(1) * ones(e))
        C.add(f"plain|{lay}|fc|biases", P * ones(bias).sum())
        C.add(f"plain|{lay}|fc|bends", (ones(Ht) * ones(share_of(Ht))).sum())
    return OUT16, SH, STOP, GATE, PILE, LPL

def knife_measure(Mz, D, MD, c0r, nA, pl, F, bias, pile, LPL):
    """MEASUREMENT ONLY (no count goes into the machine): the stop stage of every edge under
      (i)  settle.py's own rule in signed digits: the per-edge bound (m_nb of each neuron), tested at every stage;
      (ii) the shared bound (m_b at its largest over the neurons), tested at every stage, no gate.
    The intervals are made with base2's power (the values clipped at 2^45 only where they are far too wide to settle:
    a clipped lower end is still below the edge's table, a clipped upper end still above the true value)."""
    Pc, N = c0r.shape; NDG, KD, DIAG = grid(LPL); NSK = len(KD)
    KK = np.arange(pl.L + 1)
    MN = np.minimum(nA.T[:, :, None], KK[None, None, :]).astype(np.float64)        # (PL a, Pc, L+1)
    TABs = (np.int64(1) << DIAG)[None] * np.minimum(nA[:, :, None], pl.mbar[None, None, :])
    for rule in ("per-edge", "shared"):
        c = c0r.copy(); cm = np.zeros((Pc, N), np.int64); prev = 0
        st = np.full((Pc, N), -1, np.int8); FLO = np.zeros((Pc, N), np.int64); FHI = np.zeros((Pc, N), np.int64)
        work = 0
        for si, k in enumerate(KD):
            for d in range(NDG - k, NDG - prev): c += D[d].astype(np.int64) << d; cm += MD[d]
            prev = k; U = DIAG < NDG - k
            if rule == "shared": Rb = np.broadcast_to((TABs * U[None]).sum((1, 2))[:, None], (Pc, N))
            else:
                Rb = np.zeros((Pc, N), np.int64)
                if U.any():
                    Cm = (U.T * (1 << np.arange(LPL))[None, :]).astype(np.float64)  # (b, a)
                    G = np.einsum("ba,apm->bpm", Cm, MN).astype(np.int64)           # (PL b, Pc, L+1)
                    for b in range(PL): Rb += np.take_along_axis(G[b], np.broadcast_to(pl.mnb[b][None, :], (Pc, N)), 1) << b
            Hlo = power(_clip45(c - Rb), F) + bias[None, :]; Hhi = power(_clip45(c + Rb), F) + bias[None, :]
            fr = st >= 0
            LOc = np.where(fr, FLO, Hlo); HIc = np.where(fr, FHI, Hhi)
            slo, shi = sh_range(LOc, HIc)
            idx = np.flatnonzero(~fr.ravel())
            okc, v = out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo, N)[idx], np.repeat(shi, N)[idx])
            new = idx[okc]; pr, nn = new // N, new % N
            st[pr, nn] = si; FLO[pr, nn] = Hlo.ravel()[new]; FHI[pr, nn] = Hhi.ravel()[new]
            work += int(cm[pr, nn].sum())
        assert (st >= 0).all()
        Mz[f"{rule} work"] = Mz.get(f"{rule} work", 0) + work
        Mz[f"{rule} hist"] = Mz.get(f"{rule} hist", 0) + np.bincount(st.ravel(), minlength=NSK)
    Mz["full"] = Mz.get("full", 0) + int(MD.astype(np.int64).sum())

# ---------- layer 0's stored tables (entry/ and layer0/), memory-mapped ----------
class ETable:                                                 # entry/: rows read by word or by position
    def __init__(self, name):
        man = json.load(open(f"{ENT}/manifest.json")); t = man["tables"][name]; self.name, self.kind = name, t["kind"]
        ids = np.load(f"{ENT}/{t['ids']}").astype(np.int64)
        self.inv = np.full(int(ids.max()) + 1, -1, np.int64); self.inv[ids] = np.arange(len(ids))
        mm = lambda f: np.asarray(np.load(f"{ENT}/{f}.npy", mmap_mode="r"))
        if self.kind == "split40": self.lo, self.hi = mm(f"{name}_lo"), mm(f"{name}_hi")
        else: self.v = mm(name)
    def read(self, keys):
        keys = np.asarray(keys, np.int64); assert keys.max() < len(self.inv)
        r = self.inv[keys]; assert (r >= 0).all(), f"{self.name}: a row never stored"
        if self.kind == "split40": return self.hi[r].astype(np.int64) * (1 << 32) + self.lo[r]
        return self.v[r].astype(np.int64)
class LTable:                                                 # layer0/: rows read by sorted key
    def __init__(self, name):
        man = json.load(open(f"{L0D}/manifest.json")); t = man["tables"][name]; self.name, self.kind = name, t["kind"]
        self.ids = np.load(f"{L0D}/{t['ids']}").astype(np.int64); assert (np.diff(self.ids) > 0).all()
        mm = lambda f: np.asarray(np.load(f"{L0D}/{f}.npy", mmap_mode="r"))
        if self.kind == "split40": self.lo, self.hi = mm(f"{name}_lo"), mm(f"{name}_hi")
        else: self.v = mm(name)
    def rows(self, keys):
        keys = np.asarray(keys, np.int64); r = np.minimum(np.searchsorted(self.ids, keys), len(self.ids) - 1)
        return np.where(self.ids[r] == keys, r, -1)
    def has(self, keys): return self.rows(keys) >= 0
    def read(self, keys):
        r = self.rows(keys); assert (r >= 0).all(), f"{self.name}: a row never stored"
        if self.kind == "split40": return self.hi[r].astype(np.int64) * (1 << 32) + self.lo[r]
        return self.v[r].astype(np.int64)
_STORE = {}
def store():
    if not _STORE:
        _STORE.update(ZP0=ETable("ZP0"), Z1023=ETable("Z1023"), T={1: ETable("T1"), 2: ETable("T2")},
                      P={1: ETable("P1"), 2: ETable("P2")}, Z0=LTable("Z0"), FT={5: LTable("FT5"), 6: LTable("FT6")},
                      FP={5: LTable("FP5"), 6: LTable("FP6")}, OWN={1: LTable("OWN1"), 2: LTable("OWN2"), 3: LTable("OWN3")})
    return _STORE
FALLBACK = {"proj pairs not stored": 0, "proj shift not stored": 0, "fc shift not stored": 0}

# ---------- the arrival, base2's attention step for step, counted (combined/machine.py's rule) ----------
def arrival(QKV, C=None, who=("plain", "fr"), lay="", first_counted=0):
    """who counts every row for "plain", and rows from first_counted on for "fr" (layer 0: position 0 is read)."""
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64)
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        if C is not None:
            rowsc = np.tril(ones(q16) @ ones(k16).T).sum(1); ov = ones(v).sum(1)
            C.add(f"plain|{lay}|arrival|scores", rowsc); C.add(f"fr|{lay}|arrival|scores", rowsc[first_counted:])
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]; Lp = log2fix([a.sum()])[0]
            out[p, h*64:(h+1)*64] = power(num, -Lp)
            if C is not None:
                for w_ in who:
                    if w_ == "fr" and p < first_counted: continue
                    C.add(f"{w_}|{lay}|arrival|shares' total", ones(a).sum())
                    C.add(f"{w_}|{lay}|arrival|shares x values", (ones(a) * ov[:p+1]).sum())
                    C.add(f"{w_}|{lay}|arrival|divides", ones(num).sum() * E_ones([Lp])[0])
    return out

def count_meet(C, lay, part, pl, x, plain_x=None, fr=True):
    """plain: the light's 1s x the plate's 1s per line; fr: lit places x lit places (signed digits), and the writing."""
    px = x if plain_x is None else plain_x
    if pl.perm is not None: x = np.asarray(x)[:, pl.perm]; px = np.asarray(px)[:, pl.perm]
    C.add(f"plain|{lay}|{part}|meetings, plain", ones(px) @ pl.wrow)
    if fr:
        C.add(f"fr|{lay}|{part}|meetings, signed digits", naf(x) @ pl.nrow)
        C.add(f"fr|{lay}|{part}|light written in signed digits", ones(x).sum())
def count_both(C, lay, part, item, v): C.add(f"plain|{lay}|{part}|{item}", v); C.add(f"fr|{lay}|{part}|{item}", v)

# ---------- the machine, every text at once, layer by layer ----------
def forward(texts, C=None, check=None, chk=None, log=print, measure_layers=(), order_seed=None, ctl=None):
    """the fringes machine on every position of every text (each text's arrivals read only its own positions).
    check(key, value, rows): every pile (qkv, proj, mproj; fc's whole pile is CHECK ONLY), fc's output (out16 and its
    shift) and the stream after every half layer, keyed as recording/C_ref with "fc out16 l" / "fc sh l" added; rows
    (layer 0 only) names the positions the machine forms (position 0 reads its stream after layer 0 whole).
    chk(name, n): the check path's counts of failures (must all be 0).
    measure_layers: layers whose knife edges are also measured by settle.py's rule (knife_measure).
    order_seed: ORDER-FREE check: at every bank the lines (light and plate together) and the place pairs in a random
    order; at every fc plate the neurons in a random order (fc's columns, powers and biases); mproj's lines in a
    random order; the edges meet mproj in groups by their stop stage, the groups in a random order.
    ctl(kind, lay, pl, x, ...): CONTROL hook, called with every plate and the light that meets it (counts only).
    Returns the light at the top (x16, ex, LS) for every position, the knife's stop and gate stages, the measurements."""
    t0 = time.time(); S = store(); MEAS = {}
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in lens])
    first = pos == 0; r1 = np.flatnonzero(~first); f0 = np.flatnonzero(first); NP = len(ids)
    WTE, WPE = M["wte"], M["wpe"]
    A_rows, B_rows = WTE[ids], WPE[pos]; R = A_rows + B_rows
    orng = None if order_seed is None else np.random.default_rng(order_seed)
    def lp(L_): return None if orng is None else orng.permutation(L_)
    def keep(k, v, rows=None):
        if check is not None: check(k, v, rows)
    ck = chk
    if C is not None: count_both(C, "L0", "rest", "entry: word + position", ones(B_rows).sum())
    STOPS, GATES, LPLS = [], [], []

    def fc_knife(lay, l, R_, xlight, c0, x_plain, ex_, LS_, rows):
        """fc.l through the knife, then mproj.l: returns mproj's output added to the stream at rows."""
        FCPERM = None if orng is None else orng.permutation(3072)
        plf = Plate(f"fc.{l}", line_perm=lp(768), neuron_perm=FCPERM)
        bias_fc = BIAS[f"fc.{l}"] if FCPERM is None else BIAS[f"fc.{l}"][FCPERM]
        if c0 is not None and FCPERM is not None: c0 = c0[:, FCPERM]
        Ff = (plf.e[None, :] + ex_[:, None] + ES) * 65536 - LS_[:, None]
        if C is not None:
            count_both(C, lay, "rest", "water levels", water(x_plain))
            C.add(f"side|{lay}|fc|meetings, signed digits, whole light", naf(x_plain if plf.perm is None else x_plain[:, plf.perm]) @ plf.nrow)
            C.add(f"side|{lay}|fc|light written, whole light", ones(x_plain).sum())
        MEASl = {} if l in measure_layers else None
        if ctl is not None: ctl("fc", lay, plf, xlight, F=Ff, bias=bias_fc, c0=c0)
        out16, sh, stop, gate, fpile, lpl = knife(plf, xlight, c0, Ff, bias_fc, C, lay, plain_x=x_plain, chk=ck,
                                                  pair_rng=orng, measure=MEASl)
        LPLS.append(lpl)
        if MEASl is not None: MEAS[l] = MEASl
        if FCPERM is not None:                                # back to the machine's own neuron order
            inv = np.argsort(FCPERM); out16, stop, fpile = out16[:, inv], stop[:, inv], fpile[:, inv]
        STOPS.append(stop); GATES.append(gate)
        keep(f"fc pile {l}", fpile, rows); keep(f"fc out16 {l}", out16, rows); keep(f"fc sh {l}", sh, rows)
        del plf, fpile
        plm = Plate(f"mproj.{l}", line_perm=lp(3072)); wem, bm = plm.e, BIAS[f"mproj.{l}"]
        if ctl is not None: ctl("mproj", lay, plm, out16)
        mpile = mproj_meet(plm, out16, stop, orng); keep(f"mproj pile {l}", mpile, rows)
        if C is not None:
            count_meet(C, lay, "mproj", plm, out16); count_both(C, lay, "rest", "biases", len(out16) * ones(bm).sum())
        mo = rshift(mpile, -(wem[None, :] + (sh - ES)[:, None] + ES)) + bm
        if C is not None: count_both(C, lay, "rest", "stream adds", ones(mo).sum())
        return mo

    # ================= layer 0 =================
    pl = Plate("qkv.0", line_perm=lp(768)); we, b = pl.e, BIAS["qkv.0"]
    x16, ex, LS = gauge_log(R); s = ex + ES
    pile = np.empty((NP, pl.N), np.int64); carry = np.zeros((NP, 768), np.int64); carried = np.zeros(NP, bool)
    for sv in np.unique(s):
        m = np.flatnonzero(s == sv)
        if sv == 4:
            assert (pos[m] == 0).all(); z = S["ZP0"].read(ids[m]); pile[m] = z
            if C is not None: C.add("fr|L0|qkv|stored rows read and added", ones(z).sum())
        elif sv == 0:
            assert (pos[m] == 1023).all(); z = S["Z1023"].read(ids[m]); pile[m] = z
            if C is not None: C.add("fr|L0|qkv|stored rows read and added", ones(z).sum())
        elif sv in (1, 2):
            a_, b_ = rshift(A_rows[m], sv), rshift(B_rows[m], sv); c = x16[m] - a_ - b_; assert np.abs(c).max() <= 1
            t1, t2 = S["T"][sv].read(ids[m]), S["P"][sv].read(pos[m])
            pile[m] = t1 + t2; carry[m] = c; carried[m] = True
            if C is not None:
                C.add("fr|L0|qkv|stored rows read and added", ones(t1).sum() + ones(t2).sum())
                C.add("fr|L0|qkv|split (stored rounded rows taken off)", ones(a_).sum() + ones(b_).sum())
        else: raise ValueError(f"qkv.0 shift {sv} never occurs (entry/survey_s.py)")
    cr = np.flatnonzero(carried); nc = np.flatnonzero(~carried)
    if len(cr): pile[cr] += meet(pl, carry[cr], pair_rng=orng)
    if ctl is not None and len(cr): ctl("qkv", "L0", pl, carry[cr])
    if C is not None:
        if len(cr): count_meet(C, "L0", "qkv", pl, carry[cr], plain_x=x16[cr])
        if len(nc): C.add("plain|L0|qkv|meetings, plain", ones(x16[nc] if pl.perm is None else x16[nc][:, pl.perm]) @ pl.wrow)
        C.add("side|L0|qkv|meetings, signed digits, whole light", naf(x16 if pl.perm is None else x16[:, pl.perm]) @ pl.nrow)
        C.add("side|L0|qkv|light written, whole light", ones(x16).sum())
        count_both(C, "L0", "rest", "water levels", water(x16))
        count_both(C, "L0", "rest", "bank divides", ones(pile).sum(1) * E_ones(LS))
        count_both(C, "L0", "rest", "biases", NP * ones(b).sum())
    keep("qkv pile 0", pile)
    QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, pl
    # the arrival: every text (position 0's own output is never used by the machine, and not counted for it)
    o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C, lay="L0", first_counted=1) for t in range(len(texts))])
    # position 0: the stream after layer 0 is read whole
    R2 = np.empty_like(R)
    z0 = S["Z0"].read(ids[f0]); R2[f0] = z0
    if C is not None: C.add("fr|L0|rest|position 0: the stream after layer 0 read", ones(z0).sum())
    # proj.0 (positions 1 on): the own value's stored tip read, only o16 - own met
    pl = Plate("proj.0", line_perm=lp(768)); wep, bp = pl.e, BIAS["proj.0"]
    o16, eo = to16(o[r1], -ES); so = eo + ES; keys = ids[r1] * 1024 + pos[r1]
    rest_o = o16.copy(); stored = np.zeros((len(r1), 768), np.int64)
    for sv in np.unique(so):
        m = np.flatnonzero(so == sv)
        has = S["OWN"][sv].has(keys[m]) if sv in S["OWN"] else np.zeros(len(m), bool)
        st_, nf = m[has], m[~has]
        if len(st_):
            ref = rshift(QKV[r1[st_], 1536:], sv); stored[st_] = S["OWN"][sv].read(keys[st_]); rest_o[st_] = o16[st_] - ref
            if C is not None:
                C.add("fr|L0|proj|stored rows read and added", ones(stored[st_]).sum())
                C.add("fr|L0|proj|split (own value taken off)", ones(ref).sum())
        if len(nf): FALLBACK["proj shift not stored" if sv not in S["OWN"] else "proj pairs not stored"] += len(nf)
    ppile = stored + meet(pl, rest_o, pair_rng=orng); keep("proj pile 0", ppile, r1)
    if ctl is not None: ctl("proj", "L0", pl, rest_o)
    if C is not None:
        count_meet(C, "L0", "proj", pl, rest_o, plain_x=o16)
        C.add("side|L0|proj|meetings, signed digits, whole light", naf(o16 if pl.perm is None else o16[:, pl.perm]) @ pl.nrow)
        C.add("side|L0|proj|light written, whole light", ones(o16).sum())
    att = rshift(ppile, -(wep[None, :] + eo[:, None] + ES)) + bp; del pl, ppile
    if C is not None:
        count_both(C, "L0", "rest", "biases", len(r1) * ones(bp).sum()); count_both(C, "L0", "rest", "stream adds", ones(att).sum())
    R1 = R[r1] + att; keep("stream a 0", R1, r1)
    # fc.0 (positions 1 on): the word's and position's stored parts read, the rest met through the knife
    x1, e1, LS1 = gauge_log(R1); s1 = e1 + ES
    rest = x1.copy(); c0 = np.zeros((len(r1), 3072), np.int64); BPROJ = BIAS["proj.0"]
    for sv in np.unique(s1):
        m = np.flatnonzero(s1 == sv)
        if sv in S["FT"]:
            a_ = rshift(WTE[ids[r1[m]]], sv); b_ = rshift(WPE[pos[r1[m]]] + BPROJ[None, :], sv)
            rest[m] = x1[m] - a_ - b_; c0[m] = S["FT"][sv].read(ids[r1[m]]) + S["FP"][sv].read(pos[r1[m]])
            if C is not None:
                C.add("fr|L0|fc|stored rows read and added", ones(c0[m]).sum())
                C.add("fr|L0|fc|split (stored rounded rows taken off)", ones(a_).sum() + ones(b_).sum())
        else: FALLBACK["fc shift not stored"] += len(m)
    mo = fc_knife("L0", 0, R1, rest, c0, x1, e1, LS1, r1)
    R2[r1] = R1 + mo; del mo, rest, c0
    # the CHECK at position 0 (and the plain count there): base2.py's own layer 0 from the same light
    check_position0(R, QKV, o, f0, R2, C, ck)
    keep("stream 0", R2)
    R = R2; del QKV, o, R1, R2
    log(f"    layer  0 done ({time.time() - t0:.0f}s, {rss()} MB)")
    # ================= layers 1-11 =================
    for l in range(1, 12):
        lay = f"L{l}"
        pl = Plate(f"qkv.{l}", line_perm=lp(768)); we, b = pl.e, BIAS[f"qkv.{l}"]
        x16, ex, LS = gauge_log(R); pile = meet(pl, x16, pair_rng=orng); keep(f"qkv pile {l}", pile)
        if ctl is not None: ctl("qkv", lay, pl, x16)
        if C is not None:
            count_meet(C, lay, "qkv", pl, x16)
            count_both(C, lay, "rest", "water levels", water(x16)); count_both(C, lay, "rest", "bank divides", ones(pile).sum(1) * E_ones(LS))
            count_both(C, lay, "rest", "biases", NP * ones(b).sum())
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, pl
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C, lay=lay) for t in range(len(texts))]); del QKV
        pl = Plate(f"proj.{l}", line_perm=lp(768)); wep, bp = pl.e, BIAS[f"proj.{l}"]
        o16, eo = to16(o, -ES); pile = meet(pl, o16, pair_rng=orng); keep(f"proj pile {l}", pile)
        if ctl is not None: ctl("proj", lay, pl, o16)
        if C is not None: count_meet(C, lay, "proj", pl, o16)
        att = rshift(pile, -(wep[None, :] + eo[:, None] + ES)) + bp; del pile, pl, o
        if C is not None: count_both(C, lay, "rest", "biases", NP * ones(bp).sum()); count_both(C, lay, "rest", "stream adds", ones(att).sum())
        R = R + att; keep(f"stream a {l}", R); del att
        x16, ex, LS = gauge_log(R)
        mo = fc_knife(lay, l, R, x16, None, x16, ex, LS, None)
        R = R + mo; keep(f"stream {l}", R); del mo
        log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex, LS = gauge_log(R)
    if C is not None: count_both(C, "top", "rest", "top water level", water(x16))
    return (x16, ex, LS), STOPS, GATES, MEAS, LPLS

def mproj_meet(plm, out16, stop, orng):
    """each edge meets mproj once. Order is free: in the ORDER-FREE run the edges meet in groups by their stop stage,
    the groups in a random order, and the pile is the sum of the groups' piles."""
    if orng is None: return meet(plm, out16)
    tot = np.zeros((len(out16), plm.N), np.int64)
    for g in orng.permutation(int(stop.max()) + 1):
        xg = np.where(stop == g, out16, 0)
        if xg.any(): tot = tot + meet(plm, xg, pair_rng=orng)
    return tot

def check_position0(R, QKV, o, f0, R2, C, ck):
    """position 0 of every text: base2.py's own layer 0 from the same light (the plates' values rebuilt from the
    record), against the stored stream Z0; and the plain count there (the machine itself reads Z0)."""
    if len(f0) == 0: return
    lay = "L0"
    plp = PLATE_VALUES("proj.0"); o16, eo = to16(o[f0], -ES)
    pp = product(o16, plp[0].astype(np.float64))
    att = rshift(pp, -(plp[1][None, :] + eo[:, None] + ES)) + BIAS["proj.0"]
    R1_0 = R[f0] + att
    plf = PLATE_VALUES("fc.0"); x1, e1, LS1 = gauge_log(R1_0)
    fp = product(x1, plf[0].astype(np.float64))
    H = power(fp, (plf[1][None, :] + e1[:, None] + ES) * 65536 - LS1[:, None]) + BIAS["fc.0"]
    G = edge(H); g16, eg = to16(G, -ES)
    plm = PLATE_VALUES("mproj.0"); mp = product(g16, plm[0].astype(np.float64))
    mo = rshift(mp, -(plm[1][None, :] + eg[:, None] + ES)) + BIAS["mproj.0"]
    if ck is not None: ck("position 0: base2's own layer 0 = the stored stream Z0", int((R1_0 + mo != R2[f0]).any(1).sum()))
    if C is not None:                                         # position 0's plain count, under its own labels
        P0 = len(f0); q = " (position 0)"
        C.add(f"plain|{lay}|proj|meetings, plain{q}", ones(o16) @ ones(plp[0]).sum(1))
        C.add(f"plain|{lay}|rest|biases{q}", P0 * (ones(BIAS["proj.0"]).sum() + ones(BIAS["mproj.0"]).sum()))
        C.add(f"plain|{lay}|rest|stream adds{q}", ones(att).sum() + ones(mo).sum())
        C.add(f"plain|{lay}|rest|water levels{q}", water(x1))
        C.add(f"plain|{lay}|fc|meetings, plain{q}", ones(x1) @ ones(plf[0]).sum(1))
        C.add(f"plain|{lay}|fc|divides{q}", ones(fp).sum(1) * E_ones(LS1))
        C.add(f"plain|{lay}|fc|biases{q}", P0 * ones(BIAS["fc.0"]).sum())
        C.add(f"plain|{lay}|fc|bends{q}", (ones(H) * ones(share_of(H))).sum())
        C.add(f"plain|{lay}|mproj|meetings, plain{q}", ones(g16) @ ones(plm[0]).sum(1))
        C.add(f"side|{lay}|proj|meetings, signed digits, position 0", naf(o16) @ (naf(plp[0]).sum(1)))
        C.add(f"side|{lay}|fc|meetings, signed digits, position 0", naf(x1) @ (naf(plf[0]).sum(1)))
        C.add(f"side|{lay}|mproj|meetings, signed digits, position 0", naf(g16) @ (naf(plm[0]).sum(1)))
        C.add(f"side|{lay}|proj|light written, whole light", ones(o16).sum())
        C.add(f"side|{lay}|fc|light written, whole light", ones(x1).sum())
        C.add(f"side|{lay}|mproj|light written, whole light", ones(g16).sum())
_PV = {}
def PLATE_VALUES(name):                                      # a plate's whole numbers and powers, rebuilt from the record
    if name not in _PV: _PV[name] = table_values(name)
    return _PV[name]

# ---------- the top read, carried down the fine diagonals, the set-aside test at the pile's own scale ----------
TPL = 15; TNDG = 2 * TPL - 1                                  # plain photons: places 0..14, 29 diagonals
KS = list(range(2, TNDG, 2))                                  # the top 2, 4, ..., 28, then all
NS = len(KS)
_a, _b = np.arange(TPL)[:, None], np.arange(TPL)[None, :]
TCS = [np.broadcast_to(_a + _b >= TNDG - k, (TPL, TPL)).copy() for k in KS] + [np.ones((TPL, TPL), bool)]
TNEW = [TCS[0]] + [TCS[s] & ~TCS[s - 1] for s in range(1, NS + 1)]
assert (sum(n.astype(int) for n in TNEW) == 1).all()
STAGE_OF = np.zeros((TPL, TPL), np.int64)
for _s, _n in enumerate(TNEW): STAGE_OF[_n] = _s
POWA = (1 << np.arange(TPL)).astype(np.int64); NNL = np.arange(769); ONES_NN = ones(NNL)
def _amask(Cm): return np.array([sum(1 << a for a in range(TPL) if Cm[a, b]) for b in range(TPL)], np.int64)
NMASK = np.stack([_amask(n) for n in TNEW])

class TopRead:
    def __init__(self, check=True, word_perm=None, line_perm=None, log=print):
        t0 = time.time()
        T16, e = table_values("head"); assert np.array_equal(e, HEAD_E) and (HEAD_E == HEAD_E[0]).all()
        self.bias = HEAD_B.copy(); self.wperm = np.arange(T16.shape[1])
        if word_perm is not None: T16 = T16[:, word_perm]; self.bias = self.bias[word_perm]; self.wperm = word_perm
        self.lperm = line_perm
        if line_perm is not None: T16 = T16[line_perm]
        self.NW = T16.shape[1]
        TT = np.ascontiguousarray(T16.T)
        assert np.abs(TT.astype(np.int32)).max() < 1 << TPL
        self.TA = np.abs(TT.astype(np.int32)).astype(np.uint16); self.TS = np.sign(TT).astype(np.int8); del TT
        self.NT = np.stack([((self.TA >> b) & 1).sum(1, dtype=np.int64) for b in range(TPL)], 1)      # m_wb
        self.PPH = np.bitwise_count(self.TA).sum(0, dtype=np.int64)                                 # 1s per line
        self.CNT_ALL = np.stack([((self.TA >> b) & 1).sum(0, dtype=np.int64) for b in range(TPL)])   # words lit at b, per line
        self.PB8 = [(self.TS * ((self.TA >> b) & 1).astype(np.int8)) for b in range(TPL)]           # (NW, 768) int8
        self.WH = T16.astype(np.float64) if check else None                                          # CHECK ONLY
        self.BIAS_ONES = ones(self.bias); self.BMAX = int(np.abs(self.bias).max())
        OFF = 769 * np.arange(TPL)[None, :]; self.IX = self.NT + OFF
        self.HN = np.stack([np.bincount(self.NT[:, b], minlength=769) for b in range(TPL)]).ravel()
        log(f"  top table rebuilt from the record's sheets; planes made ({time.time() - t0:.0f}s, {rss()} MB)")

    def table(self, nvec, mask):                              # G[b, m] = sum over a with mask[a, b] of 2^a min(n_a, m)
        return ((mask.T * POWA[None, :]) @ np.minimum(nvec[:, None], NNL[None, :]))
    def table_photons(self, nvec, mask):                      # combined's count of building one table from the light
        Q = np.array([ONES_NN[:n].sum() + (769 - n) * ones(n) for n in nvec])
        return int((mask * Q[:, None]).sum())

    def read(self, X, EX, LS, C=None, who="fr", lay="top", bad=None, keepn=8):
        """the top read at every position given. Returns dict: top5, win, surv (per stage), plain top5/win (check)."""
        X = np.asarray(X, np.int64)
        if self.lperm is not None: X = X[:, self.lperm]
        N = len(X); NW = self.NW
        out = dict(top5=np.zeros((N, 5), np.int64), win=np.zeros(N, np.int64), surv=np.zeros((N, NS), np.int64),
                   ptop5=np.zeros((N, 5), np.int64), pwin=np.zeros(N, np.int64), work=np.zeros((N, NS + 1), np.int64),
                   visited=np.zeros((N, NS), np.int64))
        bad = {} if bad is None else bad
        def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
        BATCH = 25
        for c0 in range(0, N, BATCH):
            Xb = X[c0:c0 + BATCH]; P = len(Xb); ax, sx = np.abs(Xb), np.sign(Xb); assert ax.max() < 1 << TPL
            F = (HEAD_E[0] + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
            if self.WH is not None:
                PILE = product(Xb, self.WH); S = power(PILE, F[:, None]) + self.bias[None, :]
            # the counted piles, stage by stage: every (light place a, table place b) count is a whole number <= 768
            LA = np.stack([(sx * ((ax >> a) & 1)) for a in range(TPL)]).astype(np.float32).reshape(TPL * P, 768)
            DC = np.zeros((NS + 1, P, NW), np.int64)
            for b in range(TPL):
                Cab = (LA @ self.PB8[b].astype(np.float32).T).reshape(TPL, P, NW)
                for a in range(TPL): DC[STAGE_OF[a, b]] += Cab[a].astype(np.int64) << (a + b)
                del Cab
            CNT = np.cumsum(DC, axis=0)
            if self.WH is not None: nb("pieces vs the plain read's piles", (CNT[NS] != PILE).sum())
            ALV = np.zeros((NS + 1, P, NW), np.float32); ALV[0] = 1
            for p in range(P):
                q = c0 + p
                nx = ((ax[p][None, :] >> np.arange(TPL)[:, None]) & 1).sum(1)
                Fp = int(F[p]); I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
                F2 = -Fp; I2 = F2 >> 16; e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - I2
                beta = power(self.bias, F2)                                       # each word's standing amount at pile scale
                # gamma: |beta - bias 2^s / e| <= 1/2 + |bias| 2^s |e e2 - 2^(s+s2)| / (e 2^(s+s2)), at the largest |bias|
                twos = Fraction(2) ** s; twot = Fraction(2) ** (s + s2)
                g = Fraction(1, 2) + self.BMAX * twos * abs(e * e2 - twot) / (e * twot)
                gamma = int(-(-g.numerator // g.denominator)); delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator))))
                mu = 2 * gamma + delta
                ge = int(ones(e)); ge2 = int(ones(e2))
                if C is not None:
                    C.add(f"{who}|{lay}|top read|standing amounts written at pile scale (divides)", int(self.BIAS_ONES.sum()) * ge2)
                    C.add(f"{who}|{lay}|top read|standing amounts poured into the piles", ones(beta).sum())
                # the bound: built whole once (after stage 1), then carried down
                G0 = self.table(nx, ~TCS[0]); OG0 = ones(G0).ravel(); G02 = (G0 << np.arange(TPL)[:, None]).ravel()
                Rw = G02.take(self.IX).sum(1)                                     # (NW,)
                U = CNT[0, p] + beta + Rw
                if C is not None:
                    C.add(f"{who}|{lay}|top read|bound tables built from the light", self.table_photons(nx, ~TCS[0]))
                    C.add(f"{who}|{lay}|top read|bound adds (table values)", int((self.HN * OG0).sum()))
                    C.add(f"{who}|{lay}|top read|bound poured into the highs", ones(Rw).sum())
                alive = np.arange(NW)
                for st in range(NS):
                    if st > 0:
                        Gn = self.table(nx, TNEW[st]); OGn = ones(Gn).ravel(); Gn2 = (Gn << np.arange(TPL)[:, None]).ravel()
                        ix = self.IX[alive]; dR = Gn2.take(ix).sum(1)
                        U = U + DC[st, p, alive] - dR; Rw = Rw - dR
                        if C is not None:
                            C.add(f"{who}|{lay}|top read|bound tables built from the light", self.table_photons(nx, TNEW[st]))
                            C.add(f"{who}|{lay}|top read|bound adds (table values)", int(OGn.take(ix).sum()))
                            C.add(f"{who}|{lay}|top read|bound carried down (high and bound)", 2 * ones(dR).sum())
                    if self.WH is not None:                                       # CHECK: c - R <= pile <= c + R
                        cnt = U - beta[alive] - Rw
                        nb("counted pile carried = the counted pile", (cnt != CNT[st, p, alive]).sum())
                        nb("bound broken", ((PILE[p, alive] < cnt - Rw) | (PILE[p, alive] > cnt + Rw)).sum())
                    # can any word be set aside here? Every low is at most its high less twice the smallest bound, so
                    # the 5th-best low is at most (5th-best high) - 2 (smallest bound). If even the lowest high is above
                    # that less mu, no word can be set aside: nothing is read (found with compares only).
                    na = len(alive); U5 = np.partition(U, na - 5)[na - 5]
                    if U.min() > U5 - 2 * Rw.min() - mu:
                        if C is not None: C.add(f"cmp|{lay}|top read|compares", 3 * na)
                        out["visited"][q, st] = 0; out["surv"][q, st] = na; ALV[st + 1, p, alive] = 1
                        continue
                    # the 5th-best low at pile scale, from the words in order of their highs, only as far as needed
                    K = min(64, na)
                    while True:
                        top = np.argpartition(-U, K - 1)[:K] if K < na else np.arange(na)
                        ell = U[top] - 2 * Rw[top]
                        l5 = np.partition(ell, len(ell) - 5)[len(ell) - 5]
                        if K == na: break
                        rest_ = np.ones(na, bool); rest_[top] = False
                        if l5 >= U[rest_].max(): break
                        K = min(na, 2 * K)
                    LAM = l5 - mu
                    keepw = U > LAM
                    if C is not None:
                        C.add(f"{who}|{lay}|top read|lows of the words read for the 5th-best low", ones(Rw[top]).sum())
                        C.add(f"cmp|{lay}|top read|compares", na + K)
                    out["visited"][q, st] = K
                    if self.WH is not None:                                       # CHECK: every word set aside is below 5
                        s5 = np.partition(S[p], NW - 5)[NW - 5]
                        nb("a word set aside that is not below the 5th-best", (S[p, alive[~keepw]] >= s5).sum())
                    alive = alive[keepw]; U = U[keepw]; Rw = Rw[keepw]
                    out["surv"][q, st] = len(alive); ALV[st + 1, p, alive] = 1
                # the words still in, read to the end
                pile = U - Rw + DC[NS, p, alive] - beta[alive]                    # the last pairs read, their bound off
                sc = power(pile, Fp) + self.bias[alive]
                if self.WH is not None: nb("words read to the end vs the plain read", (sc != S[p, alive]).sum())
                if C is not None:
                    C.add(f"{who}|{lay}|top read|words read to the end: standing amount taken off, divide, standing amount",
                          ones(beta[alive]).sum() + ones(Rw).sum() + ones(pile).sum() * ge + self.BIAS_ONES[alive].sum())
                Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
                out["top5"][q] = self.wperm[top5(Sp[None])[0]]; out["win"][q] = self.wperm[race(Sp)]
                if self.WH is not None:
                    out["ptop5"][q] = self.wperm[top5(S[p:p + 1])[0]]; out["pwin"][q] = self.wperm[race(S[p])]
                    if C is not None:
                        C.add(f"plain|{lay}|top read|meetings, plain", int(ones(Xb[p]) @ self.PPH))
                        C.add(f"plain|{lay}|rest|top divide (every word)", ones(PILE[p]).sum() * ge)
                        C.add(f"plain|{lay}|rest|top standing amounts (every word)", int(self.BIAS_ONES.sum()))
            del DC, CNT
            # the meetings: each stage's new pairs, met by the words still in
            A2 = ALV.reshape((NS + 1) * P, NW); work = np.zeros((NS + 1, P), np.int64); allp = np.zeros(P, np.int64)
            for b in range(TPL):
                Bb = np.abs(self.PB8[b]).astype(np.float32)
                CN = (A2 @ Bb).astype(np.int64).reshape(NS + 1, P, 768)
                for st in range(NS + 1):
                    Lc = np.bitwise_count(ax & NMASK[st, b]).astype(np.int64)
                    work[st] += (Lc * CN[st]).sum(1)
                    allp += (Lc * self.CNT_ALL[b][None, :]).sum(1)
                del Bb, CN
            nb("all pairs' meetings = the plain read's", (allp != (np.bitwise_count(ax).astype(np.int64) * self.PPH[None, :]).sum(1)).sum())
            out["work"][c0:c0 + P] = work.T
            if C is not None: C.add(f"{who}|{lay}|top read|meetings, plain photons, staged", work.sum())
            del ALV, A2
        out["bad"] = bad
        return out

def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
