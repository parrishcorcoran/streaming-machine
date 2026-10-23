"""PHASE C: THE RECORDING MACHINE (the library; C_run_refs.py, C_controls.py and C_windows.py use it).

It runs from the files in recording/C/ alone. base2.py's own steps are read from base2.py's source with its number
file swapped for recording/C/rest.npz and its plate list emptied, so base2_gpt2.npz is never opened here.

THE RECORDING (C_write.py): every weight of the 48 plates and the top table in signed digits, 17 places, an UP sheet
and a DOWN sheet per place, packed over the lines, each neuron's lines in its line-up order (the order stored), each
neuron's own power of two. Move 1 alone, no turn.

THE PILE, at each of the 48 banks:
  - the token's light (x16 from the machine's own water level: gauge_log before qkv and fc, to16 before proj and
    mproj) is written in signed digits once per bank: up and down sheets for places 0..16, no two lit side by side;
  - the plate is laid from its files: each sheet's bits are put back on their own lines by the stored order (once per
    bank per run), so each line's light meets the digits of that line's own weight;
  - pile = sum over place pairs (a of the light, b of the plate) of 2^(a+b) x [ (up.up + down.down) - (up.down +
    down.up) ], each of the four a count over the lines of the lines where both sheets are lit;
  - the counting: the 0/1 sheets meet as 0/1 matrices (every count is a whole number of at most 3,072 lines, exact in
    any order), then each place pair's count is shifted by a + b and added into the pile as a whole number.
Bends, water levels, arrivals and the top read are untouched: base2.py's own gauge_log, to16, power, rshift, edge and
attention. The top read is base2's plain read; its table is rebuilt from the recording's top-table sheets.

WORK, in the unit of all earlier results (combined/machine.py's count; whole machine 3,666,596,259 a token on the 648
positions of the 30 refs texts):
  - plain bank meetings: on each line, the light's 1s x the plate's 1s (plain base 2)
  - recorded bank meetings (lit-pixel meetings): on each line, the light's lit places x the plate's lit places, both
    in signed digits
  - every other part counted exactly as combined/machine.py counts it (arrival, water levels, divides, biases, bends,
    stream adds, the top read's meetings, divide and standing amounts)
  - extra adds, stated apart: writing the light in signed digits (3n = n + 2n: the light's 1s, once per bank)
  - CONTROL: the recorded meetings with each position's light shuffled across its lines (same values)

PREDICTIONS (written before anything here ran)
  The task's:
  - every pile identical, judge5 6000/6000
  - lit-pixel meetings about 44-50% of plain in the banks
  - raw sheets larger than 16 bits a weight, packed sheets close to the 153 MB floor (not below it)
  Claude's:
  - every pile, the stream after every half layer and the light at the top identical to base2.py on all 30 texts (648
    positions); top 5 and race winner identical 648/648; judge5 6000/6000 in the top 5 and 6000/6000 the same top word.
    The place-pair sum is the product written out place by place.
  - Bank meetings 57-61% of plain, not 44-50%. Signed digits alone take the plate's lit places to 75% and the light's
    to about 76% (shrink/plates_results.json: 6.27 -> 4.73 a weight), and 0.75 x 0.77 is about 0.58
    (lit_formats.py printed 58-60% on 5-token pieces). The 38% of section 18 needed the gaps as well.
  - Whole machine 68-71% of 3,666,596,259 (only the banks change: plain banks 2,738M a token, top read 904M, the
    rest 25M, from combined/refs_counts.json). Writing the light adds under 1M a token.
  - Light shuffled across its lines: the same share within 1 point. The saving is the format's, set by the values'
    sizes, not by which line meets which.
  - Lit places a light value: about 7 -> 5.3.
  - Order-free: the same piles in any order of lines and place pairs. Shuffle control: piles break at exactly the
    moved neurons and nowhere else, while the bank's lit-pixel meetings stay identical (moving whole neurons keeps
    every line's lit total).
  - Storage, in C_store.py's docstring.
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; CDIR = f"{HERE}/recording/C"; PLACES = 17

# ---------- base2.py's own steps, its numbers from recording/C/rest.npz ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{CDIR}/rest.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _a in _src, _a; _src = _src.replace(_a, _b)
assert "base2_gpt2" not in _src
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert B["W"] == {}                                          # no plate was read by base2's own loader
M = B["M"]; ES, SC, ARR, EXP2, PHI = B["ES"], B["SC"], B["ARR"], B["EXP2"], B["PHI"]
rshift, to16, product, race, log2fix, power, gauge_log = (B[k] for k in ("rshift", "to16", "product", "race", "log2fix", "power", "gauge_log"))
attention, edge = B["attention"], B["edge"]
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
BIAS = {n: M[n + ".b"] for n in BANKS}; HEAD_B = M["head.b"]; HEAD_E = M["head.e"]
MAN = json.load(open(f"{CDIR}/manifest.json"))
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def per_token(total, n): return (total + n // 2) // n
def pct2(a, b): return f"{(10000 * a + b // 2) // b / 100:.2f}"   # a share of b in hundredths of a percent, whole numbers

# ---------- photons ----------
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def signed_digits(v):                                        # (up, down) place masks; no two lit side by side
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def E_ones(LS): return ones(EXP2[np.bitwise_and(-np.asarray(LS, dtype=np.int64), 0xFFFF)])   # the gain's photons
def water(x16):                                              # the gauge's sums: S1 adds, S2 squares, 768 S2 - S1^2
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
class Count(dict):                                           # key "who|part|name"
    def add(self, k, v): self[k] = self.get(k, 0) + int(np.sum(v))

# ---------- reading the recording ----------
def unpack_order(buf, lines, neurons, nb):
    n = lines * neurons; out = np.empty(n, np.int64); sh = np.arange(nb - 1, -1, -1, dtype=np.int64); step = 1 << 20
    for i in range(0, n, step):
        m = min(step, n - i); b0 = i * nb // 8
        bits = np.unpackbits(buf[b0:b0 + (m * nb + 7) // 8])[:m * nb].reshape(m, nb).astype(np.int64)
        out[i:i + m] = (bits << sh[None, :]).sum(1)
    return out.reshape(lines, neurons)
def read_files(name):
    r = MAN["plates"][name]; L, N = r["lines"], r["neurons"]
    up = np.fromfile(f"{CDIR}/{r['files']['up']['file']}", np.uint8).reshape(PLACES, L // 8, N)
    dn = np.fromfile(f"{CDIR}/{r['files']['down']['file']}", np.uint8).reshape(PLACES, L // 8, N)
    o = unpack_order(np.fromfile(f"{CDIR}/{r['files']['order']['file']}", np.uint8), L, N, r["order_bits"])
    e = np.fromfile(f"{CDIR}/{r['files']['power']['file']}", np.dtype(r["power_dtype"]))
    return up, dn, o, e
def sheets_on_lines(up, dn, inv, p, cols=slice(None)):       # place p's two sheets, each bit back on its own line
    u = np.take_along_axis(np.unpackbits(up[p][:, cols], axis=0), inv, 0)    # inv[i, n]: the rank where line i sits
    d = np.take_along_axis(np.unpackbits(dn[p][:, cols], axis=0), inv, 0)
    return u, d
def table_values(name):                                      # a plate's whole numbers, rebuilt from its sheets (int16)
    up, dn, o, e = read_files(name); L, N = o.shape; out = np.empty((L, N), np.int16)
    for c0 in range(0, N, 8192):
        sl = slice(c0, c0 + 8192); w = np.zeros((L, min(8192, N - c0)), np.int64); inv = np.argsort(o[:, sl], axis=0)
        for p in range(PLACES):
            u, d = sheets_on_lines(up, dn, inv, p, sl); w += (u.astype(np.int64) - d.astype(np.int64)) << p
        out[:, sl] = w
    return out, e

class Plate:
    """a bank's plate laid from its files: PS[b] = [UP sheet of place b | DOWN sheet of place b] on the lines, 0/1.
    line_perm (ORDER-FREE check): the lines in a random order (the light's lines go with them).
    neuron_src (SHUFFLE CONTROL): neuron j is given the sheets of neuron neuron_src[j] (same values, wrong neurons)."""
    def __init__(self, name, line_perm=None, neuron_src=None):
        up, dn, o, e = read_files(name); L, N = o.shape
        self.name, self.L, self.N, self.e, self.perm = name, L, N, e, line_perm
        self.PS = np.empty((PLACES, L, 2 * N), np.float32)
        nrow = np.zeros(L, np.int64); w = np.zeros((L, N), np.int64); self.top = -1; inv = np.argsort(o, axis=0)
        for p in range(PLACES):
            u, d = sheets_on_lines(up, dn, inv, p)
            if neuron_src is not None: u, d = u[:, neuron_src], d[:, neuron_src]
            self.PS[p, :, :N] = u; self.PS[p, :, N:] = d
            nrow += u.sum(1, dtype=np.int64) + d.sum(1, dtype=np.int64)
            w += (u.astype(np.int64) - d.astype(np.int64)) << p
            if u.any() or d.any(): self.top = p
        self.nrow = nrow                                      # lit places on each line, signed digits, all neurons
        self.wrow = ones(w).sum(1)                            # 1s on each line, plain base 2, all neurons
        self.lit_signed, self.lit_plain, self.count = int(nrow.sum()), int(self.wrow.sum()), w.size
        if line_perm is not None: self.PS = np.ascontiguousarray(self.PS[:, line_perm, :])
        del w
    def pile(self, x, chunk=256, pair_rng=None):
        """pile = sum over place pairs (a, b) of 2^(a+b) x [(up.up + down.down) - (up.down + down.up)], counted over the
        lines. pair_rng (ORDER-FREE check): every count of a piece of positions found first, then the 289 place pairs
        added in a random order."""
        x = np.asarray(x, np.int64); P, N = len(x), self.N
        if self.perm is not None: x = x[:, self.perm]
        if pair_rng is not None: chunk = 16
        out = np.empty((P, N), np.int64)
        for c0 in range(0, P, chunk):
            xs = x[c0:c0 + chunk]; Pc = len(xs)
            up, dn = signed_digits(xs); lit = up | dn             # the light written in signed digits, once per bank
            assert not (up & dn).any() and not (lit & (lit >> 1)).any() and not (lit >> PLACES).any()
            LS = np.empty((2, PLACES, Pc, self.L), np.float32)
            back = np.zeros_like(xs)
            for a in range(PLACES):
                ua, da = (up >> a) & 1, (dn >> a) & 1; LS[0, a] = ua; LS[1, a] = da; back += (ua - da) << a
            assert np.array_equal(back, xs)
            LS = LS.reshape(2 * PLACES * Pc, self.L)
            acc = np.zeros((Pc, N), np.int64)
            if pair_rng is not None: CNT = np.empty((PLACES, PLACES, Pc, N), np.int32)
            for b in range(PLACES):
                Rb = (LS @ self.PS[b]).reshape(2, PLACES, Pc, 2 * N)   # [light up | light down] x [plate up | plate down]
                uu, ud, du, dd = Rb[0, :, :, :N], Rb[0, :, :, N:], Rb[1, :, :, :N], Rb[1, :, :, N:]
                S = ((uu + dd) - (ud + du)).astype(np.int64)          # per light place a: whole numbers, |S| <= 2 x lines
                if pair_rng is None:
                    for a in range(PLACES): acc += S[a] << (a + b)
                else: CNT[b] = S
                del Rb, uu, ud, du, dd, S
            if pair_rng is not None:
                for k in pair_rng.permutation(PLACES * PLACES):
                    b, a = divmod(int(k), PLACES); acc += CNT[b, a].astype(np.int64) << (a + b)
                del CNT
            out[c0:c0 + Pc] = acc
        assert np.abs(out).max() < 2 ** 62
        return out

# ---------- counting the untouched parts (combined/machine.py's own formulas) ----------
def count_arrival(QKV, C):                                   # base2's attention's work, as machine.py's arrival counts it
    P = QKV.shape[0]
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        C.add("both|arrival|scores", np.tril(ones(q16) @ ones(k16).T).sum()); ov = ones(v).sum(1)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]; Lp = log2fix([a.sum()])[0]
            C.add("both|arrival|shares' total", ones(a).sum()); C.add("both|arrival|shares x values", (ones(a) * ov[:p+1]).sum())
            C.add("both|arrival|divides", ones(num).sum() * E_ones([Lp])[0])
def count_bend(H, C):
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    C.add("both|rest|bends", (ones(H) * ones(share)).sum())
def count_bank(C, pl, x16, rng=None):
    kind = pl.name.split(".")[0]; nx = naf(x16)
    C.add(f"plain|banks|{kind} meetings, plain", ones(x16) @ pl.wrow)
    C.add(f"rec|banks|{kind} lit-pixel meetings, signed digits", nx @ pl.nrow)
    C.add(f"bank|{pl.name}|plain", ones(x16) @ pl.wrow); C.add(f"bank|{pl.name}|rec", nx @ pl.nrow)
    C.add("rec|extra|light written in signed digits", ones(x16).sum())
    C.add(f"lit|light|{kind} values", x16.size); C.add(f"lit|light|{kind} plain 1s", ones(x16).sum())
    C.add(f"lit|light|{kind} signed lit places", nx.sum())
    if rng is not None:                                       # CONTROL: each position's light shuffled across its lines
        C.add(f"ctl|banks|{kind} lit-pixel meetings, light shuffled across its lines",
              sum(int(nx[p][rng.permutation(x16.shape[1])] @ pl.nrow) for p in range(len(x16))))

# ---------- the machine, every text at once, layer by layer ----------
def run(texts, C=None, check=None, plate_opts=None, stop_at=None, pair_seed=None, ctl_seed=None, log=print):
    """the recording machine on every position of every text. Each bank's plate is laid from its files once and met by
    every position of every text (each text's arrivals read only its own positions), then dropped.
    check(key, value): called with every pile and the stream after every half layer (keys as recording/C_ref).
    plate_opts: {bank: Plate keyword args}. stop_at: return that bank's pile as soon as it is made.
    pair_seed: ORDER-FREE check, place pairs in a random order at every bank. ctl_seed: the counts' shuffle control.
    Returns the light at the top (x16, ex, LS) for every position."""
    t0 = time.time(); plate_opts = plate_opts or {}
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
    E = M["wte"]; WPE = M["wpe"]
    R = np.concatenate([E[np.asarray(ids)] + WPE[:len(ids)] for ids in texts]); del E
    if C is not None: C.add("both|rest|entry: word + position", sum(ones(WPE[:n]).sum() for n in lens))
    prng = None if pair_seed is None else np.random.default_rng(pair_seed)
    crng = None if ctl_seed is None else np.random.default_rng(ctl_seed)
    def bank(name, x16):
        pl = Plate(name, **plate_opts.get(name, {}))
        pile = pl.pile(x16, pair_rng=prng)
        if C is not None: count_bank(C, pl, x16, crng)
        e = pl.e; del pl
        return pile, e
    def keep(k, v):
        if check is not None: check(k, v)
    for l in range(12):
        x16, ex, LS = gauge_log(R); pile, we = bank(f"qkv.{l}", x16); keep(f"qkv pile {l}", pile)
        if stop_at == f"qkv.{l}": return pile
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + BIAS[f"qkv.{l}"]
        if C is not None:
            C.add("both|rest|water levels", water(x16)); C.add("both|rest|bank divides", ones(pile).sum(1) * E_ones(LS))
            C.add("both|rest|biases", len(R) * ones(BIAS[f"qkv.{l}"]).sum())
        del pile
        o = np.concatenate([attention(QKV[off[t]:off[t + 1]]) for t in range(len(texts))])
        if C is not None:
            for t in range(len(texts)): count_arrival(QKV[off[t]:off[t + 1]], C)
        del QKV
        x16, ex = to16(o, -ES); pile, we = bank(f"proj.{l}", x16); keep(f"proj pile {l}", pile)
        if stop_at == f"proj.{l}": return pile
        att = rshift(pile, -(we[None, :] + ex[:, None] + ES)) + BIAS[f"proj.{l}"]; del pile, o
        if C is not None: C.add("both|rest|biases", len(R) * ones(BIAS[f"proj.{l}"]).sum()); C.add("both|rest|stream adds", ones(att).sum())
        R = R + att; keep(f"stream a {l}", R); del att
        x16, ex, LS = gauge_log(R); pile, we = bank(f"fc.{l}", x16); keep(f"fc pile {l}", pile)
        if stop_at == f"fc.{l}": return pile
        H = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + BIAS[f"fc.{l}"]
        if C is not None:
            C.add("both|rest|water levels", water(x16)); C.add("both|rest|bank divides", ones(pile).sum(1) * E_ones(LS))
            C.add("both|rest|biases", len(R) * ones(BIAS[f"fc.{l}"]).sum()); count_bend(H, C)
        del pile
        Hb = edge(H); del H
        x16, ex = to16(Hb, -ES); pile, we = bank(f"mproj.{l}", x16); keep(f"mproj pile {l}", pile)
        if stop_at == f"mproj.{l}": return pile
        mo = rshift(pile, -(we[None, :] + ex[:, None] + ES)) + BIAS[f"mproj.{l}"]; del pile, Hb
        if C is not None: C.add("both|rest|biases", len(R) * ones(BIAS[f"mproj.{l}"]).sum()); C.add("both|rest|stream adds", ones(mo).sum())
        R = R + mo; keep(f"stream {l}", R); del mo
        log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex, LS = gauge_log(R)
    if C is not None: C.add("both|rest|top water level", water(x16))
    return x16, ex, LS

# ---------- the top read, untouched (base2's plain read), its table rebuilt from the recording ----------
_TOP = None; _TOPROW = None
def top_table():
    global _TOP, _TOPROW
    if _TOP is None:
        T16, e = table_values("head"); assert np.array_equal(e, HEAD_E); _TOP = T16
        _TOPROW = np.concatenate([ones(T16[i:i + 64]).sum(1) for i in range(0, T16.shape[0], 64)])   # 1s on each line
    return _TOP
WORD_CHUNKS = [np.arange(i, min(i + 12800, 50257)) for i in range(0, 50257, 12800)]
def top_scores(x16, ex, LS, C=None):
    """base2's read: power(product(x16, top table), (head.e + ex + ES) x 65536 - LS) + standing amount, every word."""
    T16 = top_table()
    pile = np.concatenate([product(x16, T16[:, c].astype(np.float64)) for c in WORD_CHUNKS], 1)
    if C is not None:
        C.add("both|top read|top meetings, plain (untouched)", ones(x16) @ _TOPROW)
        C.add("both|rest|top divide (every word)", ones(pile).sum(1) * E_ones(LS))
        C.add("both|rest|top standing amounts (every word)", len(x16) * ones(HEAD_B).sum())
    return power(pile, (HEAD_E[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + HEAD_B
def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
