"""FOUNDATION_ROWS/COUNT.PY: THE BITS EVERY WORD SHARES IN AN INPUT WORD TABLE, WRITTEN IN BASE 2. 2026-10-07.
Weights only; nothing goes through the machine. Predictions are in PREDICTION.txt (written before this ran).

HIS WORDS (2026-10-07): "If we convert embedding table into base 2 can we find the base foundation that is in every
token? Should be the same on ALL of them, some exact data from every single one"
THE SESSION'S RESTATEMENT (not his words): at some (lane, place) every single word has the same bit; those bits are
exact data present in every row, written once as a foundation row, each word stored as foundation + exact remainder.

THE GRID. Each lane is written on ONE power of two shared by all its words: the lowest lit place any of its words has
(read from the stored bits themselves: fp32 / fp16 / bf16 sign, exponent, mantissa; or the machine's whole numbers).
Every value is then an exact whole number on its lane's grid. The lane spans the places from that grid up to its
largest value's top place.
  sign-and-size  cells: the sign (the stored sign bit, so -0 keeps its sign) + magnitude places 0 .. P-1,
                 P = top place of the lane's largest |value|
  two's compl.   cells: places 0 .. Q-1, Q the fewest places that hold every value of the lane (place Q-1 is the sign)

ITEM 1. For every cell, n1 = words with a 1 there; the minority m = min(n1, V - n1). A cell is shared by ALL words
  when m = 0; by 99.9% when 1000 m <= V; by 99% when 100 m <= V (whole-number tests). Counted by zone: sign | top
  (the 2 places under the sign / the top 3 magnitude places) | low (below the writing's window: places only values
  smaller than the lane's top by more than the writing's significant places can reach) | mid.
ITEM 2. Controls: fresh draws (a bell with each lane's own spread, with no level and with the lane's level), rounded
  straight onto the table's lane grid ("grid") and written the way the table writes values ("written", fp32 / bf16 /
  fp16, then on their own finest lane grid); the blank (Pythia step0); laneshuffle (each lane's values re-dealt over
  the words: it keeps every lane's set of values, so it MUST give the item-1 counts digit for digit, rule 13.3; it is
  the null for none of them; it is used only for which words carry the exceptions).
ITEM 3. Foundation row per lane = the majority bit at every cell all words share (exact version) or 99.9% share
  (exceptions: every (word, lane) whose bit differs at a foundation cell, stored exactly as word id + its bits at the
  foundation cells). Word = foundation + remainder (+ exception correction), exact: the cells partition the places,
  and a value is the sum of its places' weights (two's complement: the top place weighs -2^(Q-1)).
  Measured: lit places (sign apart in sign-and-size; every 1 of the Q-place pattern in two's complement); bits a value
  (a) raw places (each lane at its own width), (b) per-place floor (each cell coded with its own count:
  log2 C(V, n1) + log2(V+1) bits), (c) per-lane floor (each lane's own histogram of values, model not counted, as
  ~/fringes storage did); real bytes: lzma (preset 6) of the bit planes, decoded back and checked equal both ways
  (the whole numbers, and the stored float bits for sign-and-size).
  By definition (rule 13.4): taking away bits every word shares is one-to-one inside a lane, so the per-lane floor of
  the exact version is unchanged (asserted, not a finding), and the per-place floor changes only by the counts.
ITEM 4. The lanes holding shared signs, their level against their spread (z = mean / spread) and its rank; the lanes
  where a bell with the same level and spread also shares the sign; which words break the 99.9% foundation (ids,
  strings, how many lanes each), the predefined never-a-token / never-built rows set aside.

    python3 foundation_rows/count.py table=gpt2-machine | gpt2-fp32 | qwen | pythia-70m | pythia-160m  [bytes=0]
writes foundation_rows/count_<table>.log, .json and counts_<table>.npz
"""
import os, sys, json, time, lzma, struct, resource, math, numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; OUT = os.path.join(HERE, "foundation_rows")
HF = "/Users/abundancemachine/.cache/huggingface/hub"
A_ = dict(a.split("=", 1) for a in sys.argv[1:]); TABLE = A_["table"]; NL = int(A_.get("block", 32))
DO_BYTES = A_.get("bytes", "1") == "1"; PRESET = 6
LOGF = open(os.path.join(OUT, f"count_{TABLE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
T0 = time.time()

# ---------------- reading the stored bits exactly ----------------
def st_raw(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); v = h[key]
        f.seek(8 + n + v["data_offsets"][0]); raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
    dt = {"F32": np.uint32, "F16": np.uint16, "BF16": np.uint16}[v["dtype"]]
    return v["dtype"], np.frombuffer(raw, dt).reshape(v["shape"])
FMT = {"F32": (31, 23, 0xFF, 127), "F16": (15, 10, 0x1F, 15), "BF16": (15, 7, 0xFF, 127)}   # sign bit, mantissa places, exp mask, bias
SIG = {"F32": 24, "F16": 11, "BF16": 8, "INT": None}                                          # significant places of the writing
def parts(U, fmt):
    """sign (0/1), whole mantissa m (hidden 1 put back), power of m's unit place: value = (-1)^s m 2^e, exactly"""
    sb, mb, xm, bias = FMT[fmt]; U = U.astype(np.int64)
    s = (U >> sb) & 1; ex = (U >> mb) & xm; m = U & ((1 << mb) - 1)
    assert not (ex == xm).any(), "inf or nan stored"
    norm = ex > 0; m = np.where(norm, m | (1 << mb), m); e = np.where(norm, ex - bias - mb, 1 - bias - mb)
    return s, m, e
def tz(m): return np.bitwise_count((m & -m) - 1).astype(np.int64)                            # trailing zeros (m > 0)
def bl(v): return int(v).bit_length()

def to_bf16(x):                                       # round to nearest even, from fp32 bits (whole-number arithmetic)
    f = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)
def from_parts(s, a, g, fmt):                         # the stored bits back from sign, whole number and lane grid
    v = np.ldexp(a.astype(np.float64), g[None, :].astype(np.int64)) * np.where(s == 1, -1.0, 1.0)
    v = np.where((a == 0) & (s == 1), -0.0, v)
    if fmt == "F32": return v.astype(np.float32).view(np.uint32)
    if fmt == "F16": return v.astype(np.float16).view(np.uint16)
    if fmt == "BF16": f = v.astype(np.float32).view(np.uint32); assert not (f & 0xFFFF).any(); return (f >> 16).astype(np.uint16)

# ---------------- sources: block(l0, l1) -> (s, x) on the lane grid ----------------
class RawSrc:
    """a table as stored bits (fp32 / fp16 / bf16): each lane on its own finest grid"""
    def __init__(self, U, fmt, name):
        self.U, self.fmt, self.name = U, fmt, name; self.V, self.d = U.shape; self.S = SIG[fmt]
        g = np.zeros(self.d, np.int64)
        for l0 in range(0, self.d, NL):
            s, m, e = parts(U[:, l0:l0 + NL], fmt)
            low = np.where(m > 0, e + tz(np.maximum(m, 1)), 1 << 20); g[l0:l0 + NL] = low.min(0)
        self.g = g
    def block(self, l0, l1):
        s, m, e = parts(self.U[:, l0:l1], self.fmt); t = tz(np.maximum(m, 1)); sh = e + t - self.g[None, l0:l1]   # m = odd x 2^t
        assert (sh[m > 0] >= 0).all()
        a = np.where(m > 0, (m >> t) << np.maximum(sh, 0), 0); assert a.max() < 1 << 62
        return s.astype(np.uint8), np.where(s == 1, -a, a)
class IntSrc:
    """whole numbers at one power (the machine's writing), each lane put on its own finest grid"""
    def __init__(self, X, e0, name):
        self.X, self.name = X, name; self.V, self.d = X.shape; self.S = None; self.fmt = "INT"
        orr = np.bitwise_or.reduce(np.abs(X), axis=0); self.k = np.where(orr > 0, tz(np.maximum(orr, 1)), 0)
        self.g = e0 + self.k
    def block(self, l0, l1):
        x = np.asarray(self.X[:, l0:l1]).astype(np.int64); x = np.where(x < 0, -((-x) >> self.k[None, l0:l1]), x >> self.k[None, l0:l1])
        return (x < 0).astype(np.uint8), x
class GridDraw:
    """a fresh bell draw (lane mean mu, spread sd, real units) rounded straight onto a given lane grid g"""
    def __init__(self, mu, sd, g, V, seed, name, S):
        self.mu, self.sd, self.g, self.V, self.seed, self.name, self.S = mu, sd, g, V, seed, name, S
        self.d = len(g); self.fmt = "INT"
    def block(self, l0, l1):
        z = np.random.default_rng([self.seed, l0]).standard_normal((self.V, l1 - l0))
        x = np.rint(np.ldexp(self.mu[None, l0:l1] + self.sd[None, l0:l1] * z, -self.g[None, l0:l1])).astype(np.int64)
        return (x < 0).astype(np.uint8), x
def written_draw(mu, sd, V, fmt, seed):               # the same bell written as the table writes values
    d = len(mu); U = np.zeros((V, d), {"F32": np.uint32, "F16": np.uint16, "BF16": np.uint16}[fmt])
    for l0 in range(0, d, NL):
        z = np.random.default_rng([seed, l0]).standard_normal((V, min(NL, d - l0)))
        v = mu[None, l0:l0 + NL] + sd[None, l0:l0 + NL] * z
        U[:, l0:l0 + NL] = v.astype(np.float32).view(np.uint32) if fmt == "F32" else (v.astype(np.float16).view(np.uint16) if fmt == "F16" else to_bf16(v))
    return U
def lane_shuffle(U, seed):
    rng = np.random.default_rng(seed); W = np.empty_like(U)
    for j in range(U.shape[1]): W[:, j] = U[rng.permutation(U.shape[0]), j]
    return W

# ---------------- the tables ----------------
def tokstr(kind):
    from tokenizers import Tokenizer
    import glob
    p = {"gpt2": glob.glob(f"{HF}/models--gpt2/snapshots/*/tokenizer.json"), "qwen": glob.glob(f"{HF}/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"),
         "pythia": [f"{HERE}/foundation0/models/tokenizer/tokenizer.json"]}[kind][0]
    t = Tokenizer.from_file(p); n = t.get_vocab_size(with_added_tokens=True)
    return (lambda i: (repr(t.decode([int(i)], skip_special_tokens=False)) if i < n else "(no token)")), n
def load():
    """returns finished source, extra sources (blank), set-aside row sets {label: ids}, token strings, base seed"""
    if TABLE == "gpt2-machine":
        M = np.load(os.path.join(HERE, "base2_gpt2.npz")); src = IntSrc(M["wte"], -16, "finished (machine, 2^-16)")
        f, n = tokstr("gpt2"); return src, {}, {}, f, 101
    if TABLE == "gpt2-fp32":
        fmt, U = st_raw(f"{HF}/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors", "wte.weight")
        f, n = tokstr("gpt2"); return RawSrc(U, fmt, "finished (fp32)"), {}, {}, f, 102
    if TABLE == "qwen":
        fmt, U = st_raw(f"{HF}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors", "model.embed_tokens.weight")
        f, n = tokstr("qwen"); V = U.shape[0]
        return RawSrc(U, fmt, "finished (bf16)"), {}, {f"never a token (rows {n}..{V - 1})": np.arange(n, V)}, f, 103
    if TABLE.startswith("pythia-"):
        sz = TABLE.split("-")[1]; base = f"{HERE}/foundation0/models/pythia-{sz}"
        fmt, U = st_raw(f"{base}-step143000/model.safetensors", "gpt_neox.embed_in.weight")
        fmt0, U0 = st_raw(f"{base}-step0/model.safetensors", "gpt_neox.embed_in.weight")
        f, n = tokstr("pythia"); V = U.shape[0]
        nb = np.array(json.load(open(f"{HERE}/foundation0/foundation_check.json"))[f"{sz}@143000"]["embed"]["ids"])
        sets = {f"the 241 never-built rows (foundation_check.json)": nb, f"the {V - n} rows past the last token ({n}..{V - 1})": np.arange(n, V)}
        return RawSrc(U, fmt, "finished (fp16, step143000)"), {"blank (fp16, step0)": RawSrc(U0, fmt0, "blank (fp16, step0)")}, sets, f, 104 if sz == "70m" else 105
    raise SystemExit(TABLE)

# ---------------- item 1: the counts ----------------
def lgC(n, k): return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)
def hbits(c):
    c = c[c > 0].astype(np.float64); n = c.sum(); return float(n * np.log2(n) - (c * np.log2(c)).sum())
def count(src, sets=None):
    """per lane: sign-and-size n1 (sign first, then places 0..P-1), two's complement n1 (places 0..Q-1), lane stats,
    per-lane floors; and the same counts over the rows outside each set-aside set"""
    t0 = time.time(); V, d = src.V, src.d
    R = dict(ss=[None] * d, tc=[None] * d, mu=np.zeros(d), sd=np.zeros(d), hss=np.zeros(d), htc=np.zeros(d), negzero=0, zeros=0,
             sub={k: dict(ss=[None] * d, tc=[None] * d) for k in (sets or {})})
    keep = {k: np.setdiff1d(np.arange(V), ids) for k, ids in (sets or {}).items()}
    for l0 in range(0, d, NL):
        s, x = src.block(l0, min(d, l0 + NL)); a = np.abs(x); n = x.shape[1]
        real = np.ldexp(x.astype(np.float64), src.g[None, l0:l0 + n]); R["mu"][l0:l0 + n] = real.mean(0); R["sd"][l0:l0 + n] = real.std(0)
        R["zeros"] += int((a == 0).sum()); R["negzero"] += int(((a == 0) & (s == 1)).sum())
        for j in range(n):
            l = l0 + j; aj, xj, sj = a[:, j], x[:, j], s[:, j]
            def cnt(aj, xj, sj):
                P = bl(aj.max()); hi, lo = int(xj.max()), int(xj.min())
                Q = max(bl(hi) if hi > 0 else 0, bl(-lo - 1) if lo < 0 else 0) + 1
                ss = np.array([int(sj.sum())] + [int(((aj >> p) & 1).sum()) for p in range(P)], np.int64)
                tc = np.array([int(((xj >> p) & 1).sum()) for p in range(Q)], np.int64)
                return ss, tc
            R["ss"][l], R["tc"][l] = cnt(aj, xj, sj)
            R["hss"][l] = hbits(np.unique(2 * aj + sj, return_counts=True)[1]); R["htc"][l] = hbits(np.unique(xj, return_counts=True)[1])
            for k, kp in keep.items():
                R["sub"][k]["ss"][l], R["sub"][k]["tc"][l] = cnt(aj[kp], xj[kp], sj[kp])
        if l0 % (NL * 8) == 0: print(f"    {src.name}: lanes {l0 + n}/{d}  {time.time() - t0:.0f}s  {rss()} MB", flush=True)
    R["V"] = V; R["Vsub"] = {k: len(kp) for k, kp in keep.items()}; R["secs"] = time.time() - t0
    return R

def zone_of(kind, width, S):
    """zone of each cell: 'sign', 'top', 'mid', 'low'. ss: cell 0 = sign, cell 1+p = place p; tc: cell p = place p"""
    if kind == "ss":
        P = width - 1; z = ["sign"]
        for p in range(P): z.append("top" if P - 1 - p <= 2 else ("low" if S is not None and p < P - S else "mid"))
    else:
        Q = width; z = []
        for p in range(Q): z.append("sign" if p == Q - 1 else ("top" if Q - 1 - p <= 2 else ("low" if S is not None and p < Q - 1 - S else "mid")))
    return z
ZONES = ("sign", "top", "mid", "low")
def summarize(R, S, V=None, which=None):
    V = V or R["V"]; out = {}
    for kind in ("ss", "tc"):
        rows = R[kind] if which is None else R["sub"][which][kind]
        o = dict(cells=0, places_per_lane=[]);
        for lev in ("all", "999", "99"): o[lev] = {z: [0, 0] for z in ZONES}            # [shared as 0, shared as 1]
        for n1 in rows:
            o["cells"] += len(n1); o["places_per_lane"].append(len(n1) - (1 if kind == "ss" else 0))
            zs = zone_of(kind, len(n1), S)
            for c, v in enumerate(n1):
                m = min(int(v), V - int(v)); b = int(2 * v > V)
                if m == 0: o["all"][zs[c]][b] += 1
                if 1000 * m <= V: o["999"][zs[c]][b] += 1
                if 100 * m <= V: o["99"][zs[c]][b] += 1
        ppl = np.array(o["places_per_lane"]); o["places_per_lane"] = [int(ppl.min()), float(np.median(ppl)), int(ppl.max()), float(ppl.mean())]
        out[kind] = o
    return out
def fmt_sum(o, kind):
    def t(lev):
        z = o[lev]; tot = sum(sum(v) for v in z.values())
        return f"{tot:6d} (" + " ".join(f"{k} {z[k][0] + z[k][1]}" + (f"[{z[k][1]} lit]" if z[k][1] else "") for k in ZONES if z[k][0] + z[k][1]) + ")"
    return (f"{'sign-and-size' if kind == 'ss' else 'two''s compl.':13s} cells {o['cells']:7d}  places/lane {o['places_per_lane'][0]}-{o['places_per_lane'][2]} (median {o['places_per_lane'][1]:g})\n"
            f"        ALL words: {t('all')}\n        99.9%:     {t('999')}\n        99%:       {t('99')}")

# ---------------- item 3: foundation rows, remainders, exceptions ----------------
def foundation(R, kind, lev):
    """per lane: list of foundation cells and their majority bits"""
    V = R["V"]; F = []
    for n1 in R[kind]:
        cells = [c for c, v in enumerate(n1) if (min(int(v), V - int(v)) == 0 if lev == "all" else 1000 * min(int(v), V - int(v)) <= V)]
        F.append((cells, [int(2 * int(n1[c]) > V) for c in cells]))
    return F
def masks(F, kind, widths):
    """per lane: magnitude/pattern mask and bits (int), sign-in-foundation flag and bit (ss only)"""
    d = len(F); mk = np.zeros(d, np.int64); bt = np.zeros(d, np.int64); sF = np.zeros(d, bool); sB = np.zeros(d, np.int64)
    for l, (cells, bits) in enumerate(F):
        for c, b in zip(cells, bits):
            if kind == "ss":
                if c == 0: sF[l] = True; sB[l] = b
                else: mk[l] |= 1 << (c - 1); bt[l] |= b << (c - 1)
            else: mk[l] |= 1 << c; bt[l] |= b << c
    return mk, bt, sF, sB
def remainder_pass(src, R, Fs):
    """for each (kind, lev) foundation: exceptions per lane and per word, lit places, per-lane floor of the remainder"""
    V, d = src.V, src.d; res = {}
    W = {k: np.array([len(n1) for n1 in R[k]]) for k in ("ss", "tc")}
    pre = {key: masks(F, key[0], W[key[0]]) for key, F in Fs.items()}
    for key in Fs: res[key] = dict(exc_lane=np.zeros(d, np.int64), exc_word=np.zeros(V, np.int64), lit_T=0, lit_R=0, h_R=np.zeros(d), sign_lit_T=0, sign_lit_R=0)
    for l0 in range(0, d, NL):
        s, x = src.block(l0, min(d, l0 + NL)); a = np.abs(x); n = x.shape[1]; L = slice(l0, l0 + n)
        Q = W["tc"][L]; u = x & ((np.int64(1) << Q[None, :].astype(np.int64)) - 1)        # two's complement pattern, Q places
        for key, (mk, bt, sF, sB) in pre.items():
            kind = key[0]; r = res[key]; mkL, btL = mk[None, L], bt[None, L]
            if kind == "ss":
                exc = ((a & mkL) != btL) | (sF[None, L] & (s.astype(np.int64) != sB[None, L]))
                Rm = a & ~mkL; Rs = np.where(sF[None, L], 0, s)
                r["lit_T"] += int(np.bitwise_count(a).sum()); r["lit_R"] += int(np.bitwise_count(Rm).sum())
                r["sign_lit_T"] += int(s.sum()); r["sign_lit_R"] += int(Rs.sum())
                key_R = 2 * Rm + Rs; key_T = 2 * a + s
            else:
                exc = (u & mkL) != btL; Ru = u & ~mkL
                r["lit_T"] += int(np.bitwise_count(u).sum()); r["lit_R"] += int(np.bitwise_count(Ru).sum())
                key_R = Ru; key_T = u
            r["exc_lane"][L] = exc.sum(0); r["exc_word"] += exc.sum(1)
            for j in range(n):
                cR = np.unique(key_R[:, j], return_counts=True)[1]; r["h_R"][l0 + j] = hbits(cR)
                if key[1] == "all":                                                         # one-to-one inside a lane (13.4)
                    cT = np.unique(key_T[:, j], return_counts=True)[1]; assert np.array_equal(np.sort(cR), np.sort(cT))
    return res
def costs(R, F, kind, rp):
    """bits for the whole table, three ways, table alone and foundation + remainder + exceptions"""
    V = R["V"]; d = len(R[kind]); lv = math.log2(V + 1); idb = math.ceil(math.log2(V))
    raw_T = raw_R = pl_T = pl_R = 0.0; ln_T = float((R["hss"] if kind == "ss" else R["htc"]).sum()); ln_R = float(rp["h_R"].sum())
    for l, n1 in enumerate(R[kind]):
        w = len(n1); cells, bits = F[l]; ne = int(rp["exc_lane"][l]); nf = len(cells)
        raw_T += w * V; pl_T += sum(lgC(V, int(v)) + lv for v in n1)
        row = w + nf                                                                          # which cells, and their bits
        exc_raw = (math.ceil(lv) + ne * (idb + nf)) if nf else 0; exc_pl = (lv + lgC(V, ne) + ne * nf) if nf else 0
        raw_R += (w - nf) * V + row + exc_raw; cs = set(cells)
        pl_R += sum(lgC(V, int(n1[c])) + lv for c in range(w) if c not in cs) + row + exc_pl
        ln_R += row + exc_pl
    N = V * d
    return dict(raw_T=raw_T / N, raw_R=raw_R / N, plane_T=pl_T / N, plane_R=pl_R / N, lane_T=ln_T / N, lane_R=ln_R / N)

# ---------------- real bytes: lzma of the bit planes, decoded back ----------------
class LZReader:
    def __init__(self, comp): self.d = lzma.LZMADecompressor(); self.src = comp; self.fed = False; self.buf = bytearray()
    def read(self, n):
        while len(self.buf) < n:
            if not self.fed: chunk = self.d.decompress(self.src, max_length=1 << 24); self.fed = True
            else: chunk = self.d.decompress(b"", max_length=1 << 24)
            if not chunk and self.d.eof: raise EOFError
            self.buf += chunk
        out = bytes(self.buf[:n]); del self.buf[:n]; return out
def planes_of(kind, s, x, a, W, l0, n, mk=None):
    """the bit planes of lanes l0.., each cell packed over the words, foundation cells (mask mk) left out"""
    out = []
    Q = W["tc"][l0:l0 + n]
    for j in range(n):
        l = l0 + j; w = W[kind][l]
        for c in range(w):
            if mk is not None and c in mk[l]: continue
            if kind == "ss": b = s[:, j] if c == 0 else ((a[:, j] >> (c - 1)) & 1)
            else: b = (x[:, j] >> c) & 1
            out.append(np.packbits(b.astype(np.uint8)).tobytes())
    return b"".join(out)
def bytes_check(src, R, kind, F, table_too=True):
    """table alone, and foundation row + remainder planes + exceptions, each lzma'd; both decoded back and checked
    equal to the source (whole numbers, and the stored float bits for sign-and-size)"""
    V, d = src.V, src.d; W = {k: np.array([len(n1) for n1 in R[k]]) for k in ("ss", "tc")}; nb = (V + 7) // 8
    t0 = time.time(); hdr = src.g.astype(np.int16).tobytes() + W[kind].astype(np.uint8).tobytes()
    Fset = [set(c) for c, _ in F]; mk, bt, sF, sB = masks(F, kind, W[kind])
    flags = np.zeros(int(W[kind].sum()), np.uint8); fb = []; o = 0
    for l, (cells, bits) in enumerate(F):
        for c in cells: flags[o + c] = 1
        fb += bits; o += W[kind][l]
    cT = lzma.LZMACompressor(preset=PRESET) if table_too else None; bT = [cT.compress(hdr)] if table_too else []
    cF = lzma.LZMACompressor(preset=PRESET); bF = [cF.compress(hdr + np.packbits(flags).tobytes() + np.packbits(np.array(fb, np.uint8)).tobytes())]
    exc_ids, exc_bits, exc_n = [], [], np.zeros(d, np.int64)
    for l0 in range(0, d, NL):
        s, x = src.block(l0, min(d, l0 + NL)); a = np.abs(x); n = x.shape[1]; L = slice(l0, l0 + n)
        if table_too: bT.append(cT.compress(planes_of(kind, s, x, a, W, l0, n)))
        if kind == "ss":
            exc = ((a & mk[None, L]) != bt[None, L]) | (sF[None, L] & (s.astype(np.int64) != sB[None, L]))
            pat = (a & mk[None, L]) * 2 + np.where(sF[None, L], s, 0)                         # each word's own bits at the foundation cells
        else:
            Q = W["tc"][L]; u = x & ((np.int64(1) << Q[None, :].astype(np.int64)) - 1); exc = (u & mk[None, L]) != bt[None, L]; pat = u & mk[None, L]
        for j in range(n):
            w_ = np.flatnonzero(exc[:, j]); exc_n[l0 + j] = len(w_); exc_ids.append(w_.astype(np.int32)); exc_bits.append(pat[w_, j].astype(np.int64))
        bF.append(cF.compress(planes_of(kind, s, x, a, W, l0, n, Fset)))
    if table_too: bT.append(cT.flush())
    bF.append(cF.flush()); blobT = b"".join(bT); blobF = b"".join(bF)
    blobE = lzma.compress(exc_n.astype(np.int32).tobytes() + np.concatenate(exc_ids).tobytes() + np.concatenate(exc_bits).tobytes(), preset=PRESET)
    t1 = time.time()
    # decode: the exceptions, then lane block by lane block the table and the foundation form, each compared at once
    rE = LZReader(blobE); exn = np.frombuffer(rE.read(4 * d), np.int32).astype(np.int64); tot = int(exn.sum())
    eids = np.frombuffer(rE.read(4 * tot), np.int32).astype(np.int64); ebits = np.frombuffer(rE.read(8 * tot), np.int64)
    eoff = np.concatenate([[0], np.cumsum(exn)])
    rT = LZReader(blobT) if table_too else None; rF = LZReader(blobF)
    for r in ([rT] if table_too else []) + [rF]:
        g2 = np.frombuffer(r.read(2 * d), np.int16).astype(np.int64); w2 = np.frombuffer(r.read(d), np.uint8).astype(np.int64)
        assert np.array_equal(g2, src.g) and np.array_equal(w2, W[kind])
    ncell = int(W[kind].sum()); flags2 = np.unpackbits(np.frombuffer(rF.read((ncell + 7) // 8), np.uint8))[:ncell]
    nfb = int(flags2.sum()); fb2 = np.unpackbits(np.frombuffer(rF.read((nfb + 7) // 8), np.uint8))[:nfb]
    assert np.array_equal(flags2, flags) and np.array_equal(fb2, np.array(fb, np.uint8))
    F2 = []; o = 0; q = 0
    for l in range(d):
        w = int(W[kind][l]); fl = flags2[o:o + w]; cells = [c for c in range(w) if fl[c]]
        F2.append(dict(zip(cells, [int(b) for b in fb2[q:q + len(cells)]]))); o += w; q += len(cells)
    def unplane(buf): return np.unpackbits(np.frombuffer(buf, np.uint8))[:V].astype(np.int64)
    def build(acc, sg, l):
        if kind == "ss": return np.where(sg == 1, -acc, acc), sg.astype(np.uint8)
        Q = int(W["tc"][l]); return acc - ((acc >> (Q - 1)) & 1) * (np.int64(1) << Q), None
    okT = okF = bits_ok = True
    for l0 in range(0, d, NL):
        s, x = src.block(l0, min(d, l0 + NL)); n = x.shape[1]
        XT = np.zeros((V, n), np.int64); ST = np.zeros((V, n), np.uint8); XF = np.zeros((V, n), np.int64); SF = np.zeros((V, n), np.uint8)
        for j in range(n):
            l = l0 + j; w = int(W[kind][l])
            for form in (("T", "F") if table_too else ("F",)):
                acc = np.zeros(V, np.int64); sg = np.zeros(V, np.int64); cs = F2[l] if form == "F" else {}
                for c in range(w):
                    b = np.full(V, cs[c], np.int64) if c in cs else unplane((rF if form == "F" else rT).read(nb))
                    if kind == "ss":
                        if c == 0: sg = b
                        else: acc |= b << (c - 1)
                    else: acc |= b << c
                if form == "F":                                                              # the exceptions' own bits at the foundation cells
                    ids = eids[eoff[l]:eoff[l + 1]]; eb = ebits[eoff[l]:eoff[l + 1]]
                    if kind == "ss":
                        acc[ids] = (acc[ids] & ~mk[l]) | (eb >> 1)
                        if sF[l]: sg[ids] = eb & 1
                    else: acc[ids] = (acc[ids] & ~mk[l]) | eb
                xx, ss_ = build(acc, sg, l)
                if form == "T": XT[:, j] = xx; ST[:, j] = ss_ if ss_ is not None else 0
                else: XF[:, j] = xx; SF[:, j] = ss_ if ss_ is not None else 0
        if table_too: okT &= bool(np.array_equal(XT, x)) and (kind != "ss" or bool(np.array_equal(ST, s)))
        okF &= bool(np.array_equal(XF, x)) and (kind != "ss" or bool(np.array_equal(SF, s)))
        if kind == "ss" and isinstance(src, RawSrc):
            if table_too: bits_ok &= bool(np.array_equal(from_parts(ST, np.abs(XT), src.g[l0:l0 + n], src.fmt), src.U[:, l0:l0 + n]))
            bits_ok &= bool(np.array_equal(from_parts(SF, np.abs(XF), src.g[l0:l0 + n], src.fmt), src.U[:, l0:l0 + n]))
    return dict(bytes_T=len(blobT) if table_too else None, bytes_F=len(blobF) + len(blobE), bytes_exc=len(blobE), exc_pairs=tot,
                roundtrip_T=okT, roundtrip_F=okF, stored_bits_equal=bits_ok, secs_code=t1 - t0, secs_decode=time.time() - t1,
                bits_T=8 * len(blobT) / (V * d) if table_too else None, bits_F=8 * (len(blobF) + len(blobE)) / (V * d))

# ---------------- run ----------------
def main():
    src, extra, sets, tok, seed = load(); V, d = src.V, src.d; S = src.S
    log(f"FOUNDATION ROWS: {TABLE}  ({V} words x {d} lanes, writing {src.fmt}{'' if S is None else f', {S} significant places'})  {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"  lane grids: 2^{src.g.min()} .. 2^{src.g.max()} (median 2^{int(np.median(src.g))}); every value an exact whole number on its lane's grid")
    RES = {}; JS = dict(table=TABLE, V=V, d=d, writing=src.fmt, S=S, versions={})
    def run(name, sr, sets_=None):
        R = count(sr, sets_); o = summarize(R, S); RES[name] = R
        log(f"\n  [{name}]  {R['secs']:.0f}s  {rss()} MB   zeros {R['zeros']} (stored -0: {R['negzero']})")
        for kind in ("ss", "tc"): log("    " + fmt_sum(o[kind], kind))
        JS["versions"][name] = dict(summary=o, zeros=R["zeros"], negzero=R["negzero"])
        for k in (sets_ or {}):
            o2 = summarize(R, S, R["Vsub"][k], k); JS["versions"][name]["set_aside:" + k] = o2
            log(f"    -- with {k} set aside ({R['Vsub'][k]} words):")
            for kind in ("ss", "tc"): log("      " + fmt_sum(o2[kind], kind).replace("\n", "\n  "))
        return R
    Rf = run(src.name, src, sets)
    mu, sd = Rf["mu"], Rf["sd"]
    controls = {}
    for nm, sr in extra.items(): controls[nm] = sr
    controls["fresh0-grid (bell, lane spread, no level, on the table's grid)"] = GridDraw(np.zeros(d), sd, src.g, V, seed * 10 + 1, "fresh0-grid", S)
    controls["freshlane-grid (bell, lane level and spread, on the table's grid)"] = GridDraw(mu, sd, src.g, V, seed * 10 + 2, "freshlane-grid", S)
    if src.fmt != "INT":
        controls[f"fresh0-written (bell, no level, written {src.fmt})"] = ("w", np.zeros(d), seed * 10 + 3)
        controls[f"freshlane-written (bell, lane level, written {src.fmt})"] = ("w", mu, seed * 10 + 4)
    controls["laneshuffle (each lane re-dealt over the words)"] = ("shuffle", None, seed * 10 + 5)
    Fdefs = [("ss", "all"), ("ss", "999"), ("tc", "all"), ("tc", "999")]
    ITEM3 = {}
    def item3(name, sr, R, bytes_=False):
        Fs = {key: foundation(R, *key) for key in Fdefs}; rp = remainder_pass(sr, R, Fs); out = {}; tblb = {}
        log(f"\n  ITEM 3 [{name}]: foundation row + exact remainder (bits a value over all {V * d:,} values)")
        for key in Fdefs:
            kind, lev = key; F = Fs[key]; c = costs(R, F, kind, rp[key]); r = rp[key]
            nf = sum(len(cells) for cells, _ in F); ne = int(r["exc_lane"].sum())
            line = (f"    {'sign-and-size' if kind == 'ss' else 'two''s compl.':13s} {('ALL' if lev == 'all' else '99.9%'):5s}: foundation cells {nf:6d}  exceptions {ne:9,d} (word,lane) on {int((r['exc_word'] > 0).sum()):6d} words | "
                    f"lit places a value {r['lit_T'] / (V * d):.4f} -> {r['lit_R'] / (V * d):.4f}"
                    + (f" (signs lit {r['sign_lit_T'] / (V * d):.4f} -> {r['sign_lit_R'] / (V * d):.4f})" if kind == "ss" else "")
                    + f" | raw {c['raw_T']:.4f} -> {c['raw_R']:.4f}  per-place {c['plane_T']:.4f} -> {c['plane_R']:.4f}  per-lane {c['lane_T']:.4f} -> {c['lane_R']:.4f}")
            out[f"{kind}_{lev}"] = dict(found_cells=nf, exc=ne, exc_words=int((r["exc_word"] > 0).sum()), lit_T=r["lit_T"], lit_R=r["lit_R"], **c)
            if bytes_ and DO_BYTES:
                b = bytes_check(sr, R, kind, F, table_too=(lev == "all"))
                if lev == "all": tblb[kind] = (b["bytes_T"], b["bits_T"])
                else: b["bytes_T"], b["bits_T"] = tblb[kind]
                out[f"{kind}_{lev}"]["bytes"] = b
                line += (f"\n        real bytes (lzma {PRESET}, bit planes): table {b['bytes_T']:,} ({b['bits_T']:.4f} bits a value)  foundation form {b['bytes_F']:,} ({b['bits_F']:.4f}),"
                         f" {b['bytes_F'] - b['bytes_T']:+,} bytes;  round trip table {b['roundtrip_T']}, foundation form {b['roundtrip_F']}"
                         + (f", stored {src.fmt} bits equal {b['stored_bits_equal']}" if kind == "ss" and src.fmt != "INT" else "") + f"  ({b['secs_code']:.0f}s + {b['secs_decode']:.0f}s)")
                assert b["roundtrip_T"] and b["roundtrip_F"] and b["stored_bits_equal"]
            log(line)
        ITEM3[name] = out; return Fs, rp
    Ff, rpf = item3(src.name, src, Rf, bytes_=True)
    for nm, c in controls.items():
        if isinstance(c, tuple):
            if c[0] == "w": sr = RawSrc(written_draw(c[1], sd, V, src.fmt, c[2]), src.fmt, nm)
            else:
                sr = RawSrc(lane_shuffle(src.U, c[2]), src.fmt, nm) if isinstance(src, RawSrc) else IntSrc(lane_shuffle(src.X, c[2]), -16, nm)
        else: sr = c
        R = run(nm, sr, sets if nm.startswith("blank") else None)
        if nm.startswith("laneshuffle"):
            same = all(np.array_equal(R[k][l], Rf[k][l]) for k in ("ss", "tc") for l in range(d))
            log(f"    laneshuffle against the table, every (lane, place) count: identical = {same}  (must be: it keeps every lane's set of values; rule 13.3, it shows nothing about item 1)")
            assert same; JS["laneshuffle_identical"] = same
            Fs_sh, rp_sh = item3(nm, sr, R)
            RES["_shuffle_rp"] = rp_sh
        else:
            item3(nm, sr, R)
        del sr
    # ---------------- item 4 ----------------
    log(f"\n  ITEM 4: which lanes hold the shared bits, and which words break them")
    z = np.where(sd > 0, mu / np.maximum(sd, 1e-300), 0); rank = np.empty(d, np.int64); rank[np.argsort(-np.abs(z))] = np.arange(1, d + 1)
    V_ = V
    def sign_lanes(R, lev, V=V_):
        return [l for l in range(d) if (min(int(R["ss"][l][0]), V - int(R["ss"][l][0])) == 0 if lev == "all" else 1000 * min(int(R["ss"][l][0]), V - int(R["ss"][l][0])) <= V)]
    def mag_lanes(R, lev, V=V_):
        out = []
        for l in range(d):
            n1 = R["ss"][l][1:]
            for p, v in enumerate(n1):
                m = min(int(v), V - int(v))
                if (m == 0 if lev == "all" else 1000 * m <= V) and 2 * int(v) > V: out.append((l, p, len(n1)))
        return out
    I4 = {}
    for lev in ("all", "999"):
        sl = sign_lanes(Rf, lev); I4[f"sign_lanes_{lev}"] = sl
        log(f"    shared sign ({'ALL words' if lev == 'all' else '99.9%'}): {len(sl)} lanes" + (f": " + ", ".join(f"{l} (z {z[l]:+.2f}, rank {rank[l]})" for l in sorted(sl, key=lambda l: rank[l])[:40]) if sl else ""))
        if sl: log(f"      their |z| ranks: all within the top {max(rank[l] for l in sl)} lanes by |level / spread|; lowest |z| among them {min(abs(z[l]) for l in sl):.2f}; highest |z| of a lane without it {max([abs(z[l]) for l in range(d) if l not in set(sl)] or [0]):.2f}")
        for nm in RES:
            if nm.startswith("freshlane") or nm.startswith("blank"):
                fl = sign_lanes(RES[nm], lev); I4[f"sign_lanes_{lev}:{nm}"] = fl
                log(f"      {nm}: {len(fl)} lanes, {len(set(fl) & set(sl))} of them the table's")
        ml = mag_lanes(Rf, lev); I4[f"lit_mag_{lev}"] = ml
        log(f"    magnitude places lit in {'every word' if lev == 'all' else '99.9% of words'}: {len(ml)}" + (": " + ", ".join(f"lane {l} place {p} of {P}" for l, p, P in ml[:20]) if ml else ""))
    # the shared-sign lanes: how far each side of the lane reaches, in spreads, against a bell of V words
    sl_all, sl_999 = I4["sign_lanes_all"], I4["sign_lanes_999"]; near_words = {}
    if sl_999:
        bell = float(np.sqrt(2 * np.log(V)) - (np.log(np.log(V)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(V))))
        log(f"    each shared-sign lane (99.9% or all): z; how far the side toward zero reaches (mean - nearest word, in spreads); how far the other side reaches; minority-sign words (a bell of {V} words reaches about {bell:.2f} spreads each way)")
        rows = []
        for l in sorted(sl_999, key=lambda l: rank[l]):
            s_, x_ = src.block(l, l + 1); real = np.ldexp(x_[:, 0].astype(np.float64), int(src.g[l])); side = 1.0 if mu[l] > 0 else -1.0
            dev = side * (real - mu[l]) / sd[l]; near, far = -dev.min(), dev.max(); minority = np.flatnonzero(s_[:, 0] != (1 if side < 0 else 0))
            near_words[l] = minority
            rows.append(dict(lane=int(l), z=float(z[l]), near=float(near), far=float(far), minority=[int(i) for i in minority[:50]], n_minority=int(len(minority))))
            log(f"      lane {l:4d} z {z[l]:+8.2f}  toward zero {near:6.2f}  away {far:6.2f}  minority-sign words {len(minority):3d}" + (": " + ", ".join(f"{int(i)} {tok(int(i))[:16]}" for i in minority[:8]) if len(minority) else ""))
        I4["sign_lane_shapes"] = rows
        allmin = np.concatenate([v for v in near_words.values()]) if near_words else np.zeros(0, np.int64)
        if len(allmin): 
            u, c = np.unique(allmin, return_counts=True)
            log(f"      minority-sign words over those lanes: {len(allmin)} (word, lane) on {len(u)} words; words in 2 or more of the lanes: {int((c > 1).sum())}")
    for k in sets:
        R = Rf; Vk = R["Vsub"][k]
        sl = [l for l in range(d) if min(int(R["sub"][k]["ss"][l][0]), Vk - int(R["sub"][k]["ss"][l][0])) == 0]
        log(f"    with {k} set aside: shared sign in ALL remaining words: {len(sl)} lanes" + (f" (|z| ranks within the top {max(rank[l] for l in sl)})" if sl else ""))
        I4[f"sign_lanes_all_without:{k}"] = sl
    if TABLE.startswith("gpt2"):
        M = np.load(os.path.join(HERE, "base2_gpt2.npz")); Tw = M["head.w"].astype(np.float64); c = Tw.mean(1); share = c ** 2 / (c ** 2).sum()
        top = np.argsort(-share)[:10]; I4["T_column_lanes"] = [int(l) for l in top]
        log(f"    T's shared column, lanes by their share of its squares: " + ", ".join(f"{l} ({share[l]:.1%})" for l in top))
        for lev in ("all", "999"):
            sl = set(I4[f"sign_lanes_{lev}"]); log(f"      E's shared-sign lanes ({lev}) among T's top 10 column lanes: {sorted(sl & set(int(l) for l in top))}")
    # the words that break the 99.9% foundation
    setall = np.zeros(V, bool)
    for ids in sets.values(): setall[ids] = True
    rowsize = None
    for key in (("ss", "999"), ("tc", "999")):
        r = rpf[key]; ew = r["exc_word"]; tot = int(ew.sum())
        if tot == 0: log(f"    {key}: no exceptions"); continue
        order = np.argsort(-ew, kind="stable"); top10 = int(ew[order[:10]].sum())
        sh = RES["_shuffle_rp"][key]["exc_word"]; sh10 = int(np.sort(sh)[::-1][:10].sum())
        log(f"    breakers of the 99.9% foundation ({'sign-and-size' if key[0] == 'ss' else 'two''s compl.'}): {tot:,} (word, lane) exceptions on {int((ew > 0).sum()):,} words;"
            f" the 10 words with most carry {top10 / tot:.1%} (laneshuffle: {sh10 / tot:.1%} on {int((sh > 0).sum()):,} words)")
        for k, ids in sets.items():
            log(f"      on {k}: {int(ew[ids].sum()):,} ({ew[ids].sum() / tot:.1%}); {int((ew[ids] > 0).sum())} of those {len(ids)} rows break it")
        if rowsize is None:
            ss2 = np.zeros(V)
            for l0 in range(0, d, NL):
                s_, x_ = src.block(l0, min(d, l0 + NL)); real = np.ldexp(x_.astype(np.float64), src.g[None, l0:l0 + x_.shape[1]]); ss2 += ((real - mu[None, l0:l0 + x_.shape[1]]) ** 2).sum(1)
            rowsize = np.sqrt(ss2); rowsize = rowsize / np.median(rowsize)
        log(f"      the 20 words with most (id, string, lanes broken, distance from the table's mean row / median word's, set-aside):")
        for i in order[:20]:
            if ew[i] == 0: break
            log(f"        {int(i):7d} {tok(int(i))[:28]:30s} {int(ew[i]):5d}  {rowsize[i]:.3f}  {'yes' if setall[i] else ''}")
        I4[f"breakers_{key[0]}"] = [(int(i), int(ew[i])) for i in order[:200] if ew[i] > 0]
        log(f"      distance from the mean row / median word's: breakers (weighted by lanes) {np.average(rowsize, weights=ew):.3f}; all words median 1.000, p1 {np.percentile(rowsize, 1):.3f}, p99 {np.percentile(rowsize, 99):.3f}")
    JS["item3"] = ITEM3; JS["item4"] = I4; JS["z"] = z.tolist(); JS["secs"] = time.time() - T0
    json.dump(JS, open(os.path.join(OUT, f"count_{TABLE}.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    sv = {}
    for i, nm in enumerate(n for n in RES if not n.startswith("_")):
        for k in ("ss", "tc"):
            sv[f"v{i}_{k}_width"] = np.array([len(v) for v in RES[nm][k]]); sv[f"v{i}_{k}_n1"] = np.concatenate(RES[nm][k])
        sv[f"v{i}_name"] = np.array(nm)
    np.savez_compressed(os.path.join(OUT, f"counts_{TABLE}.npz"), **sv)
    log(f"\n  done {time.time() - T0:.0f}s, max {rss()} MB")

main()
