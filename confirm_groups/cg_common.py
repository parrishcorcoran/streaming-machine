"""confirm_groups/cg_common.py -- the independent checker's shared code for the Qwen3-0.6B group machine (my own code:
nothing imported or copied from qwen_groups.py, base2_groups.py, groups/ or groups2/; only their docstrings' rules and
the grouping keys were read). Whole numbers only in the machine path.

WHAT IS HERE
  member()      a plate of qwen/qwen_base2.npz read in place (every member is stored flat in the zip)
  naf(), ones() a value's signed digits (no two lit side by side) and its 1s, counted
  sd_digits()   the classic signed-digit rule, digit by digit: where the running value is odd the digit is
                2 - (value mod 4); take it away; halve
  rule()        each place b = 0..15 of a plate read in signed digits when L_b < G x M (L_b the plate's lit signed
                digits at b, G = ceil(N / g) groups, M neurons), up/hole otherwise: plate only
  expand()      each weight written once, bottom place to top, from w + C (C = 2^(top up/hole place)): a binary digit
                at an up/hole place, a signed digit elsewhere (places 16, 17 take what is left); checked to give w back.
                Then 2 x.w = sum over up/hole places of 2^b (ready pile) + 2 (signed part) + (K - 2C) X
  groups()      numpy's generator keyed [seed, slab, layer, g]: a random line order cut into runs of g (the last short)
  Slab          one slab (q|k|v, o, gate|up, down) laid for the group machine: the shared table, each plate's rule,
                patterns, signed digits; run() makes every pile from the ready piles and counts the work
  TopTable      the word table laid the same way at g = 16; parts() gives each word's place parts and their photons
"""
import os, sys, json, hashlib, zipfile, ctypes, subprocess, resource, time, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; NPZ = f"{QD}/qwen_base2.npz"; CG = f"{HERE}/confirm_groups"
NTH = 8
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")
def naf1(v): a = abs(int(v)); return bin(a ^ (3 * a)).count("1")

def member(name):
    """a member of the npz, read in place: its local header, then the npy header, then the flat data"""
    z = zipfile.ZipFile(NPZ); info = z.getinfo(name + ".npy"); assert info.compress_type == 0
    with open(NPZ, "rb") as f:
        f.seek(info.header_offset); h = f.read(30); assert h[:4] == b"PK\x03\x04"
        start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
        f.seek(start); ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return np.memmap(NPZ, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")

_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{CG}/cg.c", f"{CG}/cg.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        L = ctypes.CDLL(so); P, I, i = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        L.half_tables.argtypes = [I, I, P, I, P, P, I, P, P, P]
        L.uh_reads.argtypes = [I, I, I, P, P, I, I, P, P, P, P, P, i]
        L.uh_parts.argtypes = [I, I, I, P, P, I, I, P, P, P, P, P, i]
        L.sd_reads.argtypes = [I, P, I, P, P, P, P, P, P, I, P, P, P]
        L.pile_write.argtypes = [I, I, I, P, P, P]
        L.mm16.argtypes = [I, I, I, P, P, P, i]
        L.naf_check.argtypes = [I]; L.naf_check.restype = i; L.ones_check.argtypes = [I]; L.ones_check.restype = i
        for f in (L.half_tables, L.uh_reads, L.uh_parts, L.sd_reads, L.pile_write, L.mm16): f.restype = None
        _lib = L
    return _lib
def ptr(a): return a.ctypes.data
def mm16(x, Wc):
    """CHECK PATH: the direct product, every line met one by one (int64)"""
    x = np.ascontiguousarray(x, np.int64); out = np.empty((x.shape[0], Wc.shape[1]), np.int64)
    lib().mm16(x.shape[0], x.shape[1], Wc.shape[1], ptr(x), ptr(Wc), ptr(out), NTH); return out

def sd_digits(w, places=18):
    """the classic signed digits of w, place by place (int8 planes); asserts nothing is left"""
    r = np.asarray(w, np.int64).copy(); out = np.zeros((places,) + r.shape, np.int8)
    for b in range(places):
        d = np.where((r & 1) == 1, 2 - np.remainder(r, 4), 0); out[b] = d; r = (r - d) >> 1
    assert not r.any(); return out
def lit_by_place(w):
    """L_b: how many weights have a lit signed digit at place b (b = 0..17), counted without keeping the planes"""
    r = np.asarray(w, np.int64).copy(); L = np.zeros(18, np.int64)
    for b in range(18):
        d = np.where((r & 1) == 1, 2 - np.remainder(r, 4), 0); L[b] = int(np.count_nonzero(d)); r = (r - d) >> 1
    assert not r.any(); return L
def rule(L, G, M):
    """the up/hole places (0..15) of a plate: up/hole unless L_b < G x M"""
    return [b for b in range(16) if not (L[b] < G * M)]
def t_run(n): return ((1 << (n + 1)) - 1) // 3

def expand(w, uh):
    """bottom place to top from w + C: binary digits at the up/hole places (bits of E), signed digits elsewhere.
    Returns E (uint16), the signed digits [(b, int8 plane)], kcorr = K - 2C, C"""
    w = np.asarray(w, np.int64); C = 1 << max(uh); r = w + C; E = np.zeros(w.shape, np.uint16); sd = []; back = -C + np.zeros_like(w)
    for b in range(18):
        if b < 16 and b in uh:
            e = r & 1; E |= (e.astype(np.uint16) << np.uint16(b)); r = (r - e) >> 1; back += e << b
        else:
            d = np.where((r & 1) == 1, 2 - np.remainder(r, 4), 0); r = (r - d) >> 1; back += d << b
            if d.any(): sd.append((b, d.astype(np.int8)))
    assert not r.any(), "a weight left over after place 17"
    assert np.array_equal(back, w), "the digits do not give the weight back"
    K = sum(1 << b for b in uh); return E, sd, K - 2 * C, C

def groups(N, g, key, reorder=None):
    order = np.random.default_rng(key).permutation(N); runs = [order[s:s + g] for s in range(0, N, g)]
    if reorder is not None:
        r = np.random.default_rng(reorder); runs = [runs[i] for i in r.permutation(len(runs))]; runs = [r.permutation(u) for u in runs]
    return runs

def patterns(E, runs, ups):
    """(G, M, nup) codes: the half-table index of the pattern (bits over the group's lines 0..gk-2, the last line up)
    and bit 15 set where the pattern holds the last line as a hole (read as the flip: the opposite pile)"""
    G = len(runs); M = E.shape[1]; nup = len(ups); out = np.empty((G, M, nup), np.uint16); ub = np.asarray(ups, np.uint16)
    for k, ln in enumerate(runs):
        gk = len(ln); sub = E[ln]                                              # (gk, M)
        bits = (sub[:, :, None] >> ub[None, None, :]) & np.uint16(1)           # (gk, M, nup)
        pat = np.zeros((M, nup), np.uint32)
        for t in range(gk): pat |= bits[t].astype(np.uint32) << np.uint32(t)
        hb = np.uint32(1 << (gk - 1)); mask = np.uint32((1 << (gk - 1)) - 1)
        neg = (pat & hb) == 0
        idx = np.where(neg, (~pat) & mask, pat & mask)
        out[k] = (idx | (neg.astype(np.uint32) << np.uint32(15))).astype(np.uint16)
    return out

def table_coeffs(runs, N):
    """the half table's photons by the walk's relation, per line: the group total takes lines 1..gk-1 once; in the Gray
    walk over lines 0..gk-2, line t turns 2^(gk-2-t) times"""
    c = np.zeros(N, np.int64)
    for ln in runs:
        gk = len(ln)
        for t, l in enumerate(ln): c[l] = (1 if t >= 1 else 0) + ((1 << (gk - 2 - t)) if t <= gk - 2 else 0)
    return c

SIDX = {"qkv": 0, "o": 4, "gu": 5, "down": 6}                                   # the slab index in the grouping key
SEGS = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
        "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}

class Slab:
    """one slab at one layer, laid for the group machine (plate only, before any light)"""
    def __init__(self, slab, l, seed=1, g=12, reorder=None):
        self.name = f"{slab}.{l}"; self.Wc = np.ascontiguousarray(member(f"{slab}.{l}.w")); N, Mt = self.Wc.shape; self.N, self.Mt = N, Mt
        self.runs = groups(N, g, [seed, SIDX[slab], l, g], reorder)
        self.G = len(self.runs); self.H = 1 << (g - 1); assert self.G == -(-N // g)
        self.gl = np.ascontiguousarray(np.concatenate(self.runs).astype(np.int64))
        self.goff = np.ascontiguousarray(np.cumsum([0] + [len(u) for u in self.runs]).astype(np.int64))
        self.coef = table_coeffs(self.runs, N); self.used = np.zeros((self.G, self.H), np.uint8); self.plates = []
        self.gorder = None if reorder is None else np.ascontiguousarray(np.random.default_rng(list(reorder) + [5]).permutation(self.G).astype(np.int64))
        for pname, c0, c1 in SEGS[slab]:
            w = self.Wc[:, c0:c1].astype(np.int64); M = c1 - c0
            L = lit_by_place(w); uh = rule(L, self.G, M)
            E, sd, kcorr, C = expand(w, uh)
            pat = np.ascontiguousarray(patterns(E, self.runs, uh)); del E
            for k in range(self.G): self.used[k, pat[k].ravel() & 0x7FFF] = 1
            dl, dj, db, ds = [], [], [], []
            for b, d in sd:
                t, j = np.nonzero(d); dl.append(t); dj.append(j); db.append(np.full(len(t), b)); ds.append(d[t, j])
            dl, dj, db, ds = (np.concatenate(a) if a else np.zeros(0) for a in (dl, dj, db, ds))
            o = np.lexsort((dl, dj)) if reorder is None else np.random.default_rng(list(reorder) + [6, c0]).permutation(len(dl))
            P_ = dict(name=pname, c0=c0, c1=c1, M=M, uh=uh, L=L, ups=np.ascontiguousarray(np.asarray(uh, np.int32)), pat=pat,
                      dl=np.ascontiguousarray(dl[o].astype(np.int32)), dj=np.ascontiguousarray(dj[o].astype(np.int32)),
                      db=np.ascontiguousarray(db[o].astype(np.int32)), ds=np.ascontiguousarray(ds[o].astype(np.int8)),
                      kcorr=int(kcorr), C=int(C), onesW=ones(w).sum(1), nafW=naf(w).sum(1))
            self.plates.append(P_)
    def run(self, x, nth=NTH):
        """the piles of light x (P, N) made from the ready piles; per-position counts"""
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P = len(x); assert np.abs(x).max(initial=0) <= 1 << 15
        piles = np.empty((P, self.Mt), np.int64)
        cnt = dict(light=ones(x).sum(1), xsum=naf(x[:, 1:]).sum(1), table_rel=naf(x) @ self.coef, table=np.zeros(P, np.int64),
                   pwrite=np.zeros(P, np.int64))
        for pl in self.plates:
            for it in ("reads", "sdreads"): cnt[f"{pl['name']}|{it}"] = np.zeros(P, np.int64)
        Pb = max(1, min(P, int(1.2e8 // (self.G * self.H * 9))))
        for p0 in range(0, P, Pb):
            xb = np.ascontiguousarray(x[p0:p0 + Pb]); n = len(xb); xT = np.ascontiguousarray(xb.T)
            T = np.empty((self.G, self.H, n), np.int64); NT = np.empty((self.G, self.H, n), np.uint8); wk = np.zeros(n, np.int64)
            lib().half_tables(n, self.N, ptr(xb), self.G, ptr(self.gl), ptr(self.goff), self.H, ptr(T), ptr(NT), ptr(wk))
            cnt["table"][p0:p0 + n] = wk
            pw = np.zeros(n, np.int64); lib().pile_write(n, self.G, self.H, ptr(T), ptr(self.used), ptr(pw)); cnt["pwrite"][p0:p0 + n] = pw
            X = xb.sum(1)
            for pl in self.plates:
                acc = np.zeros((pl["M"], n), np.int64); rc = np.zeros(n, np.int64); sc = np.zeros(n, np.int64)
                lib().uh_reads(n, self.G, self.H, ptr(T), ptr(NT), pl["M"], len(pl["uh"]), ptr(pl["ups"]), ptr(pl["pat"]),
                               ptr(self.gorder) if self.gorder is not None else None, ptr(acc), ptr(rc), nth)
                if len(pl["dl"]):
                    lib().sd_reads(n, ptr(xT), len(pl["dl"]), ptr(pl["dl"]), ptr(pl["dj"]), ptr(pl["db"]), ptr(pl["ds"]),
                                   ptr(acc), ptr(sc), 0, None, None, None)
                acc += pl["kcorr"] * X[None, :]
                assert not (acc & 1).any(), "a doubled pile is odd"
                piles[p0:p0 + n, pl["c0"]:pl["c1"]] = (acc >> 1).T
                cnt[f"{pl['name']}|reads"][p0:p0 + n] = rc; cnt[f"{pl['name']}|sdreads"][p0:p0 + n] = sc
            del T, NT
        nX = naf(x.sum(1))
        for pl in self.plates:
            cnt[f"{pl['name']}|xoff"] = pl["M"] * naf1(pl["kcorr"]) * nX
            cnt[f"{pl['name']}|plain"] = ones(x) @ pl["onesW"]; cnt[f"{pl['name']}|signed"] = naf(x) @ pl["nafW"]
        return piles, cnt
    def group_work(self, cnt):
        """the group work of a counts dict, per position: light, table, ready piles written, X, X's take-away, reads"""
        t = cnt["light"] + cnt["table"] + cnt["pwrite"] + cnt["xsum"]
        for pl in self.plates: t = t + cnt[f"{pl['name']}|xoff"] + cnt[f"{pl['name']}|reads"] + cnt[f"{pl['name']}|sdreads"]
        return t

class TopTable:
    """the word table (1,024 lines x 151,936 words) laid for the group machine at g = 16 (plate only)"""
    def __init__(self, seed=1, g=16, reorder=None, word_perm=None, log=print):
        t0 = time.time(); HW = member("head.w"); self.N, self.M = HW.shape; self.HE = np.asarray(member("head.e")).astype(np.int64)
        self.wperm = np.arange(self.M) if word_perm is None else np.asarray(word_perm)
        self.Wc = np.empty((self.N, self.M), np.int16)                         # C order, words relabelled if asked
        for c in range(0, self.M, 16384): self.Wc[:, c:c + 16384] = np.asarray(HW[:, self.wperm[c:c + 16384]])
        self.HEp = self.HE[self.wperm]; self.EMIN = int(self.HEp.min()); self.KW = self.HEp - self.EMIN
        self.runs = groups(self.N, g, [seed, 99, self.N, g], reorder); self.G = len(self.runs); self.H = 1 << (g - 1)
        self.gl = np.ascontiguousarray(np.concatenate(self.runs).astype(np.int64))
        self.goff = np.ascontiguousarray(np.cumsum([0] + [len(u) for u in self.runs]).astype(np.int64))
        self.gorder = None if reorder is None else np.ascontiguousarray(np.random.default_rng(list(reorder) + [5]).permutation(self.G).astype(np.int64))
        L = np.zeros(18, np.int64); self.PPH = np.zeros(self.N, np.int64)
        for c in range(0, self.M, 8192):
            w = self.Wc[:, c:c + 8192].astype(np.int64); L += lit_by_place(w); self.PPH += ones(w).sum(1)
        self.L = L; self.uh = rule(L, self.G, self.M); self.ups = np.ascontiguousarray(np.asarray(self.uh, np.int32))
        self.pat = np.empty((self.G, self.M, len(self.uh)), np.uint16); self.used = np.zeros((self.G, self.H), np.uint8)
        dl, dj, db, ds = [], [], [], []
        for c in range(0, self.M, 8192):
            w = self.Wc[:, c:c + 8192].astype(np.int64)
            E, sd, kcorr, C = expand(w, self.uh); self.pat[:, c:c + w.shape[1], :] = patterns(E, self.runs, self.uh); del E
            for b, d in sd:
                t, j = np.nonzero(d); dl.append(t); dj.append(j + c); db.append(np.full(len(t), b)); ds.append(d[t, j])
        for k in range(self.G): self.used[k, self.pat[k].ravel() & 0x7FFF] = 1
        self.kcorr, self.C = int(kcorr), int(C)
        dl, dj, db, ds = (np.concatenate(a) for a in (dl, dj, db, ds))
        self.sp = sorted(set(int(b) for b in np.unique(db)))                   # the signed places the table lights at all
        slot = {b: i for i, b in enumerate(self.sp)}
        o = np.lexsort((dl, dj)) if reorder is None else np.random.default_rng(list(reorder) + [6]).permutation(len(dl))
        self.dl, self.dj, self.db, self.ds = (np.ascontiguousarray(a[o].astype(t)) for a, t in ((dl, np.int32), (dj, np.int32), (db, np.int32), (ds, np.int8)))
        self.dsl = np.ascontiguousarray(np.array([slot.get(b, -1) for b in range(18)], np.int32)[self.db])
        # the read order: every place the table lights, from the top down
        self.places = sorted(set(self.uh) | set(self.sp), reverse=True)
        # B(b): the most the places below b can add to the doubled pile per unit of light (|x| summed)
        self.B = {}
        for b in range(19):
            v = sum(1 << q for q in self.uh if q < b); below = [q for q in range(min(b, 18)) if q not in self.uh]
            runs_ = [];
            for q in below:
                if runs_ and runs_[-1][1] == q: runs_[-1][1] = q + 1
                else: runs_.append([q, q + 1])
            v += sum(2 * (1 << a) * t_run(c_ - a) for a, c_ in runs_); self.B[b] = v
        log(f"  top table laid: up/hole places {self.uh}, signed places lit {self.sp}, {len(self.dl):,} lit signed digits, "
            f"kcorr {self.kcorr}, groups {self.G}; ({time.time() - t0:.0f}s, {rss()} MB)")
    def parts(self, x, nth=NTH):
        """per word: each read place's part of the doubled pile and its photons. x (n, N). Returns (PART, HC) (n, M, nplaces)
        in self.places order, the table's walk photons and the ready piles written, per position"""
        x = np.ascontiguousarray(np.asarray(x, np.int64)); n = len(x); xT = np.ascontiguousarray(x.T); nup = len(self.uh); nsp = len(self.sp)
        T = np.empty((self.G, self.H, n), np.int64); NT = np.empty((self.G, self.H, n), np.uint8); wk = np.zeros(n, np.int64)
        lib().half_tables(n, self.N, ptr(x), self.G, ptr(self.gl), ptr(self.goff), self.H, ptr(T), ptr(NT), ptr(wk))
        pw = np.zeros(n, np.int64); lib().pile_write(n, self.G, self.H, ptr(T), ptr(self.used), ptr(pw))
        A = np.zeros((self.M, nup, n), np.int64); HA = np.zeros((self.M, nup, n), np.int32)
        lib().uh_parts(n, self.G, self.H, ptr(T), ptr(NT), self.M, nup, ptr(self.ups), ptr(self.pat),
                       ptr(self.gorder) if self.gorder is not None else None, ptr(A), ptr(HA), nth)
        del T, NT
        S = np.zeros((self.M, nsp, n), np.int64); HS = np.zeros((self.M, nsp, n), np.int32)
        lib().sd_reads(n, ptr(xT), len(self.dl), ptr(self.dl), ptr(self.dj), ptr(self.db), ptr(self.ds), None, None, nsp, ptr(self.dsl), ptr(S), ptr(HS))
        PART = np.empty((n, self.M, len(self.places)), np.int64); HC = np.empty((n, self.M, len(self.places)), np.int32)
        for i, b in enumerate(self.places):
            if b in self.uh: u = self.uh.index(b); PART[:, :, i] = (A[:, u, :] << b).T; HC[:, :, i] = HA[:, u, :].T
            else: s = self.sp.index(b); PART[:, :, i] = S[:, s, :].T; HC[:, :, i] = HS[:, s, :].T
        return PART, HC, wk, pw
