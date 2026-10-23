"""confirm_groups/audit_common.py -- helpers of the counting-convention audit (audit_*.py). Written for the audit: no
import of, or code from, base2_groups.py, qwen_groups.py, groups/ or groups2/. The reference steps (base2.py's and
qwen/qwen_base2.py's to16, rshift, power, log2fix, gauge, edge, bend, turn, arrival) are written again here in whole
numbers from those files' own lines; their tables are read from base2_gpt2.npz and qwen/qwen_base2.npz.

PHOTON COUNTING (the unit of every earlier result, as base2_fringes2.py states it):
  a value added costs its photons in the form it is held in: ones(v) for a binary value, naf(v) for a value written in
  signed digits; a product a x b costs ones(a) x ones(b) (each 1 of one side adds the other once); writing a binary
  value in signed digits costs its 1s (3v = v + 2v, one add). Shifts, signs and table reads cost nothing.
"""
import os, sys, json, ctypes, subprocess, zipfile, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; CG = f"{HERE}/confirm_groups"

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")
def naf1(v): a = abs(int(v)); return bin(a ^ (3 * a)).count("1")

# ---------- the C kernels (audit_k.c) ----------
_LIB = None
def lib():
    global _LIB
    if _LIB is None:
        src, so = f"{CG}/audit_k.c", f"{CG}/audit_k.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _LIB = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
        _LIB.prod_c.argtypes = [I, I, I, V, V, V]; _LIB.prod_f.argtypes = [I, I, I, V, V, V]
        _LIB.bank.argtypes = [I, I, I, I, I] + [V] * 8
    return _LIB

def product(x, w):
    """x (P, N) whole numbers |x| < 2^31, w (N, M) int16 in C or Fortran order: the exact int64 product"""
    x = np.ascontiguousarray(x, np.int32); P, N = x.shape; assert w.shape[0] == N and w.dtype == np.int16
    M = w.shape[1]; out = np.empty((P, M), np.int64)
    if w.flags["C_CONTIGUOUS"]: lib().prod_c(P, N, M, x.ctypes.data, w.ctypes.data, out.ctypes.data)
    else:
        assert w.flags["F_CONTIGUOUS"]; lib().prod_f(P, N, M, x.ctypes.data, w.ctypes.data, out.ctypes.data)
    return out

# ---------- reading an npz member in place (members stored flat) ----------
def member(npz, name):
    z = zipfile.ZipFile(npz); info = z.getinfo(name + ".npy"); assert info.compress_type == 0
    with open(npz, "rb") as f:
        f.seek(info.header_offset); h = f.read(30); assert h[:4] == b"PK\x03\x04"
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
        off = f.tell()
    return np.memmap(npz, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")

# ---------- the reference steps, whole numbers (base2.py and qwen_base2.py share these four) ----------
def rshift(X, s):
    s = np.asarray(s, dtype=np.int64); up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)
def to16(X, e):
    X = np.asarray(X, np.int64)
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)], np.int64); sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), e + sh
def log2fix(V, LOG2):
    out = []
    for v in np.atleast_1d(V):
        v = int(v); t = v.bit_length() - 1
        m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
        out.append(t * 65536 + int(LOG2[m]))
    return np.array(out, dtype=np.int64)
def power(X, F, EXP2):
    F = np.asarray(F, dtype=np.int64); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    return rshift(np.asarray(X, np.int64) * EXP2[f], 16 - I)
def gain_ones(F, EXP2):
    """the photons of the gain a divide multiplies by: ones(EXP2[fine part of F])"""
    return ones(EXP2[np.bitwise_and(np.asarray(F, np.int64), 0xFFFF)])

# ---------- the group plate, laid by the rules the builder's docstrings state (written again here) ----------
def naf_digits_lit(W):
    """per place 0..17, how many weights have a lit signed digit there (the classic rule: odd -> 2 - (r mod 4))"""
    r = np.asarray(W, np.int64).copy(); L = np.zeros(18, np.int64)
    for b in range(18):
        odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); L[b] = int(odd.sum()); r = (r - d) >> 1
    assert not r.any(); return L

class Plate:
    """w (N, M) whole numbers; segs [(name, a, b)] the plates of a slab, each its own place rule; g the group size;
    key the grouping key [seed, slab index, layer, g] (numpy's generator, a random line order cut into runs of g)."""
    def __init__(self, w, segs, g, key):
        w = np.asarray(w, np.int64); N, M = w.shape; self.N, self.M, self.g, self.segs = N, M, g, segs
        order = np.random.default_rng(key).permutation(N)
        runs = [order[s:s + g] for s in range(0, N, g)]; self.G = len(runs); self.gmax = max(len(u) for u in runs)
        self.glines = np.ascontiguousarray(np.concatenate(runs), np.int32)
        self.goff = np.concatenate([[0], np.cumsum([len(u) for u in runs])]).astype(np.int64)
        # the per-place rule, plate by plate: signed digits at b when L_b < G x M_seg
        up = np.zeros(M, np.int64); self.signed_places = {}
        for name, a, b in segs:
            Lb = naf_digits_lit(w[:, a:b]); U = self.G * (b - a)
            sp = [q for q in range(16) if Lb[q] < U]; self.signed_places[name] = sp
            up[a:b] = sum(1 << q for q in range(16) if q not in sp)
        self.up = up
        # the digits: bottom place to top, binary at up/hole places, signed elsewhere (16, 17 signed), offset C
        C = np.array([(1 << (int(u).bit_length() - 1)) if u else 0 for u in up], np.int64)
        r = w + C[None, :]; E = np.zeros_like(w); Wsig = np.zeros_like(w); nsig = np.zeros_like(w); self.planes = {}
        for b in range(18):
            isup = ((up >> b) & 1).astype(bool) if b < 16 else np.zeros(M, bool)
            odd = (r & 1) == 1
            e = np.where(isup[None, :], odd.astype(np.int64), 0)
            d = np.where(~isup[None, :] & odd, 2 - (r & 3), 0)
            E |= e << b; Wsig += d << b; nsig += (d != 0)
            if d.any(): self.planes[b] = np.ascontiguousarray(d, np.int16)        # the signed digits at place b: -1/0/+1
            r = (r - e - d) >> 1
        assert not r.any(), "digits left above place 17"
        assert np.array_equal(E - C[None, :] + Wsig, w), "the digits do not give the plate back"
        self.kcorr = up - 2 * C                                          # 2 x.w = sum_b 2^b V_b + 2 x.Wsig + kcorr X
        self.sig_line = [nsig[:, a:b].sum(1) for _, a, b in segs]       # lit signed digits per line, per segment
        # patterns: bit t = the binary digit of the group's line t
        pat = np.zeros((self.G, M, 16), np.uint16)
        for k in range(self.G):
            Lk = self.glines[self.goff[k]:self.goff[k + 1]]
            for t, i in enumerate(Lk):
                bits = (E[i][:, None] >> np.arange(16)[None, :]) & 1          # (M, 16)
                pat[k] |= (bits << t).astype(np.uint16)
        self.pat = np.ascontiguousarray(pat)
        # which half-table entries the plate reads (itself or its flip), plate only
        TH = 1 << (self.gmax - 1); self.readh = np.zeros((self.G, TH), np.uint8)
        upm = ((up[:, None] >> np.arange(16)[None, :]) & 1).astype(bool)
        for k in range(self.G):
            gk = int(self.goff[k + 1] - self.goff[k]); top = 1 << (gk - 1); full = (1 << gk) - 1
            p = pat[k][upm].astype(np.int64); keyv = np.where(p & top, p, p ^ full) - top
            self.readh[k, np.unique(keyv)] = 1
        # the half table's adds per line (group total: lines 1..gk-1 once; Gray walk: line t turns 2^(gk-2-t))
        ch = np.zeros(N, np.int64)
        for k in range(self.G):
            Lk = self.glines[self.goff[k]:self.goff[k + 1]]; gk = len(Lk)
            for t in range(gk): ch[Lk[t]] = (1 if t >= 1 else 0) + ((1 << (gk - 2 - t)) if t <= gk - 2 else 0)
        self.ch = ch; self.anyup = bool(up.any())
        self.wrow = [ones(w[:, a:b]).sum(1) for _, a, b in segs]; self.nrow = [naf(w[:, a:b]).sum(1) for _, a, b in segs]
        self.up32 = np.ascontiguousarray(up, np.uint32)

    def run(self, x):
        """the light x (P, N): the pile (exact) and every count, both ways, per position"""
        x = np.ascontiguousarray(x, np.int32); P = len(x)
        UP = np.empty((P, self.M), np.int64); cnt = np.empty((P, 4), np.int64)
        lib().bank(P, self.N, self.M, self.G, self.gmax, x.ctypes.data, self.glines.ctypes.data, self.goff.ctypes.data,
                   self.up32.ctypes.data, self.pat.ctypes.data, self.readh.ctypes.data, UP.ctypes.data, cnt.ctypes.data)
        X = x.astype(np.int64).sum(1)
        sig = np.zeros((P, self.M), np.int64)
        for b, pl in self.planes.items(): sig += product(x, pl) << b
        D = UP + 2 * sig + self.kcorr[None, :] * X[:, None]; assert not (D & 1).any(); pile = D >> 1
        o, s = ones(x), naf(x)
        c = dict(reads_naf=cnt[:, 0], reads_ones=cnt[:, 1], pwrite_ones=cnt[:, 2], pwrite_naf=cnt[:, 3],
                 light_ones=o.sum(1),
                 table_naf=(s @ self.ch) if self.anyup else np.zeros(P, np.int64),
                 table_ones=(o @ self.ch) if self.anyup else np.zeros(P, np.int64),
                 xsum_naf=s[:, 1:].sum(1) if self.anyup else np.zeros(P, np.int64),
                 xsum_ones=o[:, 1:].sum(1) if self.anyup else np.zeros(P, np.int64))
        xo_n = np.zeros(P, np.int64); xo_1 = np.zeros(P, np.int64)
        for kv in np.unique(self.kcorr):
            if kv == 0: continue
            nj = int((self.kcorr == kv).sum()); xo_n += nj * (naf(int(kv) * X) + (naf1(kv) - 1) * naf(X))
            xo_1 += nj * (ones(int(kv) * X) + (ones1(kv) - 1) * ones(X))
        c["xoff_naf"], c["xoff_ones"] = xo_n, xo_1
        c["sdreads_naf"] = sum(s @ sl for sl in self.sig_line); c["sdreads_ones"] = sum(o @ sl for sl in self.sig_line)
        c["plain"] = sum(o @ wr for wr in self.wrow); c["plain_sd"] = sum(s @ nr for nr in self.nrow) + o.sum(1)
        c["plain_sd_meet"] = sum(s @ nr for nr in self.nrow); c["plain_platesd"] = sum(o @ nr for nr in self.nrow)
        return pile, c

def forms(c):
    """the bank's work under each convention, per position"""
    common_n = c["table_naf"] + c["reads_naf"] + c["sdreads_naf"] + c["xsum_naf"] + c["xoff_naf"]
    return dict(
        delivered=c["light_ones"] + c["pwrite_ones"] + common_n,                       # as base2_groups / qwen_groups count it
        write_at_sd=c["light_ones"] + c["pwrite_naf"] + common_n,                      # the rewrite priced at the digits placed
        write_both=c["light_ones"] + c["pwrite_ones"] + c["pwrite_naf"] + common_n,    # read the 1s AND place the digits
        no_write=common_n,                                                             # the confirm steps' count (reads at signed digits, no writing)
        all_ones=c["table_ones"] + c["reads_ones"] + c["sdreads_ones"] + c["xsum_ones"] + c["xoff_ones"])   # nothing rewritten: every add at its 1s
