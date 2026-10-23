"""flat_build/2/machine.py -- PICK 2 BUILT: the plate's sign run read once in every (gate, group) pair, on GPT-2's
whole-number machine, all 48 banks, the machine's own light at all 648 refs positions. Prediction in PREDICTION.txt
beside this file, written before this ran. The session's build; his words are only in PREDICTION.txt's quotes.

THE MACHINE (C): every place of every plate up/hole (E = w + 2^15, 16 places, no signed digits anywhere); the
confirmed random groups (key [1, slab, layer, g]); per (gate, group) cell, c = the smallest c with all the cell's
weights in [-2^c, 2^c): places 0..c-1 read as they are, the sign sheet (place 15) read once at place c, places
c+1..15 skipped. 2 x.w = sum over the reads of 2^place x (ready pile) - X; the halving is a shift.
THE BASELINE (A): groups2/gbank.py's GPlate (the confirmed per-place machine), imported unchanged, same light, same g.

    python3 flat_build/2/machine.py [g=12|R] [layers=a-b]
      g=12: g = 12 at every bank (pick 2 alone); g=R: 11 at proj and mproj, 12 at qkv and fc (stacked on pick 1).
    -> flat_build/2/out/machine_g<g>.json, .log
Runs per bank: A main, C main, A shuffled light, C shuffled light, C order-free.
"""
import os, sys, json, time, ctypes, subprocess, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D2 = f"{HERE}/flat_build/2"; OUT = f"{D2}/out"
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import gbank as GB
sys.argv = _argv
R = GB.R
SLABS = R.SLABS
GREADERS = {"qkv": 12, "proj": 11, "fc": 12, "mproj": 11}

def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

_LIB = None
def lib():
    global _LIB
    if _LIB is None:
        src, so = f"{D2}/runm.c", f"{D2}/runm.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _LIB = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
        _LIB.bank_run.argtypes = [I] * 5 + [V] * 10; _LIB.bank_run.restype = None
    return _LIB

def bitlen(v):
    v = np.asarray(v, np.int64); out = np.zeros(v.shape, np.int64)
    for s in range(17): out += (v >> s) > 0
    return out

class RunPlate:
    """the plate laid for the run machine (plate only, before any light)"""
    def __init__(self, w, g, key, reorder=None):
        w = np.asarray(w, np.int64); N, M = w.shape; self.N, self.M, self.g = N, M, g
        assert w.min() >= -(1 << 15) and w.max() < (1 << 15)
        E = w + (1 << 15); assert E.min() >= 0 and E.max() < (1 << 16)
        self.glines, self.goff = R.make_groups(N, g, key, reorder)
        sizes = np.diff(self.goff); self.G = G = len(sizes); self.gmax = int(sizes.max()); self.sizes = sizes
        rl = bitlen(np.where(w >= 0, w, ~w))                                   # per weight: smallest c, w in [-2^c, 2^c)
        cc = np.zeros((G, M), np.int64); pat = np.zeros((G, M, 16), np.uint16)
        lineg = np.zeros(N, np.int64)
        for k in range(G):
            L = self.glines[self.goff[k]:self.goff[k + 1]]; lineg[L] = k
            cc[k] = rl[L].max(0)
            for t, i in enumerate(L):
                for b in range(16): pat[k, :, b] |= (((E[i] >> b) & 1) << t).astype(np.uint16)
        assert cc.max() <= 15
        self.cc = cc
        # THE CELL IDENTITY, every line and gate: places 0..c-1 as they are, the sign sheet once at place c = 2w + 1
        cl = cc[lineg]                                                         # (N, M): the cell's c for each weight
        part = np.zeros((N, M), np.int64)
        for b in range(15): part += np.where(b < cl, (2 * ((E >> b) & 1) - 1) << b, 0)
        part += (2 * ((E >> 15) & 1) - 1) << cl
        assert np.array_equal(part, 2 * w + 1), "the run does not give the cell back"
        # and the sheets c..14 are the sign sheet flipped (the flat), in every cell
        for b in range(15):
            m = b >= cl
            assert np.array_equal(((E >> b) & 1)[m], 1 - ((E >> 15) & 1)[m]), "a sheet above c is not the flipped sign sheet"
        del part, cl
        # which half-table entries the plate reads (plate only)
        TH = 1 << (self.gmax - 1); self.readh = np.zeros((G, TH), np.uint8)
        for k in range(G):
            gk = int(sizes[k]); top = 1 << (gk - 1); full = (1 << gk) - 1
            m = np.arange(16)[None, :] < cc[k][:, None]
            p = np.concatenate([pat[k][m].astype(np.int64), pat[k][:, 15].astype(np.int64)])
            key_ = np.where(p & top, p, p ^ full) - top
            self.readh[k, np.unique(key_)] = 1
        self.pat = np.ascontiguousarray(pat); self.cc8 = np.ascontiguousarray(cc, np.int8)
        self.ch = GB.half_coeffs(self.glines, self.goff, N)                    # the table's photons by the relation
        self.nreads = int((cc + 1).sum())                                      # reads a position (plate only)
        self.table_adds = int(((sizes - 1) + (1 << (sizes - 1)) - 1).sum())
        self.wrow = ones(w).sum(1); self.w = w
        self.w16 = w.astype(np.int16)

    def read(self, x, chunk=96, place_order=None):
        x = np.ascontiguousarray(x, np.int32); P, N = x.shape; assert N == self.N
        assert np.abs(x.astype(np.int64)).max(initial=0) < 1 << 24
        pile = np.empty((P, self.M), np.int64); C = {}
        for c0 in range(0, P, chunk):
            xs = np.ascontiguousarray(x[c0:c0 + chunk]); Pc = len(xs)
            A = np.empty((Pc, self.M, 16), np.int64); H = np.empty((Pc, self.M, 16), np.int32)
            wr = np.empty(Pc, np.int64); tw = np.empty(Pc, np.int64)
            lib().bank_run(Pc, N, self.M, self.G, self.gmax, xs.ctypes.data, self.glines.ctypes.data, self.goff.ctypes.data,
                           self.pat.ctypes.data, self.cc8.ctypes.data, self.readh.ctypes.data,
                           A.ctypes.data, H.ctypes.data, wr.ctypes.data, tw.ctypes.data)
            X = xs.astype(np.int64).sum(1)
            D = np.zeros((Pc, self.M), np.int64)
            for b in (range(16) if place_order is None else place_order): D += A[:, :, b] << b
            D -= X[:, None]
            assert not (D & 1).any(), "the halving met an odd doubled pile"
            pile[c0:c0 + Pc] = D >> 1
            x64 = xs.astype(np.int64); sx = naf(x64)
            table_rel = sx @ self.ch
            assert np.array_equal(table_rel, tw), "the walked table's photons differ from the relation"
            c = dict(light=ones(x64).sum(1), table=tw, pwrite=wr, reads=H.sum((1, 2), dtype=np.int64),
                     sdreads=np.zeros(Pc, np.int64), xsum=sx[:, 1:].sum(1), xoff=self.M * naf(X))
            c["adds"] = np.full(Pc, self.nreads + self.table_adds + (N - 1) + self.M, np.int64)
            c["nreads"] = np.full(Pc, self.nreads, np.int64)
            c["plain"] = ones(x64) @ self.wrow; c["plain_adds"] = (x64 != 0).astype(np.int64) @ self.wrow
            for k, v in c.items(): C.setdefault(k, []).append(v)
            del A, H
        return pile, {k: np.concatenate(v) for k, v in C.items()}

def a_read(gp, x):
    """the confirmed machine (gbank) on light x, with its adds counted the same way"""
    pile, cnt = gp.read(x)
    x64 = np.asarray(x, np.int64); P, N = x64.shape
    nup = int(np.bitwise_count(gp.up.astype(np.uint64)).sum())
    sizes = np.diff(gp.goff)
    nreads = gp.G * nup
    table_adds = int(((sizes - 1) + (1 << (sizes - 1)) - 1).sum()) if gp.anyup else 0
    litpl = np.bincount(gp.tline, minlength=N).astype(np.int64)
    c = dict(light=cnt["light"], table=cnt["table"], pwrite=cnt["pwrite"],
             reads=sum(v for k, v in cnt.items() if k.startswith("reads|")),
             sdreads=sum(v for k, v in cnt.items() if k.startswith("sdreads|")), xsum=cnt["xsum"], xoff=cnt["xoff"])
    nk = int((gp.kcorr != 0).sum())
    c["adds"] = nreads + (x64 != 0).astype(np.int64) @ litpl + table_adds + (N - 1 if gp.anyup else 0) + nk
    c["nreads"] = np.full(P, nreads, np.int64); c["nsd"] = np.full(P, int(litpl.sum()), np.int64)
    c["plain"] = ones(x64) @ gp.wrow; c["plain_adds"] = (x64 != 0).astype(np.int64) @ gp.wrow
    return pile, c

WORK = ("light", "table", "pwrite", "reads", "sdreads", "xsum", "xoff")
def tot(c): return {k: int(np.asarray(v).sum()) for k, v in c.items()}

if __name__ == "__main__":
    A_ = dict(s.split("=") for s in sys.argv[1:]); GM = A_.get("g", "12"); lay = A_.get("layers")
    L0, L1 = (int(v) for v in lay.split("-")) if lay else (0, 11)
    tag = f"machine_g{GM}" + (f"_L{L0}-{L1}" if lay else ""); os.makedirs(OUT, exist_ok=True)
    LOG = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    JC = json.load(open(f"{HERE}/flat_design/judge/cellrule_counts.json"))
    Wz = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    light = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
    log(f"PICK 2 BUILT (flat_build/2): the sign run in every (gate, group) pair; g mode {GM}; layers {L0}..{L1}")
    res = {}
    for l in range(L0, L1 + 1):
        for kind in ("qkv", "proj", "fc", "mproj"):
            tl = time.time()
            w = np.asarray(Wz[f"{kind}.{l}.w"]).astype(np.int64); N, M = w.shape
            x = np.ascontiguousarray(light[kind][l], np.int32); P = len(x); ref = np.asarray(REF[f"{kind}_pile_{l}"])
            g = 12 if GM == "12" else GREADERS[kind]; key = [1, SLABS.index(kind), l, g]
            xs = np.random.default_rng([31337] + key).permuted(x, axis=1)
            direct_s = GB.plain_product(xs, w.astype(np.int16))
            rec = dict(g=g)
            # A: the confirmed per-place machine
            gp = GB.GPlate(w, [(kind, 0, M)], g, key)
            pA, cA = a_read(gp, x); badA = int((pA != ref).any(1).sum()); assert badA == 0, ("A", kind, l, badA)
            pAs, cAs = a_read(gp, xs); badAs = int((pAs != direct_s).any(1).sum()); assert badAs == 0
            rec["A"] = tot(cA); rec["A_shuffle"] = tot(cAs); rec["A_h"] = int(gp.up[0]).bit_length()
            del gp, pA, pAs
            # C: the run
            rp = RunPlate(w, g, key)
            pC, cC = rp.read(x); badC = int((pC != ref).any(1).sum()); assert badC == 0, ("C", kind, l, badC)
            pCs, cCs = rp.read(xs); badCs = int((pCs != direct_s).any(1).sum()); assert badCs == 0
            rec["C"] = tot(cC); rec["C_shuffle"] = tot(cCs)
            rec["C_c_hist"] = np.bincount(rp.cc.ravel(), minlength=16).tolist(); rec["C_cells"] = int(rp.cc.size)
            rec["C_entries_read"] = int(rp.readh.sum()); rec["C_entries"] = int(sum(1 << (int(s) - 1) for s in rp.sizes))
            # C order-free: groups and lines reordered, place parts summed in a random order
            ro = RunPlate(w, g, key, reorder=[4242] + key)
            assert sorted(map(tuple, (sorted(ro.glines[ro.goff[k]:ro.goff[k + 1]]) for k in range(ro.G)))) == \
                   sorted(map(tuple, (sorted(rp.glines[rp.goff[k]:rp.goff[k + 1]]) for k in range(rp.G))))
            porder = list(np.random.default_rng([777] + key).permutation(16))
            pCo, cCo = ro.read(x, place_order=porder)
            assert np.array_equal(pCo, pC), "order-free piles differ"
            rec["C_order"] = tot(cCo)
            # the judge's read counts (plate only), at g by readers
            jr = JC[f"{kind}.{l}"]
            rec["judge_formC_reads"] = jr["formC_everywhere_uh_reads"] if jr["g"] == g else None
            rec["C_reads_a_position"] = rp.nreads
            if rec["judge_formC_reads"] is not None:
                assert rp.nreads == jr["formC_everywhere_uh_reads"], ("judge's count", kind, l, rp.nreads, jr["formC_everywhere_uh_reads"])
            res[f"{kind}.{l}"] = rec
            wa = sum(rec["A"][k] for k in WORK); wc = sum(rec["C"][k] for k in WORK); pl = rec["A"]["plain"]
            was = sum(rec["A_shuffle"][k] for k in WORK); wcs = sum(rec["C_shuffle"][k] for k in WORK); pls = rec["A_shuffle"]["plain"]
            wco = sum(rec["C_order"][k] for k in WORK)
            log(f"  {kind:5s}.{l:<2d} g={g} h={rec['A_h']}: exact (A, C, both shuffled, C order-free); reads a position "
                f"C {rp.nreads:,}" + (f" = judge {jr['formC_everywhere_uh_reads']:,}" if rec['judge_formC_reads'] is not None else "") +
                f"; work % of plain: A {100 * wa / pl:.3f} C {100 * wc / pl:.3f} (C-A {100 * (wc - wa) / pl:+.3f}); shuffled: "
                f"A {100 * was / pls:.3f} C {100 * wcs / pls:.3f} (C-A {100 * (wcs - was) / pls:+.3f}); order-free C-main {100 * (wco - wc) / pl:+.4f}"
                f"  ({time.time() - tl:.1f}s, {rss_mb()} MB)")
            json.dump(dict(gmode=GM, banks=res), open(f"{OUT}/{tag}.json", "w"))
            del rp, ro, pC, pCs, pCo
    log(f"done ({time.time() - t0:.0f}s, max memory {rss_mb()} MB)")
