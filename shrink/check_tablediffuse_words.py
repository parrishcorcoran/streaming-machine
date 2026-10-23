"""CHECK of the tablediffuse (diffuse_tables) family -- step 2, the words.
Loads D and T exactly as step 1 rebuilt them from the files alone (scratchpad/chk_tdf_<tag>.npz) and asks the harness
words(entry=D, top=T) -- the harness's own route. Also: (a) is the printing rule T = print(E) on the untouched tables,
(b) does the decoded D equal the checker's OWN development of E (numpy, written here from the stated rule),
(c) the whole machine: plates_v12_A8.bin parsed here and decoded by plates_rc.c compiled fresh, put in the machine,
then each tables file. (d) optional 'phases': the screen-phase copies redeveloped here and judged."""
import sys, os, struct, ctypes, json, gc, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); os.chdir("/Users/abundancemachine/streaming-machine")
import table_bits as tb
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
FILES = {"E8c32": ("tables_E8_c32.tdf", "E8", 32), "D8c34": ("tables_D8_c34.tdf", "D8", 34), "cubec34": ("tables_cube_c34.tdf", "cube", 34)}
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
args = sys.argv[1:]

def out(tag, d): print(tag, json.dumps(d), flush=True)

def tprint(D, G):
    T = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], 4096):
        sv = D[r:r + 4096].astype(np.int64) * G[None, :]; cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
        T[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
    return T

def rm13():
    B = np.zeros((16, 8), np.int64)
    for a in range(8):
        for b in range(2):
            for x in range(8): B[a * 2 + b, x] = (bin(a & x).count("1") + b) & 1
    return B

def qnear(v, e):
    q = np.int64(1) << e; a = np.abs(v); a2 = (a + (q >> 1)) & ~(q - 1); return np.where(v < 0, -a2, a2)

def my_develop(E, e, kind, grp, o=None):
    """the checker's own development (numpy, whole numbers) of E (rows, 768) int64; o: screen phase per line"""
    n = E.shape[0]; D = np.empty((n, 768), np.int64); B = rm13()
    for r in range(0, n, 4096):
        v = E[r:r + 4096].astype(np.int64)
        if o is not None: v = v - o[None, :]
        if kind == "cube":
            d = qnear(v, e[None, :])
        else:
            vg = v[:, grp].reshape(-1, 96, 8); eg = e[grp].reshape(96, 8)[None]
            if kind == "D8":
                dd = qnear(vg, eg); k = dd >> eg; u = (vg - dd) << (16 - eg)
                par = k.sum(2) & 1; j = np.argmax(np.abs(u), axis=2)
                uj = np.take_along_axis(u, j[:, :, None], 2)[:, :, 0]
                step = np.where(uj >= 0, 1, -1) * par
                np.put_along_axis(k, j[:, :, None], (np.take_along_axis(k, j[:, :, None], 2)[:, :, 0] + step)[:, :, None], 2)
            else:
                x = vg << (16 - eg); ke = ((x + 65536) >> 17) << 1; ko = ((x >> 17) << 1) + 1
                ce = (x - (ke << 16)) ** 2; co = (x - (ko << 16)) ** 2
                cost = ce.sum(2)[:, :, None] + np.einsum("rgj,cj->rgc", co - ce, B)
                best = np.argmin(cost, axis=2); Bb = B[best]; k = np.where(Bb == 1, ko, ke)
            dg = (k << eg).reshape(-1, 768); d = np.empty_like(dg); d[:, grp] = dg
        D[r:r + 4096] = d + (o[None, :] if o is not None else 0)
    return D

# --- truth, and the printing rule on the untouched tables
assert tb.words() == 30
G0 = np.load("shrink/develop_gain24.npy").astype(np.int64)
HEAD = (None, tb.W["head"][1], tb.W["head"][2])
def judge(D, T):
    """words(entry=D, top=T) with the top put in the head the way the harness does (T.T as float64), without the harness's
    int64 copy and without keeping the old head (the truth is already taken): same values, less memory"""
    tb.W["head"] = None; gc.collect(); tb.W["head"] = (np.asarray(T).T.astype(np.float64), HEAD[1], HEAD[2])
    return int(tb.words(entry=D))
if "tables" in args:
  Tp = tprint(tb.E, G0); diff = int((Tp != tb.T).sum()); tsize = int(tb.T.size); tb.E = None; tb.T = None; gc.collect()
  out("PRINT_RULE", dict(print_E_equals_T_values_differing=diff, of=tsize, words_E_with_printed_T=int(tb.words(top=Tp)),
                         head_e_distinct=int(np.unique(tb.W["head"][1]).size)))
  del Tp
E = tb._WTE0; tb.E = None; tb.T = None; gc.collect()
for tag, (fn, kind, c) in (FILES.items() if "tables" in args else []):
    z = np.load(os.path.join(SP, f"chk_tdf_{tag}.npz")); D = z["D"]; T = z["T"]; G = z["G"]
    raw = open(os.path.join(SP, fn), "rb").read(16 + 3072 + 1536)
    grp = np.frombuffer(raw, np.uint16, 768, 16 + 3072).astype(np.int64) if kind != "cube" else None
    place = np.array([int(g).bit_length() - 1 for g in G], np.int64); e = np.minimum(np.maximum(c - place, 0), 12)
    mine = my_develop(E, e, kind, grp)
    nd = int((mine != D).sum()); del mine
    changed = int((D.astype(np.int64) != E).sum())
    w = int(tb.words(entry=D, top=T))
    out("TABLES_FROM_FILE", dict(tag=tag, file_bytes=os.path.getsize(os.path.join(SP, fn)), words=w,
                                 gains_equal_npy=bool(np.array_equal(G, G0)),
                                 groups_equal_seed7=None if grp is None else bool(np.array_equal(grp, np.random.default_rng(7).permutation(768))),
                                 decoded_vs_checker_development_differ=nd, changed_share_vs_E=round(changed / D.size, 4),
                                 e_hist=np.bincount(e).tolist()))
    del D, T, z; gc.collect()

if "phases" in args:
    for kind, c in (("E8", 32), ("D8", 34), ("cube", 32), ("cube", 34)):
        tag = f"{kind}c{c}"
        G = G0; place = np.array([int(g).bit_length() - 1 for g in G], np.int64); e = np.minimum(np.maximum(c - place, 0), 12)
        grp = np.random.default_rng(7).permutation(768).astype(np.int64)
        res = []
        for sd in (1, 2, 3, 4):
            r16 = np.random.default_rng(sd).integers(0, 1 << 16, size=768, dtype=np.int64); o = (r16 << e) >> 16
            D = my_develop(E, e, kind, grp, o); T = tprint(D, G)
            res.append(judge(D, T)); del D, T; gc.collect()
        out("PHASES", dict(tag=tag, seeds=[1, 2, 3, 4], words=res))

if "machine" in args:
    lib = ctypes.CDLL(os.path.join(SP, "chk_tdf_plates_rc.dylib")); lib.plate_decode.restype = ctypes.c_int
    lib.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
    pf = os.path.join(SP, "plates_v12_A8.bin"); data = open(pf, "rb").read(); p = 0; hs = struct.calcsize("<BHHbhhI")
    e_same = True; changed = 0; nval = 0; seen = []
    for _ in range(48):
        idx, R, C, A, tbb, ob, nb = struct.unpack_from("<BHHbhhI", data, p); p += hs
        ee = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tbb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        b = np.frombuffer(data[p:p + nb], np.uint8).copy(); v = np.zeros((R, C), np.int32); p += nb
        assert lib.plate_decode(b.ctypes.data, len(b), R, C, v.ctypes.data, 1) == 0
        s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
        w = np.clip(np.left_shift(v.astype(np.int64), s), -32768, 32767)
        n = NAMES[idx]; seen.append(n); e_same &= bool(np.array_equal(ee, np.asarray(tb.W[n][1], np.int64)))
        changed += int((w != tb.plate(n)).sum()); nval += w.size
        tb.W[n] = (w.astype(np.float64), ee.astype(np.asarray(tb.W[n][1]).dtype), tb.W[n][2]); del w, v
    assert p == len(data) and sorted(seen) == sorted(NAMES)
    out("PLATES_FILE", dict(file_bytes=len(data), powers_equal_machine=e_same, changed_share=round(changed / nval, 4),
                            words_plates_only=int(tb.words())))
    for tag, (fn, kind, c) in FILES.items():
        z = np.load(os.path.join(SP, f"chk_tdf_{tag}.npz")); tbytes = os.path.getsize(os.path.join(SP, fn))
        w = judge(z["D"], z["T"])
        out("WHOLE_MACHINE", dict(tag=tag, plates_bytes=len(data), tables_bytes=tbytes,
                                  MB=round((len(data) + tbytes) / 1e6, 3), words=w))
        del z; gc.collect()
