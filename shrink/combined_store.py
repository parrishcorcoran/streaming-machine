"""COMBINED -- write the whole developed machine as ONE real file. Its size is the stored size of that point.

File (little endian), "CMF1" then six sections, each: a 4-byte tag, a uint32 length, the payload.
  PLAT  the 48 plates, round one's per-plate layout (plates_best_store.write): header <BHHbhhI (plate, lines in,
        neurons out, A, neuron-term base, line-offset base, photon bytes); each neuron's power of two (int8); each
        neuron's step term (uint8 above its base); each line's offset (uint8 above its base); the photons (plates_rc).
        The reader rebuilds every step s_ij = max(0, (t_j + off_i - A) >> 1) and every weight v << s_ij.
  BIAS  the 48 plates' biases, exact: per plate a uint32 length + the photon coder on the 1 x C row.
  TABL  the entry table: <BBBBBII (c, cap, halftoned?, symbol width, the coder's LIM, words, coded bytes); if halftoned the 768 line
        exposures (uint8); the lens (768 whole-number gains at 2^24, int32, lzma, with its uint32 length); the stored
        whole numbers k = D >> e_i, zigzagged, range coded with the line as context (combined_rc). The top table is NOT
        stored: the reader prints it from D through the lens.
  HEAD  the head's bias (photon coder, 1 x 50,257, with its uint32 length), the head's power of two (int8, one for
        every word).
  POSN  the position table, exact: each position as its change from the position before, photon coder (1024 x 768).
  ARIT  the arithmetic tables, exact: for phi, arrival, log2, exp2: <qI (first value, count), uint32 length, the
        photon coder on the changes; then log2(768) and eps40 as int64.
    python3 shrink/combined_store.py NAME            (NAME from POINTS below; files go to the scratchpad)"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
import hashlib
_s = open(HERE + "/combined_v6_light_allocation.py").read(); exec(_s[_s.index("MID = range"):_s.index("RUNS = {")])

def sec(tag, payload): return tag + struct.pack("<I", len(payload)) + payload

def write_machine(path, cfg, c, cap=12, half=None):
    M = tb.M; parts = []
    # PLAT
    buf = []
    for idx, n in enumerate(NAMES):
        rule, A = cfg[n]; q = plate(n); R, C = q.shape; e = colexp(n)
        t, off = terms(n, q, rule); s = steps(t, off, A); v = cut(q, s); ph = encode(v)
        tb0 = int(t.min()); ob0 = int(off.min())
        assert (t - tb0).max() < 256 and (off - ob0).max() < 256 and e.min() > -128
        buf.append(struct.pack("<BHHbhhI", idx, R, C, A, tb0, ob0, len(ph)) + e.astype(np.int8).tobytes()
                   + (t - tb0).astype(np.uint8).tobytes() + (off - ob0).astype(np.uint8).tobytes() + ph)
    parts.append(sec(b"PLAT", b"".join(buf)))
    # BIAS
    parts.append(sec(b"BIAS", b"".join(struct.pack("<I", len(b)) + b for b in (rc(M[n + ".b"].reshape(1, -1)) for n in NAMES))))
    # TABL
    D, _ = tables(c, cap, half); e = exposure(c, cap, half)
    z = table_symbols(D, e); w = dl.nb(int(z.max())); line = np.tile(np.arange(768, dtype=np.uint16), D.shape[0])
    bq = crc_encode(z, line, w); del z, line
    g = lzma.compress(G.astype(np.int32).tobytes(), preset=9)
    parts.append(sec(b"TABL", struct.pack("<BBBBBII", c, cap, 0 if half is None else 1, w, LIM, D.shape[0], len(bq))
                     + (b"" if half is None else e.astype(np.uint8).tobytes()) + struct.pack("<I", len(g)) + g + bq))
    # HEAD
    he = M["head.e"]; assert (he == he[0]).all(); hb = rc(M["head.b"].reshape(1, -1))
    parts.append(sec(b"HEAD", struct.pack("<I", len(hb)) + hb + struct.pack("<b", int(he[0]))))
    # POSN
    P = M["wpe"].astype(np.int64); dP = P.copy(); dP[1:] = P[1:] - P[:-1]
    parts.append(sec(b"POSN", struct.pack("<HH", *P.shape) + rc(dP)))
    # ARIT
    ar = []
    for k in ("phi", "arrival", "log2", "exp2"):
        a = M[k].astype(np.int64); b = rc(np.diff(a).reshape(1, -1))
        ar.append(struct.pack("<qII", int(a[0]), len(a), len(b)) + b)
    ar.append(struct.pack("<qq", int(M["log2_768"]), int(M["eps40"])))
    parts.append(sec(b"ARIT", b"".join(ar)))
    data = b"CMF1" + b"".join(parts)
    open(path, "wb").write(data)
    # what the writer developed, for the reader to compare against (hashes of the developed plates and table)
    hp = hashlib.sha256()
    for n in NAMES:
        rule, A = cfg[n]; v, s = dev_plate(n, rule, A); hp.update(shift_back(v, s).astype(np.int16).tobytes())
    ht = hashlib.sha256()
    for r in range(0, D.shape[0], dl.CH): ht.update(np.ascontiguousarray(D[r:r + dl.CH], np.int32).tobytes())
    sizes = {p[:4].decode(): len(p) for p in parts}
    return len(data), sizes, hp.hexdigest()[:16], ht.hexdigest()[:16]

POINTS = {
    "baseline_83": ("v0 baseline (plates v12 A=8 + tables c=34)", cfg_all("v12", 8), 34, None),
    "W1_30": ("v10 W1 layers 4-7 A=5, rest A=8, c=35", alloc(5, 8), 35, None),
    "R1_29": ("v8 R1 Q1 + end qkv A=5", alloc(5, 5, qkv_end=5), 35, None),
    "T1_28": ("v9 T1 R1 + end fc A=7", alloc(5, 5, qkv_end=5, fc_end=7), 35, None),
}
if __name__ == "__main__":
    for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else list(POINTS)):
        label, cfg, c, half = POINTS[name]
        path = SP + f"combined_{name}.cmf"
        size, sizes, hp, ht = write_machine(path, cfg, c, half=half)
        rec = dict(version=f"FILE {name}: {label}", file=path, file_bytes=size, total_MB=round(size / 1e6, 3),
                   section_bytes=sizes, plates_sha=hp, entry_sha=ht, c=c, half=half,
                   A_by_plate={n: cfg[n][1] for n in NAMES}, rule=cfg[NAMES[0]][0], words=None,
                   note="written; words come from combined_read.py, which reads only this file")
        log(rec)
