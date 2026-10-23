"""PLATES -- the best developments written to ONE real file each and read back by a decoder that sees only that
file: per plate a header, each neuron's power of two, each neuron's step term, each line's offset (whole numbers),
and the photon-coded stored numbers. The file size IS the stored size. The decoder rebuilds every step
s_ij = max(0, (t_j + off_i - A) >> 1), every weight (v << s_ij, saturated to 16 bits), puts the plates in the
machine and asks for the words.
   python3 shrink/plates_best_store.py            (files go to the scratchpad given in OUTDIR)"""
from plates_lib import *
import struct, sys
import table_bits as _tb
OUTDIR = os.environ.get("PLATES_OUT", "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad")
src = open(os.path.join(HERE, "plates_v14_next_path.py")).read(); src = src[src.index("def next_path"):src.index("def places")]
g = {}; exec("from plates_lib import *\n" + src, g); NP = g["NP"]

def terms(n, q, rule):
    """per neuron step term t_j and per line offset off_i, half places (what the file carries)"""
    qa = np.abs(q.astype(np.int64)); R, C = q.shape
    t = halfplace(qa.sum(0)) + 2 * (2 - bl(np.array(R)))
    trh = halfplace(qa.sum(1)); off = trh - mode(trh)
    if rule == "v14" and n in NP:
        I = NP[n]; cols = np.arange(C) if n.startswith("fc") else np.arange(1536, 2304)
        t = t.copy(); t[cols] = mode(t[cols]) - (I - mode(I))
    return t, off

def steps(t, off, A): return np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)

def write(path, rule, A, k=None):
    with open(path, "wb") as f:
        for idx, n in enumerate(NAMES):
            q = plate(n); R, C = q.shape; e = colexp(n)
            t, off = terms(n, q, rule); s = steps(t, off, A)
            v = cut(q, s)
            if k: v = topk(v, k, chance=False)
            buf = encode(v)
            tb = int(t.min()); ob = int(off.min())
            assert (t - tb).max() < 256 and (off - ob).max() < 256 and e.min() > -128
            f.write(struct.pack("<BHHbhhI", idx, R, C, A, tb, ob, len(buf)))
            f.write(e.astype(np.int8).tobytes()); f.write((t - tb).astype(np.uint8).tobytes()); f.write((off - ob).astype(np.uint8).tobytes())
            f.write(buf)
    return os.path.getsize(path)

def read_into_machine(path):
    data = open(path, "rb").read(); p = 0; hs = struct.calcsize("<BHHbhhI")
    for _ in NAMES:
        idx, R, C, A, tb, ob, nb = struct.unpack_from("<BHHbhhI", data, p); p += hs
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        v = decode(data[p:p + nb], R, C); p += nb
        n = NAMES[idx]; assert np.array_equal(e, colexp(n))
        w = shift_back(v, steps(t, off, A))
        _tb.W[n] = (w.astype(np.float64), _tb.W[n][1], _tb.W[n][2])
    assert p == len(data)

assert words() == 30
for label, rule, A, k in [("v12 A=8", "v12", 8, None), ("v14 A=6", "v14", 6, None), ("v14 A=6 + 2 photons", "v14", 6, 2)]:
    path = os.path.join(OUTDIR, f"plates_{label.replace(' ', '_').replace('=', '').replace('+', 'p')}.bin")
    size = write(path, rule, A, k)
    try:
        read_into_machine(path); wd = int(words())
    finally:
        restore()
    rec = {"version": f"FILE {label}", "file": path, "file_bytes": size, "total_MB": round(size / 1e6, 3),
           "bits_per_value": round(8 * size / NVAL, 4), "words": wd, "decoded_from_file_only": True}
    log(rec); print(json.dumps(rec), flush=True)
