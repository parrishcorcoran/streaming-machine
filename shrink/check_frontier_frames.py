"""CHECKER: other frames of the same exposure as a stored .cmf file (my own construction, not the family's
judge_phase). The steps come from the FILE (each plate's A, neuron terms, line offsets; the table's c, cap and lens).
Frame 0 = nearest on the plain grid (must reproduce the file's own values: hashes compared). Frame k>0 = the same
grid slid by a whole-number phase per neuron (plates) and per line (entry table), from a fixed seed: same step, same
grain power, another grain pattern. The top table is printed from each frame's entry table through the stored lens.
    python3 shrink/check_frontier_frames.py FILE SEED [SEED ...]"""
import sys, os, struct, lzma, json, time, hashlib, numpy as np
SM = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, SM); os.chdir(SM)
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]

def file_steps(path):
    raw = open(path, "rb").read(); secs = {}; p = 4
    while p < len(raw):
        tag = raw[p:p + 4].decode(); (ln,) = struct.unpack_from("<I", raw, p + 4); secs[tag] = raw[p + 8:p + 8 + ln]; p += 8 + ln
    d = secs["PLAT"]; q = 0; st = {}
    while q < len(d):
        idx, R, C, A, t0, o0, nb = struct.unpack_from("<BHHbhhI", d, q); q += 14 + C
        t = np.frombuffer(d, np.uint8, C, q).astype(np.int64) + t0; q += C
        off = np.frombuffer(d, np.uint8, R, q).astype(np.int64) + o0; q += R + nb
        s = (t[None, :] + off[:, None] - A) >> 1; s[s < 0] = 0; st[NAMES[idx]] = s
    d = secs["TABL"]; c, cap, half, nbits, lim, rows, nbq = struct.unpack_from("<BBBBBII", d, 0); q = 13
    assert not half
    (lg,) = struct.unpack_from("<I", d, q); q += 4
    G = np.frombuffer(lzma.decompress(d[q:q + lg]), np.int32).astype(np.int64)
    tp = np.zeros(768, np.int64); g = G.copy()
    while (g > 0).any(): tp += g > 0; g >>= 1
    ex = np.minimum(np.maximum(c - (tp - 1), 0), cap)
    return st, ex, G

def slide(q, s, ph):                     # nearest on the grid k*2^s - ph (ph in [0, 2^s)), whole numbers
    q = q.astype(np.int64); half = np.where(s > 0, np.int64(1) << np.maximum(s - 1, 0), 0)
    k = (q + ph + half) >> s
    return (k << s) - ph

if __name__ == "__main__":
    path = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]
    st, ex, G = file_steps(path)
    import table_bits as tb
    tb.E = None; tb.T = None; M = tb.M
    assert tb.words() == 30
    tb._WTE0 = None; tb.W["head"] = (None, tb.W["head"][1], tb.W["head"][2])
    for sd in seeds:
        t0 = time.time(); rng = np.random.default_rng(90000 + sd); hp = hashlib.sha256()
        for n in NAMES:
            q = M[n + ".w"]; s = st[n]
            ph = np.zeros(s.shape, np.int64) if sd == 0 else (rng.integers(0, 1 << 20, size=q.shape[1], dtype=np.int64)[None, :] & ((np.int64(1) << s) - 1))
            w = np.clip(slide(q, s, ph), -32768, 32767).astype(np.int16); hp.update(w.tobytes())
            tb.W[n] = None; tb.W[n] = (w.astype(np.float64), M[n + ".e"], M[n + ".b"])
        E0 = M["wte"]; phl = np.zeros(768, np.int64) if sd == 0 else (rng.integers(0, 1 << 20, size=768, dtype=np.int64) & ((np.int64(1) << ex) - 1))
        D = np.empty(E0.shape, np.int32)
        for r in range(0, E0.shape[0], 4096):
            v = E0[r:r + 4096].astype(np.int64)
            if sd == 0:                                     # nearest, ties away from zero on the magnitude (the family's rule)
                a = np.abs(v); qq = (np.int64(1) << ex)[None, :]; a2 = (a + (qq >> 1)) & ~(qq - 1); D[r:r + 4096] = np.where(v < 0, -a2, a2)
            else:
                D[r:r + 4096] = slide(v, np.broadcast_to(ex[None, :], v.shape), np.broadcast_to(phl[None, :], v.shape))
        del E0
        ht = hashlib.sha256()
        for r in range(0, D.shape[0], 4096): ht.update(np.ascontiguousarray(D[r:r + 4096], np.int32).tobytes())
        Tp = np.empty(D.shape, np.int32)
        for r in range(0, D.shape[0], 4096):
            sv = D[r:r + 4096].astype(np.int64) * G[None, :]; cc = 768 * sv - sv.sum(1, keepdims=True); y = (cc + (1 << 23)) >> 24
            Tp[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
        wd = int(tb.words(entry=D, top=Tp)); del D, Tp
        print(json.dumps(dict(file=os.path.basename(path), frame_seed=sd, words=wd, plates_sha=hp.hexdigest()[:16],
                              entry_sha=ht.hexdigest()[:16], secs=round(time.time() - t0, 1))), flush=True)
