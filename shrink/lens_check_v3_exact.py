"""CHECKER for the exact LENS answer (v3 + v10): E film (v10 rANS) + T's edge flags (range coder) + exceptions + lens.
Every piece written to disk, the originals forgotten, then E and T decoded from those files only and checked exactly.
Bytes counted from the files (shape header and lens included)."""
import os, sys, time, json, math, lzma, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/lenscheck"
import lens_lib as LL, lens_rc
t0 = time.time()
s10 = open("/Users/abundancemachine/streaming-machine/shrink/lens_v10_rans_film.py").read()
ns = {}; exec(s10[:s10.index("E, T, TE, HB, WPE = load()")], ns)
s3 = open("/Users/abundancemachine/streaming-machine/shrink/lens_v3_edge_context.py").read()
s3 = s3[s3.index("K, NB = 24, 48"):s3.index("That, side, ctx = lens_edges(E, G, TE)")]
s3 = s3.replace("E, T, TE, HB, WPE = load()\n", "").replace("G = whole(lens_float(\"ln_f.weight\"), K)\n", "")

M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
E = M["wte"].copy(); T = np.ascontiguousarray(M["head.w"].T); TE = M["head.e"].astype(np.int64)
N, L = E.shape
G = LL.whole(LL.lens_float("ln_f.weight"), 24)
files = {}
def put(name, b): open(f"{SP}/X_{name}.bin", "wb").write(b); files[name] = len(b)

# ---- encode ----
P = ns["encode_film"](E)
for k, v in P.items():
    if k != "shape": put("E_" + k, v)
put("E_shape", np.array(P["shape"], np.uint32).tobytes()); del P
put("G", lzma.compress(G.astype(np.int64).tobytes(), preset=9)); put("TE", lzma.compress(TE.tobytes(), preset=9))
g = dict(np=np, math=math, div_round=LL.div_round, E=E, G=G, TE=TE, N=N, L=L, time=time, t0=t0); exec(s3, g)
That, side, ctx = g["lens_edges"](E, G, TE); NB = g["NB"]
R = T.astype(np.int64) - That
flip = (R == side); exc = (R != 0) & ~flip
print(f"nonzero {np.mean(R != 0)*100:.3f}%  exceptions {int(exc.sum())}  [{time.time()-t0:.0f}s]", flush=True)
bits = np.ascontiguousarray(flip.T).ravel().astype(np.uint8).tolist()
cx = np.ascontiguousarray(ctx.T).ravel().tolist()
put("flags", lens_rc.encode(bits, cx, 2 * NB)); del bits, cx
ei = np.flatnonzero(np.ascontiguousarray(exc.T).ravel()); ev = np.ascontiguousarray(R.T).ravel()[ei]
put("exc", lzma.compress(np.concatenate([np.diff(ei, prepend=0), ev]).astype(np.int64).tobytes()))
print(f"encoded [{time.time()-t0:.0f}s]  {files}", flush=True)
E0, T0 = E, T
del That, side, ctx, R, flip, exc, ei, ev, M

# ---- decode from the files only ----
rd = lambda n: open(f"{SP}/X_{n}.bin", "rb").read()
P2 = {k: rd("E_" + k) for k in ["words", "ptr", "states", "tables", "kg", "grain", "esc"]}
P2["shape"] = tuple(int(x) for x in np.frombuffer(rd("E_shape"), np.uint32))
E2 = ns["decode_film"](P2); del P2
G2 = np.frombuffer(lzma.decompress(rd("G")), np.int64).copy(); TE2 = np.frombuffer(lzma.decompress(rd("TE")), np.int64).copy()
g2 = dict(np=np, math=math, div_round=LL.div_round, E=E2, G=G2, TE=TE2, N=N, L=L, time=time, t0=t0); exec(s3, g2)
Th2, side2, ctx2 = g2["lens_edges"](E2, G2, TE2)
fl = np.frombuffer(lens_rc.decode(rd("flags"), np.ascontiguousarray(ctx2.T).ravel().tolist(), 2 * NB), np.uint8)
T2 = Th2.astype(np.int64) + fl.reshape(L, N).T * side2.astype(np.int64)
z = np.frombuffer(lzma.decompress(rd("exc")), np.int64); k = len(z) // 2
pos = np.cumsum(z[:k]); T2l = np.ascontiguousarray(T2.T).ravel()
T2l[pos] = np.ascontiguousarray(Th2.T).ravel()[pos].astype(np.int64) + z[k:]
T2 = T2l.reshape(L, N).T
r = {"files_bytes": files, "total_bytes": sum(files.values()), "total_MB": sum(files.values()) / 1e6,
     "E_roundtrip": bool(np.array_equal(E2, E0)), "T_roundtrip": bool(np.array_equal(T2, T0.astype(np.int64))),
     "E_film_bytes": sum(v for k2, v in files.items() if k2.startswith("E_")),
     "bits_per_word_line_both_tables": sum(files.values()) * 8 / (N * L)}
print(json.dumps(r, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
json.dump(r, open(f"{SP}/part4.json", "w"), indent=1)
