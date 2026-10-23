"""CHECKER for LENS v5a (store T only; E rebuilt by the inverse lens + each word's DC). Part 1: the stored bytes.
No harness loaded here (memory). Steps:
  1. code T with the v10 film coder, write every stored piece to disk, forget everything, read the pieces back and
     decode -- check T exactly; count every byte (shape header included).
  2. gains G (768) and TE (50,257) and the DC D (50,257): lzma on my own packing, bytes counted, decoded back.
  3. E^ rebuilt ONLY from the decoded pieces with my own whole-number inverse lens (Python long division for H),
     compared with the claimant's inv_lens; saved for part 2 (words).
"""
import os, sys, time, json, lzma, pickle, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/lenscheck"
t0 = time.time()
src = open("/Users/abundancemachine/streaming-machine/shrink/lens_v10_rans_film.py").read()
src = src[:src.index("E, T, TE, HB, WPE = load()")]
ns = {}; exec(src, ns)                                     # the claimant's coder, functions only
encode_film, decode_film = ns["encode_film"], ns["decode_film"]
import lens_lib as LL

M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
T = np.ascontiguousarray(M["head.w"].T); TE = M["head.e"].astype(np.int64); E = M["wte"]
N, L = T.shape
res = {}

# ---- 1. T film: encode, write pieces, reload, decode ----
P = encode_film(T)
pieces = {k: v for k, v in P.items() if k != "shape"}
for k, v in pieces.items(): open(f"{SP}/T_{k}.bin", "wb").write(v)
shape_hdr = np.array(P["shape"], np.uint32).tobytes(); open(f"{SP}/T_shape.bin", "wb").write(shape_hdr)
del P, pieces
fileb = {k: os.path.getsize(f"{SP}/T_{k}.bin") for k in ["words", "ptr", "states", "tables", "kg", "grain", "esc", "shape"]}
res["T_parts_bytes_on_disk"] = fileb; res["T_film_bytes"] = sum(fileb.values())
P2 = {k: open(f"{SP}/T_{k}.bin", "rb").read() for k in ["words", "ptr", "states", "tables", "kg", "grain", "esc"]}
P2["shape"] = tuple(int(x) for x in np.frombuffer(open(f"{SP}/T_shape.bin", "rb").read(), np.uint32))
Tb = decode_film(P2).astype(np.int16); del P2
res["T_roundtrip_ok"] = bool(np.array_equal(Tb, T))
print(f"T film {res['T_film_bytes']} B ({res['T_film_bytes']*8/T.size:.4f} bits/value) roundtrip {res['T_roundtrip_ok']} [{time.time()-t0:.0f}s]", flush=True)

# per-line order-0 floor of T (reference only)
fl = 0.0
for i in range(L): fl += LL.order0_bits(T[:, i]) * N
res["T_per_line_order0_floor_MB"] = fl / 8 / 1e6

# ---- 2. side information ----
G = LL.whole(LL.lens_float("ln_f.weight"), 24)                  # the lens written once as whole numbers
bG = lzma.compress(G.astype(np.int64).tobytes(), preset=9)
bTE = lzma.compress(TE.astype(np.int64).tobytes(), preset=9)
Gs, Ts_ = LL.pack(G), LL.pack(TE)
res["gains_bytes_mine"] = len(bG); res["TE_bytes_mine"] = len(bTE)
res["gains_plus_TE_bytes_claimant_packing"] = LL.nbytes(*Gs) + LL.nbytes(*Ts_)
# the DC per word, from E, my own whole-number arithmetic
Gi = G.astype(np.int64)
assert int(np.abs(E).max()) * int(np.abs(Gi).max()) * 768 < (1 << 62)
D = np.empty(N, np.int64)
for a in range(0, N, 4096):
    S = (E[a:a + 4096] * Gi[None, :]).sum(1)
    q = (np.abs(S) + 384) // 768; q = np.where(S < 0, -q, q)       # round(S/768), half up in size
    D[a:a + 4096] = (q + (1 << 19)) >> 20                           # to 2^-20, rounded
Dp = LL.pack(D); bD_mine = lzma.compress(D.tobytes(), preset=9)
res["DC_bytes_claimant_packing"] = LL.nbytes(*Dp); res["DC_bytes_plain_lzma"] = len(bD_mine)
res["DC_bits_range"] = [int(D.min()), int(D.max())]
# decode side info back from bytes
G2 = np.frombuffer(lzma.decompress(bG), np.int64); TE2 = np.frombuffer(lzma.decompress(bTE), np.int64)
D2 = LL.unpack(*Dp)
res["side_roundtrip_ok"] = bool(np.array_equal(G2, G) and np.array_equal(TE2, TE) and np.array_equal(D2, D))

# ---- 3. E^ from decoded pieces only, my own inverse lens ----
H = np.array([((1 << 48) + (int(g) >> 1)) // int(g) for g in G2], np.int64)   # Python whole-number division
assert (G2 > 0).all() and (TE2 + 20 >= 0).all()
Eh = np.empty(E.shape, np.int64)
for a in range(0, N, 4096):
    v = (Tb[a:a + 4096].astype(np.int64) << (TE2[a:a + 4096] + 20)[:, None]) + D2[a:a + 4096, None]
    assert int(np.abs(v).max()) * int(H.max()) < (1 << 62)
    Eh[a:a + 4096] = (v * H[None, :] + (1 << 27)) >> 28
# claimant's inverse lens on the same inputs, for comparison
Hc = np.array([LL.div_round(np.array([1 << 48], np.int64), int(g))[0] for g in G], dtype=np.int64)
res["H_same_as_claimant"] = bool(np.array_equal(H, Hc))
dE = Eh - E
res["Eh_mean_abs_diff_2^-16"] = float(np.abs(dE).mean()); res["Eh_max_abs_diff"] = int(np.abs(dE).max())
res["Eh_exact_places_pct"] = float((dE == 0).mean() * 100)
del dE
np.save(f"{SP}/Eh.npy", Eh); np.save(f"{SP}/Tb.npy", Tb)
# shuffle control input (inverse gains on randomly permuted lines, seed 0) -- same as claimant's control
Hs = H[np.random.default_rng(0).permutation(L)]
Ehs = np.empty(E.shape, np.int64)
for a in range(0, N, 4096):
    v = (Tb[a:a + 4096].astype(np.int64) << (TE2[a:a + 4096] + 20)[:, None]) + D2[a:a + 4096, None]
    Ehs[a:a + 4096] = (v * Hs[None, :] + (1 << 27)) >> 28
np.save(f"{SP}/Ehs.npy", Ehs); del Ehs

side = res["gains_bytes_mine"] + res["TE_bytes_mine"] + min(res["DC_bytes_claimant_packing"], res["DC_bytes_plain_lzma"])
res["total_bytes"] = res["T_film_bytes"] + side
res["total_MB"] = res["total_bytes"] / 1e6
res["bits_per_word_line"] = res["total_bytes"] * 8 / T.size
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
json.dump(res, open(f"{SP}/part1.json", "w"), indent=1)
