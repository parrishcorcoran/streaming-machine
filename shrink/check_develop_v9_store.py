"""CHECKER for the develop family's best claim: recount the stored bytes of the developed E (lens exposure, cap 12) and
of everything needed to print T from it, with (a) an independent range coder (check_develop_rc.c), (b) develop_lib's own
range coder, decoded back here, (c) lzma and bz2 on the packed whole numbers. Every stream is decoded and checked equal.
Side information counted: the 768 gains (the lens), c, cap, K, shape, stream lengths; and, separately, the per-word head
offset table (head.b) that the top race also reads and the claim leaves out.
    python3 check_develop_v9_store.py
"""
import os, sys, json, struct, ctypes, lzma, bz2, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import develop_lib as dl
HERE = "/Users/abundancemachine/streaming-machine/shrink"
lib = ctypes.CDLL(os.path.join(HERE, "check_develop_rc.dylib")); lib.ck_encode.restype = ctypes.c_int64
P = ctypes.c_void_p

def ck_enc(sym, ctx, K, nctx):
    sym = np.ascontiguousarray(sym, np.uint16); ctx = np.ascontiguousarray(ctx, np.uint16); cap = len(sym) * 2 + 65536
    buf = np.empty(cap, np.uint8)
    n = lib.ck_encode(sym.ctypes.data_as(P), ctx.ctypes.data_as(P), ctypes.c_int64(len(sym)), ctypes.c_int(K),
                      ctypes.c_int64(nctx), buf.ctypes.data_as(P), ctypes.c_int64(cap))
    assert n <= cap; return buf[:n].tobytes()

def ck_dec(data, ctx, K, nctx):
    ctx = np.ascontiguousarray(ctx, np.uint16); b = np.frombuffer(data, np.uint8).copy(); out = np.empty(len(ctx), np.uint16)
    lib.ck_decode(b.ctypes.data_as(P), ctypes.c_int64(len(b)), ctx.ctypes.data_as(P), ctypes.c_int64(len(ctx)), ctypes.c_int(K),
                  ctypes.c_int64(nctx), out.ctypes.data_as(P))
    return out

M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
E = np.asarray(M["wte"], np.int64)
p = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); a, b = h["ln_f.weight"]["data_offsets"]
f.seek(8 + n + a); G = np.round(np.frombuffer(f.read(b - a), "<f4").astype(np.float64) * 2 ** 24).astype(np.int64)
place = np.array([int(g).bit_length() - 1 for g in G], np.int64)
gain_raw = 768 * 4                                       # 768 gains, 29 bits each, stored as 4 bytes (no coder)
gain_xz = len(lzma.compress(G.astype(np.int32).tobytes()))

# the per-word head offset (head.b): read by the top race for every word, not in E or T
hb = np.asarray(M["head.b"], np.int64); zb = dl.zigzag(hb - int(hb.min()))
hb_xz = len(lzma.compress(hb.astype(np.int32).tobytes(), preset=9)); hb_bz = len(bz2.compress(hb.astype(np.int32).tobytes(), 9))
print(f"head.b: 50,257 offsets, {int(hb.min())}..{int(hb.max())} ({int(hb.max() - hb.min()).bit_length()} bits of spread); "
      f"xz {hb_xz} bytes, bz2 {hb_bz} bytes", flush=True)

line = np.tile(np.arange(768, dtype=np.uint16), E.shape[0])
res = {"gain_raw_bytes": gain_raw, "gain_xz_bytes": gain_xz, "head_b_xz_bytes": hb_xz, "head_b_bz2_bytes": hb_bz}
for c in (32, 34, 37):
    e = np.minimum(np.maximum(c - place, 0), 12); q = (np.int64(1) << e)[None, :]
    D = np.empty(E.shape, np.int32); z16 = np.empty(E.size, np.uint16)
    for r in range(0, E.shape[0], 4096):
        v = E[r:r + 4096]; m = np.abs(v); m2 = ((m + (q >> 1)) >> e) << e; d = np.where(v < 0, -m2, m2); D[r:r + 4096] = d
        Qc = d >> e[None, :]; assert np.array_equal(Qc << e[None, :], d)
        zz = dl.zigzag(Qc).ravel(); assert int(zz.max()) < 65536; z16[r * 768:r * 768 + zz.size] = zz
    assert np.array_equal(D, dl.develop_lines(E, c, 12)[0])
    K = int(z16.max()) + 1
    # per-line order-0 floor (a reference, not a coder)
    floor = 0.0
    zl = z16.reshape(-1, 768)
    for i in range(768):
        cnt = np.bincount(zl[:, i]); cnt = cnt[cnt > 0]; floor += float((cnt * (np.log2(cnt.sum()) - np.log2(cnt))).sum())
    floor_B = floor / 8
    # (a) independent coder
    blob = ck_enc(z16, line, K, 768); back = ck_dec(blob, line, K, 768)
    ok_a = bool(np.array_equal(back, z16)); del back
    side = 1 + 1 + 2 + 4 + 4 + 4                         # c, cap, K, rows, lines, stream length
    Db = dl.unzigzag(ck_dec(blob, line, K, 768).astype(np.int64)).reshape(-1, 768) << e[None, :]
    ok_a &= bool(np.array_equal(Db, D.astype(np.int64))); del Db
    tot_a = len(blob) + side + gain_raw
    # (b) develop_lib's own coder, decoded here
    w = dl.nb(K - 1); bq = dl.rc_encode(z16, line, w, 768)
    Db = dl.unzigzag(dl.rc_decode(bq, line, w, 768).astype(np.int64)).reshape(-1, 768) << e[None, :]
    ok_b = bool(np.array_equal(Db, D.astype(np.int64))); del Db
    tot_b = len(bq) + side + gain_raw
    # (c) lzma / bz2 on the packed whole numbers (two byte planes when K > 256)
    if K <= 256: planes = [z16.astype(np.uint8).tobytes()]
    else: planes = [(z16 & 255).astype(np.uint8).tobytes(), (z16 >> 8).astype(np.uint8).tobytes()]
    lz = [lzma.compress(x, preset=6) for x in planes]; bzb = [bz2.compress(x, 9) for x in planes]
    ok_c = bool(np.array_equal(np.frombuffer(lzma.decompress(lz[0]), np.uint8).astype(np.uint16)
                               | (np.frombuffer(lzma.decompress(lz[1]), np.uint8).astype(np.uint16) << 8 if K > 256 else 0), z16))
    tot_lz = sum(map(len, lz)) + side + gain_raw; tot_bz = sum(map(len, bzb)) + side + gain_raw
    r = dict(K=K, own_rc_MB=round(tot_a / 1e6, 3), own_rc_ok=ok_a, their_rc_MB=round(tot_b / 1e6, 3), their_rc_ok=ok_b,
             lzma_MB=round(tot_lz / 1e6, 3), bz2_MB=round(tot_bz / 1e6, 3), lzma_ok=ok_c,
             line_floor_MB=round(floor_B / 1e6, 3), their_over_floor_pct=round(100 * (len(bq) / floor_B - 1), 2),
             bits_per_value_their=round(8 * tot_b / D.size, 3), bits_per_word_their=round(8 * tot_b / D.shape[0], 1),
             with_head_b_MB=round((tot_b + min(hb_xz, hb_bz)) / 1e6, 3),
             photons_per_value=round(float(dl.naf_photons(D[::7]).mean()), 3))
    res[c] = r; print("RESULT", c, json.dumps(r), flush=True)
    del D, z16, zl
json.dump(res, open(os.path.join(HERE, "check_develop_v9_store.json"), "w"), indent=1)
