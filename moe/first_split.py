"""Where the GPU machine and granite_base2.py first part: q|k|v, the stream after the arrival's write... layer by layer."""
import sys, json, numpy as np
c = int(sys.argv[1]); sys.argv = sys.argv[:1]
import granite_base2 as g
refs = json.load(open("refs_granite.json")); gw = {r["case"]: r["words"] for r in json.load(open(f"gpu_words_{c}.json"))}
seq = refs[c]["ids"] + gw[c][:-1]; P = len(seq); pos = np.arange(P); D, L = g.D, g.L
raw = open(f"dump_{c}.bin", "rb").read(); o = 0
def take(n, dt, shape):
    global o; a = np.frombuffer(raw[o:o + n * np.dtype(dt).itemsize], dt).reshape(shape); o += n * np.dtype(dt).itemsize; return a
Rf = take(P * D, np.int64, (P, D)); LIT = take(L * P * 8, np.int32, (L, P, 8))
QKVg = take(L * P * 2048, np.int64, (L, P, 2048)); Rg = take(L * P * D, np.int64, (L, P, D))
R = np.asarray(g.EMBED[seq]).astype(np.int64)
for l in range(L):
    QKV = g.gauged_slab(R, g.plate(f"qkv.{l}"))
    dq = np.argwhere(QKV != QKVg[l])
    att = g.attention(QKV, pos); R = R + g.plain_slab(att, g.plate(f"o.{l}"))
    out, lit = g.experts(R, l); R = R + out
    dl = np.argwhere(lit != LIT[l]); dr = np.argwhere(R != Rg[l])
    print(f"layer {l:2d}: q|k|v differing {len(dq):6d} (first at position {dq[0][0] if len(dq) else '-'}, column {dq[0][1] if len(dq) else '-'}) | "
          f"lit slots differing {len(dl):3d} | stream after the layer differing {len(dr):6d}" + (f" (first at position {dr[0][0]}, largest gap {int(np.abs(R - Rg[l]).max())})" if len(dr) else ""), flush=True)
    if len(dq) and l == 0 or (len(dr) and not len(dq) and l < 3):
        p, cc = (dq[0] if len(dq) else dr[0]); print("   python", (QKV if len(dq) else R)[p, cc], "gpu", (QKVg[l] if len(dq) else Rg[l])[p, cc])
    if len(dr) and l >= 2: break
