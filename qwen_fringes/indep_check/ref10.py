"""Checker's reference: qwen/qwen_base2.py's own steps, its own int64 products (x16 @ w), run text by text as its forward
does, on 10 refs texts drawn at random (seed 20261005). Nothing from qwen_fringes.py is imported.
Only three things differ from running qwen_base2.py directly, none of them arithmetic:
  - the npz path; W is loaded one plate at a time (int64, Fortran order so numpy's int64 product is quick; same numbers);
  - the product x16 @ w is passed through HOOK, which records its digest (and the light for 3 texts) and returns it unchanged;
  - the head is read in column blocks of 16,384 words (gauged_slab is column-separable), and at every position, not only
    the last one.
Records per text: every pile's digest, every attention output's, every stream's (both halves of every layer), the top
light's, and at every position the score row's sha256, the top 5 (by score, ties by word number) and the race winner.
For the 3 recount texts: the light at every plate, the qkv slab's output (for the arrival recount) and the top light.
"""
import sys, json, hashlib, time, resource, zipfile, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; NPZ = f"{QD}/qwen_base2.npz"
OUT = f"{HERE}/qwen_fringes/indep_check"
T10 = [3, 4, 5, 7, 8, 9, 12, 16, 18, 22]; T3 = [4, 7, 12]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def dg(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()

class OnePlate(dict):
    def __getitem__(self, name):
        if name not in self.keys():
            self.clear(); dict.__setitem__(self, name, (np.asfortranarray(MM[name + ".w"].astype(np.int64)), MM[name + ".e"]))
        return dict.__getitem__(self, name)
MM = np.load(NPZ)
REC = dict(t=None, dig={}, light={})
def HOOK(name, x16, pile):
    if name != "head":
        REC["dig"][f"pile|{name}|{REC['t']}"] = dg(pile)
        if REC["t"] in T3: REC["light"][f"{REC['t']}|{name}"] = np.asarray(x16, np.int16)
    return pile

src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a, b in [('M = np.load("qwen_base2.npz")', 'M = MM'),
             ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = OnePlate()'),
             ("pile = x16 @ w", "pile = HOOK(name, x16, x16 @ w)"),
             ("return rshift(x16 @ w,", "return rshift(HOOK(name, x16, x16 @ w),")]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
G = dict(MM=MM, OnePlate=OnePlate, HOOK=HOOK); sys.argv = ["x"]; exec(src, G)
to16, gauge, gauged_slab, plain_slab, attention, bend, race, W = (G[k] for k in ("to16", "gauge", "gauged_slab", "plain_slab", "attention", "bend", "race", "W"))
L, D = G["L"], G["D"]

def embed_rows(ids):                                           # M["embed"][ids], read from the stored member in place
    info = zipfile.ZipFile(NPZ).getinfo("embed.npy")
    with open(NPZ, "rb") as f:
        f.seek(info.header_offset); h = f.read(30)
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f); shape, fo, dt = np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f); off = f.tell()
    assert info.compress_type == 0 and not fo
    E = np.memmap(NPZ, dtype=dt, mode="r", offset=off, shape=shape); return np.asarray(E[ids]).copy()

meta = json.load(open(f"{HERE}/qwen06b/light/meta.json")); t0 = time.time()
HW = MM["head.w"]; HE = MM["head.e"]; NW = HW.shape[1]
read = {}; tops = {}; extra = {}
for t in T10:
    REC["t"] = t; ids = meta["texts"][t]; R = embed_rows(ids)
    for l in range(L):
        QKV = gauged_slab(R, f"qkv.{l}")
        if t in T3: extra[f"{t}|qkv_out.{l}"] = QKV
        o = attention(QKV, l); REC["dig"][f"arrival|{l}|{t}"] = dg(o)
        R = R + plain_slab(o, f"o.{l}"); REC["dig"][f"stream|{l}.0|{t}"] = dg(R)
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}"); REC["dig"][f"stream|{l}.1|{t}"] = dg(R)
    x16, ex = to16(R); LS = gauge(x16, ex, D)
    REC["dig"][f"top light|{t}"] = dg(np.concatenate([x16, np.asarray(ex)[:, None], np.asarray(LS)[:, None]], 1))
    tops[t] = (np.asarray(x16, np.int64), np.asarray(ex, np.int64), np.asarray(LS, np.int64))
    S = np.empty((len(R), NW), np.int64)                       # the read at every position, qwen_base2's own gauged_slab
    for c in range(0, NW, 16384):
        W.clear(); dict.__setitem__(W, "head", (np.asfortranarray(HW[:, c:c + 16384].astype(np.int64)), HE[c:c + 16384]))
        S[:, c:c + 16384] = gauged_slab(R, "head")
    W.clear()
    lab = np.arange(NW)
    read[t] = dict(sha=[hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).hexdigest() for s in S],
                   top5=[lab[np.lexsort((lab, -s))[:5]].tolist() for s in S], win=[race(s) for s in S])
    del S
    print(f"text {t}: {len(ids)} positions done ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
json.dump(dict(T10=T10, T3=T3, dig=REC["dig"], read=read), open(f"{OUT}/ref10.json", "w"))
np.savez(f"{OUT}/ref10_lights.npz", **{k.replace("|", "__"): v for k, v in REC["light"].items()},
         **{k.replace("|", "__"): v for k, v in extra.items()},
         **{f"top__{t}__{n}": a for t, (x, e, s) in tops.items() for n, a in (("x16", x), ("ex", e), ("LS", s))})
print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
