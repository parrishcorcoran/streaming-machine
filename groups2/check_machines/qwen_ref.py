"""CHECKER (independent of qwen_groups.py and qwen_fringes2.py): qwen_base2.py's OWN steps, run fresh, on the checker's
draw: refs texts [5, 12, 19, 20, 28] of qwen06b/light/meta.json and greedy texts [5, 12, 19, 20, 28] of
qwen_fringes/words_ref.json. qwen_base2.py's source (up to the tokenizer) is executed with four changes, none to a value:
the plates are read one at a time (W loads one plate when asked, never all at once: 3 GB limit), and the slab and
arrival products are routed through recorders that compute exactly qwen_base2's own int64 product (x16 @ w, q16 @ k16.T,
a @ v) and note the light, the pile and the arrival's counts; the top table's product is the same int64 product taken
in column blocks. Recorded per text: the sha256 of every pile, arrival output, stream (both halves of every layer) and
the light at the top, in the digest form of qwen_fringes2 (int64 bytes); the top scores' sha256 per position, top 5
(by score, ties by word number) and race winner. For texts 5 and 12 (refs) the light and pile arrays of every slab and
the arrival's counts, for the checker's own work counter.
    python3 qwen_ref.py -> qwen_ref.npz, qwen_ref.json
"""
import os, sys, json, time, hashlib, zipfile, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; OUT = f"{HERE}/groups2/check_machines"; NPZ = f"{QD}/qwen_base2.npz"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def memb(name):                                   # an uncompressed npz member, mapped in place
    zi = zipfile.ZipFile(NPZ).getinfo(name + ".npy"); assert zi.compress_type == 0
    with open(NPZ, "rb") as f:
        f.seek(zi.header_offset); h = f.read(30); f.seek(zi.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        v = np.lib.format.read_magic(f); shape, fo, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f); off = f.tell()
    return np.memmap(NPZ, dtype=dt, mode="r", offset=off, shape=shape, order="F" if fo else "C")
class OneW(dict):                                 # W[name] -> (the plate as qwen_base2 makes it, int64; its powers); not kept
    def __getitem__(self, name):
        if name == "head": return (memb("head.w"), np.asarray(memb("head.e")))
        return (np.asarray(memb(name + ".w")).astype(np.int64), np.asarray(memb(name + ".e")))
REC = dict(on=None, text=None, keep=False, arrays={}, dig={}, arr={})
def PROD(x16, w, name):
    if name == "head":
        out = np.empty((len(x16), w.shape[1]), np.int64)
        for j in range(0, w.shape[1], 16384): out[:, j:j + 16384] = x16 @ np.asarray(w[:, j:j + 16384]).astype(np.int64)
        return out
    pile = x16 @ w; t = REC["text"]
    REC["dig"][f"pile|{name}|{t}"] = digest(pile)
    if REC["keep"]: REC["arrays"][f"{t}|light|{name}"] = np.asarray(x16, np.int64); REC["arrays"][f"{t}|pile|{name}"] = pile
    return pile
def addc(k, v): REC["arr"][k] = REC["arr"].get(k, 0) + int(v)
def QKH(q16, k16, h, l):
    if REC["keep"]:
        t = REC["text"]; oq, ok_, nq, nk = ones(q16), ones(k16), naf(q16), naf(k16)
        addc(f"{t}|arr.{l}|plain_scores", np.tril(oq @ ok_.T).sum()); addc(f"{t}|arr.{l}|sd_scores", np.tril(nq @ nk.T).sum())
        addc(f"{t}|arr.{l}|write_q", oq.sum())
        if h % 2 == 0: addc(f"{t}|arr.{l}|write_k", ok_.sum())
    return q16 @ k16.T
def AVH(a, v, p, h, l):
    if REC["keep"]:
        t = REC["text"]; addc(f"{t}|arr.{l}|plain_values", ones(a) @ ones(v[:p + 1]).sum(1)); addc(f"{t}|arr.{l}|sd_values", naf(a) @ naf(v[:p + 1]).sum(1))
        if h % 2 == 0: addc(f"{t}|arr.{l}|write_v", ones(v[p]).sum())
    return a[None, :] @ v[:p + 1]
src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a_, b_ in [('M = np.load("qwen_base2.npz")', f'M = np.load("{NPZ}")'),
               ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = OneW()'),
               ("w, we = W[name]; x16, ex = to16(R); pile = x16 @ w", "w, we = W[name]; x16, ex = to16(R); pile = PROD(x16, w, name)"),
               ("w, we = W[name]; x16, ex = to16(X); return rshift(x16 @ w,", "w, we = W[name]; x16, ex = to16(X); return rshift(PROD(x16, w, name),"),
               ("sc = rshift(q16 @ k16.T,", "sc = rshift(QKH(q16, k16, h, l),"),
               ("out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],", "out[p, h * HD:(h + 1) * HD] = power(AVH(a, v, p, h, l),")]:
    assert src.count(a_) == 1, a_; src = src.replace(a_, b_)
B = dict(OneW=OneW, PROD=PROD, QKH=QKH, AVH=AVH); argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = argv
assert (B["GAUGE"], B["TURN"], B["BEND"], B["ARRIVAL"]) == ("int",) * 4
L, D = B["L"], B["D"]; EMB = memb("embed")
def top5(S): lab = np.arange(S.shape[1]); return np.array([lab[np.lexsort((lab, -s))[:5]] for s in S])

def run(ids, t, keep):
    REC["text"], REC["keep"] = t, keep; R = np.asarray(EMB[np.asarray(ids)]).astype(np.int64)
    for l in range(L):
        o = B["attention"](B["gauged_slab"](R, f"qkv.{l}"), l); REC["dig"][f"arrival|{l}|{t}"] = digest(o)
        R = R + B["plain_slab"](o, f"o.{l}"); REC["dig"][f"stream|{l}.0|{t}"] = digest(R)
        GU = B["gauged_slab"](R, f"gu.{l}"); R = R + B["plain_slab"](B["bend"](GU[:, :3072], GU[:, 3072:]), f"down.{l}")
        REC["dig"][f"stream|{l}.1|{t}"] = digest(R)
    x16, ex = B["to16"](R); LS = B["gauge"](x16, ex, D)
    REC["dig"][f"top light|{t}"] = digest(np.concatenate([x16, np.asarray(ex)[:, None], np.asarray(LS)[:, None]], 1))
    S = B["gauged_slab"](R, "head")
    return dict(sha=[digest(s) for s in S], top5=top5(S), win=[B["race"](s) for s in S],
                x16=x16, ex=np.asarray(ex, np.int64), LS=np.asarray(LS, np.int64))

if __name__ == "__main__":
    t0 = time.time(); SEL = [5, 12, 19, 20, 28]
    MT = json.load(open(f"{HERE}/qwen06b/light/meta.json")); WR = json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))
    out = dict(sel=SEL, refs={}, words={}); arrays = {}
    for t in SEL:
        r = run(MT["texts"][t], f"r{t}", keep=t in (5, 12))
        out["refs"][t] = dict(sha=r["sha"], top5=r["top5"].tolist(), win=r["win"])
        for k in ("x16", "ex", "LS"): arrays[f"r{t}|top|{k}"] = r[k]
        print(f"refs text {t}: {len(MT['texts'][t])} positions ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    for t in SEL:
        r = run(WR["texts"][t], f"w{t}", keep=False)
        out["words"][t] = dict(sha=r["sha"], top5=r["top5"].tolist(), win=r["win"])
        print(f"greedy text {t}: {len(WR['texts'][t])} positions ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    out["dig"] = REC["dig"]; out["arr"] = REC["arr"]
    json.dump(out, open(f"{OUT}/qwen_ref.json", "w"))
    arrays.update({k: v for k, v in REC["arrays"].items()})
    np.savez(f"{OUT}/qwen_ref.npz", **{k.replace("|", "__"): v for k, v in arrays.items()})
    print(f"saved ({time.time() - t0:.0f}s, max memory {rss()} MB)")
