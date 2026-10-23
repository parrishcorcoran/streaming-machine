"""CAPTURE: THE LIGHT AT EVERY PLATE OF THE ALL-BASE-2 QWEN3-0.6B, ON THE 30 REFS TEXTS, EVERY POSITION.

WHERE THE LIGHT COMES FROM. The task (2026-10-05) said a whole base-2 Qwen machine "does not exist here" and asked for the
light to be captured from the float model. It does exist: qwen/qwen_base2.py ("QuantumQwen3", 2026-10-02), its whole
numbers in qwen/qwen_base2.npz (written by qwen/convert.py: every plate a 16-place whole number with one power of two per
neuron, each water level's gain folded into the plate after it, as base2_convert.py does for GPT-2), checked in
qwen/check30.log: 28 of 30 texts give fp32's own 16 greedy words exactly. Rule 13.2 names "the float machine in place of
base 2" as a mistake, so the light here is that machine's OWN light: every value a whole number, every plate meeting
the light as the machine writes it (to16: 16 places, one power of two per position).

HOW IT IS RUN. qwen_base2.py's own source is executed (its steps unchanged: water levels, turns, the arrival, the gated
bend, the read). Four things are replaced, none of which changes a number:
  - the loader: its W dict held every plate as int64 (4.8 GB); here each plate is read when its layer comes (int16)
  - the plate product x16 @ w: done in float64, exact (every sum below 2^47, asserted below 2^52), and recorded
  - the word rows: read in place from the stored file (no 1.24 GB load)
  - the read at the top: at EVERY position (the machine reads only the last; every step is row by row and the arrival
    reads only earlier positions, so position p's values are the same in a run of the whole text)
  Every text is run layer by layer together (the arrival reads only its own text's positions).
  Counted on the side (not used): the arrival's plain photon meetings, q16.k16 and shares x values, as base2 counts.

THE TEXTS. refs_many.json's GPT-2 ids (ids + ref, the text every GPT-2 count used) decoded to text and written in Qwen's
own words: 613 positions (GPT-2: 648).

THE HARNESS CHECK (before the capture): the same run on qwen/refs_qwen.json (Qwen's own prompt words plus fp32's 16
greedy words). The read's winner at each of the 16 places must give check30.log's words.
PREDICTION (written before the run): 28 texts give all 16 words; "My favorite food is" and "Breaking news: a major
earthquake" give the same words up to the place where check30.log's machine left fp32, and there give check30.log's
machine word (' to' and ' active').

WRITTEN (qwen06b/light/): for each slab kind (qkv: q, k, v; o; gu: gate, up; down) the light x16 as it meets the plate
(28 layers x 613 positions x lines, int32) and its power; the light at the top (x16, its power, the water level's log
LS); meta.json: the Qwen words, every slab's pile digest per text, the read at the top per position (top 5 by score
then word number, the race winner, the score row's sha256), the arrival's counts.
    python3 qwen06b/capture.py
"""
import os, sys, json, time, zipfile, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")          # Accelerate's spurious flags; sums asserted
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; OUT = f"{HERE}/qwen06b"; NPZ = f"{QD}/qwen_base2.npz"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)

def member(name):
    """an array stored in the npz, read in place (the zip stores every member flat: compress type 0)"""
    info = zipfile.ZipFile(NPZ).getinfo(name + ".npy"); assert info.compress_type == 0
    with open(NPZ, "rb") as f:
        f.seek(info.header_offset); h = f.read(30); assert h[:4] == b"PK\x03\x04"
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return np.memmap(NPZ, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")

# ---------- qwen_base2.py's own steps ----------
class LazyW(dict):                                    # W[name] -> (None, the plate's powers); the plate is read by PRODUCT
    def __getitem__(self, name):
        if name not in self.keys(): dict.__setitem__(self, name, (None, np.asarray(Mz[name + ".e"])))
        return dict.__getitem__(self, name)
Mz = np.load(NPZ)
PLATE = {}; REC = dict(on=False, light={}, ex={}, dig={}, arr_s=None, arr_v=None, rows=None)
HEAD_W = member("head.w")                             # (1024, 151936) int16, in place
def PRODUCT(x16, ex, name):
    """the plate meeting: whole-number products and sums, exact (float64, every sum below 2^47)"""
    x16 = np.asarray(x16, np.int64); assert np.abs(x16).max(initial=0) <= 1 << 15
    if name == "head":
        out = np.empty((len(x16), HEAD_W.shape[1]), np.int64); xf = x16.astype(np.float64)
        for c in range(0, HEAD_W.shape[1], 16384):
            p = xf @ np.asarray(HEAD_W[:, c:c + 16384], np.float64); assert np.abs(p).max(initial=0) < 2 ** 52
            out[:, c:c + 16384] = p.astype(np.int64)
        return out
    if name not in PLATE:
        PLATE.clear(); PLATE[name] = np.asarray(Mz[name + ".w"]).astype(np.float64)
    p = x16.astype(np.float64) @ PLATE[name]; assert np.abs(p).max(initial=0) < 2 ** 52
    pile = p.astype(np.int64)
    if REC["on"]:
        kind, l = name.split(".")
        REC["light"].setdefault(kind, {})[int(l)] = x16.astype(np.int32); REC["ex"].setdefault(kind, {})[int(l)] = np.asarray(ex, np.int64)
        REC["dig"][name] = [digest(pile[a:b]) for a, b in REC["rows"]]
    return pile
def ARR_S(q16, k16):                                  # plain photon meetings of q.k at each position (causal)
    if REC["arr_s"] is not None:
        c = ones(q16) @ ones(k16).T; REC["arr_s"].append(np.tril(c).sum(1))
def ARR_V(a, v):                                      # plain photon meetings of shares x values at one position
    if REC["arr_v"] is not None: REC["arr_v"].append(int(ones(a) @ ones(v).sum(1)))

_src = open(f"{QD}/qwen_base2.py").read(); _src = _src[:_src.index("from transformers import AutoTokenizer")]
for _a, _b in [('M = np.load("qwen_base2.npz")', 'M = Mz'),
               ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = LazyW()'),
               ("w, we = W[name]; x16, ex = to16(R); pile = x16 @ w", "w, we = W[name]; x16, ex = to16(R); pile = PRODUCT(x16, ex, name)"),
               ("w, we = W[name]; x16, ex = to16(X); return rshift(x16 @ w,", "w, we = W[name]; x16, ex = to16(X); return rshift(PRODUCT(x16, ex, name),"),
               ("q16, eq = to16(q); k16, ek = to16(k); sc = rshift(q16 @ k16.T,", "q16, eq = to16(q); k16, ek = to16(k); ARR_S(q16, k16); sc = rshift(q16 @ k16.T,"),
               ("out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],", "ARR_V(a, v[:p + 1]); out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],")]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = dict(Mz=Mz, LazyW=LazyW, PRODUCT=PRODUCT, ARR_S=ARR_S, ARR_V=ARR_V)
_argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
to16, gauge, gauged_slab, plain_slab, attention, bend, race = (B[k] for k in ("to16", "gauge", "gauged_slab", "plain_slab", "attention", "bend", "race"))
L, D, ES = B["L"], B["D"], B["ES"]
EMB = member("embed")                                 # (151936, 1024) int64 at 2^-16, in place

def top5(S):                                          # by score, ties by word number (one fixed rule, both sides)
    return np.array([np.lexsort((np.arange(len(s)), -s))[:5] for s in S])

def forward(texts, record=False, log=print):
    """the machine's own forward, every text layer by layer; the read at the top at every position"""
    t0 = time.time(); lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    rows = [(int(off[i]), int(off[i + 1])) for i in range(len(texts))]
    REC.update(on=record, light={}, ex={}, dig={}, rows=rows, arr_s=[] if record else None, arr_v=[] if record else None)
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts])
    R = np.asarray(EMB[ids]).astype(np.int64)
    for l in range(L):
        QKV = gauged_slab(R, f"qkv.{l}")
        o = np.concatenate([attention(QKV[a:b], l) for a, b in rows]); del QKV
        R = R + plain_slab(o, f"o.{l}"); del o
        GU = gauged_slab(R, f"gu.{l}")
        R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}"); del GU
        if l % 7 == 6: log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    REC["on"] = False
    x16, ex = to16(R); LS = gauge(x16, ex, D)
    T5, WIN, SHA = [], [], []
    for a, b in rows:                                 # the read at the top: the machine's own gauged_slab, every position
        S = gauged_slab(R[a:b], "head")
        T5.append(top5(S)); WIN += [race(s) for s in S]; SHA += [hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).hexdigest() for s in S]
        del S
    log(f"    the read at the top done ({time.time() - t0:.0f}s, {rss()} MB)")
    return dict(x16=x16, ex=ex, LS=np.asarray(LS, np.int64), top5=np.concatenate(T5), win=np.array(WIN), sha=SHA, rows=rows)

if __name__ == "__main__":
    from transformers import AutoTokenizer, GPT2Tokenizer
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    t0 = time.time()
    # ---- the harness check: Qwen's own prompts and fp32's 16 greedy words ----
    rq = json.load(open(f"{QD}/refs_qwen.json"))
    seqs = [r["ids"] + r["ref"] for r in rq]
    print(f"HARNESS CHECK on qwen/refs_qwen.json: {len(seqs)} texts, {sum(map(len, seqs))} positions", flush=True)
    res = forward(seqs)
    n_ident = 0; firsts = []
    for i, r in enumerate(rq):
        a, b = res["rows"][i]; k = len(r["ids"])
        got = [int(res["win"][a + k - 1 + t]) for t in range(len(r["ref"]))]
        same = got == r["ref"]
        if same: n_ident += 1
        else:
            t = next(t for t in range(len(got)) if got[t] != r["ref"][t])
            firsts.append((r["prompt"], t, qtok.decode(got[:t + 1]), qtok.decode(r["ref"][:t + 1])))
    print(f"  texts whose 16 words all equal fp32's: {n_ident} of {len(rq)} (check30.log: 28 of 30)")
    for p, t, g, w in firsts: print(f"  {p!r}: first difference at word {t}: machine {g!r}   fp32 {w!r}")
    # ---- the capture: refs_many.json's GPT-2 texts in Qwen's words ----
    g2 = GPT2Tokenizer.from_pretrained("gpt2")
    refs = json.load(open(f"{HERE}/refs_many.json"))
    texts = [qtok(g2.decode(r["ids"] + r["ref"])).input_ids for r in refs]
    print(f"CAPTURE on refs_many.json's texts in Qwen's words: {len(texts)} texts, {sum(map(len, texts))} positions", flush=True)
    res = forward(texts, record=True)
    NP = sum(map(len, texts)); mx = {}
    for kind in ("qkv", "o", "gu", "down"):
        lt = np.stack([REC["light"][kind][l] for l in range(L)]); assert lt.shape[:2] == (L, NP)
        ex = np.stack([REC["ex"][kind][l] for l in range(L)])
        mx[kind] = int(np.abs(lt).max()); np.save(f"{OUT}/light/{kind}_light.npy", lt); np.save(f"{OUT}/light/{kind}_ex.npy", ex)
        print(f"  {kind:5s} light {lt.shape} largest |x16| {mx[kind]}; values at 2^15 exactly: {int((np.abs(lt) == 1 << 15).sum())}")
        del lt
    assert np.abs(res["x16"]).max() <= 1 << 15
    np.save(f"{OUT}/light/top_light.npy", res["x16"].astype(np.int32)); np.save(f"{OUT}/light/top_ex.npy", np.asarray(res["ex"], np.int64))
    np.save(f"{OUT}/light/top_LS.npy", res["LS"])
    arr_s = np.concatenate(REC["arr_s"]) if REC["arr_s"] else np.zeros(0)
    meta = dict(texts=texts, rows=res["rows"], dig=REC["dig"], top5=res["top5"].tolist(), win=res["win"].tolist(), sha=res["sha"],
                arrival_scores_total=int(arr_s.sum()), arrival_values_total=int(sum(REC["arr_v"])), largest=mx,
                top_largest=int(np.abs(res["x16"]).max()), npos=NP)
    json.dump(meta, open(f"{OUT}/light/meta.json", "w"))
    print(f"  the arrival's plain photon meetings a token: scores {meta['arrival_scores_total'] / NP:,.0f}, shares x values {meta['arrival_values_total'] / NP:,.0f}")
    print(f"  the read at the top: race winner = top 5's first {int((res['top5'][:, 0] == res['win']).sum())}/{NP}")
    print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
