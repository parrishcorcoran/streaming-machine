"""qwen_fringes2/check.py: A SECOND CODE PATH for qwen_fringes2.py (same session: NOT an independent rebuild).

A. THE COUNTS RECOUNTED ANOTHER WAY, on capture's record of the light (qwen06b/light, qwen_base2.py's own run; the
   machine's light equals it at every slab): at a sample of slabs and positions, each line's weights sorted across the
   whole slab (and across each plate) by np.sort, the signed digits of every gap counted by the textbook digit-by-digit
   rule (a remainder of 1 or 3 mod 4 gives a digit +1 or -1; no 3x trick), the light's signed digits the same way; the
   pours as the 1s of every exact product |x w| counted by unpacking bytes (no bitwise_count). Compared with
   qwen_fringes2/machine_counts.npz: the slab's walk, the per-plate walks, the pours, the pairs.
B. THE RULE recomputed: per line (weights only) the slab's gap digits against its plates' summed, at every line of every
   sampled slab; and per slab from the counts file, the sums the rule compares.
C. THE ARRIVAL USES ITS OWN PRODUCTS, AND NO FLOAT IS ANYWHERE IN THE MACHINE PATH (one text, number 9, 17 positions):
   (i) the place-pair weights doubled: the machine's arrival outputs must change at every layer (it uses its own sums);
   (ii) a trace of every function of qwen_fringes2.py and of qwen_base2.py's executed source: at every return, every
   local value and the returned value are inspected; any float (a Python float, a numpy float, a float array) is listed.
   forward()'s clock reading (t0) is the one known float and is not part of the machine.
D. THE TOTAL recomputed from the counts files by a separate sum: the whole machine a token and its share of plain.
"""
import sys, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen_fringes2"; CAP = f"{HERE}/qwen06b"
sys.path.insert(0, HERE)
log_f = open(f"{OUT}/check.log", "w")
def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

def naf_count(a):
    """signed digits of |a| by the digit-by-digit rule (a second way: no 3x trick)"""
    n = np.abs(np.asarray(a, np.int64)).copy(); c = np.zeros(n.shape, np.int64)
    while n.any():
        odd = (n & 1) == 1; d = np.where(odd, 2 - (n & 3), 0)          # +1 when n = 1 mod 4, -1 when n = 3 mod 4
        c += odd; n = (n - d) >> 1
    return c
def ones_count(a):
    """the 1s of |a| by unpacking its bytes (a second way)"""
    u = np.ascontiguousarray(np.abs(np.asarray(a, np.int64)).astype(np.uint64))
    return np.unpackbits(u.view(np.uint8).reshape(*u.shape, 8), axis=-1).sum(axis=(-1)).astype(np.int64)

Z = np.load(f"{OUT}/machine_counts.npz"); CNT = {k.replace("__", "|"): Z[k] for k in Z.files}
import zipfile
NPZ = f"{HERE}/qwen/qwen_base2.npz"; Mz = np.load(NPZ)
rng = np.random.default_rng(20261005); POS = np.sort(rng.choice(613, 24, replace=False))
SAMPLE = [("qkv", 3), ("qkv", 17), ("qkv", 27), ("gu", 0), ("gu", 12), ("gu", 26)]
PL = {"qkv": [(0, 2048), (2048, 3072), (3072, 4096)], "gu": [(0, 3072), (3072, 6144)]}
log(f"A. COUNTS RECOUNTED at positions {POS.tolist()}")
diffs = 0; nl_worse = {}; t = 0
for slab, l in SAMPLE:
    W = np.asarray(Mz[f"{slab}.{l}.w"]).astype(np.int64); X = np.load(f"{CAP}/light/{slab}_light.npy", mmap_mode="r")[l][POS].astype(np.int64)
    ws = np.sort(W, axis=1); gn_slab = naf_count(ws[:, 0]) + naf_count(np.diff(ws, axis=1)).sum(1)
    gn_pl = np.zeros(W.shape[0], np.int64)
    for c0, c1 in PL[slab]:
        wp = np.sort(W[:, c0:c1], axis=1); gn_pl += naf_count(wp[:, 0]) + naf_count(np.diff(wp, axis=1)).sum(1)
    nx = naf_count(X)
    walk, walkp = nx @ gn_slab, nx @ gn_pl
    pours = np.array([ones_count(X[p][:, None] * W).sum() for p in range(len(POS))])
    pairs = (X != 0).sum(1) * W.shape[1]
    got = {k: CNT[f"{slab}.{l}|{k}"][POS] for k in ("walk", "walk_plates", "pours", "pairs")}
    d = {k: int((got[k] != v).sum()) for k, v in (("walk", walk), ("walk_plates", walkp), ("pours", pours), ("pairs", pairs))}
    diffs += sum(d.values()); nl_worse[f"{slab}.{l}"] = (int((gn_slab > gn_pl).sum()), W.shape[0])
    log(f"  {slab}.{l}: positions differing: {d}; walk per pair here {walk.sum() / pairs.sum():.3f} (plates {walkp.sum() / pairs.sum():.3f})")
log(f"  A: {diffs} differing values")
log("B. THE RULE")
log(f"  lines whose slab gap digits exceed their plates' summed (weights only, recounted): {nl_worse}")
MJ = json.load(open(f"{OUT}/machine.json"))
log(f"  the machine's own record for the same slabs: { {k: MJ['lines'][k] for k in nl_worse} }")
dec = {}
for s in ("qkv", "gu"):
    for l in range(28):
        a = int(CNT[f"{s}.{l}|walk"].sum() + CNT[f"{s}.{l}|pours"].sum()); b = int(CNT[f"{s}.{l}|walk_plates"].sum() + CNT[f"{s}.{l}|pours"].sum())
        dec[f"{s}.{l}"] = a < b
log(f"  slabs where the rule picks the slab's line-up: {sum(dec.values())} of {len(dec)}")
del Mz

log("C. THE ARRIVAL AND THE FLOAT TRACE (text 9)")
import qwen_fringes2 as Q
meta = json.load(open(f"{CAP}/light/meta.json")); RF = json.load(open(f"{HERE}/qwen_fringes/ref.json"))
text = meta["texts"][9]
res = Q.forward([text], log=lambda s: None)
same = sum(res["dig"][f"arrival|{l}|0"] == RF["dig"][f"arrival|{l}|9"] for l in range(28))
log(f"  plain run of text 9: arrival outputs equal to the reference {same} of 28; checks {Q.ST['chk']}")
_tw = Q.TW; Q.TW = lambda na, nb: 2 * _tw(na, nb)
res2 = Q.forward([text], log=lambda s: None); Q.TW = _tw
ch = sum(res2["dig"][f"arrival|{l}|0"] != RF["dig"][f"arrival|{l}|9"] for l in range(28))
log(f"  (i) the place-pair weights doubled: arrival outputs changed at {ch} of 28 layers; the arrival's checks: "
    f"{ {k: v for k, v in Q.ST['chk'].items() if k.startswith('arrival')} }")
found = {}; seen_known = {}; returns = {}
def is_float(v):
    if isinstance(v, (float, np.floating, complex, np.complexfloating)): return True
    if isinstance(v, np.ndarray): return v.dtype.kind in "fc"
    return False
def tracer(frame, event, arg):
    fn = frame.f_code.co_filename
    if not (fn.endswith("qwen_fringes2.py") or fn == "<string>"): return None
    def local(frame, event, arg):
        if event == "return":
            name = frame.f_code.co_name; returns[name] = returns.get(name, 0) + 1
            for k, v in list(frame.f_locals.items()):
                if is_float(v) and name == "forward" and k == "t0": seen_known["forward:t0"] = type(v).__name__; continue   # the positive control
                if is_float(v): found.setdefault(f"{name}:{k}", type(v).__name__ + (f" {v.dtype}" if isinstance(v, np.ndarray) else ""))
            if is_float(arg): found.setdefault(f"{name}:<returned>", type(arg).__name__)
        return local
    return local
sys.settrace(tracer); res3 = Q.forward([text], log=lambda s: None); sys.settrace(None)
same3 = sum(res3["dig"][k] == res["dig"][k] for k in res["dig"])
log(f"  (ii) the traced run: digests equal to the plain run {same3} of {len(res['dig'])}; floats found in the machine path: {found if found else 'none'}")
log(f"       positive control: the known float seen by the trace: {seen_known if seen_known else 'NOT SEEN (the trace did not inspect forward)'}")
log(f"       returns inspected: {sum(returns.values()):,} in {len(returns)} functions: " + ", ".join(f"{k} {v}" for k, v in sorted(returns.items(), key=lambda kv: -kv[1])))
log("D. THE TOTAL")
C1 = {k: v for k, v in CNT.items()}
plates = sum(int(v.sum()) for k, v in C1.items() if k.split("|")[1] in ("walk", "pours", "writing") and k.split(".")[0] in ("qkv", "o", "gu", "down"))
T = np.load(f"{OUT}/top_machine.npz"); top = int(T["walk"].sum() + T["writing"].sum() + T["pours"].sum() + T["bound"].sum() + T["end"].sum())
arr = sum(int(v.sum()) for k, v in C1.items() if k.startswith("arr.") and k.split("|")[1] in ("sd_scores", "sd_values", "write_q", "write_k", "write_v"))
plain = sum(int(v.sum()) for k, v in C1.items() if k.split("|")[1] == "plain") + int(T["plain_meet"].sum() + T["plain_div"].sum()) + \
        sum(int(v.sum()) for k, v in C1.items() if k.startswith("arr.") and k.split("|")[1] in ("plain_scores", "plain_values"))
log(f"  a token: plates {plates / 613:,.0f} + read {top / 613:,.0f} + arrival {arr / 613:,.0f} = {(plates + top + arr) / 613:,.0f}; plain {plain / 613:,.0f}; "
    f"share {(plates + top + arr) / plain * 100:.2f}%")
