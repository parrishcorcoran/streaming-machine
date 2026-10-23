"""FOUNDATION_ROWS/CHECK.PY: an independent recount of count.py's item 1, by other means. 2026-10-07.
  - the tables read through numpy's own float types (fp32 / fp16 as they are; bf16 as the top half of an fp32), not
    through the sign / exponent / mantissa fields count.py takes apart
  - each lane's grid from frexp: the lowest lit place of a value is its exponent minus 53 plus the trailing zeros of
    its 53-place mantissa; whole numbers by ldexp (exact: every lane fits under 2^53, asserted)
  - cells shared by ALL words from the lane's AND and OR over its words (a place is shared exactly where AND = OR);
    the sign from np.signbit (so -0 keeps its sign, as the stored bit does)
  - 99.9% and 99% from the bit planes of the whole numbers' bytes (np.unpackbits), not from shifts
  - word = foundation + remainder for the exact foundation (sign-and-size): the shared 1s subtracted, checked to give
    back every value, and the remainder dark at every shared place
Compares every total with count_<table>.json, and the per-lane counts with counts_<table>.npz.
    python3 foundation_rows/check.py table=gpt2-machine | gpt2-fp32 | qwen | pythia-70m | pythia-160m
"""
import os, sys, json, struct, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = os.path.join(HERE, "foundation_rows"); HF = "/Users/abundancemachine/.cache/huggingface/hub"
TABLE = dict(a.split("=", 1) for a in sys.argv[1:])["table"]; NL = 8
LOGF = open(os.path.join(OUT, f"check_{TABLE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

def tensor(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); v = h[key]
        f.seek(8 + n + v["data_offsets"][0]); raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
    if v["dtype"] == "F32": return np.frombuffer(raw, np.float32).reshape(v["shape"])
    if v["dtype"] == "F16": return np.frombuffer(raw, np.float16).reshape(v["shape"])
    if v["dtype"] == "BF16": return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
def tables():
    if TABLE == "gpt2-machine":
        X = np.load(os.path.join(HERE, "base2_gpt2.npz"))["wte"]; return {"finished": ("int", X)}
    if TABLE == "gpt2-fp32": return {"finished": ("float", tensor(f"{HF}/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors", "wte.weight"))}
    if TABLE == "qwen": return {"finished": ("float", tensor(f"{HF}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors", "model.embed_tokens.weight"))}
    sz = TABLE.split("-")[1]; b = f"{HERE}/foundation0/models/pythia-{sz}"
    return {"finished": ("float", tensor(f"{b}-step143000/model.safetensors", "gpt_neox.embed_in.weight")),
            "blank": ("float", tensor(f"{b}-step0/model.safetensors", "gpt_neox.embed_in.weight"))}

def lane_block(kind, T, l0, l1):
    """sign (signbit), whole numbers on each lane's own finest grid"""
    if kind == "int":
        x = np.asarray(T[:, l0:l1]).astype(np.int64); orr = np.bitwise_or.reduce(np.abs(x), axis=0)
        k = np.array([(int(o) & -int(o)).bit_length() - 1 if o else 0 for o in orr]); sgn = (x < 0)
        return sgn, np.where(sgn, -((-x) >> k), x >> k)
    v = np.asarray(T[:, l0:l1]).astype(np.float64); sgn = np.signbit(v); av = np.abs(v)
    m, e = np.frexp(av); M = np.ldexp(m, 53).astype(np.int64)                                  # 53-place whole mantissa
    low = np.where(M > 0, e - 53 + np.log2(np.where(M > 0, M & -M, 1)).astype(np.int64), 10 ** 6); g = low.min(0)
    a = np.ldexp(av, -g[None, :]); assert (a < 2.0 ** 53).all() and (a == np.round(a)).all()
    a = a.astype(np.int64); return sgn, np.where(sgn, -a, a)

def recount(kind, T):
    V, d = T.shape; tot = {w: {"all": 0, "999": 0, "99": 0} for w in ("ss", "tc")}; per = {"ss": [], "tc": []}; recon_ok = True
    for l0 in range(0, d, NL):
        sgn, x = lane_block(kind, T, l0, min(d, l0 + NL)); a = np.abs(x); n = x.shape[1]
        P = np.array([int(v).bit_length() for v in a.max(0)])
        hi, lo = x.max(0), x.min(0); Q = np.array([max(int(h).bit_length() if h > 0 else 0, int(-l_ - 1).bit_length() if l_ < 0 else 0) + 1 for h, l_ in zip(hi, lo)])
        u = x & ((np.int64(1) << Q.astype(np.int64)) - 1)
        for w, Y, Wd in (("ss", a, P), ("tc", u, Q)):
            AND, OR = np.bitwise_and.reduce(Y, axis=0), np.bitwise_or.reduce(Y, axis=0)
            planes = np.unpackbits(np.ascontiguousarray(Y).view(np.uint8).reshape(V, n, 8), axis=2, bitorder="little").sum(0, dtype=np.int64)   # (n, 64) ones per place
            for j in range(n):
                cnt = planes[j, :Wd[j]]
                if w == "ss": cnt = np.concatenate([[int(sgn[:, j].sum())], cnt])
                assert cnt.sum() >= 0
                shared = [int(((int(AND[j]) >> p) & 1) == ((int(OR[j]) >> p) & 1)) for p in range(Wd[j])]
                if w == "ss": shared = [int(sgn[:, j].all() or not sgn[:, j].any())] + shared
                m = np.minimum(cnt, V - cnt)
                assert np.array_equal(np.array(shared, bool), m == 0)                                    # the two ways agree
                tot[w]["all"] += int((m == 0).sum()); tot[w]["999"] += int((1000 * m <= V).sum()); tot[w]["99"] += int((100 * m <= V).sum())
                per[w].append(cnt)
                if w == "ss":                                                                            # word = foundation + remainder
                    ones = 0
                    for p in range(Wd[j]):
                        if shared[1 + p] and (int(AND[j]) >> p) & 1: ones |= 1 << p
                    R = a[:, j] - ones; recon_ok &= bool((R >= 0).all() and ((R & ones) == 0).all() and np.array_equal(R + ones, a[:, j]))
    return tot, per, recon_ok

J = json.load(open(os.path.join(OUT, f"count_{TABLE}.json"))); Z = np.load(os.path.join(OUT, f"counts_{TABLE}.npz"))
names = {str(Z[k]): k.split("_")[0] for k in Z.files if k.endswith("_name")}
log(f"CHECK {TABLE}: independent recount")
allok = True
for lab, (kind, T) in tables().items():
    tot, per, rok = recount(kind, T)
    vname = [n for n in J["versions"] if n.startswith(lab)][0]; S = J["versions"][vname]["summary"]; vi = names[vname]
    for w in ("ss", "tc"):
        mine = {lev: sum(a + b for a, b in S[w][lev].values()) for lev in ("all", "999", "99")}
        cn = Z[f"{vi}_{w}_n1"]; wd = Z[f"{vi}_{w}_width"]; same_lanes = np.array_equal(np.concatenate(per[w]), cn) and np.array_equal(np.array([len(c) for c in per[w]]), wd)
        ok = mine == tot[w] and same_lanes; allok &= ok
        log(f"  {vname:32s} {w}: check all/99.9/99 = {tot[w]['all']}/{tot[w]['999']}/{tot[w]['99']}  count.py = {mine['all']}/{mine['999']}/{mine['99']}  per-lane counts identical {same_lanes}  -> {'AGREE' if ok else 'DIFFER'}")
    log(f"  {vname:32s} word = foundation + remainder (exact foundation, sign-and-size), every value: {rok}"); allok &= rok
log(f"ALL AGREE: {allok}")
