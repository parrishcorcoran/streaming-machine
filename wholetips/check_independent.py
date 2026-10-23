"""INDEPENDENT CHECK of wholetips.py, with its own code (nothing imported from wholetips.py).

Rebuilds tips k = 1, tips k = 2 and the 4-bit grid from the stored numbers in GPT-2's safetensors file:
  tips: each number's single-wave digits written out one place at a time (the plain carry rule: an odd remainder
    takes digit 2 - (remainder mod 4), i.e. +1 or -1, then halve), checked to rebuild the number and to have no two
    lit places side by side; the top k lit places kept, the rest dropped.
  grid: one grid per row (a plate's row = one neuron's inputs, a word's 768, a position's 768, a bias or gain
    vector one row), step = fp16(row's largest / 7), round to nearest in float64, clipped to +-7.
Recounts the stored bits (floor, count cells charged 32 bits, grid row scales 16 bits) for wte, wpe, h.0.attn.c_attn,
h.5.ln_1.weight, and the whole machine by part. Reruns the words: the first 60 judge5 windows (1,200 positions), and
also all 300 windows and the 30 greedy texts, so the reported counts can be compared exactly.
    python3 wholetips/check_independent.py
"""
import glob, json, os, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
import torch
from safetensors.numpy import load_file
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/wholetips"
T0 = time.time()
def peak(): return f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"
def say(s): print(f"[{time.time() - T0:6.0f}s {peak()}] {s}", flush=True)

def wait_free():
    while True:
        rows = [r.split() for r in os.popen("ps -axo pid=,rss=,comm=").read().split("\n") if r.strip()]
        other = [r for r in rows if "ython" in " ".join(r[2:]) and int(r[0]) != os.getpid() and int(r[1]) > 1_500_000]
        if not other: return
        say(f"waiting on pid {other[0][0]}"); time.sleep(60)
wait_free()

# ---------------- stored numbers ----------------
S = load_file(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors"))[0])
S = {k: v for k, v in S.items() if not (k.endswith(".attn.bias") and v.ndim == 4)}       # drop the causal mask buffer
MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval()
P = dict(MODEL.named_parameters())
NAME = {k: "transformer." + k for k in S}
assert set(NAME.values()) == set(P), (set(NAME.values()) ^ set(P))
assert MODEL.lm_head.weight.data_ptr() == P["transformer.wte.weight"].data_ptr()
for k in S: assert np.array_equal(S[k], P[NAME[k]].detach().numpy()), k
NV = sum(v.size for v in S.values()); say(f"stored numbers {NV:,}")
assert NV == 124_439_808
def part(k):
    if k == "wte.weight": return "word table"
    if k == "wpe.weight": return "position rows"
    if S[k].ndim == 2: return "plates"
    return "small pieces"
PARTS = ["word table", "position rows", "plates", "small pieces"]

# ---------------- tips ----------------
ND = 27
def digits(m):
    """m int64 >= 0 (< 2^24) -> (N, ND) int8 single-wave digits, place 0 first"""
    D = np.zeros((m.size, ND), np.int8); r = m.copy()
    for i in range(ND):
        d = np.where(r & 1, 2 - (r & 3), 0); D[:, i] = d; r = (r - d) >> 1
    assert (r == 0).all()
    return D
POW = (2.0 ** np.arange(ND))
def tips(A, k, counts):
    x = A.ravel().astype(np.float64); out = np.empty(x.size, np.float32)
    CH = 1 << 19
    for i0 in range(0, x.size, CH):
        xx = x[i0:i0 + CH]; neg = xx < 0
        mant, ex = np.frexp(np.abs(xx)); m = (mant * 2 ** 24).astype(np.int64)
        assert np.array_equal(m.astype(np.float64), mant * 2 ** 24)
        D = digits(m)
        assert np.array_equal((D * POW).sum(1), m.astype(np.float64))
        nz = D != 0
        assert not (nz[:, 1:] & nz[:, :-1]).any()                  # no two lit side by side
        fromtop = np.cumsum(nz[:, ::-1], axis=1)[:, ::-1]           # lit places at or above each place
        keep = nz & (fromtop <= k)
        val = np.ldexp((D * keep * POW).sum(1), ex - 24); val = np.where(neg, -val, val)
        o = val.astype(np.float32); assert np.array_equal(o.astype(np.float64), val); out[i0:i0 + CH] = o
        # counts: round 1 = (absolute place of top lit, sign) or dark; round r = (gap, digit sign) or stop
        nlit = nz.sum(1)
        idx = [np.full(xx.size, -1) for _ in range(k)]
        for j in range(k):                                          # place of the j-th lit from the top
            hit = nz & (fromtop == j + 1)
            has = hit.any(1); idx[j] = np.where(has, ND - 1 - np.argmax(hit[:, ::-1], 1), -1)
        lit0 = idx[0] >= 0
        place = idx[0] + ex - 24
        s1 = np.where(lit0, place * 2 + neg, 10 ** 6)               # dark = its own symbol
        u, c = np.unique(s1, return_counts=True); acc(counts, "r1", u, c)
        alive = lit0
        for j in range(1, k):
            go = alive & (idx[j] >= 0)
            sgn = np.where(go, D[np.arange(xx.size), np.maximum(idx[j], 0)], 0)
            gap = idx[j - 1] - idx[j]
            assert (gap[go] >= 2).all()
            s = np.where(go, gap * 2 + (sgn < 0), -1)               # -1 = stop
            u, c = np.unique(s[alive], return_counts=True); acc(counts, f"r{j + 1}", u, c)
            alive = go
    return out.reshape(A.shape)
def acc(counts, key, u, c):
    d = counts.setdefault(key, {})
    for a, b in zip(u.tolist(), c.tolist()): d[a] = d.get(a, 0) + b
def floor(d):
    c = np.array(list(d.values()), np.float64); N = c.sum()
    return float(-(c * np.log2(c / N)).sum()), len(c)

# ---------------- grid ----------------
def rows(k, A):
    if A.ndim == 1: return A[None, :]
    return A.T if part(k) == "plates" else A
def unrows(k, R, shape):
    if len(shape) == 1: return R[0]
    return R.T if part(k) == "plates" else R
def grid(k, A, b):
    RA = rows(k, A); L = 2 ** (b - 1) - 1; V = np.empty(RA.shape, np.float32); cnt = {}
    for r0 in range(0, RA.shape[0], 2048):
        R = RA[r0:r0 + 2048].astype(np.float64)
        sc = (np.abs(R).max(1) / L).astype(np.float16).astype(np.float64)
        q = np.clip(np.rint(R / sc[:, None]), -L, L)
        v = (q * sc[:, None]).astype(np.float32); assert np.array_equal(v.astype(np.float64), q * sc[:, None]); V[r0:r0 + 2048] = v
        u, c = np.unique(q, return_counts=True); acc({"g": cnt}, "g", u, c)
    return np.ascontiguousarray(unrows(k, V, A.shape)), cnt, RA.shape[0]

def build(ver):
    CUT, BITS, DETAIL = {}, {}, {}
    for k, A in S.items():
        if ver[0] == "tips":
            cnt = {}; CUT[k] = tips(A, ver[1], cnt)
            per = {r: floor(cnt[r]) for r in cnt}
            BITS[k] = sum(e + 32 * n for e, n in per.values()); DETAIL[k] = {r: per[r][0] / A.size for r in per}
            DETAIL[k]["cells_bits"] = sum(32 * n for _, n in per.values()) / A.size
        else:
            CUT[k], cnt, nr = grid(k, A, ver[1]); e, n = floor(cnt)
            BITS[k] = e + 32 * n + 16 * nr; DETAIL[k] = dict(levels=e / A.size, cells_and_scales=(32 * n + 16 * nr) / A.size)
    return CUT, BITS, DETAIL

# ---------------- words ----------------
WIN = json.load(open(f"{HERE}/judge5_texts.json")); REFS = json.load(open(f"{HERE}/refs_many.json"))
IDS = torch.tensor([w["ids"] for w in WIN])
def install(CUT, only=None):
    for k in S: P[NAME[k]].data.copy_(torch.from_numpy(np.ascontiguousarray(CUT[k] if (only is None or only(k)) else S[k])))
def words():
    T5 = np.concatenate([torch.topk(MODEL(IDS[i:i + 20]).logits[:, 3:23], 5, -1).indices.numpy() for i in range(0, 300, 20)])
    G = []
    for r in REFS:
        seq = list(r["ids"])
        for _ in range(16): seq.append(int(MODEL(torch.tensor([seq])).logits[0, -1].argmax()))
        G.append(seq[len(r["ids"]):])
    return T5, G
def score(T5, G, REF, n):
    a = T5[:n].reshape(-1, 5); f = REF[0][:n].reshape(-1, 5)
    out = dict(A=int((a == f[:, :1]).any(1).sum()), B=int((f == a[:, :1]).any(1).sum()), C=int((a[:, 0] == f[:, 0]).sum()), of=len(a))
    if n == 300:
        out["greedy"] = sum(int(x == y) for g, h in zip(G, REF[1]) for x, y in zip(g, h))
        out["texts"] = sum(g == h for g, h in zip(G, REF[1])); out["rec"] = sum(int(g[0] == h[0]) for g, h in zip(G, REF[1]))
    return out
def fmt(s):
    t = f"A {s['A']}/{s['of']} ({100 * s['A'] / s['of']:.2f}%)  B {s['B']} ({100 * s['B'] / s['of']:.2f}%)  C {s['C']} ({100 * s['C'] / s['of']:.2f}%)"
    return t + (f"  greedy {s['greedy']}/480  texts {s['texts']}/30  rec {s['rec']}/30" if "greedy" in s else "")

REF = words()
want = [r["ref"] for r in REFS]
say(f"uncut vs refs_many: {sum(int(x == y) for g, h in zip(REF[1], want) for x, y in zip(g, h))}/480; record inputs are the 5 prompt ids: "
    f"{all(len(r['ids']) == 5 for r in REFS)}")
b2 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); u = REF[0].reshape(-1, 5)[:, 0]
say(f"uncut word in judge5_ref top 5: {int((b2 == u[:, None]).any(1).sum())}/6000; same top {int((b2[:, 0] == u).sum())}/6000")

RESULTS = {}
CHECK_TENSORS = ["wte.weight", "wpe.weight", "h.0.attn.c_attn.weight", "h.5.ln_1.weight"]
for ver in [("tips", 1), ("tips", 2), ("grid", 4)]:
    wait_free(); tag = f"{ver[0]}{ver[1]}"
    CUT, BITS, DETAIL = build(ver)
    tot = sum(BITS.values())
    byp = {p: sum(BITS[k] for k in S if part(k) == p) / sum(S[k].size for k in S if part(k) == p) for p in PARTS}
    say(f"== {tag}: {tot / NV:.4f} bits a number, {tot / 8e6:.3f} MB; by part " + ", ".join(f"{p} {byp[p]:.3f}" for p in PARTS))
    for k in CHECK_TENSORS:
        say(f"   {k}: {BITS[k] / S[k].size:.4f} bits a number  {json.dumps({a: round(b, 4) for a, b in DETAIL[k].items()})}")
    if ver[0] == "tips":
        rr = {r: sum(DETAIL[k][r] * S[k].size for k in S if r in DETAIL[k]) / NV for r in ["r1", "r2"] if r in DETAIL["wte.weight"]}
        say(f"   rounds over the whole machine: {rr}; count cells {sum(DETAIL[k]['cells_bits'] * S[k].size for k in S) / NV:.4f}")
    else:
        say(f"   levels {sum(DETAIL[k]['levels'] * S[k].size for k in S) / NV:.4f}; cells and scales {sum(DETAIL[k]['cells_and_scales'] * S[k].size for k in S) / NV:.4f}")
    def ss(a, b=None):
        a = a.ravel(); b = None if b is None else b.ravel(); t = 0.0
        for i in range(0, a.size, 1 << 22):
            x = a[i:i + (1 << 22)].astype(np.float64); d = x if b is None else x - b[i:i + (1 << 22)].astype(np.float64)
            t += float((d * d).sum())
        return t
    err = {p: (sum(ss(S[k], CUT[k]) for k in S if part(k) == p) / sum(ss(S[k]) for k in S if part(k) == p)) ** 0.5 for p in PARTS}
    say("   error share " + ", ".join(f"{p} {err[p]:.4f}" for p in PARTS))
    install(CUT); T5, G = words()
    s60, s300 = score(T5, G, REF, 60), score(T5, G, REF, 300)
    say(f"   WHOLE first 60 windows: {fmt(s60)}"); say(f"   WHOLE all 300: {fmt(s300)}")
    R = dict(bits=tot / NV, MB=tot / 8e6, by_part_bits=byp, w60=s60, w300=s300, parts={})
    for p in PARTS:
        install(CUT, lambda k: part(k) == p); T5, G = words(); R["parts"][p] = score(T5, G, REF, 300)
        say(f"   only {p}: {fmt(R['parts'][p])}")
    if ver == ("tips", 1):                                  # the gains against the biases, at k = 1
        for nm, fn in [("gains (ln .weight) only", lambda k: part(k) == "small pieces" and "ln" in k and k.endswith("weight")),
                       ("ln offsets (ln .bias) only", lambda k: part(k) == "small pieces" and "ln" in k and k.endswith("bias")),
                       ("plate biases only", lambda k: part(k) == "small pieces" and "ln" not in k)]:
            install(CUT, fn); T5, G = words(); R["parts"][nm] = score(T5, G, REF, 300); say(f"   {nm}: {fmt(R['parts'][nm])}")
    install(S); del CUT
    RESULTS[tag] = R
json.dump(RESULTS, open(f"{OUT}/check_independent.json", "w"), indent=1)
say("done")
