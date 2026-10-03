"""The all-base-2 GPT-2, checked stage by stage (SPEC.md). Every value is a whole number; products are whole-number
products (done here exactly, every sum below 2^53; on the GPU they are bit planes). Switches put a stage back to float
so each can be checked on its own:
    gauge=int|float   edge=int|float   arrival=int|float
    python3 base2.py [gauge=float] [edge=float] [arrival=float] [words=8]
"""
import sys, math, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

A = dict(a.split("=") for a in sys.argv[1:])
GAUGE, EDGE, ARRIVAL, WORDS = A.get("gauge", "int"), A.get("edge", "int"), A.get("arrival", "int"), int(A.get("words", 8))
M = np.load("base2_gpt2.npz")
ES, SC = 16, 10
PHI, ARR, LOG2, EXP2 = M["phi"], M["arrival"], M["log2"], M["exp2"]
LOG768, EPS40 = int(M["log2_768"]), int(M["eps40"])
W = {k: (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"]) for k in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]}

def rshift(X, s):                                    # X x 2^-s, rounded to nearest: add half, shift (s may be 0 or negative)
    s = np.asarray(s, dtype=np.int64)
    up = np.left_shift(X, np.maximum(-s, 0))
    sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

def to16(X, e):                                      # each position on 16 bits: its own power of two
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)])   # OR of the magnitudes: the top digit any of them reaches
    sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), e + sh

def product(x16, w16):                               # whole-number products and sums (exact: below 2^53)
    p = x16.astype(np.float64) @ w16
    assert np.abs(p).max() < 2 ** 52
    return p.astype(np.int64)

def race(vals):                                      # who arrives first: top digits first; at each digit, if anyone still in
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)   # has a 1 there, everyone with a 0 drops out
    alive = np.ones(len(u), dtype=bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])

def log2fix(V):                                      # log2 of a positive whole number: top 1's position + a table read
    out = []
    for v in V:
        v = int(v); t = v.bit_length() - 1
        m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
        out.append(t * 65536 + int(LOG2[m]))
    return np.array(out, dtype=np.int64)

def power(X, F):                                     # X x 2^(F / 65536): a table read of the fine part, then a shift
    F = np.asarray(F, dtype=np.int64)
    I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    return rshift(X * EXP2[f], 16 - I)

def gauge_log(R):                                    # log2 of the position's size, with the size never computed
    x16, ex = to16(R, -ES)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    V = 768 * S2 - S1 * S1
    V = V + np.array([EPS40 >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else EPS40 << -(40 + 2 * int(e)) for e in ex])
    return x16, ex, np.right_shift(log2fix(V), 1) - LOG768 + ex * 65536    # halved: the root; minus log2 768

def gauged_bank(R, name):                            # the bank after a folded layer norm: product, then the gauge, then the bias
    w, we, b = W[name]
    x16, ex, LS = gauge_log(R)
    pile = product(x16, w)
    if GAUGE == "float":
        Rf = R / 2.0 ** ES; sig = np.sqrt(((Rf - Rf.mean(1, keepdims=True)) ** 2).mean(1) + 1e-5)
        val = pile * 2.0 ** (we[None, :] + ex[:, None]) / sig[:, None]
        return np.round(val * 2 ** ES).astype(np.int64) + b
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    return power(pile, F) + b

def plain_bank(X, name):                             # a bank with no layer norm before it: product, shift, bias
    w, we, b = W[name]
    x16, ex = to16(X, -ES)
    return rshift(product(x16, w), -(we[None, :] + ex[:, None] + ES)) + b

def edge(H):                                         # the bend: where the value arrives picks the share
    if EDGE == "float":
        h = H / 2.0 ** ES; return np.round(0.5 * h * (1 + np.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3))) * 2 ** ES).astype(np.int64)
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    return rshift(H * share, 16)

def attention(QKV):                                  # the arrival, head by head
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64)
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        if ARRIVAL == "float":
            s = (q / 2.0**ES) @ (k / 2.0**ES).T / 8; s = np.where(np.tril(np.ones((P, P))) > 0, s, -1e30)
            a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            out[:, h*64:(h+1)*64] = np.round(a @ (v / 2.0**ES) * 2**ES).astype(np.int64); continue
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))   # the 1/8 is a shift of 3
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row                       # the clock starts at whoever arrives first
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]
            out[p, h*64:(h+1)*64] = power(num, -log2fix([a.sum()])[0])            # divided by the total: its log negated
    return out

def forward(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]
    x16, ex, LS = gauge_log(R[-1:])
    pile = product(x16, w)
    if GAUGE == "float":
        Rf = R[-1:] / 2.0 ** ES; sig = np.sqrt(((Rf - Rf.mean()) ** 2).mean() + 1e-5)
        lg = pile * 2.0 ** (we[None, :] + ex[:, None]) / sig + b / 2.0 ** ES
        return race(np.round(lg[0] * 2 ** ES).astype(np.int64))
    return race(power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)   # the word: whoever arrives first

tok = GPT2Tokenizer.from_pretrained("gpt2"); ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
print(f"all base 2: gauge {GAUGE}, edge {EDGE}, arrival {ARRIVAL}")
same_all = True
for prompt in ["The capital of France is", "Once upon a time", "In 1492, Columbus"]:
    ids = tok(prompt).input_ids
    want = ref.generate(torch.tensor([ids]), max_new_tokens=WORDS, do_sample=False, pad_token_id=50256)[0, len(ids):].tolist()
    got = []
    for _ in range(WORDS): got.append(forward(ids + got))
    same = got == want; same_all &= same
    print(f"  {prompt!r:28s} {'IDENTICAL' if same else 'DIFFERENT'}  {tok.decode(got)!r}" + ("" if same else f"   model: {tok.decode(want)!r}"))
print("all three identical" if same_all else "not all identical")
