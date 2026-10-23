"""Base-2 GPT-2 with every slab as a TURN on a whole-number circle (exact turns mod two primes, combined): the plate is
frozen on the circle once; the new piece's chord goes through. MLP: turn 1 (axis 1: each neuron's fc column in its own
segment, the chord in at slot 0, each sum lands at its segment start), the bend in place there, turn 2 (axis 2: the
fired values are impulses at the segment starts, mproj laid so the next chord comes out contiguous). The saved base
(earlier pieces) runs as before; only the new piece goes through the turns. Piles must be identical.
    python3 turns_base2.py [words=4]
"""
import sys, json, time, numpy as np
WORDS = int(sys.argv[1].split("=")[1]) if len(sys.argv) > 1 else 4
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2"); refs = json.load(open("refs_many.json"))
PR = [(998244353, 3), (985661441, 3)]
_rev = {}
def ntt(a, p, g, inv=False):
    n = a.shape[-1]
    if n not in _rev:
        j, r = 0, np.zeros(n, np.int64)
        for i in range(1, n):
            bit = n >> 1
            while j & bit: j ^= bit; bit >>= 1
            j |= bit; r[i] = j
        _rev[n] = r
    a = a[..., _rev[n]] % p; length = 2
    while length <= n:
        w = pow(g, (p - 1) // length, p); w = pow(w, p - 2, p) if inv else w
        ws = np.ones(length // 2, np.uint64)
        for k in range(1, length // 2): ws[k] = (int(ws[k - 1]) * w) % p
        a = a.reshape(a.shape[:-1] + (n // length, length))
        u = a[..., :length // 2]; v = (a[..., length // 2:] * ws) % p
        a = np.concatenate([(u + v) % p, (u + p - v) % p], -1).reshape(a.shape[:-2] + (n,)); length <<= 1
    return (a * pow(n, p - 2, p)) % p if inv else a
def pow2(n): return 1 << (n - 1).bit_length()
def signed(res):
    (p1, _), (p2, _) = PR; a1, a2 = res[0].astype(object), res[1].astype(object); M2 = p1 * p2
    v = a1 + p1 * (((a2 - a1) * pow(p1, -1, p2)) % p2)
    return np.array([int(z) - M2 if z > M2 // 2 else int(z) for z in v.ravel()], dtype=np.int64).reshape(res[0].shape)
class Axis1:                                  # neuron j's column in segment j; chord in at slot 0; sum j lands at slot j*nin
    def __init__(s, w):
        s.nin, s.nout = w.shape; s.N = pow2(s.nin * s.nout + s.nin); wave = np.zeros(s.N, np.int64); wave[:s.nin * s.nout] = w.T.reshape(-1)
        s.spec = [ntt((wave % p).astype(np.uint64), p, g) for p, g in PR]
    def turn(s, x):                           # x: one chord (nin) -> all nout sums
        a = np.zeros(s.N, np.int64); a[0] = x[0]; a[s.N - s.nin + 1:] = x[:0:-1]       # the chord reversed onto the circle
        out = [ntt((ntt((a % p).astype(np.uint64), p, g) * sp) % p, p, g, inv=True) for (p, g), sp in zip(PR, s.spec)]
        return signed(np.stack(out)[:, ::1][:, np.arange(s.nout) * s.nin])
class Axis2:                                  # impulses at slot j*nin_seg; the plate laid so the output comes out contiguous at 0..nout-1
    def __init__(s, w, seg):
        s.nin, s.nout = w.shape; s.seg = seg; s.N = pow2(s.nin * seg + s.nout); wave = np.zeros(s.N, np.int64)
        for j in range(s.nin): idx = (np.arange(s.nout) - j * seg) % s.N; wave[idx] = w[j]
        s.spec = [ntt((wave % p).astype(np.uint64), p, g) for p, g in PR]
    def turn(s, h):
        a = np.zeros(s.N, np.int64); a[np.arange(s.nin) * s.seg] = h
        out = [ntt((ntt((a % p).astype(np.uint64), p, g) * sp) % p, p, g, inv=True) for (p, g), sp in zip(PR, s.spec)]
        return signed(np.stack(out)[:, :s.nout])
t0 = time.time(); CIRC = {}
for l in range(12):
    CIRC[f"qkv.{l}"] = Axis1(M[f"qkv.{l}.w"].astype(np.int64)); CIRC[f"proj.{l}"] = Axis1(M[f"proj.{l}.w"].astype(np.int64))
    CIRC[f"fc.{l}"] = Axis1(M[f"fc.{l}.w"].astype(np.int64)); CIRC[f"mproj.{l}"] = Axis2(M[f"mproj.{l}.w"].astype(np.int64), 768)
hw = M["head.w"].astype(np.int64); CH = [Axis1(c) for c in np.array_split(hw, 16, axis=1)]
print(f"the plate frozen on its circles, once ({time.time() - t0:.0f} s)", flush=True)
exact = product                                       # the saved base: earlier pieces as before
def product_turn(x16, w, name):
    out = exact(x16[:-1], w) if x16.shape[0] > 1 else np.zeros((0, w.shape[1]), np.int64)
    new = CIRC[name].turn(x16[-1].astype(np.int64)) if name != "head" else np.concatenate([c.turn(x16[-1].astype(np.int64)) for c in CH])
    return np.vstack([out, new[None]])
CHECK = [0, 0]
def gb(R, name):
    w, we, b = W[name]; x16, ex, LS = gauge_log(R); pile = product_turn(x16, M[name + ".w"].astype(np.int64), name)
    CHECK[0] += 1; CHECK[1] += int(np.array_equal(pile[-1], exact(x16[-1:], w)[0]))
    return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
def pb(X, name):
    w, we, b = W[name]; x16, ex = to16(X, -ES); pile = product_turn(x16, M[name + ".w"].astype(np.int64), name)
    CHECK[0] += 1; CHECK[1] += int(np.array_equal(pile[-1], exact(x16[-1:], w)[0]))
    return rshift(pile, -(we[None, :] + ex[:, None] + ES)) + b
def fwd(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + pb(attention(gb(R, f"qkv.{l}")), f"proj.{l}")
        R = R + pb(edge(gb(R, f"fc.{l}")), f"mproj.{l}")          # turn 1, the bend in place, turn 2
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    pile = np.concatenate([c.turn(x16[0].astype(np.int64)) for c in CH])[None]
    CHECK[0] += 1; CHECK[1] += int(np.array_equal(pile[0], exact(x16, w)[0]))
    return race(power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
r = refs[0]; got = []
for k in range(WORDS):
    t1 = time.time(); got.append(fwd(r["ids"] + got)); print(f"  word {k + 1}: {tok.decode([got[-1]])!r}  ({time.time() - t1:.0f} s)", flush=True)
want = r["ref"][:WORDS]
print(f"{r['prompt']!r} -> {tok.decode(got)!r}   {'IDENTICAL' if got == want else 'DIFFERENT'} to fp32 ({tok.decode(want)!r})")
print(f"turns whose piles equal the machine's exactly: {CHECK[1]} of {CHECK[0]}")
