"""Is every strike's twist the same base twist, turned? The tiny model (GPT-2's whole depth collapsed for a 5-word
strike: 768 x 3840 notes) for 30 strikes, each as 16-bit whole numbers laid end to end on one circle. For each strike
against the first: every turn of the circle checked at once (exact whole-number turns on two prime circles), the best
turn found, and how much lines up there (the share of the first twist's square that the turned twist explains).
    python3 twist_turns.py
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/Users/abundancemachine/basics/code"); import os; os.chdir("/Users/abundancemachine/basics/code")
from _layerlib import collapse_column, tok, dev
os.chdir("/Users/abundancemachine/streaming-machine"); refs = json.load(open("refs_many.json"))
PR = [(998244353, 3), (469762049, 3)]; P1, P2 = PR[0][0], PR[1][0]; INV = pow(P1, -1, P2); MOD = P1 * P2
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
    a = a[_rev[n]] % p; length = 2
    while length <= n:
        w = pow(g, (p - 1) // length, p); w = pow(w, p - 2, p) if inv else w
        ws = np.ones(length // 2, np.uint64)
        for k in range(1, length // 2): ws[k] = (int(ws[k - 1]) * w) % p
        a = a.reshape(n // length, length); u = a[:, :length // 2]; v = (a[:, length // 2:] * ws) % p
        a = np.concatenate([(u + v) % p, (u + p - v) % p], 1).reshape(n); length <<= 1
    return (a * pow(n, p - 2, p)) % p if inv else a
def twist(ids):                                                       # the tiny model for one strike, as 16-bit whole numbers
    J = collapse_column(torch.tensor([ids], device=dev))[0].double().numpy()
    return np.round(J / np.abs(J).max() * 32767).astype(np.int64).reshape(-1)
strikes = [(r["ids"] + r["ref"])[:5] for r in refs]
T = [twist(s) for s in strikes]; n = T[0].size; N = 1 << (2 * n - 1).bit_length()
def lap(v, rev=False):
    a = np.zeros(N, np.int64)
    if rev: a[0] = v[0]; a[N - n + 1:] = v[:0:-1]
    else: a[:n] = v
    return a
specA = [ntt((lap(T[0], rev=True) % p).astype(np.uint64), p, g) for p, g in PR]
a2 = int((T[0] * T[0]).sum())
print(f"each twist: {n:,} notes; circle of {N:,}")
for k in range(1, len(T)):
    res = [ntt((ntt((lap(T[k]) % p).astype(np.uint64), p, g) * sa) % p, p, g, inv=True) for (p, g), sa in zip(PR, specA)]
    r1, r2 = res[0].astype(np.uint64), res[1].astype(np.uint64)
    t = (((r2 + P2 - (r1 % P2)) % P2) * np.uint64(INV)) % np.uint64(P2)      # the two circles together: every turn's true overlap
    v = r1.astype(np.int64) + np.int64(P1) * t.astype(np.int64); v = np.where(v > MOD // 2, v - MOD, v)
    b2 = int((T[k] * T[k]).sum()); best = int(np.argmax(np.abs(v))); at0 = int(v[0])
    share = lambda c: c * c / (a2 * b2)
    print(f"  {tok.decode(strikes[k])!r:34s} best turn {best if best < N // 2 else best - N:+10,d}: lines up {100 * share(int(v[best])):5.1f}%   unturned: {100 * share(at0):5.1f}%")
