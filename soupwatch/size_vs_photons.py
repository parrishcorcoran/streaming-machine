"""How big the change is, against how many photons it carries. The same soups as soup_watch.py, re-run.

For each wave w -> w+1, snapshot, and region (the front; ahead of it with the word changed; ahead with the word the
same), summed over every value of every position there:
  size      sum |d|  against  sum |R(w+1)|                  (how much the light moved, in its own units)
  places    sum of the places d spans (top 1's place + 1)   against  the same for R(w+1)
  photons   sum of d's 1s  against  R(w+1)'s 1s
and the same three with d taken against a different random position (shuffle A, as in soup_watch.py).
Predicted: the size share is small, the places share is large (each halving of the size removes one place
only), and the photons follow the places, not the size.

    python3 soupwatch/size_vs_photons.py A B
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soup_lib as L
A, B = int(sys.argv[1]), int(sys.argv[2]); N = 16
refs = json.load(open(L.HERE + "refs_many.json"))
REG = ("front", "ahead, own word changed", "ahead, own word same")
GROUPS = [("entry", [0]), ("layers 0-2", list(range(1, 7))), ("layers 3-7", list(range(7, 17))), ("layers 8-11", list(range(17, 25)))]
acc = {r: np.zeros((25, 8), dtype=object) for r in REG}           # object: Python whole numbers, no overflow
for r in REG: acc[r][:] = 0
def places(X):                                                     # the top 1's place + 1, by shifting (0 for 0)
    X = np.abs(X); n = np.zeros(X.shape, np.int64)
    while X.any(): n += X != 0; X = X >> 1
    return n


def tally(r, new, old, sh):
    d = new - old; ds = new - sh
    for s in range(25):
        acc[r][s] += [int(np.abs(d[s]).sum()), int(np.abs(new[s]).sum()), int(places(d[s]).sum()), int(places(new[s]).sum()),
                      int(L.bc(d[s]).sum()), int(L.bc(new[s]).sum()), int(np.abs(ds[s]).sum()), int(places(ds[s]).sum())]


for t in range(A, B):
    prompt = (refs[t]["ids"] + refs[t]["ref"])[:5]; S = len(prompt); P = S + N; reads = list(range(S - 1, P - 1))
    T = prompt + [prompt[-1]] * N; prev = None; wv = 0
    while True:
        wv += 1; cur = L.light(T, reads); newT = prompt + cur["tops"]
        if prev is not None:
            g = np.random.default_rng(1000 * t + wv); qa = (np.arange(P) + g.integers(1, P, size=P)) % P
            a, b = np.array(PT), np.array(T); f = np.flatnonzero(a != b)[0]
            for p in range(f, P):
                r = "front" if p == f else ("ahead, own word changed" if a[p] != b[p] else "ahead, own word same")
                tally(r, cur["snaps"][:, p], prev["snaps"][:, p], prev["snaps"][:, qa[p]])
        if newT == T: break
        prev, PT, T = cur, T, newT
    print(f"text {t} done", flush=True)
print("\nsize = sum |d| / sum |R(w+1)|;  places = places d spans / places R(w+1) spans;  photons = 1s of d / 1s of R(w+1);  SHUFFLE A in brackets")
for r in REG:
    print(f"\n  {r}")
    for name, ss in GROUPS:
        v = acc[r][ss].sum(0)
        print(f"  {name:12s} size {v[0]:,}/{v[1]:,} ({100 * v[0] // v[1]}%) [{100 * v[6] // v[1]}%]   places {v[2]:,}/{v[3]:,} ({100 * v[2] // v[3]}%) [{100 * v[7] // v[3]}%]"
              f"   photons {v[4]:,}/{v[5]:,} ({100 * v[4] // v[5]}%)")
