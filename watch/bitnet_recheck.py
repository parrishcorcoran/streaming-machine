"""A checker's own count of how the 1s move in the BitNet records (bwatch/text_NN.npz), written apart from
bitnet_watch.py. Whole numbers only. Every count is built from one line's own trajectory and summed, then the whole
thing is redone with the lines (and neurons) in a random order (seed 123 / 321); the control pairings follow line
identity, so the controls must come out identical too. Controls: Sattolo cycles (never a line with itself), seed 2026.

    python3 bitnet_recheck.py A B     passages A..B-1 -> brecheck/c_NN.pkl
    python3 bitnet_recheck.py sum     add them up and print"""
import numpy as np, sys, os, pickle, time
DATA = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/bwatch/"
OUT = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/brecheck/"
L, NEU, DEEP, KEEP = 2560, 6912, 30, 24

def pc(v): return np.bitwise_count(v).astype(np.int64)
def blen(v):                                   # places held by a whole number >= 0: smear the top photon down, count
    v = np.array(v, np.int64, copy=True)
    for s in (1, 2, 4, 8, 16, 32): v |= v >> s
    return pc(v)
def mv(X, k):                                  # move X k places (up for k >= 0)
    return (X << k) if k >= 0 else (X >> -k)
def cycle(rng, m):                             # Sattolo: one random cycle through all m, so nobody is paired with itself
    p = np.arange(m)
    for i in range(m - 1, 0, -1):
        j = int(rng.integers(0, i)); p[i], p[j] = p[j], p[i]
    return p
def notnear(rng, m):                           # rows 0..m-1 (= positions t-1); never t-1 itself, never t
    while True:
        p = rng.permutation(m); r = np.arange(m)
        if not np.any((p == r) | (p == r + 1)): return p
def add(C, k, v): C[k] = C.get(k, 0) + v

def m1(C, tg, x, ex, y, ey):
    lit = (x != 0) & (y != 0); d = (blen(np.abs(y)) + ey - blen(np.abs(x)) - ex)[lit]
    add(C, (tg, "lit"), int(lit.sum())); add(C, (tg, "stay"), int((d == 0).sum()))
    add(C, (tg, "up1"), int((d == 1).sum())); add(C, (tg, "dn1"), int((d == -1).sum()))
    add(C, (tg, "up2+"), int((d >= 2).sum())); add(C, (tg, "dn2+"), int((d <= -2).sum()))
    add(C, (tg, "sign turned"), int((lit & ((x < 0) != (y < 0))).sum()))

def m2(C, tg, a, b, hrec=None):
    s = a + b; A, B, Sm = np.abs(a), np.abs(b), np.abs(s)
    h = max(0, int(blen(Sm.max())) - KEEP)
    if hrec is not None: assert h == hrec
    chg = A ^ Sm; mask = (1 << h) - 1; carry = pc(chg & ~B)
    add(C, (tg, "write photons"), int(pc(B).sum())); add(C, (tg, "changed"), int(pc(chg).sum()))
    add(C, (tg, "carry"), int(carry.sum())); add(C, (tg, "absorbed"), int(pc(B & ~chg).sum()))
    for lo, hi in ((0, 0), (1, 4), (5, 8), (9, 16), (17, 99)): add(C, (tg, f"carry {lo}-{hi}"), int(((carry >= lo) & (carry <= hi)).sum()))
    add(C, (tg, "written"), int((b != 0).sum())); add(C, (tg, "opposite"), int(((a != 0) & (b != 0) & ((a < 0) != (b < 0))).sum()))
    add(C, (tg, "after photons"), int(pc(Sm).sum())); add(C, (tg, "fall"), int(pc(Sm & mask).sum()))
    add(C, (tg, "write below"), int(pc(B & mask).sum()))
    add(C, (tg, "cut/add differ"), int((((a >> h) + (b >> h)) != (s >> h)).sum()))

def m3(C, tg, x, y):                           # brute force over every k that can matter (|X| < 2^36)
    X, Y = np.abs(x), np.abs(y); hits = np.zeros(X.shape, np.int64); tip = np.zeros_like(X); bx = blen(X)
    for k in range(-37, 38):
        c = mv(X, k); ok = (X > 0) & (Y > 0) & (bx + k <= 62) & (np.abs(Y - c) <= (Y - 1) >> 3)   # 8|Y - c| < Y, without overflow
        hits += ok; tip = np.where(ok, np.abs(Y - c), tip)
    assert hits.max() <= 1
    hold = hits == 1
    add(C, (tg, "write lit"), int((Y > 0).sum())); add(C, (tg, "holds"), int(hold.sum()))
    add(C, (tg, "holds same sign"), int((hold & ((x < 0) == (y < 0))).sum()))
    add(C, (tg, "tip ph"), int(pc(tip)[hold].sum())); add(C, (tg, "write ph where holds"), int(pc(Y)[hold].sum()))

def m4(C, tg, x, y):
    X, Y, sx, sy = np.abs(x), np.abs(y), np.sign(x), np.sign(y); bx = blen(X)
    add(C, (tg, "write ph"), int(pc(Y).sum()))
    for ax, nm in ((1, "pos"), (0, "line")):   # predicted: gain from top places, sign from the count of agreeing signs
        k = blen(Y.max(ax, keepdims=True)) - blen(X.max(ax, keepdims=True))
        ag, dg = (sx * sy > 0).sum(ax, keepdims=True), (sx * sy < 0).sum(ax, keepdims=True)
        s = np.where(ag >= dg, 1, -1)
        c = np.where(k >= 0, X << np.clip(k, 0, 62), X >> np.clip(-k, 0, 62))
        add(C, (tg, nm + " predicted"), int(pc(np.abs(y - s * sx * c)).sum()))
    rows, cols = [], []                        # searched (labelled a search): every k in -40..40, both signs
    for k in range(-40, 41):
        c = sx * mv(X, k) if k <= 0 else sx * (X << k)
        over = (k > 0) & (bx + k > 62)
        tP = np.where(over, 64, pc(np.abs(y - c))); tM = np.where(over, 64, pc(np.abs(y + c)))
        rows += [tP.sum(1), tM.sum(1)]; cols += [tP.sum(0), tM.sum(0)]
    add(C, (tg, "pos searched"), int(np.min(rows, 0).sum())); add(C, (tg, "line searched"), int(np.min(cols, 0).sum()))

def passage(i, order, norder):
    f = np.load(DATA + f"text_{i:02d}.npz"); n = int(f["n"]); inv = np.argsort(order)
    S = f["states"].astype(np.int64)[:, :, order]; Wr = f["writes"].astype(np.int64)[:, :, order]; sh3 = f["shifts"]
    dark = np.unpackbits(f["dark"], axis=-1, count=NEU).astype(bool)
    rng = np.random.default_rng(2026 + i)      # controls by line identity, then laid into this order
    cyc = [[cycle(rng, L) for _ in range(3)] for _ in range(60)]; pos = [notnear(rng, n - 1) for _ in range(60)]
    ncy = [cycle(rng, NEU) for _ in range(DEEP)]; npos = [notnear(rng, n - 1) for _ in range(DEEP)]
    lay = lambda p: inv[p[order]]
    C = {}; e = 0
    for w in range(60):
        kind = ("look", "neuron")[w % 2]; sa, sb, sh = (int(v) for v in sh3[w])
        x, y, b = S[w], S[w + 1], Wr[w] << sb; a = x << sa
        assert np.array_equal((a + b) >> sh, y)
        P, P5s, P5w = (lay(p) for p in cyc[w]); ne = e - sa + sh
        m1(C, ("M1a", kind), x, e, y, ne); m1(C, ("ctl M1a", kind), x, e, y[:, P], ne)
        m1(C, ("M1b", kind), a, 0, a + b, 0); m1(C, ("ctl M1b", kind), a[:, P], 0, a[:, P] + b, 0)
        m2(C, ("M2", kind), a, b, sh); m2(C, ("ctl M2", kind), a[:, P], b)
        m3(C, ("M3", kind), a, b); m3(C, ("ctl M3", kind), a[:, P], b)
        m4(C, ("M4", kind), a, b); m4(C, ("ctl M4", kind), a[:, P], b)
        for what, arr, Q in (("stream", y, P5s), ("write", Wr[w], P5w)):
            pv, cu = arr[:-1], arr[1:]
            for ctl, pp in (("", pv), ("ctl lines ", pv[:, Q]), ("ctl pos ", pv[pos[w]])):
                tg = (ctl + "M5 " + what, kind); m1(C, tg, pp, 0, cu, 0); m3(C, tg, pp, cu); m4(C, tg, pp, cu)
        e = ne
    for z in range(DEEP):
        no = norder[z]; ninv = np.argsort(no); dk = dark[z][:, no]; Pn = ninv[ncy[z][no]]
        add(C, ("dark", z, "all"), dk.size); add(C, ("dark", z, "dark"), int(dk.sum())); add(C, ("dark", z, "pairs"), dk[1:].size)
        add(C, ("dark", z, "both"), int((dk[:-1] & dk[1:]).sum())); add(C, ("dark", z, "ctl neurons"), int((dk[:-1][:, Pn] & dk[1:]).sum()))
        add(C, ("dark", z, "ctl pos"), int((dk[:-1][npos[z]] & dk[1:]).sum()))
    return C

def same(A, B): return A.keys() == B.keys() and all(A[k] == B[k] for k in A)

if sys.argv[1] != "sum":
    rho = np.random.default_rng(123).permutation(L); nr = np.random.default_rng(321); nro = [nr.permutation(NEU) for _ in range(DEEP)]
    for i in range(int(sys.argv[1]), int(sys.argv[2])):
        t = time.time(); A = passage(i, np.arange(L), [np.arange(NEU)] * DEEP); B = passage(i, rho, nro)
        pickle.dump((A, B), open(OUT + f"c_{i:02d}.pkl", "wb"))
        print(f"passage {i}: random order identical {same(A, B)} ({len(A)} counters, {time.time() - t:.0f}s)", flush=True)
    sys.exit()

R, Q, ok, files = {}, {}, True, sorted(f for f in os.listdir(OUT) if f.startswith("c_"))
for fn in files:
    A, B = pickle.load(open(OUT + fn, "rb")); ok &= same(A, B)
    for k, v in A.items(): add(R, k, v)
    for k, v in B.items(): add(Q, k, v)
print(f"{len(files)} passages; random line/neuron order identical per passage: {ok}; summed: {same(R, Q)} ({len(R)} counters)")
def r(a, b): return f"{a}/{b} = {(2000 * a + b) // (2 * b)} in 1000"
for kind in ("look", "neuron"):
    print(f"\n=== {kind} writes")
    for c in ("", "ctl "):
        for m in ("M1a", "M1b"):
            g = lambda k: R[((c + m, kind), k)]
            print(f"{c + m:9s} stay {r(g('stay'), g('lit'))}  up1 {g('up1')} dn1 {g('dn1')} up2+ {g('up2+')} dn2+ {g('dn2+')} sign turned {g('sign turned')}")
    for c in ("", "ctl "):
        g = lambda k: R[((c + "M2", kind), k)]
        print(f"{c}M2 write ph {g('write photons')} changed {g('changed')}; carry/changed {r(g('carry'), g('changed'))}; carry/write {r(g('carry'), g('write photons'))};"
              f" absorbed {r(g('absorbed'), g('write photons'))}")
        print(f"     carry hist " + " ".join(f"{q}:{g('carry ' + q)}" for q in ("0-0", "1-4", "5-8", "9-16", "17-99")) +
              f"; write below water {r(g('write below'), g('write photons'))}; fall {r(g('fall'), g('after photons'))};"
              f" opposite {r(g('opposite'), g('written'))}; cut/add differ {r(g('cut/add differ'), g('written'))}")
    for c in ("", "ctl "):
        g = lambda k: R[((c + "M3", kind), k)]
        print(f"{c}M3 holds {r(g('holds'), g('write lit'))} same sign {g('holds same sign')} tip {r(g('tip ph'), g('write ph where holds'))}")
    for c in ("", "ctl "):
        g = lambda k: R[((c + "M4", kind), k)]; wp = g("write ph")
        print(f"{c}M4 per pos: predicted {r(g('pos predicted'), wp)} searched {r(g('pos searched'), wp)}; per line: predicted {r(g('line predicted'), wp)} searched {r(g('line searched'), wp)}")
for what in ("stream", "write"):
    for kind in ("look", "neuron"):
        print(f"\n=== M5 {what} ({kind})")
        for c in ("", "ctl lines ", "ctl pos "):
            g = lambda k: R[((c + "M5 " + what, kind), k)]; wp = g("write ph")
            print(f"{c or 'real ':11s} stay {r(g('stay'), g('lit'))}; holds {r(g('holds'), g('write lit'))} tip {r(g('tip ph'), g('write ph where holds'))};"
                  f" pos pred {r(g('pos predicted'), wp)} srch {r(g('pos searched'), wp)}; line pred {r(g('line predicted'), wp)} srch {r(g('line searched'), wp)}")
print("\n=== dark")
for z in range(DEEP):
    g = lambda k: R[("dark", z, k)]
    print(f"depth {z:2d}: dark {r(g('dark'), g('all'))}; t-1&t {r(g('both'), g('pairs'))}; ctl neurons {r(g('ctl neurons'), g('pairs'))}; ctl pos {r(g('ctl pos'), g('pairs'))}")
