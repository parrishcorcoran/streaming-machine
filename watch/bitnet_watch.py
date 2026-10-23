"""How the 1s move in BitNet b1.58 on circles: watched from the records bitnet_collect.py wrote (no machine loaded).

A line's value is a whole number; its photons are the 1s of its size, its sign rides alongside. Every count here is
built from one line's own trajectory (its value before and after each of the 60 writes), then summed over lines, so
the lines can be taken in any order. That is checked, not assumed: every number is computed twice, the second time
with the 2,560 lines (and the 6,912 neurons) in a random order (seed 7), and the two sets of counts must be identical.

Every count also has a shuffle control: each line is paired with ANOTHER line's value (a random derangement, fixed
seed per passage and write; never chosen by likeness), or along the stream with another position's. The shuffle keeps
every value's distribution exactly; what the real pairing does beyond it is structure.

 M1  the wavefront: a line's top photon place after each write (all on one gauge, by the add's own shifts), and how
     many places it moves per write.
 M2  the coin pusher: at each add (stream + write, lined up by shifts), how many places change where the write had no
     photon (carries and borrows running), and how many photons fall below the water level (the add keeps the top 24
     places of the whole tensor).
 M3  is the write on a line that line's own value moved by a shift (times +-2^k, any k) plus a tip under 1/8 of the
     write? An existence test: the three shifts that can land within 1/8 are each checked.
 M4  one shift for a whole position (all 2,560 lines) or one shift for a line (every position): the gain predicted
     from the top places (the write's top place minus the stream's) and the sign from the count of agreeing signs --
     no search; and, labelled as a search, the gain that leaves the fewest tip photons after trying every k in
     -40..40 and both signs. Photons left as tip over the write's photons.
 M5  M1, M3, M4 along the stream: the same write's light at consecutive positions (pulse to pulse), and the writes.
 Dark: per depth, the neurons whose gate pile ends at or below the edge (the bend sends exactly nothing), and how
     often the same neuron is dark at consecutive positions, against a neuron shuffle.

    python3 bitnet_watch.py A B      measure passages A..B-1, save the counts
    python3 bitnet_watch.py combine  add every saved count and print"""
import numpy as np, os, sys, time, pickle
DATA = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/bwatch/"
L, NEU, DEEP, WATER, KS = 2560, 6912, 30, 24, np.arange(-40, 41)

def bitlen(v):                                 # places a whole number >= 0 occupies (0 for dark), by halving
    v = v.copy(); n = np.zeros(v.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = (v >> s) > 0; n += m * s; v = np.where(m, v >> s, v)
    return n + (v > 0)
def topbits(v): return int(bitlen(np.abs(v).max().reshape(1))[0])
def pc(v): return np.bitwise_count(v).astype(np.int64)        # photons of a whole number >= 0
def shift(X, k):                               # X moved k places up (k >= 0) or -k places down
    return np.where(k >= 0, X << np.clip(k, 0, 62), X >> np.clip(-k, 0, 62))
def derange(rng, m):
    while True:
        p = rng.permutation(m)
        if not np.any(p == np.arange(m)): return p
def derange_pos(rng, m):                       # previous position for cur rows 1..m: never its own neighbour t-1, never t itself
    while True:
        p = rng.permutation(m); r = np.arange(m)
        if not np.any((p == r) | (p == r + 1)): return p
def add(C, key, v): C[key] = C.get(key, 0) + v

def topmove(C, tag, x, ex, y, ey):             # M1: top photon place of x (gauge 2^ex) and of y (gauge 2^ey), per line
    lx, ly = x != 0, y != 0; both = lx & ly
    dT = (bitlen(np.abs(y)) + ey - bitlen(np.abs(x)) - ex)[both]
    add(C, (tag, "hist"), np.bincount(np.clip(dT, -64, 64) + 64, minlength=129))
    add(C, (tag, "lit both"), int(both.sum())); add(C, (tag, "stay"), int((dT == 0).sum()))
    add(C, (tag, "dark both"), int((~lx & ~ly).sum())); add(C, (tag, "dark to lit"), int((~lx & ly).sum()))
    add(C, (tag, "lit to dark"), int((lx & ~ly).sum())); add(C, (tag, "sign turned"), int((both & ((x > 0) != (y > 0))).sum()))

def ownwave(C, tag, x, y):                     # M3: is y = +-(x moved k places) + tip, |tip| < |y|/8, for some k?
    X, Y = np.abs(x), np.abs(y); Dk = bitlen(Y) - bitlen(X); hold = np.zeros(X.shape, bool); tip = np.zeros_like(X)
    for dk in (-1, 0, 1):
        c = shift(X, Dk + dk); ok = (X > 0) & (Y > 0) & (8 * np.abs(Y - c) < Y)
        assert not np.any(hold & ok); tip = np.where(ok, np.abs(Y - c), tip); hold |= ok
    add(C, (tag, "write lit"), int((Y > 0).sum())); add(C, (tag, "holds"), int(hold.sum()))
    add(C, (tag, "holds, same sign"), int((hold & ((x > 0) == (y > 0))).sum()))
    add(C, (tag, "tip photons where holds"), int(pc(tip)[hold].sum())); add(C, (tag, "write photons where holds"), int(pc(Y)[hold].sum()))
    add(C, (tag, "write photons"), int(pc(Y).sum()))

def gains(C, tag, x, y, search=True):          # M4: one shift per row (all lines) / per line (all rows): photons left as tip
    X, Y, sx, sy = np.abs(x), np.abs(y), np.sign(x), np.sign(y)
    add(C, (tag, "gain write photons"), int(pc(Y).sum()))
    for ax, nm in ((1, "row"), (0, "line")):
        k = bitlen(Y.max(ax, keepdims=True)) - bitlen(X.max(ax, keepdims=True))
        agree, dis = ((sx * sy) > 0).sum(ax, keepdims=True), ((sx * sy) < 0).sum(ax, keepdims=True)
        s = np.where(agree >= dis, 1, -1)
        add(C, (tag, nm + " predicted tip"), int(pc(np.abs(y - s * sx * shift(X, k))).sum()))
        add(C, (tag, nm + " predicted sign agrees"), int((s > 0).sum())); add(C, (tag, nm + " count"), int(s.size))
    if not search: return
    bx = bitlen(X)[None]; K = KS[:, None, None]; c = sx[None] * shift(X[None], K)
    bad = (K > 0) & (bx + K > 62)
    tP = np.where(bad, 64, pc(np.abs(y[None] - c))); tM = np.where(bad, 64, pc(np.abs(y[None] + c)))
    add(C, (tag, "row searched tip"), int(np.minimum(tP.sum(2).min(0), tM.sum(2).min(0)).sum()))
    add(C, (tag, "line searched tip"), int(np.minimum(tP.sum(1).min(0), tM.sum(1).min(0)).sum()))

def measure(i, order, norder):
    f = np.load(DATA + f"text_{i:02d}.npz"); n = int(f["n"]); inv = np.argsort(order)
    S = f["states"].astype(np.int64)[:, :, order]; Wr = f["writes"].astype(np.int64)[:, :, order]; sh3 = f["shifts"]
    dark = np.unpackbits(f["dark"], axis=-1, count=NEU).astype(bool); below = np.unpackbits(f["below"], axis=-1, count=NEU).astype(bool)
    rng = np.random.default_rng(1000 + i)                                     # the control pairings, by line identity
    pi = [derange(rng, L) for _ in range(60)]; pi5 = [derange(rng, L) for _ in range(60)]; pi5w = [derange(rng, L) for _ in range(60)]
    sig = [derange_pos(rng, n - 1) for _ in range(60)]; pn = [derange(rng, NEU) for _ in range(DEEP)]
    sigd = [derange_pos(rng, n - 1) for _ in range(DEEP)]
    part = lambda p: inv[p[order]]                                            # the same pairs, in this line order
    e = np.zeros(61, np.int64)
    for w in range(60): e[w + 1] = e[w] - sh3[w, 0] + sh3[w, 2]              # the stream's gauge, by the add's own shifts
    C = {}
    for w in range(60):
        kind = ("look", "neuron")[w % 2]; sa, sb, sh = (int(v) for v in sh3[w]); P = part(pi[w])
        before, after = S[w], S[w + 1]
        for ctl, aft in (("", after), ("ctl ", after[:, P])):
            topmove(C, (ctl + "M1", kind), before, e[w], aft, e[w + 1])
            topmove(C, (ctl + "M1 depth", w), before, e[w], aft, e[w + 1])
        a, b = before << sa, Wr[w] << sb                                      # the add's own places
        s = a + b; assert np.array_equal(s >> sh, after) and max(0, topbits(s) - WATER) == sh
        for ctl, aa in (("", a), ("ctl ", a[:, P])):
            ss = aa + b; A, B, Sm = np.abs(aa), np.abs(b), np.abs(ss)
            topmove(C, (ctl + "M1 add", kind), aa, 0, ss, 0); topmove(C, (ctl + "M1 add depth", w), aa, 0, ss, 0)
            h = max(0, topbits(Sm) - WATER); mask = (1 << h) - 1; chg = A ^ Sm
            for tg in ((ctl + "M2", kind), (ctl + "M2 depth", w)):
                add(C, (tg, "write photons"), int(pc(B).sum())); add(C, (tg, "places changed"), int(pc(chg).sum()))
                add(C, (tg, "changed where write had none"), int(pc(chg & ~B).sum())); add(C, (tg, "write photons absorbed"), int(pc(B & ~chg).sum()))
            add(C, ((ctl + "M2", kind), "carry hist"), np.bincount(pc(chg & ~B).ravel(), minlength=65))
            tg = (ctl + "M2", kind)
            add(C, (tg, "lines written"), int((b != 0).sum())); add(C, (tg, "opposite signs"), int(((aa != 0) & (b != 0) & ((aa > 0) != (b > 0))).sum()))
            add(C, (tg, "sign turned"), int(((aa != 0) & (ss != 0) & ((aa > 0) != (ss > 0))).sum()))
            add(C, (tg, "photons after add"), int(pc(Sm).sum())); add(C, (tg, "fall below water"), int(pc(Sm & mask).sum()))
            add(C, (tg, "write photons below water"), int(pc(B & mask).sum())); add(C, (tg, "stream photons below water"), int(pc(A & mask).sum()))
            add(C, (tg, "water then add differs"), int((((aa >> h) + (b >> h)) != (ss >> h)).sum()))
            add(C, (tg, "negatives rounded away"), int(((ss < 0) & ((Sm & mask) != 0)).sum())); add(C, (tg, "water level sum"), h)
            ownwave(C, (ctl + "M3", kind), aa, b); ownwave(C, (ctl + "M3 depth", w), aa, b)
            gains(C, (ctl + "M4", kind), aa, b)
        # M5: along the stream, the light after this write and the write itself, position t-1 against t
        for what, arr, P5 in (("stream", after, part(pi5[w])), ("write", Wr[w], part(pi5w[w]))):
            prev, cur = arr[:-1], arr[1:]
            for ctl, pv in (("", prev), ("ctl lines ", prev[:, P5]), ("ctl positions ", prev[sig[w]])):
                tg = (ctl + "M5 " + what, kind)
                topmove(C, tg, pv, 0, cur, 0); ownwave(C, tg, pv, cur); gains(C, tg, pv, cur)
    for z in range(DEEP):                                                     # the dark neurons, any neuron order
        no = norder[z]; ninv = np.argsort(no); dk, bl = dark[z][:, no], below[z][:, no]; Pn = ninv[pn[z][no]]
        add(C, ("dark", z, "neurons x positions"), dk.size); add(C, ("dark", z, "dark"), int(dk.sum())); add(C, ("dark", z, "below edge"), int(bl.sum()))
        add(C, ("dark", z, "pairs"), int(dk[1:].size)); add(C, ("dark", z, "dark at t-1 and t"), int((dk[:-1] & dk[1:]).sum()))
        add(C, ("dark", z, "ctl dark at t-1 and t"), int((dk[:-1][:, Pn] & dk[1:]).sum()))
        add(C, ("dark", z, "ctl pos dark at t-1 and t"), int((dk[:-1][sigd[z]] & dk[1:]).sum()))
    return C

def same(A, B):
    if A.keys() != B.keys(): return False
    return all(np.array_equal(np.asarray(A[k]), np.asarray(B[k])) for k in A)

if sys.argv[1] != "combine":
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    rho = np.random.default_rng(7).permutation(L); nrng = np.random.default_rng(8); nro = [nrng.permutation(NEU) for _ in range(DEEP)]
    ident = np.arange(L); nid = [np.arange(NEU)] * DEEP
    for i in range(a, b):
        C0 = measure(i, ident, nid); C1 = measure(i, rho, nro)
        pickle.dump({"real order": C0, "random order": C1}, open(DATA + f"counts_{i:02d}.pkl", "wb"))
        print(f"  passage {i}: lines in a random order give identical counts: {same(C0, C1)}  ({time.time() - t0:.0f}s)", flush=True)
    sys.exit()

# ---- combine: add every passage's counts and print
def sh(a, b): return f"{a}/{b} (= {(2000 * a + b) // (2 * b) if b else 0} in 1000)"
files = sorted(f for f in os.listdir(DATA) if f.startswith("counts_"))
R, Q, alleq = {}, {}, True
for fn in files:
    d = pickle.load(open(DATA + fn, "rb")); alleq &= same(d["real order"], d["random order"])
    for k, v in d["real order"].items(): add(R, k, v)
    for k, v in d["random order"].items(): add(Q, k, v)
print(f"\n  {len(files)} passages. Every count recomputed with the 2,560 lines and the 6,912 neurons in a random order (seed 7):"
      f"\n  identical, passage by passage: {alleq};  summed: {same(R, Q)}  ({len(R)} counters compared)")
def hist(hv, lo=-6, hi=6):
    h = hv; s = [f"{d:+d}:{int(h[d + 64])}" for d in range(lo, hi + 1)]
    return f"<{lo}:{int(h[:lo + 64].sum())}  " + "  ".join(s) + f"  >{hi}:{int(h[hi + 65:].sum())}"
def m1(tg, label):
    lb = R[(tg, "lit both")]; h = R[(tg, "hist")]
    print(f"    {label:34s} stays put {sh(R[(tg, 'stay')], lb)};  up 1: {int(h[65])}, down 1: {int(h[63])}, "
          f"up 2+: {int(h[66:].sum())}, down 2+: {int(h[:63].sum())}")
    print(f"    {'':34s} dark->dark {R[(tg, 'dark both')]}, dark->lit {R[(tg, 'dark to lit')]}, lit->dark {R[(tg, 'lit to dark')]}, "
          f"sign turned {R[(tg, 'sign turned')]}")
    print(f"    {'':34s} moves: {hist(h)}")
print("\n  M1  the wavefront: each line's top photon place, before and after each write (one gauge: the add's own shifts)")
for kind in ("look", "neuron"):
    for ctl in ("", "ctl "): m1((ctl + "M1", kind), f"{kind} writes{', after-value SHUFFLED' if ctl else ''}")
print("    the same at the add itself (before the water), control: the write that lands is another line's write")
for kind in ("look", "neuron"):
    for ctl in ("", "ctl "): m1((ctl + "M1 add", kind), f"{kind} writes{', write SHUFFLED' if ctl else ''}")
print("\n  M2  the coin pusher: the add, stream + write, at the add's own places")
for kind in ("look", "neuron"):
    for ctl in ("", "ctl "):
        tg = (ctl + "M2", kind); wp = R[(tg, "write photons")]; ch = R[(tg, "places changed")]; cn = R[(tg, "changed where write had none")]
        print(f"    {kind} writes{', SHUFFLED' if ctl else ''}:")
        print(f"      write photons {wp};  places changed {ch};  changed where the write had no photon {sh(cn, ch)};  "
              f"carry places per write photon {sh(cn, wp)};  write photons absorbed (place unchanged) {sh(R[(tg, 'write photons absorbed')], wp)}")
        hh = R[(tg, "carry hist")]; buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 9), (9, 17), (17, 65)]
        print("      carry places per line and write: " + "  ".join(f"{a if b - a == 1 else f'{a}-{b - 1}'}:{int(hh[a:b].sum())}" for a, b in buckets))
        print(f"      opposite signs {sh(R[(tg, 'opposite signs')], R[(tg, 'lines written')])};  sign turned {R[(tg, 'sign turned')]};  "
              f"photons after the add {R[(tg, 'photons after add')]}, fall below the water {sh(R[(tg, 'fall below water')], R[(tg, 'photons after add')])}")
        print(f"      of the write's photons below the water {sh(R[(tg, 'write photons below water')], wp)};  stream photons below the water "
              f"{R[(tg, 'stream photons below water')]};  negatives rounded away {R[(tg, 'negatives rounded away')]};  water level, places "
              f"above the add's grain, summed {R[(tg, 'water level sum')]} over {30 * len(files)} adds")
        print(f"      add then water vs water then add (each pile cut first): differ on {sh(R[(tg, 'water then add differs')], R[(tg, 'lines written')])} lines")
print("\n  M3  is a write on a line that line's own value moved (+-2^k, any k) plus a tip under 1/8 of the write?")
for kind in ("look", "neuron"):
    for ctl in ("", "ctl "):
        tg = (ctl + "M3", kind)
        print(f"    {kind} writes{', SHUFFLED' if ctl else '':10s}: holds {sh(R[(tg, 'holds')], R[(tg, 'write lit')])};  "
              f"same sign {R[(tg, 'holds, same sign')]};  tip photons / write photons where it holds "
              f"{sh(R[(tg, 'tip photons where holds')], R[(tg, 'write photons where holds')])}")
def m4(tg, label):
    wp = R[(tg, "gain write photons")]
    print(f"    {label:38s} tip photons / write photons:  one shift per position: predicted {sh(R[(tg, 'row predicted tip')], wp)}, "
          f"searched {sh(R[(tg, 'row searched tip')], wp)}")
    print(f"    {'':38s}                               one shift per line:     predicted {sh(R[(tg, 'line predicted tip')], wp)}, "
          f"searched {sh(R[(tg, 'line searched tip')], wp)}")
print("\n  M4  shared motion: one whole-number shift (and sign) for all lines of a position, or for all positions of a line")
for kind in ("look", "neuron"):
    for ctl in ("", "ctl "): m4((ctl + "M4", kind), f"{kind} writes{', SHUFFLED' if ctl else ''}")
print("\n  M5  along the stream: the same write's light at position t-1 against t")
for what in ("stream", "write"):
    for kind in ("look", "neuron"):
        for ctl in ("", "ctl lines ", "ctl positions "):
            tg = (ctl + "M5 " + what, kind); lab = f"{what} after {kind} writes" if what == "stream" else f"{kind} writes themselves"
            lab += {"": "", "ctl lines ": ", lines SHUFFLED", "ctl positions ": ", positions SHUFFLED"}[ctl]
            m1(tg, lab)
            print(f"    {'':34s} own wave moved + tip < 1/8: {sh(R[(tg, 'holds')], R[(tg, 'write lit')])}; tip/photons where it holds "
                  f"{sh(R[(tg, 'tip photons where holds')], R[(tg, 'write photons where holds')])}")
            m4(tg, "")
print("\n  per depth: M1 at the add stays put (look | neuron), M3 holds (look | neuron), M2 carry places per write photon (look | neuron); real / SHUFFLED")
for z in range(DEEP):
    row = []
    for nm, a_, b_ in (("M1 add depth", "stay", "lit both"), ("M3 depth", "holds", "write lit"), ("M2 depth", "changed where write had none", "write photons")):
        for w in (2 * z, 2 * z + 1):
            r, c = R[((nm, w), a_)], R[(("ctl " + nm, w), a_)]; br, bc = R[((nm, w), b_)], R[(("ctl " + nm, w), b_)]
            row.append(f"{(2000 * r + br) // (2 * br)}/{(2000 * c + bc) // (2 * bc)}")
    print(f"    depth {z:2d}:  M1 {row[0]:>9s} {row[1]:>9s}   M3 {row[2]:>9s} {row[3]:>9s}   M2 {row[4]:>9s} {row[5]:>9s}   (in 1000)")
print("\n  dark neurons per depth (gate pile at or below the edge: the bend sends exactly nothing)")
for z in range(DEEP):
    g = lambda k: R[("dark", z, k)]
    print(f"    depth {z:2d}: dark {sh(g('dark'), g('neurons x positions'))}  (below the edge {g('below edge')}, on it {g('dark') - g('below edge')});"
          f"  same neuron dark at t-1 and t {sh(g('dark at t-1 and t'), g('pairs'))}, SHUFFLED neurons {sh(g('ctl dark at t-1 and t'), g('pairs'))}"
          f", SHUFFLED positions {sh(g('ctl pos dark at t-1 and t'), g('pairs'))}")
