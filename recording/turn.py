"""PHASE A, part 3: THE EXACT TURN AND ITS LIT PIXELS. Whole numbers and exact rationals in the machine path; float
only to print percentages.

THE TURN at a plate, set by a direction d (768 whole numbers, lines summing to exactly 0: the centered foundation):
    H = I - 2 v v^T / (v . v)            (a reflection: H H = I exactly, H^T H = I exactly, for ANY rational v)
  - keeps every length exactly, for any v (it is a reflection).
  - keeps the all-ones direction fixed exactly iff v . 1 = 0.
  - sends d to a target u exactly iff |u| = |d| and v = d - u.
  A pure reflection cannot do both: one line e_k is not orthogonal to the all-ones direction, so a reflection that
  keeps the all-ones direction fixed can only send d inside the centered space. So the turn used is the reflection
  INSIDE THE CENTERED SPACE: the target is the centered line t = 768 e_k - 1 (767 on line k, -1 on every other line,
  lines summing to 0), k = the line where |d| is largest, signed like d there:
        u = sg (C / 2^m) t,   C = floor(sqrt(|d|^2 4^m / (768 x 767)))   (m = 48; the length match needs a square root,
        v = 2^m d - sg C t    (whole numbers; v . 1 = 0 exactly)           so C is its whole-number floor)
  H d = u + (a leftover of size below 2^-40 of |d|, from the floor), reported exactly.
  On the light: x' = H x = (D x - 2 (v.x) v) / D, D = v . v.  On a plate: H W (lines in), the same per neuron column.

BARE-LINE VARIANT (F| and S|; NOT allowed by the rules, shown only to size what the centering costs): target = the bare
line e_k, v = 2^m d - sg C e_k. It moves the all-ones direction, so the water level changes.

COUNTED IN THE MACHINE'S OWN WAY, positions >= 1 of the 30 texts:
  light: H x rounded to whole numbers at the stream's place (2^-16), then the machine's 16-place water level (to16),
         photons = 1s of |value| on each line. Also: kept at the UNTURNED light's places (no coarser shift), and the
         water level (gauge) compared.
  plate: H W per neuron, written as whole numbers at the neuron's own 16-place scale (the smallest power of two that
         holds its largest value, as base2_convert.cols16 does), photons per weight.
  WORK = sum over lines of (light photons on the line) x (plate photons on that line), per position, summed.
  Against the unturned machine (to16 of x, the stored plate). Directions: F, S, controls R (random), F~, S~ (shuffled
  lines), and the bare-line variants F|, S|.
EXACT CHECKS (qkv.l, fc.l): for every position of texts 0-2, (H x) met with (H W) == x met with W as exact rationals
  (the turned light's whole-number numerators split into 24-bit pieces, each met with W in int64, recombined exactly);
  and directly, with the turned plate's numerators formed entry by entry, for the last position of texts 0-2 (F, S).
  For every position of every text: H (H x) == x, |H x|^2 == |x|^2, 1 . H x == 1 . x (exactly).
ORDER-FREE CHECK: the S turn at qkv.6 and fc.6 with the lines of light, plate and direction shuffled together must
  give every count exactly.
PREDICTION (written before running, from the plan): lit pixels of the turned light fall by less than the energy share,
  because lit pixels go with the log of a value's size; the turned plates gain photons; the net work changes by a few
  percent at most unless the share along the direction is very large.
  Claude's, after seeing the shares (S 15-26% from layer 3 on, F under 1% mostly; S from layer 3 on is already 92%
  on one line, 447): the S turn moves little light, light photons change by under 2%, plate photons per weight change
  by under 1%, net work within +-2% at every plate; F the same or smaller.
    python3 recording/turn.py PLATE[:DIR,DIR] ...   -> recording/turn/PLATE__DIR.json     python3 recording/turn.py report
"""
import sys, os, json, time, math, resource, numpy as np
DOC = __doc__                                                # kept before the machine's source is run
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE); os.makedirs("recording/turn", exist_ok=True)
ARGS = sys.argv[1:]
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
i0 = src.index("W = {k: (M[k + \".w\"].astype(np.float64)"); i1 = src.index("\n", i0)
src = src[:i0] + "W = {}   # plates are read one at a time below, as the stored whole numbers" + src[i1:]
exec(src)                                                    # the machine's own to16, rshift, gauge_log
MB, LIM = 48, 32767
DIRS = ["F", "S", "R", "F~", "S~", "F|", "S|"]

def obj(a): return np.array([int(v) for v in np.asarray(a).ravel()], dtype=object).reshape(np.shape(a))
def centered(f): f = obj(f); return 768 * f - sum(f)
def dot(a, b): return sum(int(x) * int(y) for x, y in zip(a, b))
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
_r = np.random.default_rng(0).integers(-(1 << 40), 1 << 40, 1000)
assert all(int(o) == bin(abs(int(v))).count("1") for o, v in zip(ones(_r), _r))

def pieces(rows, bits=24):
    """whole numbers (object) -> int64 pieces P_j with rows = sum_j P_j 2^(bits j); every piece below 2^bits in size"""
    vals = np.asarray(rows, dtype=object); out = []; mask = (1 << bits) - 1
    while True:
        if all(-(1 << bits) <= int(x) < (1 << bits) for x in vals.ravel()): out.append(vals.astype(np.int64)); return out
        out.append((vals & mask).astype(np.int64)); vals = vals >> bits

def meet_exact(rows, w, bits=24):
    """rows (object whole numbers, n x 768) met with plate w (whole numbers below 2^15, 768 x N), exactly: object n x N"""
    acc = None; w64 = w if w.dtype == np.int64 else w.astype(np.int64)
    for j, Pj in enumerate(pieces(rows, bits)):
        part = obj(Pj @ w64) * (1 << (bits * j))                     # each piece < 2^24, w < 2^15, 768 lines: < 2^49
        acc = part if acc is None else acc + part
    return acc

def make_turn(d, bare):
    """v, D, k, sg, target t, C for the reflection sending d to (C/2^m) sg t"""
    dd = dot(d, d)
    if dd == 0: return None
    k = max(range(768), key=lambda i: abs(int(d[i])))                # the line where |d| is largest (first if tied)
    sg = 1 if d[k] > 0 else -1
    if bare: t = np.zeros(768, dtype=object); t[k] = 1; tt = 1
    else: t = np.full(768, -1, dtype=object); t[k] = 767; tt = 768 * 767
    C = math.isqrt(dd * 4 ** MB // tt)
    v = d * (1 << MB) - sg * C * t
    return dict(v=v, D=dot(v, v), k=k, sg=sg, t=t, C=C, dd=dd)

def col_shift(mx, D):                                        # smallest s with mx / D <= 32767 x 2^s
    if mx == 0: return 0
    def ok(s): return mx <= ((LIM * D) << s) if s >= 0 else (mx << -s) <= LIM * D
    s = mx.bit_length() - (LIM * D).bit_length()
    while not ok(s): s += 1
    while ok(s - 1): s -= 1
    return s

L = np.load("recording/light.npz"); PLATES = list(L["plates"]); POS, TXT = L["pos"], L["txt"]
later = np.flatnonzero(POS >= 1); first = np.flatnonzero(POS == 0)
CHK3 = np.flatnonzero(TXT < 3); LAST3 = [int(np.flatnonzero(TXT == t)[-1]) for t in range(3)]

def directions(k):
    i = PLATES.index(k)
    dF, dS = centered(L[f"F|{k}"]), centered(L[f"S The|{k}"])
    r = np.random.default_rng(20261004 + i).integers(-(1 << 20), 1 << 20, 768)          # the same seeds as share.py
    pF, pS = np.random.default_rng(7700 + i).permutation(768), np.random.default_rng(8800 + i).permutation(768)
    return {"F": dF, "S": dS, "R": centered(r), "F~": dF[pF], "S~": dS[pS], "F|": dF, "S|": dS}

def light_side(X, T, perm=None):
    """the turned light, exactly, then counted the machine's way; checks H H x = x, lengths, all-ones"""
    v, D = T["v"], T["D"]; Xo = obj(X); a = Xo.dot(v)
    Xp = D * Xo - 2 * np.multiply.outer(a, v)                 # D x H x, whole numbers
    back = D * Xp - 2 * np.multiply.outer(Xp.dot(v), v)       # D^2 x H H x
    c = dict(involution=int(sum(bool((back[p] == D * D * Xo[p]).all()) for p in range(len(X)))),
             length=int(((Xp * Xp).sum(1) == D * D * (Xo * Xo).sum(1)).sum()),
             allones=int((Xp.sum(1) == D * Xo.sum(1)).sum()), n=len(X))
    xr = ((2 * Xp + D) // (2 * D)).astype(np.int64)          # rounded to the stream's place, half up (as rshift does)
    return Xp, xr, c

def one_plate(k, only=None):
    t0 = time.time(); X = L[f"X|{k}"]; w = M[k + ".w"]; e = M[k + ".e"]; N = w.shape[1]; w64 = w.astype(np.int64)
    x16, ex = to16(X, -ES); _, _, LS = gauge_log(X)
    ph = ones(x16); PP = np.zeros(768, np.int64)
    for c0 in range(0, N, 4096): PP += ones(w[:, c0:c0 + 4096]).sum(1)
    base = dict(light_ph=int(ph[later].sum()), work=int((ph[later] @ PP).sum()), plate_ph=int(PP.sum()), weights=768 * N)
    json.dump(base, open(f"recording/turn/{k}__base.json", "w"))
    for name, d in directions(k).items():
        if only and name not in only: continue
        T = make_turn(d, bare=name.endswith("|"))
        if T is None: json.dump(None, open(f"recording/turn/{k}__{name}.json", "w")); continue
        v, D = T["v"], T["D"]
        # where d lands: H d against the target
        Hd = D * d - 2 * dot(v, d) * v                       # D x H d
        tgt = T["sg"] * T["C"] * T["t"] * D                  # D x 2^m x target
        left = Hd * (1 << MB) - tgt
        info = dict(k=T["k"], sg=T["sg"], D_bits=D.bit_length(), v_bits=max(int(x).bit_length() for x in v),
                    leftover=float(Fraction(dot(left, left), (D << MB) ** 2 * T["dd"])),
                    on_line=float(Fraction(int(Hd[T["k"]]) ** 2, dot(Hd, Hd))), v_dot_ones=int(sum(v) != 0))
        # the light
        Xp, xr, c = light_side(X, T)
        t16, tex = to16(xr, -ES); _, _, tLS = gauge_log(xr); tph = ones(t16)
        keep = rshift(xr, (ex + ES)[:, None])                  # at the unturned light's own places
        kph = ones(keep)
        info.update(checks=c, light_ph=int(tph[later].sum()), light_ph_keep=int(kph[later].sum()),
                    over16_keep=int((np.abs(keep[later]) > LIM).sum()),
                    shift_up=int((tex[later] > ex[later]).sum()), shift_down=int((tex[later] < ex[later]).sum()),
                    water_same=int(((tex[later] == ex[later]) & (tLS[later] == LS[later])).sum()), n=len(later))
        # the plate: H W per neuron, at the neuron's own 16-place scale
        b = meet_exact(v[None, :], w64)[0]                   # v . W per neuron, exact
        TP = np.zeros(768, np.int64); shifts = np.zeros(N, np.int64); exact_bits = []
        Q = np.zeros((768, N), np.int64) if not k.startswith("head") else None
        direct_ok = direct_n = 0; Xd = Xp[LAST3] if name in ("F", "S") and not k.startswith("head") else None
        for c0 in range(0, N, 1024):
            c1 = min(N, c0 + 1024); wc = w[:, c0:c1]
            Num = D * obj(wc) - np.multiply.outer(2 * v, b[c0:c1])      # D x (H W), whole numbers
            mx = np.abs(Num).max(0)
            s = [col_shift(int(m), D) for m in mx]; shifts[c0:c1] = s
            exact_bits += [int(m).bit_length() + 1 for m in mx]
            up = np.array([1 << max(-x, 0) for x in s], dtype=object); dn = np.array([D << max(x, 0) for x in s], dtype=object)
            q = ((2 * Num * up + dn) // (2 * dn)).astype(np.int64)
            assert np.abs(q).max() <= LIM
            TP += ones(q).sum(1)
            if Q is not None: Q[:, c0:c1] = q
            if Xd is not None:                               # direct: turned light met with the turned plate's numerators
                lhs = Xd.dot(Num); rhs = D * D * meet_exact(obj(X[LAST3]), wc)
                direct_ok += int((lhs == rhs).sum()); direct_n += lhs.size
        info.update(plate_ph=int(TP.sum()), work=int((tph[later] @ TP).sum()), work_keep=int((kph[later] @ TP).sum()),
                    cols_scale_up=int((shifts > 0).sum()), cols_scale_down=int((shifts < 0).sum()),
                    exact_bits_mean=float(np.mean(exact_bits)), direct_ok=direct_ok, direct_n=direct_n)
        if not k.startswith("head"):
            # exact pile check, every position of texts 0-2: (D H x) . (D H W) == D^2 (x . W)
            XpC = Xp[CHK3]; lhs = D * meet_exact(XpC, w64) - 2 * np.multiply.outer(XpC.dot(v), b)
            rhs = D * D * meet_exact(obj(X[CHK3]), w64)
            info.update(pile_ok=int((lhs == rhs).sum()), pile_n=int(lhs.size))
            # the 16-place turned machine's pile against the machine's pile (exact rationals): same value?
            if name in ("F", "S", "R"):
                p_t = t16[CHK3] @ Q; p_m = x16[CHK3].astype(np.int64) @ w64
                et = (tex[CHK3][:, None] + (e + shifts)[None, :]); em = (ex[CHK3][:, None] + e[None, :])
                lo = np.minimum(et, em)
                same = obj(p_t) * obj(np.left_shift(1, et - lo)) == obj(p_m) * obj(np.left_shift(1, em - lo))
                info.update(pile16_same=int(same.sum()), pile16_n=int(same.size))
        json.dump(info, open(f"recording/turn/{k}__{name}.json", "w"))
        print(f"   {k:8s} {name:3s} k={T['k']:3d} light {100 * info['light_ph'] / base['light_ph']:6.2f}%  plate {100 * info['plate_ph'] / base['plate_ph']:6.2f}%"
              f"  work {100 * info['work'] / base['work']:6.2f}%  checks {c}  pile {info.get('pile_ok', '-')}/{info.get('pile_n', '-')}"
              f"  direct {direct_ok}/{direct_n}  ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB)", flush=True)
    # order-free check: lines of light, plate and direction shuffled together (S turn), every count the same
    if k in ("qkv.6", "fc.6") and (not only or "S" in only):
        pm = np.random.default_rng(31).permutation(768); d = directions(k)["S"][pm]; T = make_turn(d, False)
        _, xr, _ = light_side(X[:, pm], T); t16, _ = to16(xr, -ES); tph = ones(t16)
        wp = w[pm]; b = meet_exact(T["v"][None, :], wp.astype(np.int64))[0]; TP = np.zeros(768, np.int64)
        for c0 in range(0, N, 1024):
            c1 = min(N, c0 + 1024); Num = T["D"] * obj(wp[:, c0:c1]) - np.multiply.outer(2 * T["v"], b[c0:c1])
            s = [col_shift(int(m), T["D"]) for m in np.abs(Num).max(0)]
            up = np.array([1 << max(-x, 0) for x in s], dtype=object); dn = np.array([T["D"] << max(x, 0) for x in s], dtype=object)
            TP += ones(((2 * Num * up + dn) // (2 * dn)).astype(np.int64)).sum(1)
        S0 = json.load(open(f"recording/turn/{k}__S.json"))
        of = bool(int(tph[later].sum()) == S0["light_ph"] and int(TP.sum()) == S0["plate_ph"] and int((tph[later] @ TP).sum()) == S0["work"])
        json.dump(of, open(f"recording/turn/{k}__order.json", "w"))
        print(f"   {k:8s} order-free (lines of light, plate, direction shuffled together): {'same counts' if of else 'DIFFERENT'}")

if ARGS and ARGS[0] != "report":
    print(DOC.split("PREDICTION")[1].split("    python3")[0].join(["PREDICTION", ""]), flush=True)
    for a in ARGS:
        k, _, sub = a.partition(":"); one_plate(k, sub.split(",") if sub else None)
    sys.exit()

# ---------------------------------------------------------------- report
R = {}
for k in PLATES:
    R[k] = dict(base=json.load(open(f"recording/turn/{k}__base.json")),
                dirs={n: json.load(open(f"recording/turn/{k}__{n}.json")) for n in DIRS})
    if os.path.exists(f"recording/turn/{k}__order.json"): R[k]["order_free"] = json.load(open(f"recording/turn/{k}__order.json"))
def pc(a, b): return f"{100 * a / b:6.2f}%"
print("THE EXACT TURN AND ITS LIT PIXELS (positions >= 1 of the 30 texts; counts at the machine's 16-place water level)")
print(DOC.split("PREDICTION")[1].split("    python3")[0].join(["PREDICTION", ""]))
print("\nWHERE d LANDS: the line k, |v| in places, D's places, d's energy on line k after the turn, the leftover (exact, printed)")
for k in PLATES:
    print(f"   {k:8s} " + "  ".join(f"{n}:" + ("none" if R[k]['dirs'][n] is None else
          f"k={R[k]['dirs'][n]['k']} on-line {100 * R[k]['dirs'][n]['on_line']:.2f}% left {R[k]['dirs'][n]['leftover']:.1e}") for n in ("F", "S", "F|", "S|")))
print("\nEXACT CHECKS")
tot = {q: [0, 0] for q in ("involution", "length", "allones", "pile", "direct", "order")}
for k in PLATES:
    for n, I in R[k]["dirs"].items():
        if I is None: continue
        c = I["checks"]; tot["involution"][0] += c["involution"]; tot["involution"][1] += c["n"]
        tot["length"][0] += c["length"]; tot["length"][1] += c["n"]
        if not n.endswith("|"): tot["allones"][0] += c["allones"]; tot["allones"][1] += c["n"]
        if "pile_ok" in I: tot["pile"][0] += I["pile_ok"]; tot["pile"][1] += I["pile_n"]
        tot["direct"][0] += I["direct_ok"]; tot["direct"][1] += I["direct_n"]
    if "order_free" in R[k]: tot["order"][0] += R[k]["order_free"]; tot["order"][1] += 1
bare_ones = sum(R[k]["dirs"][n]["checks"]["allones"] for k in PLATES for n in ("F|", "S|") if R[k]["dirs"][n])
bare_n = sum(R[k]["dirs"][n]["checks"]["n"] for k in PLATES for n in ("F|", "S|") if R[k]["dirs"][n])
print(f"   H H x == x (every position, every text, every plate, every turn): {tot['involution'][0]}/{tot['involution'][1]}")
print(f"   |H x|^2 == |x|^2: {tot['length'][0]}/{tot['length'][1]};   1 . H x == 1 . x (centered turns): {tot['allones'][0]}/{tot['allones'][1]}"
      f"   (bare-line turns: {bare_ones}/{bare_n}, expected to fail)")
print(f"   (H x) met with (H W) == x met with W, exact rationals, every position of texts 0-2, qkv.l and fc.l, every turn: {tot['pile'][0]}/{tot['pile'][1]} piles")
print(f"   the same, with the turned plate's numerators formed entry by entry, last position of texts 0-2, F and S: {tot['direct'][0]}/{tot['direct'][1]} piles")
print(f"   order-free (S turn, lines of light, plate and direction shuffled together; qkv.6, fc.6): {tot['order'][0]}/{tot['order'][1]} same counts")
print("\nLIGHT: photons of the turned light at the machine's 16-place water level, % of the unturned light's photons"
      "\n   (and kept at the unturned light's places; positions whose 16-place shift went up; positions whose water level is unchanged)")
for k in PLATES:
    B = R[k]["base"]; row = []
    for n in DIRS:
        I = R[k]["dirs"][n]
        row.append(f"{n}:{'  none ':>8s}" if I is None else f"{n}:{pc(I['light_ph'], B['light_ph'])}")
    print(f"   {k:8s} " + " ".join(row))
print("\n   S turn detail: photons at the machine's water level | at the unturned places | lines then over 16 places |"
      " shift up / down | water level unchanged")
for k in PLATES:
    B = R[k]["base"]; I = R[k]["dirs"]["S"]
    print(f"   {k:8s} {pc(I['light_ph'], B['light_ph'])} | {pc(I['light_ph_keep'], B['light_ph'])} | {I['over16_keep']:5d} |"
          f" {I['shift_up']:3d} / {I['shift_down']:3d} | {I['water_same']}/{I['n']}")
print("\nPLATE: photons per weight, unturned -> turned at the neuron's own 16-place scale; columns whose scale went up/down;"
      " places per weight to hold H W exactly over one shared D")
for k in PLATES:
    B = R[k]["base"]; row = []
    for n in ("F", "S", "R", "S~"):
        I = R[k]["dirs"][n]
        row.append(f"{n}: none" if I is None else f"{n}: {I['plate_ph'] / B['weights']:.3f} ({I['cols_scale_up']}/{I['cols_scale_down']}, exact {I['exact_bits_mean']:.0f} pl)")
    print(f"   {k:8s} unturned {B['plate_ph'] / B['weights']:.3f} | " + " | ".join(row))
print("\nWORK = sum over lines of light photons x plate photons, % of the unturned machine's work at the same plate")
print("   plate     " + "  ".join(f"{n:>8s}" for n in DIRS) + "   | S share (mean, from share.py)")
SH = json.load(open("recording/share_stats.json"))
tw = {n: [0, 0] for n in DIRS}
for k in PLATES:
    B = R[k]["base"]; row = []
    for n in DIRS:
        I = R[k]["dirs"][n]
        if I is None: row.append(f"{'none':>8s}"); continue
        row.append(f"{pc(I['work'], B['work']):>8s}"); tw[n][0] += I["work"]; tw[n][1] += B["work"]
    print(f"   {k:8s}  " + "  ".join(row) + f"   | {SH[k]['S'][0] * 100:6.2f}%")
print("   all      " + "  ".join(f"{pc(a, b):>8s}" for a, b in tw.values()) + "   (sum over plates that have the direction)")
print("\nTHE 16-PLACE TURNED MACHINE (preview for phase B): piles of the 16-place turned light met with the 16-place turned plate,"
      "\n   equal as rationals to the machine's own pile, every position of texts 0-2")
for k in PLATES:
    if k == "head": continue
    print(f"   {k:8s} " + "  ".join(f"{n}: " + ("none" if R[k]['dirs'][n] is None else f"{R[k]['dirs'][n]['pile16_same']}/{R[k]['dirs'][n]['pile16_n']}") for n in ("F", "S", "R")))
