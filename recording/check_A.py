"""INDEPENDENT CHECK OF PHASE A. My own light, directions, shares, exact turn and photon counts. Whole numbers and exact
rationals in the machine path; float only to print percentages.
  python3 recording/check_A.py light     the machine run on F, S (5 words) and the 30 refs texts, my own recording
  python3 recording/check_A.py share     sizes, stone vs word, shares at qkv/fc of layers 0, 6, 11 and the top read
  python3 recording/check_A.py turn      the exact turn and photon counts at those plates (F, S, R, F~, S~)
  python3 recording/check_A.py exact     (H x) met with (H W) == x met with W at every qkv.l, fc.l, 2 texts, by
                                         residues modulo primes whose product exceeds the piles' size (so exact)
Differences from the checked scripts, on purpose: the turn's length match uses 2^64 and the nearest whole C (not 2^48
and the floor); my own seeds for the random and shuffled controls; the light is centred explicitly (768 x - sum x)
in the share; the exact identity is proved by residues, not by 24-bit pieces; texts 5 and 17 for the identity.
"""
import sys, os, json, math, time, resource, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
STAGE = sys.argv[1]; ARGS = sys.argv[2:]
SCR = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
if STAGE != "light":
    i0 = src.index("W = {k: (M[k + \".w\"]"); i1 = src.index("\n", i0); src = src[:i0] + "W = {}" + src[i1:]
exec(src)
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
T0 = time.time()
def say(*a): print(*a, f"  [{time.time() - T0:.0f}s {gb():.2f}GB]", flush=True)

POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def photons(A):
    A = np.abs(np.asarray(A, dtype=np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += POP[A & 0xFFFF]; A = A >> 16
    return n
def O(a): return np.array([int(x) for x in np.asarray(a).ravel()], dtype=object).reshape(np.shape(a))
def cen(f): f = O(f); return 768 * f - int(f.sum())            # 768 x (f - its mean), whole numbers, lines sum to 0
def idot(a, b): return int(np.dot(O(a), O(b)))
PL = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["top"]
CHECK = ["qkv.0", "fc.0", "qkv.6", "fc.6", "qkv.11", "fc.11", "top"]
WORDS = {" The": 383, " the": 262, " In": 554, ".": 13, " Once": 4874}
LIGHT = f"{SCR}/check_light.npz"

# ------------------------------------------------------------------------------------------------ the light
if STAGE == "light":
    def run(R):
        seen = {}
        for l in range(12):
            seen[f"qkv.{l}"] = R
            R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            seen[f"fc.{l}"] = R
            R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        seen["top"] = R
        w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
        return seen, power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    out = {}
    F, _ = run(np.zeros((1, 768), np.int64))
    for k in PL: out[f"F|{k}"] = F[k][0]
    for s, i in WORDS.items():
        S, _ = run(M["wte"][[i]] + M["wpe"][:1])
        for k in PL: out[f"S{s}|{k}"] = S[k][0]
    refs = json.load(open("refs_many.json")); TX = [r["ids"] + r["ref"] for r in refs]
    X = {k: [] for k in PL}; pos = []; txt = []; right = n = 0
    for t, seq in enumerate(TX):
        seen, top = run(M["wte"][seq] + M["wpe"][:len(seq)])
        for k in PL: X[k].append(seen[k])
        pos += list(range(len(seq))); txt += [t] * len(seq)
        for p in range(len(refs[t]["ids"]) - 1, len(seq) - 1): right += race(top[p]) == seq[p + 1]; n += 1
    say(f"machine word == refs word at {right}/{n} judged positions")
    np.savez(LIGHT, pos=np.array(pos), txt=np.array(txt), **{f"X|{k}": np.concatenate(X[k]) for k in PL}, **out)
    T = np.load("recording/light.npz")
    same = sum(np.array_equal(np.concatenate(X[k]), T[f"X|{'head' if k == 'top' else k}"]) for k in PL)
    sameF = sum(np.array_equal(out[f"F|{k}"], T[f"F|{'head' if k == 'top' else k}"]) for k in PL)
    say(f"my light == recording/light.npz at {same}/25 plates (all 648 positions); my F == theirs {sameF}/25")
    sys.exit()

L = np.load(LIGHT); POS, TXT = L["pos"], L["txt"]; later = np.flatnonzero(POS >= 1)
def light(k): return L[f"X|{k}"]
def dirs(k):
    i = PL.index(k)
    dF, dS = cen(L[f"F|{k}"]), cen(L[f"S The|{k}"])
    r = cen(np.random.default_rng(424242 + i).integers(-(1 << 20), 1 << 20, 768))
    return {"F": dF, "S": dS, "R": r, "F~": dF[np.random.default_rng(515151 + i).permutation(768)],
            "S~": dS[np.random.default_rng(616161 + i).permutation(768)]}

# ------------------------------------------------------------------------------------------------ sizes and shares
if STAGE == "share":
    print("SIZES (stream units = whole numbers / 2^16) and the energy share of the largest line")
    for k in CHECK + ["qkv.1", "qkv.3", "qkv.9"]:
        row = []
        for n, key in (("F", "F"), ("S", "S The")):
            f = O(L[f"{key}|{k}"]); ff = int(np.dot(f, f))
            if ff == 0: row.append(f"{n}: 0"); continue
            j = int(np.argmax(np.abs(L[f"{key}|{k}"])))
            row.append(f"{n}: |{n}|={math.sqrt(ff) / 65536:8.2f} line {j:3d} holds {100 * float(Fraction(int(f[j]) ** 2, ff)):5.1f}%")
        dF, dS = cen(L[f"F|{k}"]), cen(L[f"S The|{k}"]); a, b = idot(dF, dF), idot(dS, dS)
        cfs = "-" if a == 0 else f"{math.copysign(math.sqrt(float(Fraction(idot(dF, dS) ** 2, a * b))), idot(dF, dS)):+.3f}"
        print(f"  {k:7s} " + " | ".join(row) + f" | cos(F,S) centred {cfs}")
    print("THE STONE AGAINST THE WORD (centred; same-sign lines %, |S_w - S_The| / |S_The| %)")
    for k in ("qkv.0", "qkv.6", "fc.11", "top"):
        dS = cen(L[f"S The|{k}"]); nn = idot(dS, dS); row = []
        for w in (" the", " In", ".", " Once"):
            dw = cen(L[f"S{w}|{k}"]); same = int(sum((x > 0) == (y > 0) for x, y in zip(dw, dS)))
            row.append(f"{w!r}: {100 * same / 768:5.1f}% {100 * math.sqrt(float(Fraction(idot(dw - dS, dw - dS), nn))):6.2f}%")
        print(f"  {k:7s} " + "  ".join(row))
    print(f"\nSHARE of the centred light's energy along d, exact (Fractions), positions >= 1 ({len(later)}): mean / median / 90th %")
    ST = {}
    for k in CHECK:
        X = O(light(k)[later]); Xc = 768 * X - X.sum(1)[:, None]; xx = (Xc * Xc).sum(1); row = []
        ST[k] = {}
        for n, d in dirs(k).items():
            dd = idot(d, d)
            if dd == 0: row.append(f"{n}: none"); continue
            xd = Xc.dot(d); s = sorted(Fraction(int(a) ** 2, int(b) * dd) for a, b in zip(xd, xx)); m = len(s)
            st = (sum(s) / m, (s[m // 2 - 1] + s[m // 2]) / 2, s[math.ceil(0.9 * m) - 1]); ST[k][n] = [float(x) for x in st]
            row.append(f"{n}: {100 * float(st[0]):6.2f} {100 * float(st[1]):6.2f} {100 * float(st[2]):6.2f}")
        say(f"  {k:7s} " + " | ".join(row))
    # order-free: shuffle lines of light and direction together (S at qkv.6)
    pm = np.random.default_rng(3).permutation(768); X = O(light("qkv.6")[later]); Xc = 768 * X - X.sum(1)[:, None]
    d = dirs("qkv.6")["S"]; dd = idot(d, d)
    a = [Fraction(int(p) ** 2, int(q) * dd) for p, q in zip(Xc.dot(d), (Xc * Xc).sum(1))]
    b = [Fraction(int(p) ** 2, int(q) * dd) for p, q in zip(Xc[:, pm].dot(d[pm]), (Xc[:, pm] * Xc[:, pm]).sum(1))]
    print(f"  order-free (qkv.6, S, lines of light and direction shuffled together): {'same shares' if a == b else 'DIFFERENT'}")
    json.dump(ST, open(f"{SCR}/check_share.json", "w"))
    sys.exit()

# ------------------------------------------------------------------------------------------------ the turn
MB, LIM = 64, 32767
def make_turn(d, mb=MB):
    di = [int(x) for x in d]; dd = sum(x * x for x in di)
    if dd == 0: return None
    k = max(range(768), key=lambda i: abs(di[i])); sg = 1 if di[k] > 0 else -1
    tt = 768 * 767; Q = dd << (2 * mb)                       # C / 2^mb ~ |d| / |t|, C the nearest whole number
    C = math.isqrt(Q // tt)
    while (2 * C + 1) ** 2 * tt <= 4 * Q: C += 1
    while C > 0 and (2 * C - 1) ** 2 * tt > 4 * Q: C -= 1
    t = np.full(768, -1, dtype=object); t[k] = 767
    v = O(di) * (1 << mb) - sg * C * t
    assert int(v.sum()) == 0                                 # keeps the all-ones direction fixed
    D = int(np.dot(v, v)); Hd = D * O(di) - 2 * int(np.dot(v, O(di))) * v
    return dict(v=v, D=D, k=k, on_line=float(Fraction(int(Hd[k]) ** 2, int(np.dot(Hd, Hd)))), t=t)

def turned_light(X, T):
    v, D = T["v"], T["D"]; Xo = O(X); a = Xo.dot(v)
    Num = D * Xo - 2 * np.multiply.outer(a, v)               # D x (H x)
    return Num, ((2 * Num + D) // (2 * D)).astype(np.int64)  # rounded, half up, at the stream's place

def fits(m, s, LD): return m <= (LD << s) if s >= 0 else (m << -s) <= LD
def scale(m, LD):                                            # smallest s with m / 2^s <= 32767 x D
    if m == 0: return 0
    s = m.bit_length() - LD.bit_length()
    while not fits(m, s, LD): s += 1
    while fits(m, s - 1, LD): s -= 1
    return s

def turned_plate(w, T, chunk=1024):
    v, D = T["v"], T["D"]; LD = LIM * D; N = w.shape[1]; PP = np.zeros(768, np.int64); fin = coa = 0; xb = []
    for c0 in range(0, N, chunk):
        wc = O(w[:, c0:c0 + chunk]); b = v.dot(wc)
        Num = D * wc - 2 * np.multiply.outer(v, b)           # D x (H W) for these neurons
        mx = np.abs(Num).max(0); s = [scale(int(m), LD) for m in mx]
        fin += sum(x < 0 for x in s); coa += sum(x > 0 for x in s); xb += [int(m).bit_length() for m in mx]
        up = O([1 << max(-x, 0) for x in s]); dn = O([D << max(x, 0) for x in s])
        q = ((2 * Num * up + dn) // (2 * dn)).astype(np.int64)
        assert np.abs(q).max() <= LIM
        PP += photons(q).sum(1)
    return PP, fin, coa, float(np.mean(xb))

if STAGE == "turn":
    which = ARGS[0].split(",") if ARGS else CHECK
    print("THE TURN (m = 64, nearest C) at the machine's 16-place water level, positions >= 1; % of the unturned machine")
    RES = {}
    for k in which:
        X = light(k); wn = "head" if k == "top" else k; w = M[wn + ".w"]
        x16, ex = to16(X, -ES); lph = photons(x16)[later]
        PP0 = np.zeros(768, np.int64)
        for c0 in range(0, w.shape[1], 4096): PP0 += photons(w[:, c0:c0 + 4096]).sum(1)
        s0 = [scale(int(m), LIM) for m in np.abs(w.astype(np.int64)).max(0)]
        base = dict(light=int(lph.sum()), plate=int(PP0.sum()), work=int((lph @ PP0).sum()), nw=768 * w.shape[1])
        say(f"  {k:7s} unturned: plate photons/weight {base['plate'] / base['nw']:.3f} (stored columns already at their"
            f" own scale: {sum(x == 0 for x in s0)}/{len(s0)})")
        RES[k] = dict(base=base)
        for n, d in dirs(k).items():
            T = make_turn(d)
            if T is None: say(f"  {k:7s} {n:3s} none"); continue
            Num, xr = turned_light(X, T)
            t16, tex = to16(xr, -ES); tph = photons(t16)[later]
            keep = photons(rshift(xr, (ex + ES)[:, None]))[later]
            up = int((tex[later] > ex[later]).sum())
            PP, fin, coa, xb = turned_plate(w, T)
            r = dict(light=int(tph.sum()), keep=int(keep.sum()), plate=int(PP.sum()), work=int((tph @ PP).sum()), up=up,
                     fin=fin, coa=coa, exact_places=xb + 1, Dbits=T["D"].bit_length(), k=T["k"],
                     on_line=T["on_line"])
            RES[k][n] = r
            say(f"  {k:7s} {n:3s} line {T['k']:3d} (d on it after: {100 * T['on_line']:.2f}%)  light {100 * r['light'] / base['light']:7.2f}%"
                f" (kept places {100 * r['keep'] / base['light']:7.2f}%, water level up one place at {up}/{len(later)})"
                f"  plate/wt {base['plate'] / base['nw']:.3f}->{r['plate'] / base['nw']:.3f} (finer {fin}, coarser {coa})"
                f"  WORK {100 * r['work'] / base['work']:7.2f}%   D {r['Dbits']} places, exact plate {xb + 1:.0f} places/wt")
    json.dump(RES, open(f"{SCR}/check_turn_{'_'.join(which)}.json", "w"))
    sys.exit()

# ------------------------------------------------------------------------------------------------ exact identity by residues
if STAGE == "exact":
    def isprime(n): return n > 1 and all(n % p for p in range(2, math.isqrt(n) + 1))
    P = []; c = (1 << 26) - 1
    while len(P) < 24:
        if isprime(c): P.append(c)
        c -= 2
    rows = np.flatnonzero((TXT == 5) | (TXT == 17)); ok = tot = 0; direct_ok = direct_n = 0
    which = ARGS[0].split(",") if ARGS else PL[:-1]
    print(f"rows: {len(rows)} positions of texts 5 and 17")
    for k in which:
        X = light(k)[rows]; w = M[k + ".w"].astype(np.int64)
        for n in (ARGS[1].split(",") if len(ARGS) > 1 else ("F", "S", "R")):
            T = make_turn(dirs(k)[n])
            if T is None: continue
            v, D = T["v"], T["D"]; vm = max(abs(int(x)) for x in v); xm = int(np.abs(X).max())
            bnd_l = 768 * (D * xm + 2 * 768 * xm * vm * vm) * (D * LIM + 2 * vm * 768 * vm * LIM)
            bnd_r = D * D * 768 * xm * LIM; need = bnd_l + bnd_r; prod = 1; use = []
            for p in P:
                use.append(p); prod *= p
                if prod > 2 * need: break
            assert prod > 2 * need, "not enough primes"
            good = np.ones((len(rows), w.shape[1]), bool)
            for p in use:
                vp = np.array([int(x) % p for x in v], np.int64); Dp = D % p
                Xp = X % p; a = (Xp @ vp) % p
                XN = (Dp * Xp - 2 * np.outer(a, vp)) % p           # (D H x) mod p
                wp = w % p; bp = (vp @ wp) % p
                WN = (Dp * wp - 2 * np.outer(vp, bp)) % p          # (D H W) mod p
                good &= ((XN @ WN) % p) == ((Dp * Dp % p) * ((Xp @ wp) % p)) % p
            ok += int(good.sum()); tot += good.size
            # direct, whole numbers, last row of each text, first 48 neurons
            Num, _ = turned_light(X[[-1]], T); wc = O(w[:, :48]); WNum = D * wc - 2 * np.multiply.outer(v, v.dot(wc))
            direct_ok += int((Num.dot(WNum) == D * D * O(X[[-1]]).dot(wc)).sum()); direct_n += 48
        say(f"  {k:7s} piles equal so far {ok}/{tot}  (primes used {len(use)}, D {D.bit_length()} places)  direct {direct_ok}/{direct_n}")
    print(f"(H x) met with (H W) == x met with W, exact (residues, product of primes > twice the largest pile): {ok}/{tot}; direct whole numbers {direct_ok}/{direct_n}")
    # how long the exact turned plate must be depends on how finely the turn lands d on its line
    if len(ARGS) < 3: sys.exit()
    print("\nHOW LONG THE EXACT TURN IS: D's places for the S turn at qkv.6, by how finely v is written (d on its line after)")
    d = dirs("qkv.6")["S"]
    for mb in (64, 48, 16, 0, -12, -20, -24):
        dr = d if mb >= 0 else O([(int(x) + (1 << (-mb - 1))) >> -mb for x in d])
        if mb < 0: dr[int(np.argmax(np.abs(dr)))] -= int(dr.sum())        # lines sum to 0 again
        T = make_turn(dr, max(mb, 0))
        v, D = T["v"], T["D"]; Hd = D * d - 2 * int(np.dot(v, d)) * v       # where the TRUE d lands
        onl = float(Fraction(int(Hd[T["k"]]) ** 2, int(np.dot(Hd, Hd))))
        print(f"  m = {mb:4d}: v {max(abs(int(x)) for x in v).bit_length():3d} places, D {D.bit_length():4d} places,"
              f" exact plate about {D.bit_length() + 16} places/wt; d on line {T['k']} after the turn {100 * onl:.4f}% (767/768 = 99.8698%)")

# ------------------------------------------------------------------------------------------------ extras
if STAGE == "own":                                           # each text's own position-0 stream as the direction
    first = {int(TXT[i]): i for i in np.flatnonzero(POS == 0)}
    for k in ARGS[0].split(","):
        X = light(k); s = []
        for p in later:
            d = cen(X[first[int(TXT[p])]]); x = O(X[p]); xc = 768 * x - int(x.sum())
            s.append(Fraction(int(np.dot(xc, d)) ** 2, int(np.dot(xc, xc)) * int(np.dot(d, d))))
        s.sort(); m = len(s)
        print(f"  OWN {k:7s} mean {100 * float(sum(s) / m):6.2f}  median {100 * float((s[m // 2 - 1] + s[m // 2]) / 2):6.2f}  90th {100 * float(s[math.ceil(0.9 * m) - 1]):6.2f}")
if STAGE == "totals":                                        # my work totals against the checked run's, same plates
    mine = {}
    for f in os.listdir(SCR):
        if f.startswith("check_turn_") and f.endswith(".json"): mine.update(json.load(open(f"{SCR}/{f}")))
    for n in ("S", "F", "R", "F~", "S~"):
        for grp, ks in (("qkv,fc", [k for k in mine if k != "top"]), ("with top", list(mine))):
            a = sum(mine[k][n]["work"] for k in ks if n in mine[k]); b = sum(mine[k]["base"]["work"] for k in ks if n in mine[k])
            ta = sum(json.load(open(f"recording/turn/{'head' if k == 'top' else k}__{n}.json"))["work"] for k in ks if n in mine[k])
            print(f"  {n:3s} {grp:9s} plates {sorted(ks)}: mine {100 * a / b:7.2f}%   theirs on the same plates {100 * ta / b:7.2f}%")
