"""LAYER 0 AS A MODEL OF THE EXTRA (pilot). The plates never change. Everything entering layer 0 is a combination of
fixed rows: a token's light is its word's entry row + its position's row. So each row's meeting with layer 0's plates
is stored once (one whole-number tip per neuron, ups and downs already cancelled), and per token only what is NOT
stored meets the plate.

The machine's water level rounds the light BEFORE the meeting: x16 = rshift(word + position, s), s picked per position
from the OR of the 768 magnitudes. Rounding is not additive, so the stored rows are the word's and the position's rows
rounded at that same s, their tips stored for every s, and the rounded light = word_s + position_s + a CARRY (-1, 0
or +1 per line). The carry is the only light that still meets the plate.

Versions (entry / proj / neurons-in):
  V0  plain machine, counted
  V1  stored tips, no carry                        / plain                   / stored parts + extra
  V2  tips of the unrounded rows, pile rounded     / plain                   / stored parts + extra
  V3  stored tips + carry                          / stored proj(value) tips, mixed with the machine's own shares,
                                                     one divide per head after the mix              / stored parts + extra
  V4  stored tips + carry                          / V3 + the rounding remainders meeting the plate  / stored parts + extra
  V5  stored tips + carry                          / plain                   / stored parts + extra
  V6  stored tips + carry                          / own value stored, the rest meets / stored parts + extra
  V7  (work only) V3 with neurons-in's mix also taken through stored fc(proj(value)) tips mixed by the shares
"neurons-in: stored parts + extra" = the rounded light at fc.0 split into word_s + (position + proj bias)_s, both
stored with their tips, + the extra (the arrival's part and the carries), which meets the plate.

Checks: every pile and the stream after layer 0 against the plain machine (np.array_equal) on all 30 refs texts (every
position of prompt + reference words); random-order proof; photons per token for layer 0 each way and for the whole
machine, with a shuffled-word control; judge5 for the versions named.
    python3 extra/extra_layer0.py                    # exactness + counts for every version
    python3 extra/extra_layer0.py judge=V5 n=300     # judge5 (top 5) for a version
    python3 extra/extra_layer0.py windows=V5,V6      # every pile and the stream after layer 0 on all 300 judge windows
Results (2026-10-04): exact = V4, V5, V6 (30/30 texts, 300/300 windows, judge5 6000/6000 and 6000/6000); V1 6000/5996,
V2 6000/5999, V3 6000/6000 but streams differ. Photons per token: whole machine 3,666,596,259; plain layer 0 219,153,343;
V5 157,188,684 (-1.69% of the whole); V6 155,565,994 (-1.73%); V4 229,156,068 (+0.27%); V3 152,966,576 (grows with position).
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every product is checked whole below 2^53
HERE = "/Users/abundancemachine/streaming-machine"
ARGS = dict(a.split("=") for a in sys.argv[1:]); sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
W, M, ES, SC, ARR, EXP2 = J.W, J.M, J.ES, J.SC, J.ARR, J.EXP2
rshift, to16, product, gauge_log, power, log2fix, race = J.rshift, J.to16, J.product, J.gauge_log, J.power, J.log2fix, J.race
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]

# ---------- photons ----------
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):                                         # photons (1s) per value, sign apart
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
def ones_obj(A):                                     # the same for Python whole numbers of any width
    return np.array([bin(abs(int(v))).count("1") for v in np.asarray(A).ravel()], np.int64).reshape(np.shape(A))
WROW = {}                                            # photons of each plate line, summed over its neurons
for name in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    w = M[name + ".w"]; WROW[name] = np.concatenate([ones(w[i:i + 64]).sum(1) for i in range(0, w.shape[0], 64)])
def E_ones(LS): return ones(EXP2[np.bitwise_and(-np.asarray(LS), 0xFFFF)])   # photons of the shared gain 2^fraction

# ---------- the store: only the words and positions the texts use ----------
WORDS = sorted({t for w in J.WINDOWS for t in w["ids"]} | {t for t in sum(TEXTS, [])})
IDX = np.full(50257, -1, np.int64); IDX[WORDS] = np.arange(len(WORDS))
WTE = M["wte"][WORDS].copy(); PMAX = 24; WPE = M["wpe"][:PMAX].copy(); BPROJ = W["proj.0"][2]
PLATE_INT = {n: M[n + ".w"].astype(np.int64) for n in ("qkv.0", "fc.0")}
_STORE = {}
def exact_meet(X, name):                             # stored tips: whole-number meetings, every partial sum below 2^53
    w = W[name][0]; assert (np.abs(X).astype(np.float64) @ np.abs(w)).max() < 2 ** 52
    return product(X, w)
def store(kind, s):
    key = (kind, int(s))
    if key not in _STORE:
        if kind == "A": v = rshift(WTE, s)                                   # word rows rounded at s
        elif kind == "B": v = rshift(WPE, s)                                 # position rows rounded at s
        elif kind == "B1": v = rshift(WPE + BPROJ[None, :], s)               # position + proj bias, rounded at s
        elif kind == "TQ": v = exact_meet(store("A", s), "qkv.0")
        elif kind == "PQ": v = exact_meet(store("B", s), "qkv.0")
        elif kind == "TF": v = exact_meet(store("A", s), "fc.0")
        elif kind == "PF": v = exact_meet(store("B1", s), "fc.0")
        elif kind == "TQfull": v = exact_meet(WTE, "qkv.0")                  # full resolution (V2)
        elif kind == "PQfull": v = exact_meet(WPE, "qkv.0")
        _STORE[key] = v
    return _STORE[key]
def take(kind, s, rows):                             # per position: the stored row at that position's own s
    out = None
    for sv in np.unique(s):
        m = s == sv; t = store(kind, sv)
        if out is None: out = np.zeros((len(rows), t.shape[1]), np.int64)
        out[m] = t[rows[m]]
    return out

# ---------- counted pieces of the machine (each mirrors base2.py step for step) ----------
class Count(dict):                                   # photons: total per step, and per position index
    def __init__(self): super().__init__(); self.bypos = {}
    def add(self, k, v):
        v = np.asarray(v, np.int64); self[k] = self.get(k, 0) + int(v.sum())
        bp = self.bypos.setdefault(k, np.zeros(PMAX, np.int64)); bp[:len(v)] += v
def water(x16):                                      # the gauge's sums: S1 adds, S2 squares, 768 S2 - S1^2
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
def gauged(R, name, C, tag, pile_fn=None):
    w, we, b = W[name]; x16, ex, LS = gauge_log(R); C.add(f"{tag} water level", water(x16))
    if pile_fn is None: pile = product(x16, w); C.add(f"{tag} meeting", ones(x16) @ WROW[name])
    else: pile = pile_fn(x16, ex, C)
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    C.add(f"{tag} divide", ones(pile).sum(1) * E_ones(LS)); C.add(f"{tag} bias", np.full(len(R), ones(b).sum()))
    return power(pile, F) + b, pile, x16, ex
def plain(X, name, C, tag):
    w, we, b = W[name]; x16, ex = to16(X, -ES); pile = product(x16, w)
    C.add(f"{tag} meeting", ones(x16) @ WROW[name]); C.add(f"{tag} bias", np.full(len(X), ones(b).sum()))
    return rshift(pile, -(we[None, :] + ex[:, None] + ES)) + b, pile, x16, ex
def arrival(QKV, C, tag, need_o=True):               # J.attention step for step, keeping the shares and the mixes
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64); det = []
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        C.add(f"{tag} scores", np.tril(ones(q16) @ ones(k16).T).sum(1))
        A = np.zeros((P, P), np.int64); L = np.zeros(P, np.int64); NUM = np.zeros((P, 64), np.int64); ov = ones(v).sum(1)
        mix, tot, div = np.zeros(P, np.int64), np.zeros(P, np.int64), np.zeros(P, np.int64)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]; Lp = log2fix([a.sum()])[0]
            out[p, h*64:(h+1)*64] = power(num, -Lp)
            A[p, :p+1] = a; L[p] = Lp; NUM[p] = num
            mix[p] = (ones(a) * ov[:p+1]).sum(); tot[p] = ones(a).sum(); div[p] = ones(num).sum() * E_ones([Lp])[0]
        C.add(f"{tag} shares' total", tot)
        if need_o: C.add(f"{tag} mix (shares x values)", mix); C.add(f"{tag} divide", div)
        det.append((A, L, NUM))
    return out, det
def bend(H, C, tag):
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, J.PHI[np.clip(idx, 0, 65535)]))
    C.add(f"{tag} bend", (ones(H) * ones(share)).sum(1)); return rshift(H * share, 16)

# ---------- whole numbers wider than 64 bits for the mixes of V3/V4 ----------
def split_meet(A, X):                                # A @ X exactly, X wide: X = hi 2^21 + lo, every partial sum below 2^53
    X = np.asarray(X, np.int64); lo = np.bitwise_and(X, (1 << 21) - 1); hi = np.right_shift(X, 21)
    for part in (lo, hi): assert (np.abs(A).astype(np.float64) @ np.abs(part).astype(np.float64)).max() < 2 ** 52
    return (A.astype(np.float64) @ hi.astype(np.float64)).astype(np.int64).astype(object) * (1 << 21) + \
           (A.astype(np.float64) @ lo.astype(np.float64)).astype(np.int64).astype(object)
def obj_rshift(X, s):                                # X 2^-s rounded to nearest, s > 0 per element, any width
    s = np.asarray(s, np.int64); assert (s > 0).all()
    half = np.array([1 << (int(t) - 1) for t in s.ravel()], object).reshape(s.shape)
    return (X + half) >> s.astype(object)

# ---------- layer 0, every version ----------
VERSIONS = {"V0": ("plain", "plain", "plain"), "V1": ("nocarry", "plain", "extra"), "V2": ("fullres", "plain", "extra"),
            "V3": ("carry", "tips", "extra"), "V4": ("carry", "remainder", "extra"), "V5": ("carry", "plain", "extra"),
            "V6": ("carry", "ownref", "extra")}
def layer0(ids, version, C=None, shuffle_rows=None, chained=None):
    entry, projw, fcw = VERSIONS[version]; C = Count() if C is None else C
    P = len(ids); wi = IDX[np.asarray(ids)]; assert (wi >= 0).all(); pos = np.arange(P)
    a, b = WTE[wi], WPE[:P]; R = a + b; C.add("entry: word + position", ones(b).sum(1))
    wq = W["qkv.0"][0]; got = {}
    def qkv_pile(x16, ex, C):
        s = ex + ES; got["s entry"] = s
        if entry == "nocarry":
            t1, t2 = take("TQ", s, wi), take("PQ", s, pos); C.add("qkv stored tips", ones(t1).sum(1) + ones(t2).sum(1)); return t1 + t2
        if entry == "fullres":
            t1, t2 = store("TQfull", 0)[wi], store("PQfull", 0)[pos]; C.add("qkv stored tips", ones(t1).sum(1) + ones(t2).sum(1))
            return rshift(t1 + t2, s[:, None])
        A_, B_ = take("A", s, wi), take("B", s, pos); c = x16 - A_ - B_; got["carry"] = c
        t1, t2 = take("TQ", s, wi), take("PQ", s, pos)
        C.add("qkv carry split", ones(A_).sum(1) + ones(B_).sum(1)); C.add("qkv stored tips", ones(t1).sum(1) + ones(t2).sum(1))
        C.add("qkv meeting", ones(c) @ WROW["qkv.0"])
        if shuffle_rows is not None:                 # control: the same light split with a random other word's row
            C.add("CONTROL qkv meeting, shuffled word", ones(x16 - rshift(shuffle_rows, s[:, None]) - B_) @ WROW["qkv.0"])
        return t1 + t2 + product(c, wq)
    QKV, got["qkv pile"], x16, ex = gauged(R, "qkv.0", C, "qkv", None if entry == "plain" else qkv_pile)
    got["qkv"] = QKV; got["x16"] = x16
    o, det = arrival(QKV, C, "arrival", need_o=(projw != "tips")); got["o"] = o
    wp, wpe_, bp = W["proj.0"]
    if projw == "plain":
        att, got["proj pile"], o16, eo = plain(o, "proj.0", C, "proj")
    elif projw == "ownref":
        o16, eo = to16(o, -ES); so = eo + ES; ref = rshift(QKV[:, 1536:], so[:, None])     # the pair's own value, at o's s
        tip = product(ref, wp)                                                              # stored per (word, position, s)
        C.add("proj stored tips", ones(tip).sum(1)); C.add("proj split", ones(ref).sum(1))
        C.add("proj meeting", ones(o16 - ref) @ WROW["proj.0"])
        got["proj pile"] = tip + product(o16 - ref, wp)
        att = rshift(got["proj pile"], -(wpe_[None, :] + eo[:, None] + ES)) + bp; C.add("proj bias", np.full(P, ones(bp).sum()))
    else:                                            # V3 / V4: stored proj(value) tips, mixed with the machine's own shares
        v = QKV[:, 1536:]; Y = []; PVS = []          # Y_h = sum_j a_j PV_j,h   (PV_j,h = v_j,h @ wp_h, stored per (word, position))
        for h in range(12):
            A, L, NUM = det[h]; vh, wh = v[:, h*64:(h+1)*64], wp[h*64:(h+1)*64]
            assert (np.abs(vh).astype(np.float64) @ np.abs(wh)).max() < 2 ** 52
            PV = product(vh, wh); PVS.append(PV); Y.append(split_meet(A, PV)); oPV = ones(PV).sum(1)
            C.add("proj tips mix (shares x stored proj(value))", np.array([(ones(A[p, :p+1]) * oPV[:p+1]).sum() for p in range(P)]))
        if projw == "tips":                          # one divide per neuron per head after the mix; heads added; bias
            att = np.zeros((P, 768), object); dv = np.zeros(P, np.int64); hs = np.zeros(P, np.int64)
            for h in range(12):
                L = det[h][1]
                for p in range(P):
                    Fv = wpe_ * 65536 - int(L[p]); E = int(EXP2[(-int(L[p])) & 0xFFFF])
                    term = obj_rshift(Y[h][p] * E, 16 - (Fv >> 16))
                    att[p] = att[p] + term; dv[p] += ones_obj(Y[h][p]).sum() * ones(E); hs[p] += ones_obj(term).sum()
            C.add("proj divide (per head)", dv); C.add("proj heads added", hs); C.add("proj bias", np.full(P, ones(bp).sum()))
            att = (att + bp.astype(object)).astype(np.int64); got["proj pile"] = None
            if chained is not None:                  # V7, work only: fc(proj(value)) stored per pair, mixed with the shares
                wf_ = W["fc.0"][0]; mixc = np.zeros(P, np.int64)
                for h in range(12):
                    PV = PVS[h]; lo = np.bitwise_and(PV, (1 << 21) - 1); hi = np.right_shift(PV, 21)
                    for part in (lo, hi): assert (np.abs(part).astype(np.float64) @ np.abs(wf_)).max() < 2 ** 52
                    rl, rh = (lo.astype(np.float64) @ wf_).astype(np.int64), (hi.astype(np.float64) @ wf_).astype(np.int64)
                    assert np.abs(rh).max() < 2 ** 40
                    PFV = rh * (1 << 21) + rl; oP = ones(PFV).sum(1); A = det[h][0]
                    mixc += np.array([(ones(A[p, :p+1]) * oP[:p+1]).sum() for p in range(P)])
                chained.add("fc tips mix (shares x stored fc(proj(value))), divides not counted", mixc)
        else:                                        # V4, exact: o16 = (num E + H - r) / 2^K, so o16 @ w = (E Y + H colsum - r @ w) / 2^K
            o16, eo = to16(o, -ES); so = eo + ES; pile = np.zeros((P, 768), object)
            ey = np.zeros(P, np.int64); rm = np.zeros(P, np.int64); ad = np.zeros(P, np.int64); got["K"] = []; swap = True
            for h in range(12):
                L, NUM = det[h][1], det[h][2]; wh = PLATE_INT_proj[h*64:(h+1)*64]; colsum = wh.sum(0).astype(object)
                for p in range(P):
                    F = -int(L[p]); I = F >> 16; E = int(EXP2[F & 0xFFFF]); k = 16 - I; sh = int(so[p]); assert k > 0
                    K = k + sh; Hh = (1 << (k - 1)) + ((1 << (sh - 1)) << k if sh > 0 else 0); assert K < 62; got["K"].append(K)
                    acc = np.zeros(768, object)                       # the mix of stored tips, earlier tokens in a random order ...
                    for j in np.random.default_rng(1000 * h + p).permutation(p + 1): acc = acc + int(det[h][0][p, j]) * PVS[h][j].astype(object)
                    swap &= all(acc == Y[h][p]) and all(Y[h][p] == split_meet(wh.T, NUM[p][:, None])[:, 0])   # ... = the machine's mix, then met
                    r = np.array([(int(x) * E + Hh) % (1 << K) for x in NUM[p]], np.int64)       # the low K digits: the remainders
                    tot = E * Y[h][p] + Hh * colsum - split_meet(wh.T, r[:, None])[:, 0]
                    assert all(int(t) & ((1 << K) - 1) == 0 for t in tot)
                    pile[p] = pile[p] + (tot >> K)
                    ey[p] += ones_obj(Y[h][p]).sum() * ones(E); rm[p] += (ones(r) * WROW["proj.0"][h*64:(h+1)*64]).sum(); ad[p] += ones(wh.sum(0)).sum()
            C.add("proj gain x mix", ey); C.add("proj remainders meeting", rm); C.add("proj stored column sums", ad); got["swap"] = swap
            got["proj pile"] = pile.astype(np.int64); att = rshift(got["proj pile"], -(wpe_[None, :] + eo[:, None] + ES)) + bp
            C.add("proj bias", np.full(P, ones(bp).sum()))
    got["att"] = att; R1 = R + att; C.add("stream add (arrival)", ones(att).sum(1))
    wf = W["fc.0"][0]
    def fc_pile(x1, e1, C):
        s1 = e1 + ES; got["s fc"] = s1; A_, B_ = take("A", s1, wi), take("B1", s1, pos); m = x1 - A_ - B_; got["fc extra"] = m
        t1, t2 = take("TF", s1, wi), take("PF", s1, pos)
        C.add("fc split", ones(A_).sum(1) + ones(B_).sum(1)); C.add("fc stored tips", ones(t1).sum(1) + ones(t2).sum(1))
        C.add("fc meeting", ones(m) @ WROW["fc.0"])
        if shuffle_rows is not None:
            C.add("CONTROL fc meeting, shuffled word", ones(x1 - rshift(shuffle_rows, s1[:, None]) - B_) @ WROW["fc.0"])
        return t1 + t2 + product(m, wf)
    H, got["fc pile"], x1, e1 = gauged(R1, "fc.0", C, "fc", None if fcw == "plain" else fc_pile); got["fc"] = H; got["x1"] = x1
    mo, got["mproj pile"], _, _ = plain(bend(H, C, "fc"), "mproj.0", C, "mproj")
    R2 = R1 + mo; C.add("stream add (neurons)", ones(mo).sum(1)); got["stream"] = R2
    return R2, got, C
PLATE_INT_proj = M["proj.0.w"].astype(np.int64)

def rest(R, C=None, first=None):                     # layers 1..11 and the read at the top (counted when C is given)
    for l in range(1, 12):
        if C is None:
            R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        else:
            QKV = gauged(R, f"qkv.{l}", C, "L1-11 qkv")[0]; o, _ = arrival(QKV, C, "L1-11 arrival")
            att = plain(o, f"proj.{l}", C, "L1-11 proj")[0]; R = R + att; C.add("L1-11 stream adds", ones(att).sum(1))
            H = gauged(R, f"fc.{l}", C, "L1-11 fc")[0]; mo = plain(bend(H, C, "L1-11 fc"), f"mproj.{l}", C, "L1-11 mproj")[0]
            R = R + mo; C.add("L1-11 stream adds", ones(mo).sum(1))
    w, we, b = W["head"]; Rj = R if first is None else R[first[0]:first[1]]
    x16, ex, LS = gauge_log(Rj); pile = product(x16, w)
    if C is not None:
        C.add("top water level", water(x16)); C.add("top meeting", ones(x16) @ WROW["head"])
        C.add("top divide", ones(pile).sum(1) * E_ones(LS)); C.add("top standing amounts", np.full(len(Rj), ones(b).sum()))
    return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b

def forward_for(version):
    def fwd(ids): R2, _, _ = layer0(ids, version); return rest(R2, first=(J.FIRST, J.LAST))
    return fwd

if __name__ == "__main__":
    t0 = time.time()
    if "judge" in ARGS:
        n = int(ARGS.get("n", 300))
        ones = ones_obj = lambda A: np.zeros(np.shape(A), np.int64)      # judging only: no photon counting
        for v in ARGS["judge"].split(","):
            t1 = time.time(); inn, same, tot = J.judge(forward=forward_for(v), n=n)
            print(f"judge5 {v} {VERSIONS[v]}: in top 5 {inn}/{tot}, same top word {same}/{tot}   ({time.time() - t1:.0f}s)", flush=True)
        print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"); sys.exit()
    if "windows" in ARGS:
        ones = ones_obj = lambda A: np.zeros(np.shape(A), np.int64)
        KEYS = ["qkv pile", "qkv", "o", "proj pile", "att", "x1", "fc pile", "fc", "mproj pile", "stream"]
        for v in ARGS["windows"].split(","):
            t1 = time.time(); n_ok = 0
            for wdw in J.WINDOWS:
                _, g0, _ = layer0(wdw["ids"], "V0"); _, g, _ = layer0(wdw["ids"], v)
                n_ok += all(np.array_equal(g[k], g0[k]) for k in KEYS if g.get(k) is not None)
            print(f"{v}: every pile and the stream after layer 0 identical on {n_ok}/{len(J.WINDOWS)} judge windows ({time.time() - t1:.0f}s)", flush=True)
        sys.exit()
    for wdw in J.WINDOWS[:2]:                        # the counted mirror IS the machine
        R2, _, _ = layer0(wdw["ids"], "V0"); assert np.array_equal(rest(R2, Count(), (J.FIRST, J.LAST)), J.scores_all(wdw["ids"]))
    print("the counted mirror gives the machine's own scores on 2 judge windows: yes", flush=True)
    rng = np.random.default_rng(7)
    KEYS = ["x16", "qkv pile", "qkv", "o", "proj pile", "att", "x1", "fc pile", "fc", "mproj pile", "stream"]
    WHOLE = Count(); NPOS = sum(len(t) for t in TEXTS); plain_got = []
    for ids in TEXTS:
        R2, g, _ = layer0(ids, "V0"); plain_got.append(g); rest(R2, WHOLE)
    results, gots = {}, {}; V7 = Count(); KS = []; SWAP = True
    for v in VERSIONS:
        C = Count(); same = {k: 0 for k in KEYS}; diff = {}; gots[v] = []
        for ids, pg in zip(TEXTS, plain_got):
            sh = WTE[rng.integers(0, len(WORDS), len(ids))] if v == "V5" else None
            R2, g, _ = layer0(ids, v, C, shuffle_rows=sh, chained=(V7 if v == "V3" else None))
            if v == "V4": KS += g["K"]; SWAP &= g["swap"]
            if v == "V5": gots[v].append({k: g[k] for k in ("carry", "fc extra", "s entry", "s fc", "qkv pile", "fc pile")})
            for k in KEYS:
                if g.get(k) is None: continue
                eq = np.array_equal(g[k], pg[k]); same[k] += eq
                if not eq: d = np.abs(g[k].astype(np.int64) - pg[k]); diff.setdefault(k, []).append(((d > 0).mean(), int(d.max()), int(np.abs(pg[k]).max())))
        results[v] = C
        print(f"\n{v} {VERSIONS[v]}: identical to the plain machine (texts out of {len(TEXTS)}):", flush=True)
        print("   " + "  ".join(f"{k}: {same[k]}" for k in KEYS if not (k == 'proj pile' and v == 'V3')))
        for k, dd in diff.items():
            print(f"   {k:10s} differs: {np.mean([x[0] for x in dd]) * 100:6.2f}% of its values on average; largest difference {max(x[1] for x in dd):,} (largest value {max(x[2] for x in dd):,})")
    print(f"\nV4: K (digits the arrival's output is rounded off by, before the proj meeting): {dict(zip(*[x.tolist() for x in np.unique(KS, return_counts=True)]))}")
    print(f"V3/V4: stored proj(value) tips mixed by the shares, earlier tokens in a random order = the machine's mix met with the plate: {SWAP}")
    v3 = sum(x for k, x in results["V3"].items()) / NPOS; fcm = (results["V3"]["fc meeting"] + results["V3"]["fc split"]) / NPOS
    v7 = v3 - fcm + V7[next(iter(V7))] / NPOS
    print(f"V7 (work only): V3 with neurons-in's mix taken through stored fc(proj(value)) tips: layer 0 >= {v7:,.0f}  "
          f"(the chained mix alone {V7[next(iter(V7))] / NPOS:,.0f} replaces a meeting of {fcm:,.0f})")
    npos_ = np.zeros(PMAX)
    for t in TEXTS: npos_[:len(t)] += 1
    print("    chained mix by position: " + "  ".join(f"p{p} {V7.bypos[next(iter(V7))][p] / npos_[p] / 1e6:.0f}M" for p in range(0, PMAX, 4)))
    # random-order proof (V5): any order of the lines and of the parts gives the same pile
    ok, trials = True, 0
    for ids, g in zip(TEXTS, gots["V5"]):
        P = len(ids); wi = IDX[np.asarray(ids)]; pos = np.arange(P)
        for key, name, kinds, light, s in (("qkv pile", "qkv.0", ("TQ", "PQ"), g["carry"], g["s entry"]), ("fc pile", "fc.0", ("TF", "PF"), g["fc extra"], g["s fc"])):
            perm = rng.permutation(768); w = PLATE_INT[name][perm]
            parts = [take(kinds[0], s, wi), take(kinds[1], s, pos), light[:, perm] @ w]
            tot = np.zeros_like(parts[0])
            for i in rng.permutation(3): tot = tot + parts[i]
            ok &= np.array_equal(tot, g[key]); trials += 1
    print(f"\nrandom order of the 768 lines and of the three parts (V5, qkv and fc piles, {trials} trials): identical {ok}")
    C = gots["V5"]; cv = np.concatenate([g["carry"].ravel() for g in C])
    print(f"carry at the entry: values {dict(zip(*[x.tolist() for x in np.unique(cv, return_counts=True)]))}, nonzero on {100 * (cv != 0).mean():.1f}% of lines")
    print(f"shifts: entry {dict(zip(*[x.tolist() for x in np.unique(np.concatenate([g['s entry'] for g in C]), return_counts=True)]))}, neurons-in {dict(zip(*[x.tolist() for x in np.unique(np.concatenate([g['s fc'] for g in C]), return_counts=True)]))}")
    # photons per token
    print(f"\nPHOTONS PER TOKEN (mean over the {NPOS} positions of the 30 refs texts; meeting = light photons x plate photons; add = photons added)")
    L0 = sum(v for v in results["V0"].values()) / NPOS; RST = sum(WHOLE.values()) / NPOS; whole = L0 + RST
    print(f"  whole machine, plain: {whole:,.0f}   layer 0 {L0:,.0f} ({100 * L0 / whole:.2f}%), layers 1-11 + the top {RST:,.0f}")
    for k, v in sorted(WHOLE.items(), key=lambda kv: -kv[1]): print(f"      {k:46s} {v / NPOS:16,.0f}")
    for v, C in results.items():
        tot = sum(x for k, x in C.items() if not k.startswith("CONTROL")) / NPOS
        print(f"\n  {v} {VERSIONS[v]} layer 0: {tot:,.0f}   vs plain layer 0 {tot - L0:+,.0f} = {100 * (tot - L0) / L0:+.1f}% of layer 0 = {100 * (tot - L0) / whole:+.2f}% of the whole machine")
        for k, x in sorted(C.items(), key=lambda kv: -kv[1]):
            if x: print(f"      {k:46s} {x / NPOS:16,.0f}")
    npos = np.zeros(PMAX)
    for t in TEXTS: npos[:len(t)] += 1
    print("\n  layer 0 per token by position (plain / V5 / V3), and the arrival's share of plain layer 0:")
    for p in range(0, PMAX, 3):
        row = [sum(x[p] for k, x in results[v].bypos.items() if not k.startswith("CONTROL")) / npos[p] for v in ("V0", "V5", "V3")]
        arr = sum(x[p] for k, x in results["V0"].bypos.items() if k.startswith("arrival")) / npos[p]
        print(f"    position {p:2d} ({int(npos[p]):2d} texts): {row[0]:14,.0f} {row[1]:14,.0f} {row[2]:14,.0f}   arrival {100 * arr / row[0]:5.1f}%")
    print(f"\nwords stored {len(WORDS)}; stores: " + ", ".join(f"{k}@{s}" for k, s in sorted(_STORE)))
    print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
