"""INDEPENDENT CHECK of turns.py and halves.py, built the other way round: FORWARD, not reverse.

turns.py / halves.py send the lead's row DOWN every step (machine.py's reverse) and meet it with each part on the way.
This script never sends anything down. It runs the base-2 machine forward itself (base2.py's own functions, executed
from its own source, nothing edited), writes down every dimming the light got (water level per position, bend share,
arrival shares and totals), and then sends each PART of the light FORWARD through the same held dimmings, as exact
whole numbers over one power of two (Python whole numbers, plate meetings by 24-place limbs, every float sum < 2^53).
At the top each part is met with the lead's row (answer's dimmed row minus the 2nd word's). Because the held machine
is a straight-line sum, the parts sent forward and the light sent backward must give the same numbers, digit for digit.

Every held step is checked against the machine's own arithmetic: the exact value before rounding (computed here) and
the machine's whole-number output differ by at most half a unit at every bank, arrival, bend and the top (so the held
step written here is the machine's step, not a guess).

  1. the turn identity, exact, on texts 0 and 4: TN from turns.npz rebuilt here (other route: fc columns scaled to one
     power and met with mproj by limbs), and (H/2) met with mproj == g_p (R_p + rho_p) met with T_l + (1/2)(fc
     out-rounding + fc bias) met with mproj, at every position, line, layer.
  2. the lead split at the newest bends, held, layers 0, 5, 11 (and all others), 30 texts: the newest position's
     straight half H/2 and symmetric half e = Bout - H/2, each sent forward alone through layers l+1..11 (position t
     only: nothing later reads it) and met with the lead's row; rows: lead, random word's lead, answer's score.
  3. all layers at once, straight: every bend replaced by its straight half 1/2, every symmetric half injected at its
     run value; parts sent forward: each token's entry numbers apart, the symmetric halves (newest, older), fc's bias,
     fc's roundings, the arrivals' constants, mproj's constants; their sum checked == the machine's stream at every
     layer (every position, every line) and their meetings + the top constant == the target, exactly.
Nothing in the machine changes. Measurement only.
    python3 turns/check_fwd.py I0 I1     (texts I0..I1-1)
"""
import sys, os, json, time, resource
import numpy as np
from fractions import Fraction
np.seterr(all="ignore")
HERE = "/Users/abundancemachine/streaming-machine"
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv; os.chdir(_cwd)
refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [(r["ids"] + r["ref"])[:5] for r in refs]
I0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
I1 = int(sys.argv[2]) if len(sys.argv) > 2 else len(TEXTS)
DO_TURNS = "noturns" not in sys.argv
t0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1 << 20)
def O(a): return np.asarray(a).astype(np.int64).astype(object)
def pw(k): return 1 << int(k)
def F2(n, e): return Fraction(int(n)) * Fraction(2) ** int(e)

# ---------------------------------------------------------------- exact plate meeting: object whole numbers @ int16 plate
def to_limbs(N):
    """object (r, k) -> float64 (K, r, k) signed 24-place limbs"""
    r, k = N.shape; vals = N.ravel()
    mags = [abs(int(x)) for x in vals]
    nb = max(1, max(m.bit_length() for m in mags)); K = -(-nb // 24)
    buf = b"".join(m.to_bytes(3 * K, "little") for m in mags)
    u = np.frombuffer(buf, np.uint8).reshape(r * k, K, 3).astype(np.int64)
    L = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16)
    sg = np.fromiter(((int(x) > 0) - (int(x) < 0) for x in vals), np.int64, r * k)
    L = L * sg[:, None]
    return np.ascontiguousarray(L.reshape(r, k, K).transpose(2, 0, 1)).astype(np.float64), K

def from_limbs(S):
    """int64 (K, n), value sum_k S_k 2^(24k) -> object (n,) exact (carry, then bytes)"""
    K, n = S.shape; dig = np.empty((K, n), np.int64); carry = np.zeros(n, np.int64)
    for k in range(K):
        v = S[k] + carry; dig[k] = v & 0xFFFFFF; carry = v >> 24
    b = np.ascontiguousarray(dig.T.astype("<u4")).view(np.uint8).reshape(n, K, 4)[:, :, :3]
    b = np.ascontiguousarray(b).reshape(n, 3 * K)
    out = np.empty(n, dtype=object)
    for i in range(n): out[i] = int.from_bytes(b[i].tobytes(), "little") + (int(carry[i]) << (24 * K))
    return out

def mm(N, w):
    """exact N @ w: N object (..., k), w float64 holding whole numbers (k, m), |w| < 2^15"""
    sh = N.shape; flat = N.reshape(-1, sh[-1]); r, k = flat.shape
    assert np.abs(w).max() * k * (1 << 24) < 2.0 ** 53
    outs = []; step = 8
    for r0 in range(0, r, step):                      # a few rows at a time: memory stays small
        L, K = to_limbs(flat[r0:r0 + step]); rr = L.shape[1]
        S = L.reshape(K * rr, k) @ w; del L
        Si = S.astype(np.int64); assert (Si.astype(np.float64) == S).all(); del S
        outs.append(from_limbs(Si.reshape(K, rr * w.shape[1])).reshape(rr, w.shape[1])); del Si
    return np.concatenate(outs, axis=0).reshape(sh[:-1] + (w.shape[1],))

def selftest():
    rg = np.random.default_rng(3)
    N = np.empty((4, 300), dtype=object)
    for i in range(4):
        for j in range(300): N[i, j] = int(rg.integers(-2**62, 2**62)) * pw(int(rg.integers(0, 900))) * (1 if rg.random() < .5 else -1)
    w = rg.integers(-32767, 32768, (300, 50)).astype(np.float64)
    want = np.array([[sum(int(N[i, q]) * int(w[q, j]) for q in range(300)) for j in range(50)] for i in range(4)], dtype=object)
    return bool((mm(N, w) == want).all())
print(f"exact plate meeting self-test (random 900-place whole numbers against pure Python): {selftest()}", flush=True)

# ---------------------------------------------------------------- the machine run, every dimming written down (own capture)
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; rec = dict(P=P, R0=R.copy(), L=[])
    for l in range(12):
        d = dict(R=R)
        x16, ex, LS = gauge_log(R); d.update(q_LS=LS, q_x16=x16, q_sh=ex + ES)
        QKV = gauged_bank(R, f"qkv.{l}"); d["QKV"] = QKV
        A = np.zeros((12, P, P), np.int64); Lt = np.zeros((12, P), np.int64); ao = np.zeros((P, 768), np.int64)
        for h in range(12):
            q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
            q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
            sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
            for p in range(P):
                row = sc[p, :p+1]; dd = row[race(row)] - row
                a = np.where(dd < len(ARR), ARR[np.minimum(dd, len(ARR) - 1)], 0)
                A[h, p, :p+1] = a; Lt[h, p] = log2fix([a.sum()])[0]
                ao[p, h*64:(h+1)*64] = power(product(a[None, :], v[:p+1].astype(np.float64))[0], -Lt[h, p])
        assert (ao == attention(QKV)).all()
        d.update(A=A, Lt=Lt, ao=ao)
        d["p_x16"], exx = to16(ao, -ES); d["p_sh"] = exx + ES
        po = plain_bank(ao, f"proj.{l}"); d["po"] = po
        R = R + po; d["Rmid"] = R
        x16, ex, LS = gauge_log(R); d.update(f_LS=LS, f_x16=x16, f_sh=ex + ES)
        H = gauged_bank(R, f"fc.{l}"); d["H"] = H
        Bo = edge(H); d["Bo"] = Bo
        idx = rshift(H, ES - 12) + 32768
        share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)])); d["share"] = share
        assert (rshift(H * share, 16) == Bo).all()
        d["m_x16"], exx = to16(Bo, -ES); d["m_sh"] = exx + ES
        mo = plain_bank(Bo, f"mproj.{l}"); d["mo"] = mo
        R = R + mo; rec["L"].append(d)
    rec["R12"] = R
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    scores = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b
    assert race(scores) == forward(ids)
    rec["top"] = dict(LS=int(LS[0]), sh=int(ex[0] + ES), x16=x16[0], scores=scores)
    return rec

# ---------------------------------------------------------------- held steps, exact: value = N x 2^e (one power per array)
def align(a, b):
    (Na, ea), (Nb, eb) = a, b; e = min(ea, eb)
    return Na * pw(ea - e) if ea > e else Na, Nb * pw(eb - e) if eb > e else Nb, e

def add(a, b):
    Na, Nb, e = align(a, b); return (Na + Nb, e)

def gauge_factor(LS):
    """per position: EXP2[f] x 2^(kq - 16), f = (-LS) mod 2^16, kq = floor(-LS / 2^16): (object factors, exponent)"""
    f = (-np.asarray(LS, np.int64)) & 0xFFFF; kq = (-np.asarray(LS, np.int64)) >> 16; k0 = int(kq.min())
    return np.array([int(EXP2[a]) * pw(b - k0) for a, b in zip(f, kq)], dtype=object), k0 - 16

def col_factor(we):
    k0 = int(np.min(we)); return np.array([pw(x - k0) for x in we], dtype=object), k0

def gauged_lin(X, name, LS, cols=None):
    """X (..., P, 768) -> X @ w x 2^we_n x EXP2[f_p] x 2^(kq_p - 16): the water-level bank with its dimming held"""
    w, we, b = W[name]
    if cols is not None: w, we = w[:, cols], we[cols]
    N, e = X; Z = mm(N, w); cf, ck = col_factor(we); rf, rk = gauge_factor(LS)
    return (Z * cf * rf[:, None], e + ck + rk)

def plain_lin(X, name):
    w, we, b = W[name]; N, e = X; Z = mm(N, w); cf, ck = col_factor(we)
    return (Z * cf, e + ck)

def arr_lin(V, A, Lt, P, only=None):
    """arrival with every share and total held: out_p = sum_{j<=p} A[h,p,j] v_j x EXP2[f'] x 2^(I' - 16), F' = -L[h,p]"""
    N, e = V; Fp = -Lt; Ih, fh = Fp >> 16, Fp & 0xFFFF; k0 = int(Ih.min())
    out = np.empty(N.shape, dtype=object); out[...] = 0
    ps = range(P) if only is None else [only]
    for h in range(12):
        cs = slice(h * 64, (h + 1) * 64)
        for p in ps:
            acc = 0
            for j in range(p + 1):
                if only is not None and j != p: continue        # position t alone: nothing else carries the part
                if A[h, p, j]: acc = acc + N[..., j, cs] * int(A[h, p, j])
            out[..., p, cs] = acc * (int(EXP2[fh[h, p]]) * pw(Ih[h, p] - k0))
    return (out, e + k0 - 16)

def half_ok(n, e):
    """|n x 2^e| <= 1/2 everywhere"""
    lim = Fraction(1, 2)
    return all(abs(F2(x, e)) <= lim for x in np.asarray(n).ravel())

def const_of(actual, lin_val):
    """actual whole numbers minus an exact held value: (N, e)"""
    N, e = lin_val
    if e >= 0: return (O(actual) - N * pw(e), 0)
    return (O(actual) * pw(-e) - N, e)

def top_rows(rec, words):
    """the top with its dimming held: score_w = R12[t] met with row_w x EXP2[f] x 2^(HEADE_w + kq - 16) + rounding + b"""
    tp = rec["top"]; hw, he, hb = W["head"]
    f = (-tp["LS"]) & 0xFFFF; kq = (-tp["LS"]) >> 16
    rows = []
    for w_ in words:
        rows.append((O(hw[:, w_]) * int(EXP2[f]), int(he[w_]) + kq - 16))
    return rows

def meet_row(X, row):
    """X: (N (..., 768) object, e); row (object (768,), e): exact Fraction per leading index"""
    (N, e), (r, er) = X, row
    tot = (N * r).sum(axis=-1)
    return np.vectorize(lambda z: F2(z, e + er), otypes=[object])(tot) if np.ndim(tot) else F2(tot, e + er)

# ---------------------------------------------------------------- 0. the knife edge (other seed)
k = np.arange(1, 32768)
mir = bool((PHI[32768 + k].astype(np.int64) + PHI[32768 - k].astype(np.int64) == 65536).all())
Hr = np.random.default_rng(2026).integers(-(1 << 19), 1 << 19, 1_000_000)
dd = edge(Hr) - edge(-Hr) - Hr
print(f"0. PHI[32768+k]+PHI[32768-k]==65536 all k: {mir}; PHI[32768]={int(PHI[32768])}; bend(H)-bend(-H)-H zero for "
      f"{100 * (dd == 0).mean():.2f}% (seed 2026), range [{dd.min()}, {dd.max()}], nonzero only where H = 8 mod 16: "
      f"{bool(((dd != 0) <= ((Hr % 16) == 8)).all())} ({100 * ((Hr % 16) == 8).mean():.2f}% of H)", flush=True)

# ---------------------------------------------------------------- 1. the turns, rebuilt, and the identity on texts 0 and 4
if DO_TURNS:
    Z = np.load(f"{HERE}/turns/turns.npz", allow_pickle=True)
    TN, M0, WEM = Z["TN"], Z["m0"], Z["we_m"]; BN, BE = Z["bias_num"], Z["bias_exp"]
    ok_tn = 0; places = []
    for l in range(12):
        wf, wef, bf = W[f"fc.{l}"]; wm, wem, bm = W[f"mproj.{l}"]
        m0 = int(wef.min()); assert m0 == int(M0[l]) and (wem == WEM[l]).all()
        A_ = O(wf) * np.array([pw(x - m0) for x in wef], dtype=object)[None, :]        # fc columns on one power
        mine = mm(A_, wm)                                                                 # (768, 768) whole numbers
        same = bool((mine == O(TN[l])).all()); ok_tn += same
        bmine = mm(O(bf)[None, :], wm)[0]
        okb = bool((bmine == O(BN[l])).all()) and bool((BE[l] == wem - 1).all())
        places.append(max(abs(int(x)).bit_length() for x in mine.ravel()))
        if not (same and okb): print(f"   layer {l}: TN same {same}, bias part same {okb}")
    print(f"1. turns.npz TN rebuilt (fc columns on one power, met with mproj by limbs): {ok_tn}/12 layers identical; "
          f"places of |TN| per layer {places}", flush=True)
    for i in (0, 4):
        rec = run(TEXTS[i]); P = rec["P"]; n_ok = n_all = 0
        for l in range(12):
            d = rec["L"][l]; wf, wef, bf = W[f"fc.{l}"]; wm, wem, bm = W[f"mproj.{l}"]
            H, R, x16, sh, LS = d["H"], d["Rmid"], d["f_x16"], d["f_sh"], d["f_LS"]
            rho = (x16 << sh[:, None]) - R; assert (np.abs(rho) <= (1 << sh[:, None]) // 2).all()
            # fc's out-rounding: H - b - (x16 @ wf) x 2^we_f x EXP2[f] x 2^(kq + sh - 16), exact
            pile = gauged_lin((O(x16), 0), f"fc.{l}", LS); pile = (pile[0] * np.array([pw(s) for s in sh], dtype=object)[:, None], pile[1])
            orn = const_of(H - bf, pile); assert half_ok(*orn)
            # left: (H/2) met with mproj
            cf, ck = col_factor(wem); LHS = (mm(O(H), wm) * cf, ck - 1)
            # right: g_p (R_p + rho_p) @ T_l + (1/2)(out-rounding + bias) @ mproj
            rf, rk = gauge_factor(LS); cT = np.array([pw(x - int(WEM[l].min())) for x in WEM[l]], dtype=object)
            # TN needs ~38 places: meet with limbs of the light against 16-place pieces of TN
            TNo = TN[l].astype(np.int64); sg = np.sign(TNo); mag = np.abs(TNo); acc = None
            for q in range(3):
                piece = ((mag >> (16 * q)) & 0xFFFF) * sg
                part = mm(O(R + rho), piece.astype(np.float64)) * pw(16 * q)
                acc = part if acc is None else acc + part
            RHS1 = (acc * rf[:, None] * cT[None, :], rk + int(M0[l]) + int(WEM[l].min()) - 1)
            ob = add(orn, (O(np.broadcast_to(bf, H.shape)), 0))
            RHS2 = (mm(ob[0], wm) * cf, ob[1] + ck - 1)
            Lh, Rr, e1 = align(LHS, add(RHS1, RHS2))
            eq = (Lh == Rr); n_ok += int(eq.sum()); n_all += eq.size
        print(f"   text {i}: identity (H/2) met with mproj == g (R + rho) met with T_l + (1/2)(fc out-rounding + bias) met with"
              f" mproj: {n_ok}/{n_all} entries equal  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)

# ---------------------------------------------------------------- 2 and 3, per text
OUT = dict(lead=np.zeros((30, 3), object), s2=np.full((30, 12, 3, 3), np.nan), s3=np.full((30, 3, 13), np.nan),
           words=np.zeros((30, 3), np.int64))
NAMES = ["entry0", "entry1", "entry2", "entry3", "entry4", "symN", "symO", "fcB", "fcR", "arr", "mp"]
chk = dict(res=0, res_n=0, stream=0, stream_n=0, s2sum=0, s2sum_n=0, s3=0, s3_n=0, sym=0)
EX = {}
for i in range(I0, I1):
    ids = TEXTS[i]; rec = run(ids); P = rec["P"]; t = P - 1; S = rec["top"]["scores"]
    order = np.argsort(-S, kind="stable"); a, b = int(order[0]), int(order[1])
    rg = np.random.default_rng(12345 + i); r = int(rg.integers(0, len(S)))
    while r in order[:5]: r = int(rg.integers(0, len(S)))
    targets = [int(S[a]) - int(S[b]), int(S[a]) - int(S[r]), int(S[a])]
    OUT["words"][i] = [a, b, r]; OUT["lead"][i] = targets
    ra, rb, rr = top_rows(rec, [a, b, r])
    def diff(x, y):
        (nx, ex_), (ny, ey) = x, y; e = min(ex_, ey); return (nx * pw(ex_ - e) - ny * pw(ey - e), e)
    ROWS = [diff(ra, rb), diff(ra, rr), ra]
    # top: the held top is the machine's (rounding <= 1/2 per word)
    tp = rec["top"]; hw, he, hb = W["head"]
    for w_, row in zip((a, b, r), (ra, rb, rr)):
        val = meet_row((O(tp["x16"]) * pw(tp["sh"]), 0), row)
        chk["res"] += abs(Fraction(int(S[w_]) - int(hb[w_])) - val) <= Fraction(1, 2); chk["res_n"] += 1
    # ---- the held steps' constants for this run, and the residue checks
    LC = []
    for l in range(12):
        d = rec["L"][l]; c = {}
        vcols = np.arange(1536, 2304)
        Vact = d["QKV"][:, 1536:]
        # qkv (v columns): rounding of the light, rounding of the output, each <= its half unit
        pile = gauged_lin((O(d["q_x16"]) * np.array([pw(s) for s in d["q_sh"]], dtype=object)[:, None], 0), f"qkv.{l}", d["q_LS"], vcols)
        chk["res"] += half_ok(*const_of(Vact - W[f"qkv.{l}"][2][1536:], pile)); chk["res_n"] += 1
        c["v"] = const_of(Vact, gauged_lin((O(d["R"]), 0), f"qkv.{l}", d["q_LS"], vcols))
        exact_ao = arr_lin((O(Vact), 0), d["A"], d["Lt"], P)
        c["ao"] = const_of(d["ao"], exact_ao); chk["res"] += half_ok(*c["ao"]); chk["res_n"] += 1
        pj = plain_lin((O(d["p_x16"]) * np.array([pw(s) for s in d["p_sh"]], dtype=object)[:, None], 0), f"proj.{l}")
        chk["res"] += half_ok(*const_of(d["po"] - W[f"proj.{l}"][2], pj)); chk["res_n"] += 1
        c["po"] = const_of(d["po"], plain_lin((O(d["ao"]), 0), f"proj.{l}"))
        fcp = gauged_lin((O(d["f_x16"]) * np.array([pw(s) for s in d["f_sh"]], dtype=object)[:, None], 0), f"fc.{l}", d["f_LS"])
        chk["res"] += half_ok(*const_of(d["H"] - W[f"fc.{l}"][2], fcp)); chk["res_n"] += 1
        c["fcR"] = const_of(d["H"] - W[f"fc.{l}"][2], gauged_lin((O(d["Rmid"]), 0), f"fc.{l}", d["f_LS"]))
        c["fcB"] = (O(np.broadcast_to(W[f"fc.{l}"][2], d["H"].shape)), 0)
        Y2 = 2 * d["Bo"] - d["H"]; chk["sym"] += int((Y2 < 0).sum()) + int((np.abs(Y2) > np.abs(d["H"]) + 1).sum())
        c["e2"] = Y2                                          # 2 e, whole numbers
        bshare = const_of(d["Bo"], (O(d["H"]) * O(d["share"]), -16)); chk["res"] += half_ok(*bshare); chk["res_n"] += 1
        c["bend"] = bshare
        mpp = plain_lin((O(d["m_x16"]) * np.array([pw(s) for s in d["m_sh"]], dtype=object)[:, None], 0), f"mproj.{l}")
        chk["res"] += half_ok(*const_of(d["mo"] - W[f"mproj.{l}"][2], mpp)); chk["res_n"] += 1
        c["mp"] = const_of(d["mo"], plain_lin((O(d["Bo"]), 0), f"mproj.{l}"))
        LC.append(c)

    # ---- STEP 2, held: the newest position's two halves sent forward from layer l alone
    def forward_held_t(X, l_from):
        """X (rows, 768) at position t, value after layer l_from's write; sent through layers l_from+1..11 held"""
        for l in range(l_from + 1, 12):
            d = rec["L"][l]
            V = gauged_lin((X[0][:, None, :], X[1]), f"qkv.{l}", d["q_LS"][t:t+1], np.arange(1536, 2304))
            Ao = arr_lin(V, d["A"][:, t:t+1, t:t+1], d["Lt"][:, t:t+1], 1)
            X = add(X, (lambda q: (q[0][:, 0, :], q[1]))(plain_lin(Ao, f"proj.{l}")))
            Hc = gauged_lin((X[0][:, None, :], X[1]), f"fc.{l}", d["f_LS"][t:t+1])
            Bc = (Hc[0] * O(d["share"][t:t+1])[None], Hc[1] - 16)
            X = add(X, (lambda q: (q[0][:, 0, :], q[1]))(plain_lin(Bc, f"mproj.{l}")))
        return X
    for l in range(12):
        d = rec["L"][l]
        Bt = np.stack([d["H"][t], 2 * d["Bo"][t] - d["H"][t], 2 * d["Bo"][t]])      # 2 x (straight, symmetric, whole)
        X0 = plain_lin((O(Bt)[:, None, :], -1), f"mproj.{l}"); X0 = (X0[0][:, 0, :], X0[1])
        X = forward_held_t(X0, l)
        for w_, row in enumerate(ROWS):
            v = meet_row(X, row)                                                     # straight, symmetric, part
            chk["s2sum"] += (v[0] + v[1] == v[2]); chk["s2sum_n"] += 1
            OUT["s2"][i, l, w_] = [float(v[0]), float(v[1]), float(v[2])]
            if w_ == 0 and l in (0, 5, 11): EX[(i, l)] = (v[0], v[1], v[2])

    # ---- STEP 3, straight: every part sent forward through the 12 fixed turns, held water levels and arrivals
    C = len(NAMES); X = (np.empty((C, P, 768), dtype=object), 0); X[0][...] = 0
    for p in range(P): X[0][p, p, :] = O(rec["R0"][p])
    ix = {n: k_ for k_, n in enumerate(NAMES)}
    def inject(X, name, val):
        N, e = X; Nv, ev = val; m = min(e, ev)
        N = N * pw(e - m) if e > m else N.copy()
        N[ix[name]] = N[ix[name]] + (Nv * pw(ev - m) if ev > m else Nv); return (N, m)
    def total(X): return X[0].sum(axis=0), X[1]
    def same_as(X, actual):
        N, e = total(X)
        return bool((N == O(actual) * pw(-e)).all()) if e <= 0 else bool((N * pw(e) == O(actual)).all())
    for l in range(12):
        d = rec["L"][l]; c = LC[l]
        chk["stream"] += same_as(X, d["R"]); chk["stream_n"] += 1
        V = gauged_lin(X, f"qkv.{l}", d["q_LS"], np.arange(1536, 2304)); V = inject(V, "arr", c["v"])
        Ao = arr_lin(V, d["A"], d["Lt"], P); Ao = inject(Ao, "arr", c["ao"])
        Po = plain_lin(Ao, f"proj.{l}"); Po = inject(Po, "arr", c["po"])
        X = add(X, Po)
        chk["stream"] += same_as(X, d["Rmid"]); chk["stream_n"] += 1
        Hc = gauged_lin(X, f"fc.{l}", d["f_LS"]); Hc = inject(Hc, "fcB", c["fcB"]); Hc = inject(Hc, "fcR", c["fcR"])
        chk["stream"] += same_as(Hc, d["H"]); chk["stream_n"] += 1
        Bc = (Hc[0], Hc[1] - 1)                                                       # the straight half: x 1/2
        Yn = np.zeros_like(c["e2"]); Yn[t] = c["e2"][t]; Yo = c["e2"] - Yn
        Bc = inject(Bc, "symN", (O(Yn), -1)); Bc = inject(Bc, "symO", (O(Yo), -1))
        chk["stream"] += same_as(Bc, d["Bo"]); chk["stream_n"] += 1
        Mo = plain_lin(Bc, f"mproj.{l}"); Mo = inject(Mo, "mp", c["mp"])
        X = add(X, Mo)
    chk["stream"] += same_as(X, rec["R12"]); chk["stream_n"] += 1
    Xt = (X[0][:, t, :], X[1])
    for w_, row in enumerate(ROWS):
        parts = meet_row(Xt, row)                                                    # per part, exact
        tot_lin = meet_row(total(Xt), row)
        topc = Fraction(targets[w_]) - tot_lin
        ok = sum(parts, Fraction(0)) + topc == targets[w_] and sum(parts, Fraction(0)) == tot_lin
        chk["s3"] += ok; chk["s3_n"] += 1
        entry = sum(parts[:5], Fraction(0))
        OUT["s3"][i, w_] = [float(entry)] + [float(x) for x in parts[5:]] + [float(topc)] + [float(x) for x in parts[:5]]
        if w_ == 0 and i in (0, 1, 2): EX[(i, "s3")] = (entry, parts[5] + parts[6], parts[7])
    print(f"  text {i:2d} lead {targets[0]:7d} | step 2 (held, forward) lead: layer 0 str {OUT['s2'][i, 0, 0, 0] / targets[0]:8.3f}"
          f" sym {OUT['s2'][i, 0, 0, 1] / targets[0]:8.3f}; layer 5 str {OUT['s2'][i, 5, 0, 0] / targets[0]:8.3f} sym"
          f" {OUT['s2'][i, 5, 0, 1] / targets[0]:8.3f}; layer 11 str {OUT['s2'][i, 11, 0, 0] / targets[0]:8.3f} sym"
          f" {OUT['s2'][i, 11, 0, 1] / targets[0]:8.3f} | step 3 entry {OUT['s3'][i, 0, 0] / targets[0]:11.4g} sym"
          f" {(OUT['s3'][i, 0, 1] + OUT['s3'][i, 0, 2]) / targets[0]:11.4g} fcB {OUT['s3'][i, 0, 3] / targets[0]:11.4g} leads"
          f" | checks {chk}  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    if i in (0, 1, 2):
        for l in (0, 5, 11):
            s_, y_, p_ = EX[(i, l)]; D = max(s_.denominator, y_.denominator).bit_length() - 1
            print(f"     text {i} layer {l:2d} exact, over 2^{D}: straight {s_ * pw(D)}  symmetric {y_ * pw(D)}  (straight + symmetric) x 2^{D}"
                  f" = {(s_ + y_) * pw(D)}")
        en, sy, fb = EX[(i, "s3")]
        print(f"     text {i} all layers: fixed entry part = {en.numerator}/{en.denominator}\n         = {float(en):.6f} units; symmetric"
              f" halves {float(sy):.6f}; fc bias through the turns {float(fb):.6f}", flush=True)

np.savez(f"{HERE}/turns/check_fwd_{I0:02d}_{I1:02d}.npz", s2=OUT["s2"], s3=OUT["s3"], words=OUT["words"],
         lead=np.array([[int(x) if x is not None else 0 for x in row] for row in OUT["lead"]], dtype=np.int64))
print(f"CHECKS: {chk}  ({time.time() - t0:.0f}s, max {rss()} MB)")
