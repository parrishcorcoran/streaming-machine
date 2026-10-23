"""THE THIN PAGE. A token alone (its arrival over only itself) meets only fixed things: its word row, its position row,
the plates, the bends. So its whole column through all 12 layers is fixed: the OWN column. In context, the earlier
tokens change the light at every plate. The THIN PAGE is that change:
    thin = x_ctx - x_own,   x = the whole-number light that meets the plate after the water level (x16, as the machine
                            forms it), in context and alone.
A meeting is a sum of products, so   pile_ctx = pile_own + (the thin page meeting the plate),   exactly.

Texts: the 30 refs texts (ids + ref, every position: 648 positions). Banks: qkv, proj, fc, mproj of all 12 layers and
the top read (every position).
  1. every text in context (the machine untouched) and alone (the arrival sees only its own position; nothing else
     changed); position 0 alone == in context, bit for bit.
  2. the identity at every bank of every text, np.array_equal; text 0 also rebuilt in plain int64 (no float at all).
  3. photons (1s of |value|) of x_ctx, x_own, thin; lines where thin is exactly 0; positions whose water level moved;
     size (sum of |value|); work = sum over lines of (light photons on the line) x (plate photons on that line).
  4. CONTROL (pairing broken): x_ctx against a DIFFERENT word's own column at the same position (random other text).
  5. POSITION: 200 random words alone at their own position vs alone at another position (0..23).
  6. ORDER-FREE: fc.6's identity with the 768 lines in one random order on both sides.
PREDICTIONS (written before running):
  the user's:  the thin page is 5-7% of the light's 1s.
  Claude's:    50-80%, because a change small in size still keeps most of its places unless whole lines are exactly equal.
    python3 thinpage/thin_page.py      -> thinpage/thin_page.npz, report printed
"""
import sys, os, json, time, numpy as np
ARGS = sys.argv[1:]                                          # optional: how many texts (a quick look)
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")          # the machine never
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                  # uses these
exec(src)                                                    # the machine, untouched
FORWARD = forward
# the alone arrival: the same lines as the machine's arrival, except it covers only the token's own position
i0, i1 = src.index("def attention(QKV):"), src.index("def forward(ids):"); asrc = src[i0:i1]
for a, b in [("def attention(QKV):", "def attention_alone(QKV):"), ("row = sc[p, :p+1]", "row = sc[p, p:p+1]"),
             ("v[:p+1]", "v[p:p+1]")]:
    assert asrc.count(a) == 1; asrc = asrc.replace(a, b)
exec(asrc)

machine_product, machine_to16 = product, to16
BANKS = {id(W[k][0]): k for k in W}
LAST, CAP = [None], {}
def to16(X, e):                                              # the machine's water level, the result remembered
    out = machine_to16(X, e); LAST[0] = out; return out
def product(x16, w16):                                       # the machine's meeting; at a bank, the light is kept
    pile = machine_product(x16, w16)
    k = BANKS.get(id(w16))
    if k is not None:
        assert LAST[0][0] is x16
        CAP[k] = (x16, LAST[0][1], pile)
    return pile
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]
def run(R0, arrival):                                        # forward() on every position, the top read on every position
    CAP.clear(); R = R0
    for l in range(12):
        R = R + plain_bank(arrival(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    top = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return dict(CAP), top

def ones(A): return np.bitwise_count(np.abs(np.asarray(A).astype(np.int64))).astype(np.int64)
_r = np.random.default_rng(0).integers(-(1 << 20), 1 << 20, 1000)
assert all(int(o) == bin(abs(int(v))).count("1") for o, v in zip(ones(_r), _r))
PP = {}                                                      # plate photons on each line (lines in)
for k in NAMES:
    w = M[k + ".w"]; assert np.array_equal(w.astype(np.float64), W[k][0])
    PP[k] = np.concatenate([ones(w[r:r + 128]).sum(1) for r in range(0, w.shape[0], 128)]); del w

refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
if ARGS: refs, TEXTS = refs[:int(ARGS[0])], TEXTS[:int(ARGS[0])]
WTE, WPE = M["wte"], M["wpe"]
NB = len(NAMES); NPOS = sum(len(s) for s in TEXTS)
Q = ["ph_ctx", "ph_own", "ph_thin", "zero", "moved", "work_ctx", "work_own", "work_thin", "size_ctx", "size_thin", "lines"]
REC = {q: np.zeros((NPOS, NB), np.int64) for q in Q}
POS, TXT = np.zeros(NPOS, np.int64), np.zeros(NPOS, np.int64)
XC, XO, EXC, EXO = [], [], [], []                             # kept for the control (whole numbers, int32)
CHK = dict(pos0_same=0, identity_bad=0, identity_n=0, int64_bad=0, int64_n=0, forward_same=0, words_same=0, words_n=0,
           orderfree_bad=0, orderfree_n=0)
perm = np.random.default_rng(6).permutation(768); W6p = W["fc.6"][0][perm]

def fill(D, r, bi, xc, xo, exc, exo, k, thin):
    n = len(xc)
    D["ph_ctx"][r:r + n, bi] = ones(xc).sum(1); D["ph_own"][r:r + n, bi] = ones(xo).sum(1)
    ot = ones(thin); D["ph_thin"][r:r + n, bi] = ot.sum(1); D["zero"][r:r + n, bi] = (thin == 0).sum(1)
    D["moved"][r:r + n, bi] = exc != exo; D["lines"][r:r + n, bi] = xc.shape[1]
    D["work_ctx"][r:r + n, bi] = ones(xc) @ PP[k]; D["work_own"][r:r + n, bi] = ones(xo) @ PP[k]
    D["work_thin"][r:r + n, bi] = ot @ PP[k]
    D["size_ctx"][r:r + n, bi] = np.abs(xc).astype(np.int64).sum(1); D["size_thin"][r:r + n, bi] = np.abs(thin).astype(np.int64).sum(1)

t0 = time.time(); row = 0
for t, seq in enumerate(TEXTS):
    P = len(seq); R0 = WTE[seq] + WPE[:P]
    O, top_own = run(R0, attention_alone)
    C, top_ctx = run(R0, attention)                          # the machine
    CHK["forward_same"] += race(top_ctx[-1]) == FORWARD(seq)
    n0 = len(refs[t]["ids"])
    for p in range(n0 - 1, P - 1): CHK["words_same"] += race(top_ctx[p]) == seq[p + 1]; CHK["words_n"] += 1
    same0 = np.array_equal(top_ctx[0], top_own[0])
    for k in NAMES: same0 &= np.array_equal(C[k][0][0], O[k][0][0]) and np.array_equal(C[k][2][0], O[k][2][0]) and C[k][1][0] == O[k][1][0]
    CHK["pos0_same"] += bool(same0)
    xc_t, xo_t, ec_t, eo_t = {}, {}, {}, {}
    for bi, k in enumerate(NAMES):
        xc, exc, pc = C[k]; xo, exo, po = O[k]; thin = xc - xo
        CHK["identity_bad"] += not np.array_equal(pc, po + machine_product(thin, W[k][0])); CHK["identity_n"] += 1
        if t == 0:                                           # the same identity in plain int64, no float anywhere
            w64 = M[k + ".w"]; ok = True
            for c in range(0, w64.shape[1], 4096):
                wc = w64[:, c:c + 4096].astype(np.int64)
                ok &= np.array_equal(pc[:, c:c + 4096], xo.astype(np.int64) @ wc + thin.astype(np.int64) @ wc)
            CHK["int64_bad"] += not ok; CHK["int64_n"] += 1; del w64
        if k == "fc.6":                                      # order-free: the 768 lines in one random order, both sides
            ok = np.array_equal(machine_product(xo[:, perm], W6p), po) and np.array_equal(pc, po + machine_product(thin[:, perm], W6p))
            ok &= np.array_equal(ones(thin[:, perm]).sum(1), ones(thin).sum(1)) and np.array_equal(ones(thin[:, perm]) @ PP[k][perm], ones(thin) @ PP[k])
            CHK["orderfree_bad"] += not ok; CHK["orderfree_n"] += 1
        fill(REC, row, bi, xc, xo, exc, exo, k, thin)
        xc_t[k], xo_t[k], ec_t[k], eo_t[k] = xc.astype(np.int32), xo.astype(np.int32), exc.copy(), exo.copy()
    XC.append(xc_t); XO.append(xo_t); EXC.append(ec_t); EXO.append(eo_t)
    POS[row:row + P] = np.arange(P); TXT[row:row + P] = t; row += P
    del C, O
    if t % 5 == 4: print(f"  {t + 1} texts ({time.time() - t0:.0f}s)  {CHK}", flush=True)

# CONTROL: the same in-context light against a different word's own column at the same position
rng = np.random.default_rng(4)
CREC = {q: np.zeros((NPOS, NB), np.int64) for q in Q}; CVALID = np.zeros(NPOS, bool)
for r in range(NPOS):
    t, p = TXT[r], POS[r]
    cand = [u for u in range(len(TEXTS)) if u != t and len(TEXTS[u]) > p and TEXTS[u][p] != TEXTS[t][p]]
    if not cand: continue
    u = int(rng.choice(cand)); CVALID[r] = True
    for bi, k in enumerate(NAMES):
        xc, xo = XC[t][k][p:p + 1].astype(np.int64), XO[u][k][p:p + 1].astype(np.int64)
        fill(CREC, r, bi, xc, xo, EXC[t][k][p:p + 1], EXO[u][k][p:p + 1], k, xc - xo)

# EXTRA (the previous page): the in-context light against the PREVIOUS token's in-context light, same text, same bank.
# pile(p) = pile(p-1) + (x_ctx(p) - x_ctx(p-1)) meeting the plate, by the same sum of products.
VREC = {q: np.zeros((NPOS, NB), np.int64) for q in Q}
for r in range(NPOS):
    t, p = TXT[r], POS[r]
    if p == 0: continue
    for bi, k in enumerate(NAMES):
        xc, xp = XC[t][k][p:p + 1].astype(np.int64), XC[t][k][p - 1:p].astype(np.int64)
        fill(VREC, r, bi, xc, xp, EXC[t][k][p:p + 1], EXC[t][k][p - 1:p], k, xc - xp)

# POSITION: 200 words alone at their own position vs alone at another position
rng = np.random.default_rng(5)
NC = min(200, NPOS); cases = rng.choice(NPOS, NC, replace=False)
wd = np.array([TEXTS[TXT[r]][POS[r]] for r in cases]); pa = POS[cases]
pb = np.array([int(rng.choice([q for q in range(24) if q != p])) for p in pa])
A, _ = run(WTE[wd] + WPE[pa], attention_alone); A = {k: (A[k][0].copy(), A[k][1].copy()) for k in NAMES}
B, _ = run(WTE[wd] + WPE[pb], attention_alone); B = {k: (B[k][0].copy(), B[k][1].copy()) for k in NAMES}
CHK["batch_alone_equals_text_alone"] = int(sum(np.array_equal(A[k][0][i], XO[TXT[r]][k][POS[r]]) for k in NAMES for i, r in enumerate(cases)))
CHK["batch_alone_n"] = len(NAMES) * len(cases)
PREC = {q: np.zeros((NC, NB), np.int64) for q in Q}
for bi, k in enumerate(NAMES): fill(PREC, 0, bi, A[k][0], B[k][0], A[k][1], B[k][1], k, A[k][0] - B[k][0])
print(f"  all done ({time.time() - t0:.0f}s)  {CHK}", flush=True)
np.savez(f"{HERE}/thinpage/thin_page.npz", names=np.array(NAMES), pos=POS, txt=TXT, cvalid=CVALID, cases=cases, pa=pa, pb=pb,
         checks=json.dumps({k: int(v) for k, v in CHK.items()}),
         **{f"rec_{q}": REC[q] for q in Q}, **{f"ctl_{q}": CREC[q] for q in Q}, **{f"pos_{q}": PREC[q] for q in Q}, **{f"prev_{q}": VREC[q] for q in Q})

# ---------------------------------------------------------------- report
def pct(a, b): return 100.0 * a / b if b else float("nan")
print(f"\nTHE THIN PAGE: 30 texts, {NPOS} positions; every bank of 12 layers and the top read (every position)")
print(f"checks: position 0 alone == in context (every bank, x16, water level, pile, top read): {CHK['pos0_same']}/30 texts")
print(f"        identity pile_ctx == pile_own + thin meeting the plate: {CHK['identity_n'] - CHK['identity_bad']}/{CHK['identity_n']} banks"
      f"   (text 0 in plain int64: {CHK['int64_n'] - CHK['int64_bad']}/{CHK['int64_n']})")
print(f"        order-free (fc.6, lines in one random order on both sides): {CHK['orderfree_n'] - CHK['orderfree_bad']}/{CHK['orderfree_n']} texts")
print(f"        run() == base2.forward at the last position: {CHK['forward_same']}/30;  words == refs: {CHK['words_same']}/{CHK['words_n']}")
print(f"        a word alone in a batch == the same word alone in its text: {CHK['batch_alone_equals_text_alone']}/{CHK['batch_alone_n']}")
def table(D, sel, title):
    print(f"\n{title}\n  photons per token: light in context | own | thin page; thin as % of the light's photons; size %; lines exactly 0 %;"
          f" water level moved %; work %")
    for kind in ("qkv", "proj", "fc", "mproj", "head"):
        for l in (range(12) if kind != "head" else [None]):
            k = kind if l is None else f"{kind}.{l}"; bi = NAMES.index(k); s = sel
            c, o, th = D["ph_ctx"][s, bi].sum(), D["ph_own"][s, bi].sum(), D["ph_thin"][s, bi].sum(); n = s.sum()
            print(f"  {k:8s} {c / n:9.0f} {o / n:9.0f} {th / n:9.0f}   thin {pct(th, c):5.1f}%  size {pct(D['size_thin'][s, bi].sum(), D['size_ctx'][s, bi].sum()):5.1f}%"
                  f"  zero {pct(D['zero'][s, bi].sum(), D['lines'][s, bi].sum()):5.1f}%  moved {pct(D['moved'][s, bi].sum(), n):5.1f}%"
                  f"  work {pct(D['work_thin'][s, bi].sum(), D['work_ctx'][s, bi].sum()):5.1f}%")
    print("  per layer (all four banks): photons thin/light %, work thin/plain %")
    for l in range(12):
        bis = [NAMES.index(f"{b}.{l}") for b in ("qkv", "proj", "fc", "mproj")]
        print(f"    layer {l:2d}: photons {pct(D['ph_thin'][sel][:, bis].sum(), D['ph_ctx'][sel][:, bis].sum()):5.1f}%"
              f"   work {pct(D['work_thin'][sel][:, bis].sum(), D['work_ctx'][sel][:, bis].sum()):5.1f}%")
    nh = [NAMES.index(k) for k in NAMES if k != "head"]
    for lab, bis in [("without the top read", nh), ("with the top read", list(range(NB)))]:
        print(f"  overall {lab:22s}: photons {pct(D['ph_thin'][sel][:, bis].sum(), D['ph_ctx'][sel][:, bis].sum()):5.1f}%"
              f"  size {pct(D['size_thin'][sel][:, bis].sum(), D['size_ctx'][sel][:, bis].sum()):5.1f}%"
              f"  lines exactly 0 {pct(D['zero'][sel][:, bis].sum(), D['lines'][sel][:, bis].sum()):5.1f}%"
              f"  work {pct(D['work_thin'][sel][:, bis].sum(), D['work_ctx'][sel][:, bis].sum()):5.1f}%"
              f"   (plain work per token {D['work_ctx'][sel][:, bis].sum() / sel.sum():.4g} photon meetings)")
ALL = np.ones(NPOS, bool)
table(REC, ALL, "IN CONTEXT vs ALONE (the paired thin page), every position")
table(REC, POS > 0, "the same, positions 1 and on (position 0 is all own)")
table(CREC, CVALID, "CONTROL: in-context light vs a DIFFERENT word's own column at the same position")
table(VREC, POS > 0, "EXTRA, THE PREVIOUS PAGE: in-context light vs the previous token's in-context light (positions 1 and on)")
nh = [NAMES.index(k) for k in NAMES if k != "head"]; hb = NAMES.index("head")
print("\nBY POSITION: thin page photons as % of the light's photons (banks of 12 layers | top read), work %, positions moved, control")
for p in range(POS.max() + 1):
    s = POS == p; sc_ = s & CVALID
    print(f"  pos {p:2d} ({s.sum():2d} tokens): banks {pct(REC['ph_thin'][s][:, nh].sum(), REC['ph_ctx'][s][:, nh].sum()):5.1f}%"
          f"  top {pct(REC['ph_thin'][s, hb].sum(), REC['ph_ctx'][s, hb].sum()):5.1f}%"
          f"  work {pct(REC['work_thin'][s][:, nh].sum(), REC['work_ctx'][s][:, nh].sum()):5.1f}%"
          f"  zero lines {pct(REC['zero'][s][:, nh].sum(), REC['lines'][s][:, nh].sum()):5.1f}%"
          f"  water moved {pct(REC['moved'][s][:, nh].sum(), s.sum() * len(nh)):5.1f}%"
          f"   | control banks {pct(CREC['ph_thin'][sc_][:, nh].sum(), CREC['ph_ctx'][sc_][:, nh].sum()):5.1f}%")
print("\nWATER LEVEL: thin photons % split by whether the position's water level moved (banks of 12 layers)")
for lab, m in [("moved", REC["moved"][:, nh] == 1), ("not moved", REC["moved"][:, nh] == 0)]:
    print(f"  {lab:9s}: {m.sum():6d} bank-positions   thin {pct(REC['ph_thin'][:, nh][m].sum(), REC['ph_ctx'][:, nh][m].sum()):5.1f}%")
print("\nPOSITION: the same word alone at its own position vs at another position (200 cases)")
z0 = (pa == 0) | (pb == 0)
for k in ["qkv.0", "fc.0", "qkv.6", "fc.6", "qkv.11"]:
    bi = NAMES.index(k); cb = REC  # paired thin and control at the same bank, for comparison
    line = f"  {k:7s} difference photons {pct(PREC['ph_thin'][:, bi].sum(), PREC['ph_ctx'][:, bi].sum()):5.1f}% of own light"
    line += f"  (not involving 0: {pct(PREC['ph_thin'][~z0, bi].sum(), PREC['ph_ctx'][~z0, bi].sum()):5.1f}%, {(~z0).sum()} cases)"
    line += f"  lines exactly 0 {pct(PREC['zero'][:, bi].sum(), PREC['lines'][:, bi].sum()):5.1f}%  water moved {pct(PREC['moved'][:, bi].sum(), NC):5.1f}%"
    line += f"  size {pct(PREC['size_thin'][:, bi].sum(), PREC['size_ctx'][:, bi].sum()):5.1f}%"
    line += f"  | paired thin {pct(REC['ph_thin'][POS > 0, bi].sum(), REC['ph_ctx'][POS > 0, bi].sum()):5.1f}%  control {pct(CREC['ph_thin'][CVALID, bi].sum(), CREC['ph_ctx'][CVALID, bi].sum()):5.1f}%"
    print(line)
import resource
print(f"\nlargest memory held by this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
