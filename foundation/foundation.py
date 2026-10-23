"""THE FOUNDATION PAGE. The first token's pattern never changes: at position 0 the arrival covers only itself, in
context or not. Every later token is stacked on it. So the FOUNDATION column of a token is the column it gets when its
arrival, at every layer, covers only the first token and itself (position 0: only itself, so position 0 is the
machine, bit for bit). Everything else is the machine. In context, the other earlier tokens add to the light at every
plate. The PAGE over the foundation is that addition:
    page = x_ctx - x_found,   x = the whole-number light that meets the plate after the water level (x16, as the
                              machine forms it), in context and on the foundation.
A meeting is a sum of products, so   pile_ctx = pile_found + (the page meeting the plate),   exactly.
SECOND LEVEL: the arrival covers the first token, the token just before, and itself.

Texts: the 30 refs texts (ids + ref, every position: 648 positions). Banks: qkv, proj, fc, mproj of all 12 layers and
the top read (every position).
  1. every text in context (the machine untouched), on the foundation {0, itself}, on the second level
     {0, the one before, itself}, on a FOREIGN foundation (control), and alone {itself} (the old base).
     The same arrival lines with the whole cover (0..itself) are checked to be the machine, bit for bit.
  2. the identity at every bank of every text for every base, np.array_equal; text 0 also in plain int64 (no float).
  3. photons (1s of |value|) of x_ctx, x_base, page; size (sum of |value|) of each; lines where the page is exactly 0;
     positions whose water level moved; work = sum over lines of (light photons on the line) x (plate photons on it).
  4. top read: the machine's word in the base's top 5; the same first word; the same top-5 set.
  5. CONTROL: each text's own tokens, but position 0's word replaced by a different text's first word (a different
     word), arrivals over {0, itself}; its page against the real in-context light. And the old alone base.
PREDICTIONS (written before running):
  the user's:  the page over the foundation is thin.
  Claude's:    thin in size (30-60% of the light's size) but still most of the 1s (70-100% of the photons),
               because photons count 1s, and a value that changes a little still keeps most of its 1s.
    python3 foundation/foundation.py [texts]   -> foundation/foundation.npz, report printed
"""
import sys, os, json, time, resource, numpy as np
ARGS = sys.argv[1:]                                          # optional: how many texts (a quick look)
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
PRED = ["PREDICTIONS (written into the script before any run):",
        "  the user's:  the page over the foundation is thin.",
        "  Claude's:    thin in size (30-60% of the light's size) but still most of the 1s (70-100% of the photons)."]
print("\n".join(PRED), flush=True)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")          # the machine never
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                  # uses these
exec(src)                                                    # the machine, untouched
FORWARD = forward
# the restricted arrival: the machine's own arrival lines, except which earlier positions it covers
i0, i1 = src.index("def attention(QKV):"), src.index("def forward(ids):"); asrc = src[i0:i1]
for a, b in [("def attention(QKV):", "def arrival_over(QKV, cover):"), ("row = sc[p, :p+1]", "row = sc[p, cover(p)]"),
             ("v[:p+1]", "v[cover(p)]")]:
    assert asrc.count(a) == 1; asrc = asrc.replace(a, b)
exec(asrc)
COVER = {"whole":  lambda p: np.arange(p + 1),                       # 0..itself: must be the machine
         "found":  lambda p: np.array(sorted({0, p})),               # the first token and itself
         "second": lambda p: np.array(sorted({0, max(p - 1, 0), p})),# the first, the one before, itself
         "alone":  lambda p: np.array([p])}                          # only itself (the old base)
def arr(name): return lambda Q: arrival_over(Q, COVER[name])

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
PP = {}                                                      # plate photons on each line (lines in, over every neuron out)
for k in NAMES:
    w = M[k + ".w"]
    PP[k] = np.concatenate([ones(w[r:r + 128]).sum(1) for r in range(0, w.shape[0], 128)]); del w

refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
if ARGS: refs, TEXTS = refs[:int(ARGS[0])], TEXTS[:int(ARGS[0])]
WTE, WPE = M["wte"], M["wpe"]
NB = len(NAMES); NPOS = sum(len(s) for s in TEXTS); NT = len(TEXTS)
BASES = ["found", "second", "ctrl", "alone"]
Q = ["ph_ctx", "ph_base", "ph_page", "size_ctx", "size_base", "size_page", "zero", "lines", "moved",
     "work_ctx", "work_base", "work_page"]
REC = {b: {q: np.zeros((NPOS, NB), np.int64) for q in Q} for b in BASES}
T5 = {b: {q: np.zeros(NPOS, bool) for q in ("in5", "first", "set")} for b in BASES}
POS, TXT = np.zeros(NPOS, np.int64), np.zeros(NPOS, np.int64)
CHK = dict(whole_same=0, pos0_same=0, found_p1_same=0, second_p12_same=0, identity_bad=0, identity_n=0, int64_bad=0,
           int64_n=0, forward_same=0, words_same=0, words_n=0, race_is_first_of_top5=0, race_n=0)
rng = np.random.default_rng(7)
FOREIGN = []                                                 # the control's first word, one per text
for t in range(NT):
    cand = [u for u in range(len(refs)) if u != t and refs[u]["ids"][0] != TEXTS[t][0]]
    FOREIGN.append(int(refs[int(rng.choice(cand))]["ids"][0]))

def same_at(A, B, topA, topB, p):                            # every bank (x16, water level, pile) and the top read at p
    ok = np.array_equal(topA[p], topB[p])
    for k in NAMES: ok &= np.array_equal(A[k][0][p], B[k][0][p]) and np.array_equal(A[k][2][p], B[k][2][p]) and A[k][1][p] == B[k][1][p]
    return bool(ok)
def same_all(A, B, topA, topB):
    return all(same_at(A, B, topA, topB, p) for p in range(len(topA)))

def fill(D, r, bi, xc, xb, exc, exb, k, page):
    n = len(xc); oc, ob, op = ones(xc), ones(xb), ones(page)
    D["ph_ctx"][r:r + n, bi] = oc.sum(1); D["ph_base"][r:r + n, bi] = ob.sum(1); D["ph_page"][r:r + n, bi] = op.sum(1)
    D["size_ctx"][r:r + n, bi] = np.abs(xc).astype(np.int64).sum(1); D["size_base"][r:r + n, bi] = np.abs(xb).astype(np.int64).sum(1)
    D["size_page"][r:r + n, bi] = np.abs(page).astype(np.int64).sum(1)
    D["zero"][r:r + n, bi] = (page == 0).sum(1); D["lines"][r:r + n, bi] = xc.shape[1]; D["moved"][r:r + n, bi] = exc != exb
    D["work_ctx"][r:r + n, bi] = oc @ PP[k]; D["work_base"][r:r + n, bi] = ob @ PP[k]; D["work_page"][r:r + n, bi] = op @ PP[k]

def top5(top): return np.argsort(-top, axis=1, kind="stable")[:, :5]

t0 = time.time(); row = 0
for t, seq in enumerate(TEXTS):
    P = len(seq); R0 = WTE[seq] + WPE[:P]
    C, top_c = run(R0, attention)                            # the machine
    Wh, top_w = run(R0, arr("whole"))                        # the restricted arrival with the whole cover
    CHK["whole_same"] += same_all(C, Wh, top_c, top_w); del Wh, top_w
    RUNS = {"found": run(R0, arr("found")), "second": run(R0, arr("second")),
            "ctrl": run(WTE[[FOREIGN[t]] + seq[1:]] + WPE[:P], arr("found")), "alone": run(R0, arr("alone"))}
    CHK["forward_same"] += race(top_c[-1]) == FORWARD(seq)
    n0 = len(refs[t]["ids"])
    for p in range(n0 - 1, P - 1): CHK["words_same"] += race(top_c[p]) == seq[p + 1]; CHK["words_n"] += 1
    CHK["pos0_same"] += all(same_at(C, RUNS[b][0], top_c, RUNS[b][1], 0) for b in ("found", "second", "alone"))
    CHK["found_p1_same"] += same_at(C, RUNS["found"][0], top_c, RUNS["found"][1], 1)
    CHK["second_p12_same"] += same_at(C, RUNS["second"][0], top_c, RUNS["second"][1], 1) and same_at(C, RUNS["second"][0], top_c, RUNS["second"][1], 2)
    t5c = top5(top_c); ans = np.array([race(top_c[p]) for p in range(P)])
    CHK["race_is_first_of_top5"] += int((t5c[:, 0] == ans).sum()); CHK["race_n"] += P
    for b in BASES:
        B, top_b = RUNS[b]; t5b = top5(top_b)
        T5[b]["in5"][row:row + P] = [ans[p] in t5b[p] for p in range(P)]
        T5[b]["first"][row:row + P] = t5b[:, 0] == ans
        T5[b]["set"][row:row + P] = [set(t5b[p]) == set(t5c[p]) for p in range(P)]
        for bi, k in enumerate(NAMES):
            xc, exc, pc = C[k]; xb, exb, pb = B[k]; page = xc - xb
            CHK["identity_bad"] += not np.array_equal(pc, pb + machine_product(page, W[k][0])); CHK["identity_n"] += 1
            if t == 0 and b == "found":                      # the same identity in plain int64, no float anywhere
                w64 = M[k + ".w"]; ok = True
                for c in range(0, w64.shape[1], 4096):
                    wc = w64[:, c:c + 4096].astype(np.int64)
                    ok &= np.array_equal(pc[:, c:c + 4096], xb.astype(np.int64) @ wc + page.astype(np.int64) @ wc)
                CHK["int64_bad"] += not ok; CHK["int64_n"] += 1; del w64
            fill(REC[b], row, bi, xc, xb, exc, exb, k, page)
    POS[row:row + P] = np.arange(P); TXT[row:row + P] = t; row += P
    del C, RUNS
    if t % 5 == 4 or t == NT - 1:
        print(f"  {t + 1} texts ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB)  {CHK}", flush=True)

# the alone base, against the earlier thin-page run (its own records): must be the same numbers
OLD = {}
if NT == 30:
    d = np.load("thinpage/thin_page.npz")
    for mine, theirs in [("ph_ctx", "ph_ctx"), ("ph_base", "ph_own"), ("ph_page", "ph_thin"), ("size_ctx", "size_ctx"),
                         ("size_page", "size_thin"), ("zero", "zero"), ("moved", "moved"), ("work_ctx", "work_ctx"),
                         ("work_page", "work_thin")]:
        OLD[mine] = bool(np.array_equal(REC["alone"][mine], d["rec_" + theirs]))
print(f"  all done ({time.time() - t0:.0f}s)  {CHK}", flush=True)
np.savez(f"{HERE}/foundation/foundation.npz", names=np.array(NAMES), pos=POS, txt=TXT, foreign=np.array(FOREIGN),
         checks=json.dumps({k: int(v) for k, v in CHK.items()}),
         **{f"{b}_{q}": REC[b][q] for b in BASES for q in Q}, **{f"{b}_top5_{q}": T5[b][q] for b in BASES for q in T5[b]})

# ---------------------------------------------------------------- report
def pct(a, b): return 100.0 * a / b if b else float("nan")
NH = [NAMES.index(k) for k in NAMES if k != "head"]; HB = NAMES.index("head"); ALLB = list(range(NB))
LAB = {"found": "FOUNDATION {0, itself}", "second": "SECOND LEVEL {0, one before, itself}",
       "ctrl": "CONTROL: foreign first word, {0, itself}", "alone": "OLD BASE: alone {itself}"}
print(f"\nTHE FOUNDATION PAGE: {NT} texts, {NPOS} positions; every bank of 12 layers and the top read (every position)")
print("\n".join(PRED))
print(f"\nchecks: the restricted arrival with the whole cover == the machine (every bank, x16, water level, pile, top read, every position): {CHK['whole_same']}/{NT} texts")
print(f"        position 0 on the foundation, second level and alone == in context (every bank, top read): {CHK['pos0_same']}/{NT} texts")
print(f"        position 1 on the foundation == in context (its cover is the whole context): {CHK['found_p1_same']}/{NT};"
      f"  positions 1, 2 on the second level == in context: {CHK['second_p12_same']}/{NT}")
print(f"        identity pile_ctx == pile_base + page meeting the plate: {CHK['identity_n'] - CHK['identity_bad']}/{CHK['identity_n']} banks"
      f" (4 bases x {NB} banks x {NT} texts)   (text 0 foundation in plain int64: {CHK['int64_n'] - CHK['int64_bad']}/{CHK['int64_n']})")
print(f"        run() == base2.forward at the last position: {CHK['forward_same']}/{NT};  words == refs: {CHK['words_same']}/{CHK['words_n']}"
      f";  the machine's word == first of its top 5: {CHK['race_is_first_of_top5']}/{CHK['race_n']}")
if OLD: print(f"        alone base here == thinpage/thin_page.npz records: " + ", ".join(f"{k} {'same' if v else 'DIFFERENT'}" for k, v in OLD.items()))
print(f"        control first words (one per text): {FOREIGN}")

def table(b, sel, title):
    D = REC[b]
    print(f"\n{title}\n  photons per token: light in context | base | page; page as % of the light's photons; size % (page/light);"
          f" base size %; lines exactly 0 %; water level moved %; work %")
    for kind in ("qkv", "proj", "fc", "mproj", "head"):
        for l in (range(12) if kind != "head" else [None]):
            k = kind if l is None else f"{kind}.{l}"; bi = NAMES.index(k); s = sel; n = s.sum()
            c, o, th = D["ph_ctx"][s, bi].sum(), D["ph_base"][s, bi].sum(), D["ph_page"][s, bi].sum()
            print(f"  {k:8s} {c / n:7.0f} {o / n:7.0f} {th / n:7.0f}   page {pct(th, c):5.1f}%  size {pct(D['size_page'][s, bi].sum(), D['size_ctx'][s, bi].sum()):5.1f}%"
                  f"  base size {pct(D['size_base'][s, bi].sum(), D['size_ctx'][s, bi].sum()):5.1f}%"
                  f"  zero {pct(D['zero'][s, bi].sum(), D['lines'][s, bi].sum()):5.1f}%  moved {pct(D['moved'][s, bi].sum(), n):5.1f}%"
                  f"  work {pct(D['work_page'][s, bi].sum(), D['work_ctx'][s, bi].sum()):5.1f}%")
    print("  per layer (all four banks): photons page/light %, size %, work page/light %")
    for l in range(12):
        bis = [NAMES.index(f"{x}.{l}") for x in ("qkv", "proj", "fc", "mproj")]
        print(f"    layer {l:2d}: photons {pct(D['ph_page'][sel][:, bis].sum(), D['ph_ctx'][sel][:, bis].sum()):5.1f}%"
              f"   size {pct(D['size_page'][sel][:, bis].sum(), D['size_ctx'][sel][:, bis].sum()):5.1f}%"
              f"   work {pct(D['work_page'][sel][:, bis].sum(), D['work_ctx'][sel][:, bis].sum()):5.1f}%")
    for lab, bis in [("without the top read", NH), ("with the top read", ALLB)]:
        print(f"  overall {lab:22s}: photons {pct(D['ph_page'][sel][:, bis].sum(), D['ph_ctx'][sel][:, bis].sum()):5.1f}%"
              f"  size {pct(D['size_page'][sel][:, bis].sum(), D['size_ctx'][sel][:, bis].sum()):5.1f}%"
              f"  lines exactly 0 {pct(D['zero'][sel][:, bis].sum(), D['lines'][sel][:, bis].sum()):5.1f}%"
              f"  work {pct(D['work_page'][sel][:, bis].sum(), D['work_ctx'][sel][:, bis].sum()):5.1f}%"
              f"   (light work per token {D['work_ctx'][sel][:, bis].sum() / sel.sum():.4g} photon meetings)")
table("found", POS > 0, "FOUNDATION {0, itself}: page = in-context light - foundation light, positions 1 on")
table("found", POS > 1, "FOUNDATION, positions 2 on (position 1's cover is already the whole context)")
table("second", POS > 0, "SECOND LEVEL {0, one before, itself}, positions 1 on")
table("second", POS > 2, "SECOND LEVEL, positions 3 on (positions 1, 2 cover the whole context)")
table("ctrl", POS > 0, "CONTROL: foreign first word, {0, itself}; page against the real in-context light, positions 1 on")
table("alone", POS > 0, "OLD BASE: alone {itself}, positions 1 on")

print("\nSIDE BY SIDE, overall (banks of 12 layers | with the top read): page photons %, size %, lines exactly 0 %, water moved %, work %")
for lab, sel in [("positions 1 on", POS > 0), ("positions 2 on", POS > 1), ("positions 3 on", POS > 2)]:
    print(f"  {lab} ({sel.sum()} tokens)")
    for b in BASES:
        D = REC[b]; f = lambda q, bis: D[q][sel][:, bis].sum()
        print(f"    {LAB[b]:44s} photons {pct(f('ph_page', NH), f('ph_ctx', NH)):5.1f}% | {pct(f('ph_page', ALLB), f('ph_ctx', ALLB)):5.1f}%"
              f"   size {pct(f('size_page', NH), f('size_ctx', NH)):5.1f}% | {pct(f('size_page', ALLB), f('size_ctx', ALLB)):5.1f}%"
              f"   zero {pct(f('zero', NH), f('lines', NH)):4.1f}%   moved {pct(f('moved', NH), sel.sum() * len(NH)):4.1f}%"
              f"   work {pct(f('work_page', NH), f('work_ctx', NH)):5.1f}% | {pct(f('work_page', ALLB), f('work_ctx', ALLB)):5.1f}%")
print("\nSIDE BY SIDE, per layer (positions 1 on): page photons % / size %   foundation | second | control | alone")
for l in range(12):
    bis = [NAMES.index(f"{x}.{l}") for x in ("qkv", "proj", "fc", "mproj")]; s = POS > 0
    print(f"  layer {l:2d}: " + " | ".join(f"{pct(REC[b]['ph_page'][s][:, bis].sum(), REC[b]['ph_ctx'][s][:, bis].sum()):5.1f} / {pct(REC[b]['size_page'][s][:, bis].sum(), REC[b]['size_ctx'][s][:, bis].sum()):5.1f}" for b in BASES))
print(f"  top read: " + " | ".join(f"{pct(REC[b]['ph_page'][POS > 0, HB].sum(), REC[b]['ph_ctx'][POS > 0, HB].sum()):5.1f} / {pct(REC[b]['size_page'][POS > 0, HB].sum(), REC[b]['size_ctx'][POS > 0, HB].sum()):5.1f}" for b in BASES))
print("\nBY POSITION (banks of 12 layers): page photons % / size % / water moved %   foundation | second | control | alone")
for p in range(POS.max() + 1):
    s = POS == p
    print(f"  pos {p:2d} ({s.sum():2d}): " + " | ".join(
        f"{pct(REC[b]['ph_page'][s][:, NH].sum(), REC[b]['ph_ctx'][s][:, NH].sum()):5.1f} / {pct(REC[b]['size_page'][s][:, NH].sum(), REC[b]['size_ctx'][s][:, NH].sum()):5.1f}"
        f" / {pct(REC[b]['moved'][s][:, NH].sum(), s.sum() * len(NH)):4.1f}" for b in BASES))
print("\nWATER LEVEL (positions 1 on, banks of 12 layers): page photons % and size % split by whether the water level moved")
for b in BASES:
    D = REC[b]; s = POS > 0; mv = D["moved"][s][:, NH] == 1
    pg, cx, sp, sx = D["ph_page"][s][:, NH], D["ph_ctx"][s][:, NH], D["size_page"][s][:, NH], D["size_ctx"][s][:, NH]
    print(f"  {LAB[b]:44s} moved {mv.sum():6d}: photons {pct(pg[mv].sum(), cx[mv].sum()):5.1f}% size {pct(sp[mv].sum(), sx[mv].sum()):5.1f}%"
          f"   not moved {(~mv).sum():6d}: photons {pct(pg[~mv].sum(), cx[~mv].sum()):5.1f}% size {pct(sp[~mv].sum(), sx[~mv].sum()):5.1f}%")
print("\nTOP READ: does the base keep the machine's answer? (the machine's word in the base's top 5 | same first word | same top-5 set)")
for lab, sel in [("positions 1 on", POS > 0), ("positions 2 on", POS > 1), ("positions 3 on", POS > 2)]:
    print(f"  {lab} ({sel.sum()} tokens)")
    for b in BASES:
        print(f"    {LAB[b]:44s} in top 5 {T5[b]['in5'][sel].sum():4d} ({pct(T5[b]['in5'][sel].sum(), sel.sum()):5.1f}%)"
              f"   same first {T5[b]['first'][sel].sum():4d} ({pct(T5[b]['first'][sel].sum(), sel.sum()):5.1f}%)"
              f"   same set {T5[b]['set'][sel].sum():4d} ({pct(T5[b]['set'][sel].sum(), sel.sum()):5.1f}%)")
print(f"\nlargest memory held by this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
