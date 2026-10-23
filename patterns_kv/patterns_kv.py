"""PATTERNS + KV. His sentence, exactly: "There are only 50k fringe patterns. The KV shows the rotation and the angles.
That's it."

THE SESSION'S READING OF IT (the task's construction, not his words):
  A word's OWN PATTERN = the word alone at its position (the arrival restricted to itself), through all 12 layers. It is
  fixed, storable once per word AND position (so 50,257 x positions patterns, more than his 50k: that count is the
  session's reading). THE KV = the arrival over the earlier tokens. It supplies the turn at every layer.

THIS TEST CHANGES VALUES. It is judged by the top 5, not by bit-identity.

  M1 "patterns + KV, middle from storage". For a token at position p, at every layer l:
       light(l) = own stream(l) + (sum of the KV's pushes so far)
       the KV's push at layer l = the arrival's write (this light's query against every key and value of positions
           0..p, the machine's arrival exactly as it is, keys and values from M1's own run) MINUS the alone arrival's
           write (stored, from the own pattern).
       The middle (fc -> bend -> mproj) is NOT met: its write is the stored own pattern's middle write.
       The qkv plate IS met per token (for q, k, v); the proj plate is met on the arrival's output; the top plate at
       the end. fc and mproj are never met.
  M2 "patterns + KV, nothing met". As M1, but q, k, v are also the stored own pattern's: the push = shares (own
       query against the earlier tokens' own keys) times the earlier tokens' own values, through proj, minus the
       alone write. Plates met per token: proj (on the arrival's output) and the top plate. qkv, fc, mproj never.
  Baselines: the plain machine; the own pattern alone (no KV at all).
  The arrival covers positions 0..p, the token's own key included, as the machine's arrival does; its shares are
  shared out over all of them together, so the push is (whole write) minus (alone write), never a separate sum.

MEASURES
  (a) the top 5: at every position, against the plain machine at the same position:
      word-in-plain-top-5 (judge5's rule: the machine's word is one of the plain machine's top 5),
      plain-word-in-its-top-5, same top word, top-5 set kept (the same 5 words, any order).
  (b) how far each machine's light is from the plain machine's light, at every layer: the angle in whole degrees
      between the two lights as they meet the plate after the water level (x16 with its mean blocked, as the folded
      water level does; also the raw x16). 25 places: each layer's qkv place and fc place, and the top read.
  CONTROL (pairing broken): a random other word's own pattern + its own KV (the same machine run on another text, the
      same position, a different word) against this position's plain machine. Same for the own pattern alone.
  ORDER-FREE: the arrival's sum with the earlier tokens met in a random order gives the same write; the pushes summed
      in reverse order give the same light.
  DEFINITION-FIXED (reported, never counted as findings): position 0 is the plain machine in every machine; layer
      0's qkv place is the plain light in every machine; layer 0's fc place is the plain light in M1 and M2.
  Texts: the 30 refs texts (all 648 positions) and judge5's 300 windows (positions 3..22, 6,000 judged positions).

PREDICTIONS (written before any run):
  his:      M1 and M2 keep the answer.
  Claude's: M1 keeps it in under 30% of positions and M2 in under 15%, because the middle's response to the KV (the
            glow at the gates) is missing; the own pattern alone keeps about 7%.

    python3 patterns_kv/patterns_kv.py refs                 -> patterns_kv/refs.npz
    python3 patterns_kv/patterns_kv.py judge START END      -> patterns_kv/judge_START_END.npz
    python3 patterns_kv/patterns_kv.py quick N               (N refs texts, no file written)
Whole numbers only in every machine. The angle uses one fixed table of 180 whole numbers (each boundary's cosine
times 2^62, made once), and every comparison against it is exact on whole numbers.
"""
import sys, os, json, time, math, resource, numpy as np
ARGS = sys.argv[1:]
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)                                                    # the machine, untouched
i0, i1 = src.index("def attention(QKV):"), src.index("def forward(ids):"); asrc = src[i0:i1]
for a, b in [("def attention(QKV):", "def attention_alone(QKV):"), ("row = sc[p, :p+1]", "row = sc[p, p:p+1]"),
             ("v[:p+1]", "v[p:p+1]")]:
    assert asrc.count(a) == 1; asrc = asrc.replace(a, b)
exec(asrc)                                                   # the alone arrival (as thinpage/thin_page.py)
WTE, WPE = M["wte"], M["wpe"]
MACH = ["plain", "own", "M1", "M2"]
PLACES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["top"]

def plain(R0):                                               # the machine, the light kept at every place
    R = R0; L = []
    for l in range(12):
        L.append(R); R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        L.append(R); R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    L.append(R); return L

def own(R0):                                                 # each word alone at its position: the own pattern
    R = R0; L, QKV, A, F = [], [], [], []
    for l in range(12):
        L.append(R); q = gauged_bank(R, f"qkv.{l}"); a = plain_bank(attention_alone(q), f"proj.{l}")
        QKV.append(q); A.append(a); R = R + a
        L.append(R); f = plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}"); F.append(f); R = R + f
    L.append(R); return L, QKV, A, F

def patterns_kv(O, nothing_met):                             # M1 (nothing_met False) and M2 (True)
    Lo, Qo, Ao, Fo = O
    KV = np.zeros_like(Lo[0]); L, PUSH, QKV = [], [], []
    for l in range(12):
        light = Lo[2 * l] + KV; L.append(light)              # own stream + the KV's pushes so far
        q = Qo[l] if nothing_met else gauged_bank(light, f"qkv.{l}")
        write = plain_bank(attention(q), f"proj.{l}")        # the arrival's write over positions 0..p
        push = write - Ao[l]; KV = KV + push; PUSH.append(push); QKV.append(q)
        mid = Lo[2 * l + 1] + KV; L.append(mid)
        assert np.array_equal(mid, light + write)            # the same light, written the plain way
        assert np.array_equal(Lo[2 * l + 2] + KV, mid + Fo[l])   # the middle's write: the stored own one
    L.append(Lo[24] + KV)
    rev = Lo[24].copy()
    for pu in PUSH[::-1]: rev = rev + pu                     # ORDER-FREE: the pushes summed in reverse order
    assert np.array_equal(rev, L[24])
    return L, QKV

def top_scores(R):
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b

def top5(S):                                                 # as judge5.py
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def arrival_last(QKV, order):                                # the machine's arrival lines for the last position,
    P = QKV.shape[0]; p = P - 1; out = np.zeros(768, np.int64)   # every key and value met in `order`
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q[p:p+1], -ES); k16, ek = to16(k[order], -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))[0]
        d = sc[race(sc)] - sc
        a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
        num = product(a[None, :], v[order].astype(np.float64))[0]
        out[h*64:(h+1)*64] = power(num, -log2fix([a.sum()])[0])
    return out

# ---------------------------------------------------------------- the angle, whole degrees, exact on whole numbers
BOUND = [int(round(math.cos(math.radians(k - 0.5)) * 2 ** 62)) for k in range(1, 181)]   # fixed table, made once
def _below(dot, nn, B):                                      # cos < B / 2^62, with cos = dot / sqrt(nn), exactly
    if dot >= 0:
        return B > 0 and (dot * dot << 124) < B * B * nn
    return B > 0 or (dot * dot << 124) > B * B * nn
def angle(dot, na, nb):                                      # whole degrees: the number of half-degree boundaries passed
    if na <= 0 or nb <= 0: return -1
    nn = na * nb; lo, hi = 0, 180
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _below(dot, nn, BOUND[mid - 1]): lo = mid
        else: hi = mid - 1
    return lo
assert angle(5, 25, 1) == 0 and angle(-5, 25, 1) == 180 and angle(0, 4, 9) == 90 and angle(1, 2, 1) == 45
assert angle(1, 4, 1) == 60 and angle(1, 3, 1) in (54, 55)  # cos 60 = 1/2; arccos(1/sqrt 3) = 54.7
def angles(A, B):                                            # rows of x16 (whole numbers): (raw, mean blocked)
    A = A.astype(np.int64); B = B.astype(np.int64)
    d, na, nb, sa, sb = (A * B).sum(1), (A * A).sum(1), (B * B).sum(1), A.sum(1), B.sum(1)
    raw = np.array([angle(int(x), int(y), int(z)) for x, y, z in zip(d, na, nb)], np.int64)
    cen = np.array([angle(768 * int(x) - int(s) * int(t), 768 * int(y) - int(s) ** 2, 768 * int(z) - int(t) ** 2)
                    for x, y, z, s, t in zip(d, na, nb, sa, sb)], np.int64)
    return raw, cen

def x16_all(L, rows):                                        # (rows, 25, 768): every place's light after the water level
    return np.stack([to16(L[i][rows], -ES)[0] for i in range(25)], 1).astype(np.int32)

def machines(seq, rows):                                     # every machine on one text; lights and top 5 at `rows`
    P = len(seq); R0 = WTE[seq] + WPE[:P]
    Lp = plain(R0); O = own(R0); L1, Q1 = patterns_kv(O, False); L2, _ = patterns_kv(O, True)
    LL = {"plain": Lp, "own": O[0], "M1": L1, "M2": L2}
    X = {m: x16_all(LL[m], rows) for m in MACH}
    T5, T1 = {}, {}
    for m in MACH:
        S = top_scores(LL[m][24][rows]); T5[m] = top5(S); T1[m] = S.argmax(1)
    chk = dict(pos0=0, pos0_n=0, defin=0, defin_n=0)
    if rows[0] == 0:                                         # position 0: every machine is the plain machine
        for m in MACH[1:]:
            chk["pos0"] += all(np.array_equal(LL[m][i][0], Lp[i][0]) for i in range(25)); chk["pos0_n"] += 1
    for m, places in [("own", [0]), ("M1", [0, 1]), ("M2", [0, 1])]:   # the places fixed by definition
        for i in places: chk["defin"] += np.array_equal(LL[m][i], Lp[i]); chk["defin_n"] += 1
    return X, T5, T1, chk, Q1

def measure(Xa, Xp):                                         # angles at all 25 places: (rows, 25) raw and mean blocked
    n = Xa.shape[0]; raw = np.zeros((n, 25), np.int64); cen = np.zeros((n, 25), np.int64)
    for i in range(25): raw[:, i], cen[:, i] = angles(Xa[:, i], Xp[:, i])
    return raw, cen

CMP = ["own", "M1", "M2"]                                    # real: machine vs plain;  control: other word's vs plain
def run_refs(ntexts=None, save=True):
    refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
    if ntexts: refs, TEXTS = refs[:ntexts], TEXTS[:ntexts]
    NPOS = sum(len(s) for s in TEXTS); start = np.cumsum([0] + [len(s) for s in TEXTS])
    XS = {m: np.zeros((NPOS, 25, 768), np.int32) for m in MACH}
    T5 = {m: np.zeros((NPOS, 5), np.int64) for m in MACH}; T1 = {m: np.zeros(NPOS, np.int64) for m in MACH}
    POS, TXT, WORD = np.zeros(NPOS, np.int64), np.zeros(NPOS, np.int64), np.zeros(NPOS, np.int64)
    CHK = dict(pos0=0, pos0_n=0, defin=0, defin_n=0, words_same=0, words_n=0, orderfree=0, orderfree_n=0)
    t0 = time.time()
    for t, seq in enumerate(TEXTS):
        P = len(seq); r0 = start[t]; rows = np.arange(P)
        X, t5, t1, chk, Q1 = machines(seq, rows)
        for k, v in chk.items(): CHK[k] += v
        for m in MACH: XS[m][r0:r0 + P] = X[m]; T5[m][r0:r0 + P] = t5[m]; T1[m][r0:r0 + P] = t1[m]
        POS[r0:r0 + P] = rows; TXT[r0:r0 + P] = t; WORD[r0:r0 + P] = seq
        n0 = len(refs[t]["ids"])
        for p in range(n0 - 1, P - 1): CHK["words_same"] += int(t1["plain"][p] == seq[p + 1]); CHK["words_n"] += 1
        if t == 0:                                           # ORDER-FREE: M1's arrival, last position, keys in a random order
            rng = np.random.default_rng(11)
            for l in range(12):
                ref = arrival_last(Q1[l], np.arange(P))
                ok = np.array_equal(ref, attention(Q1[l])[-1])
                ok &= np.array_equal(arrival_last(Q1[l], rng.permutation(P)), ref)
                CHK["orderfree"] += bool(ok); CHK["orderfree_n"] += 1
        if t % 5 == 4: print(f"  {t + 1} texts ({time.time() - t0:.0f}s)  {CHK}", flush=True)
    rng = np.random.default_rng(4)                           # CONTROL partner: another text, same position, other word
    PART = np.full(NPOS, -1, np.int64)
    for r in range(NPOS):
        t, p = TXT[r], POS[r]
        cand = [u for u in range(len(TEXTS)) if u != t and len(TEXTS[u]) > p and TEXTS[u][p] != TEXTS[t][p]]
        if cand: PART[r] = start[int(rng.choice(cand))] + p
    out = dict(pos=POS, txt=TXT, word=WORD, part=PART, checks=json.dumps({k: int(v) for k, v in CHK.items()}))
    for m in MACH: out[f"t5_{m}"] = T5[m]; out[f"t1_{m}"] = T1[m]
    for m in CMP:
        out[f"raw_{m}"], out[f"cen_{m}"] = measure(XS[m], XS["plain"])
    ok = PART >= 0
    for m in CMP + ["plain"]:                                # the other word's machine against this plain light
        raw = np.full((NPOS, 25), -1, np.int64); cen = np.full((NPOS, 25), -1, np.int64)
        raw[ok], cen[ok] = measure(XS[m][PART[ok]], XS["plain"][ok])
        out[f"craw_{m}"], out[f"ccen_{m}"] = raw, cen
    print(f"  refs done ({time.time() - t0:.0f}s)  {CHK}", flush=True)
    if save: np.savez(f"{HERE}/patterns_kv/refs.npz", **out)
    return out

def run_judge(s, e):
    WIN = json.load(open("judge5_texts.json"))[s:e]; REF = np.load("judge5_ref.npz")["top5"][s:e]
    n = len(WIN); rows = np.arange(3, 23)
    out = {f"t5_{m}": np.zeros((n, 20, 5), np.int64) for m in MACH}
    out.update({f"t1_{m}": np.zeros((n, 20), np.int64) for m in MACH})
    for m in CMP: out[f"raw_{m}"] = np.zeros((n, 20, 25), np.int64); out[f"cen_{m}"] = np.zeros((n, 20, 25), np.int64)
    for m in CMP + ["plain"]:
        out[f"craw_{m}"] = np.full((n, 20, 25), -1, np.int64); out[f"ccen_{m}"] = np.full((n, 20, 25), -1, np.int64)
    CHK = dict(ref_same=0, ref_n=0, defin=0, defin_n=0)
    words = np.array([w["ids"] for w in WIN])[:, 3:23]
    t0 = time.time(); first = prev = None
    def control(i, j, Xi, Xj):                               # window i's plain light vs window j's machines, other words only
        ok = words[i] != words[j]
        for m in CMP + ["plain"]:
            raw, cen = measure(Xj[m][ok], Xi["plain"][ok]); out[f"craw_{m}"][i][ok] = raw; out[f"ccen_{m}"][i][ok] = cen
    for i, win in enumerate(WIN):
        X, t5, t1, chk, _ = machines(win["ids"], rows)
        CHK["defin"] += chk["defin"]; CHK["defin_n"] += chk["defin_n"]
        CHK["ref_same"] += int(np.array_equal(t5["plain"], REF[i])); CHK["ref_n"] += 1
        for m in MACH: out[f"t5_{m}"][i] = t5[m]; out[f"t1_{m}"][i] = t1[m]
        for m in CMP: out[f"raw_{m}"][i], out[f"cen_{m}"][i] = measure(X[m], X["plain"])
        if prev is not None: control(i, i - 1, X, prev)      # partner: the window before
        if first is None: first = X
        prev = X
        if i % 10 == 9: print(f"  windows {s}..{s + i} ({time.time() - t0:.0f}s)  {CHK}", flush=True)
    control(0, n - 1, first, prev)                           # the first window's partner: the last one
    out["checks"] = json.dumps(CHK); out["words"] = words; out["ref"] = REF
    print(f"  judge {s}..{e} done ({time.time() - t0:.0f}s)  {CHK}", flush=True)
    np.savez(f"{HERE}/patterns_kv/judge_{s}_{e}.npz", **out)

if ARGS[0] == "refs": run_refs()
elif ARGS[0] == "judge": run_judge(int(ARGS[1]), int(ARGS[2]))
elif ARGS[0] == "quick":
    o = run_refs(int(ARGS[1]), save=False)
    p = o["pos"] > 0; ref = o["t5_plain"]
    for m in CMP:
        inn = (ref[p] == o[f"t1_{m}"][p][:, None]).any(1).sum(); same = (o[f"t1_{m}"][p] == o["t1_plain"][p]).sum()
        print(f"  {m}: in plain top 5 {inn}/{p.sum()}  same top word {same}/{p.sum()}"
              f"  angle at top (mean blocked) median {int(np.median(o[f'cen_{m}'][p][:, 24]))}")
print(f"largest memory held by this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 100 // 2**30 / 100} GB")
