"""BASEDOWN/LINEUP: THE LIGHT LINED UP ACROSS THE TOKENS, AT EVERY PLATE OF EVERY LAYER (GPT-2, all 48 banks).
Whole numbers only. Nothing in the machine is changed; only how each bank's pile is formed.

HIS WORDS (2026-10-07), the only sentences quoted as his:
  "It's removable as in convert them to base 2 and line them all up, any shared 1s in a line cancel, any shared zeros
  cancel"
  "Everything these cancel with ARE the base. So every single weight they touch we can extract those also, and alllllll
  the way down all The layers"

THE SESSION'S RESTATEMENT (the session's words, not his): carry that line-up down the machine. At every plate of every
layer, the light of all the tokens flowing together is lined up line by line across the tokens. For one line j the
order of the tokens depends only on the light, so ONE order per line serves every neuron of the plate: for each neuron
n, walk the line's tokens in order, each product = the previous token's product + w[j,n] x gap (the shared 1s and 0s
cancel); then each product lands in its own token's pile. Exact by construction; the question is the count.

THE LIGHT: the machine's own light at all 48 banks, all 648 positions of the 30 refs texts (groups/light_gpt2/
{qkv,proj,fc,mproj}.npy, 12 x 648 x lines, int32, x16 as base2 writes it). The plates: base2_gpt2.npz (int16).
Every pile is checked against base2.py's own run (recording/C_ref/plain_refs.npz) and the direct product.

THE RULES (fixed here, before any run; nothing in them is read from results)
  SOUPS: (a) each text on its own (30 soups of 19-24 tokens: the soup of one text, every position at once);
         (b) all 648 positions together (one soup).
  THE LINE-UP of line j in a soup: the tokens' values on line j, written in base 2. A dark value (0) takes no part
     (its product is 0, nothing lands). The lit values are lined up from 0 outward: the positive ones from the
     smallest up, the negative ones from the one nearest 0 down (equal values in the order the tokens came). Each
     token's GAP = its value minus the previous value on its side; the first on each side is written against 0 (its
     gap is its value whole). One order per line, from the light alone, shared by every neuron of the plate.
  THE WALK for neuron n: the running product starts at 0 on each side; at each token it takes w[j,n] x gap (the gap's
     signed digits meet the weight's signed digits: naf(gap) x naf(w[j,n]) photons); a gap of 0 adds nothing.
  THE LANDING: after each step the running product (= w[j,n] x the token's value, checked) lands in the token's own
     pile at neuron n: its 1s (the product is a whole number in base 2; writing it in signed digits first would cost
     its 1s again, so the 1s are the cheapest landing; its signed digits are reported beside it, count only).
  THE LIGHT'S WRITING: every gap written in signed digits, its 1s (as base2_fringes2 counts the light's writing at
     every bank); and the gaps formed: each gap is a value minus its neighbour, the neighbour's 1s (nothing for the
     first on a side). Compares (the line-up's order) are a count, kept apart: p x ceil(log2 p) for p lit values.
  PLAIN: lit light x lit plate, ones(x) @ wrow (wrow_j = the 1s of line j's weights), as every earlier machine.
  BESIDE IT: the direct signed-digit form on the same light (naf(x) @ nrow + the light's 1s written); the group
     machine's banks as base2_groups.py measured them (qkv 19.52%, proj 25.28%, fc 21.15%, mproj 25.35% of their
     plain); basedown/combine.py rebuilds them on this same light.
  WHAT CANCELS: the share of each plate's lit places that cancels when lined up = 1 - (the gaps' 1s) / (the values'
     1s), over the lit values (and the same in signed digits).

CONTROLS AND CHECKS
  exact: every pile = recording/C_ref/plain_refs.npz's (base2.py's own run, a separate process) and = the direct
    product (walk.c's `direct`, lines met one by one, int64), every bank, every position; every running product
    = w x the token's value (walk.c counts any that differ: must be 0).
  order-free: the tokens of each soup presented in a random order and the lines walked in a random order: every pile
    identical; every count identical (the line-up's gaps are the same multiset in any presented order).
  shuffle control (count only): each position's light shuffled across its lines (every value kept, which line holds
    it broken), lined up and walked the same way; its own piles checked against its own direct product; its counts
    against its own plain.
  the landing recounted in numpy (sum over every line, neuron and token of the 1s of w x x) at layer 5 of each kind.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the work of forming every exact pile; the light's 1s enter only as the plain reference and the
       cancel share the task asked for (reported, not the measure of the machine).
  13.2 the base is base2.py's own light and plates; no part of the machine is changed.
  13.3 no agreement statistic: the shuffle control measures work.
  13.4 FIXED BY THE DEFINITIONS, so not measured by the run: the landings are the 1s of every product w x x, the same
       numbers in (a), (b), any order and any soup; so "(b) the landings do not fall" is true by definition. The run
       measures the walk, the cancel share and the landings' size against plain.
  13.5 the restatement and every rule are the session's; the task's prediction is labelled as the task's.
  13.6 every bank, every position, every text.

ANALYSIS BEFORE THE RUN (the session's)
  - One look before this file was written (qkv.3, the 648 positions sorted from the smallest value, 1s): values 5.09
    lit places, gaps 1.75: 66% cancels. The word table's 89% (task) has 50,257 words a line; here (b) has 648 and (a)
    19-24, so each gap spans about 78x (b) or about 2,500x (a) the table's: some 6 and some 11 more places a gap.
  - The same look, 200 random tokens at qkv.3: a product w x x has 10.2 1s (7.4 signed digits) against 31.6 plain
    meetings a pair (lit light x lit plate). A landing is one whole product a line, neuron and token, so the
    landings alone are about a third of plain whatever the line-up does. The group machine forms the same pile with
    about 1.2 reads a line of about 4.5 signed digits each: about 5-6 photons a line against 10 for one landing.
PREDICTIONS (written before any run of this file)
  THE TASK'S (labelled as the task's): every plate's light cancels 85-92% when lined up across tokens; the walk is 1-4%
    of plain; the landings stay at about 25-35%, so the machine is about 30-40% -- worse than the group machine --
    unless the landings are shared; with more tokens a line (b) the walk falls further, the landings do not.
  THE SESSION'S:
  - Cancel share: (b) 55-70% at every plate kind (the task's 85-92% missed: 648 tokens a line, not 50,257); (a) 15-35%.
  - The walk: (b) 15-25% of plain, (a) 35-50% (the task's 1-4% missed: the gaps keep 1-3 lit places a value, and each
    meets every lit place of every weight of the line).
  - The landings: 28-40% of plain, identical in (a) and (b) (by definition).
  - The light's writing and the gaps formed: under 0.2% of plain; compares a few per value (count only).
  - Whole: (b) 45-60% of plain, (a) 65-85%: about 2-3x the group machine's banks (19.5-25.4%), (b) about the direct
    signed-digit form (53-59%), (a) above it.
  - Exact: every pile identical at every bank and position; every running product = w x x.
  - Order-free: every pile and every count identical.
  - Shuffle control: the light shuffled across its lines cancels less (b: 5-20 points less), its walk a larger share
    of its plain; its landings about the same share of its plain.

    python3 basedown/lineup.py  -> basedown/out/lineup.log, lineup.json
"""
import os, sys, json, time, ctypes, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; BD = f"{HERE}/basedown"; OUT = f"{BD}/out"
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/lineup.log", "w"); T0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

src, so = f"{BD}/walk.c", f"{BD}/walk.dylib"
if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
    import subprocess; subprocess.check_call(["cc", "-O3", "-mcpu=apple-m1", "-shared", "-o", so, src])
LIB = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
LIB.walk.argtypes = [I, I, I] + [V] * 10; LIB.walk.restype = None
LIB.direct.argtypes = [I, I, I, V, V, V]; LIB.direct.restype = None

def direct(x, w16):
    x = np.ascontiguousarray(x, np.int32); w16 = np.ascontiguousarray(w16, np.int16); P, N = x.shape; M = w16.shape[1]
    out = np.empty((P, M), np.int64); LIB.direct(P, N, M, x.ctypes.data, w16.ctypes.data, out.ctypes.data); return out

def lineup(x, soups):
    """the entries of every line: x (P, N) the light as presented; soups: list of row arrays. Returns per line the
    lit tokens in lined-up order with their gaps, start flags, the previous value (for the gaps formed), offsets."""
    P, N = x.shape; TOK, GAP, ST, PRV, LIT = [], [], [], [], []
    for rows in soups:
        v = x[rows].astype(np.int64)                                         # (p, N)
        key = np.where(v > 0, v, np.where(v < 0, (1 << 16) - v, 1 << 20))    # positives up from 0, then negatives down
        o = np.argsort(key, axis=0, kind="stable")
        sv = np.take_along_axis(v, o, 0); neg = sv < 0
        prev = np.vstack([np.zeros((1, N), np.int64), sv[:-1]]); pneg = np.vstack([np.zeros((1, N), bool), neg[:-1]])
        start = np.zeros_like(neg); start[0] = True; start[1:] = neg[1:] != pneg[1:]
        prev = np.where(start, 0, prev)
        TOK.append(np.asarray(rows)[o]); GAP.append(sv - prev); ST.append(start); PRV.append(prev); LIT.append(sv != 0)
    TOK, GAP, ST, PRV, LIT = (np.concatenate(a, 0) for a in (TOK, GAP, ST, PRV, LIT))   # soups stacked, (P, N)
    L = LIT.T; off = np.concatenate([[0], np.cumsum(L.sum(1))]).astype(np.int64)
    e = dict(tok=np.ascontiguousarray(TOK.T[L], np.int32), gap=np.ascontiguousarray(GAP.T[L], np.int32),
             start=np.ascontiguousarray(ST.T[L], np.uint8), prev=PRV.T[L], off=off)
    assert np.abs(e["gap"]).max(initial=0) <= 1 << 15
    e["compares"] = sum(int(np.sum(np.where(c > 1, c * np.ceil(np.log2(np.maximum(c, 2))), 0)))
                        for rows in soups for c in [(x[rows] != 0).sum(0)])
    return e

def run_walk(x, w16, e, lorder):
    x = np.ascontiguousarray(x, np.int32); P, N = x.shape; M = w16.shape[1]
    pile = np.zeros((P, M), np.int64); run = np.zeros(M, np.int64); out = np.zeros(6, np.int64)
    lorder = np.ascontiguousarray(lorder, np.int64)
    LIB.walk(P, N, M, x.ctypes.data, w16.ctypes.data, lorder.ctypes.data, e["off"].ctypes.data, e["tok"].ctypes.data,
             e["gap"].ctypes.data, e["start"].ctypes.data, pile.ctypes.data, run.ctypes.data, out.ctypes.data)
    return pile, dict(zip(("walk", "land", "land_sd", "bad", "entries", "nzgap"), (int(v) for v in out)))

def measure(x, w16, wrow, nrow, soups, lorder):
    e = lineup(x, soups); pile, c = run_walk(x, w16, e, lorder)
    lv = x[x != 0]
    lineidx = np.repeat(np.arange(x.shape[1]), np.diff(e["off"]))
    c["walk_py"] = int((naf(e["gap"]) * nrow[lineidx]).sum())                # the walk recounted from the entries
    c["write"] = int(ones(e["gap"]).sum()); c["formed"] = int(ones(e["prev"]).sum()); c["compares"] = int(e["compares"])
    c["v1"] = int(ones(lv).sum()); c["g1"] = int(ones(e["gap"]).sum()); c["vsd"] = int(naf(lv).sum()); c["gsd"] = int(naf(e["gap"]).sum())
    c["plain"] = int((ones(x) @ wrow).sum()); c["fr"] = int((naf(x) @ nrow).sum() + ones(x).sum())
    c["work"] = c["walk"] + c["land"] + c["write"] + c["formed"]
    return pile, c

REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = REF["npos"].astype(int); P = int(NPOS.sum())
Wz = np.load(f"{HERE}/base2_gpt2.npz")
LIGHT = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
starts = np.concatenate([[0], np.cumsum(NPOS)])
SOUPS = {"a": [np.arange(starts[i], starts[i + 1]) for i in range(30)], "b": [np.arange(P)]}
SLAB = ["qkv", "proj", "fc", "mproj"]
GROUP_PCT = dict(qkv=19.52, proj=25.28, fc=21.15, mproj=25.35)

log("BASEDOWN/LINEUP: the light lined up across the tokens at every plate (GPT-2, 48 banks, 648 refs positions)")
log(f"  soups (a) each text ({NPOS.min()}-{NPOS.max()} tokens), (b) all {P} positions together; whole numbers only")
REC = []; CHK = dict(ref=0, direct=0, bad=0, walk_py=0, of_piles=0, of_counts=0, ctl_direct=0, ctl_bad=0, land_np=0)
for l in range(12):
    for kind in SLAB:
        tb = time.time(); name = f"{kind}.{l}"
        x = np.ascontiguousarray(LIGHT[kind][l], np.int32); w16 = np.ascontiguousarray(Wz[f"{name}.w"], np.int16)
        ref = np.asarray(REF[f"{kind}_pile_{l}"]); N, M = w16.shape
        wrow = ones(w16).sum(1); nrow = naf(w16).sum(1)
        d0 = direct(x, w16); CHK["direct"] += int((d0 != ref).any(1).sum())
        rng = np.random.default_rng([7, SLAB.index(kind), l])
        xs = rng.permuted(x, axis=1).astype(np.int32); ds = direct(xs, w16)          # the shuffle control's light
        rec = dict(bank=name, kind=kind, layer=l)
        for sm in ("a", "b"):
            soups = SOUPS[sm]
            pile, c = measure(x, w16, wrow, nrow, soups, np.arange(N))
            CHK["ref"] += int((pile != ref).any(1).sum()); CHK["bad"] += c["bad"]; CHK["walk_py"] += int(c["walk_py"] != c["walk"])
            rec[sm] = c
            # ORDER-FREE: each soup's tokens in a random order, the lines walked in a random order
            perm = np.concatenate([rng.permutation(r) for r in soups]); lo = rng.permutation(N)
            pp, cp = measure(np.ascontiguousarray(x[perm]), w16, wrow, nrow, soups, lo)
            back = np.empty_like(pp); back[perm] = pp
            CHK["of_piles"] += int((back != ref).any(1).sum())
            CHK["of_counts"] += sum(int(cp[k] != c[k]) for k in ("walk", "land", "land_sd", "write", "formed", "entries", "nzgap"))
            # SHUFFLE CONTROL: the light shuffled across its lines
            ps, cs = measure(xs, w16, wrow, nrow, soups, np.arange(N))
            CHK["ctl_direct"] += int((ps != ds).any(1).sum()); CHK["ctl_bad"] += cs["bad"]
            rec[sm + "_ctl"] = cs
        if l == 5:                                                                  # the landing recounted in numpy
            w64 = w16.astype(np.int64); tot = 0
            for t in range(P): tot += int(ones(x[t].astype(np.int64)[:, None] * w64).sum())
            rec["land_np"] = tot; CHK["land_np"] += int(tot != rec["b"]["land"])
        REC.append(rec); a, b = rec["a"], rec["b"]
        log(f"  {name:8s} plain {b['plain']:>13,}  cancel (a) {100 * (1 - a['g1'] / a['v1']):5.1f}% (b) {100 * (1 - b['g1'] / b['v1']):5.1f}%"
            f"  walk (a) {100 * a['walk'] / a['plain']:5.2f}% (b) {100 * b['walk'] / b['plain']:5.2f}%  landings {100 * b['land'] / b['plain']:5.2f}%"
            f"  whole (a) {100 * a['work'] / a['plain']:5.2f}% (b) {100 * b['work'] / b['plain']:5.2f}%"
            f"  | ctl cancel (b) {100 * (1 - rec['b_ctl']['g1'] / rec['b_ctl']['v1']):5.1f}%  ({time.time() - tb:.0f}s, {rss()} MB)")
        json.dump(dict(rec=REC, chk=CHK), open(f"{OUT}/lineup.json", "w"))

def S(sm, k, kinds=SLAB): return sum(r[sm][k] for r in REC if r["kind"] in kinds)
log("")
log("CHECKS: " + ", ".join(f"{k} {v}" for k, v in CHK.items()))
log("  ref = piles differing from plain_refs (rows, 96 runs); direct = the direct product against plain_refs; bad = running products")
log("  differing from w x x; walk_py = banks where the walk recounted in numpy differs; of_* = order-free piles / counts differing;")
log("  ctl_* = the shuffle control's piles against its own direct product; land_np = banks where numpy's landing count differs")
for sm, lab in (("a", "(a) each text"), ("b", "(b) all 648 together"), ("a_ctl", "(a) SHUFFLE CONTROL"), ("b_ctl", "(b) SHUFFLE CONTROL")):
    log(""); log(f"{lab}: a token = total / {P}")
    for kinds, kl in [([k], k) for k in SLAB] + [(SLAB, "ALL")]:
        pl = S(sm, "plain", kinds); f = lambda k: 100 * S(sm, k, kinds) / pl
        cut = 100 * (1 - S(sm, "g1", kinds) / S(sm, "v1", kinds)); cutsd = 100 * (1 - S(sm, "gsd", kinds) / S(sm, "vsd", kinds))
        gp = f" | group machine {GROUP_PCT[kl]:.2f}%" if kl in GROUP_PCT else ""
        log(f"  {kl:5s} plain {pl / P:>13,.0f}  cancel 1s {cut:5.1f}% (signed {cutsd:5.1f}%)  walk {f('walk'):6.2f}%  landings {f('land'):6.2f}%"
            f" (signed {f('land_sd'):6.2f}%)  writing {f('write'):5.3f}%  formed {f('formed'):5.3f}%  WHOLE {f('work'):6.2f}%"
            f"  | direct signed {f('fr'):6.2f}%{gp}  compares {S(sm, 'compares', kinds) / P:,.0f}")
log("")
log("CANCEL SHARE BY PLATE (1s): min / median / max over the 12 layers")
for kind in SLAB:
    for sm in ("a", "b", "b_ctl"):
        v = [100 * (1 - r[sm]["g1"] / r[sm]["v1"]) for r in REC if r["kind"] == kind]
        log(f"  {kind:5s} {sm:5s} {min(v):5.1f}% / {float(np.median(v)):5.1f}% / {max(v):5.1f}%")
log(f"done ({time.time() - T0:.0f}s, max memory {rss()} MB)")
assert all(v == 0 for v in CHK.values()), CHK
