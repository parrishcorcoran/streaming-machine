"""The all-base-2 machine for the tiny GPT-2-shaped builds, checked against the float model's own words.
    python3 machine.py width=64 layers=2 recipe=A [steps=0,10,...] [plates=finished,pushes,scr1,scr2,scr3]
                       [order=plain|shuffle] [matmul=float64|int]

Every value a whole number. Follows ../base2.py (GPT-2) and ../foundation0/pythia_base2.py (the gauge with its gain
and offset as their own numbers, products checked whole). GPT-2's own parts as they are:
  - one after the other: R <- R + out(arrival(qkv(gauge1 R)));  R <- R + down(bend(up(gauge2 R)))
  - the gauge (water level): each lane's n x - sum x over sqrt(n sum x^2 - (sum x)^2 + eps n^2) -- the log of the size
    halved, no root and no divide -- times the gain, plus the offset
  - the score's 1/sqrt(16) = 1/4: a shift of 2 (a half power of 4, read as the table's fine part 0 and a shift)
  - the arrival: the clock starts at whoever arrives first in the row (the water level), e^-d from its table, the sum
    divided by the total through its log
  - the bend: GPT-2's own tanh form, x times the share read from the knife-edge table
  - the input word row and the position row, each on its own power of two, onto the stream's scale 2^-16
  - the output word table read at the top; the word: whoever arrives first (the race)
Every position's word is read in one pass over the held-out text (nothing in the machine reaches forward in the text).

plates (from the machine file, all on the finished plate's grid, so each is a whole number per neuron power):
  finished  .w                  pushes  .w - .s (the blank taken out, digit for digit)
  scrK      .w - .s + the blank scrambled within each neuron (snapio.scramble, the same draw as eval.py's)
  (gains, offsets and biases are the finished ones in every case; their start was not drawn)

Compared against evals/eval_<shape>_<recipe>.json (the float64 model of the same weights, eval.py):
  own       the machine's top word against the float model of the SAME plates (the machine check)
  lead      the machine's own lead of word 1 over word 2 against the float's, largest difference over the 480 positions
            (added after the first w64 L2 runs, because recipe A's model says one word everywhere at steps 10-1000, where
            top-word agreement alone tells little; no prediction was written for it before those runs)
  finished  the machine's top word against the float FINISHED model's (what taking the blank out does)
  top5      the float word inside the machine's top 5
  control   the machine's words against the previous text's float words (pairing broken)

PREDICTIONS (the session's, written before any run of this script):
  M1. finished: the machine's top word equals the float model's at 474 or more of 480 positions at every snapshot of
      every build, including step 0 (the blank: word scores spread only 0.16-0.23, but the machine's error is about
      1e-3 of that); every difference at a position where the float's lead of word 1 over word 2 is under 0.01.
  M2. control (pairing broken): 2% or less at step 0, 5-25% at later steps (the common words).
  M3. pushes and scr1-3: the machine against the float model of the same plates, 470 or more of 480; against the
      finished float's words, within 10 of eval.py's float numbers.
  M4. order=shuffle (lanes in a random order, every plate permuted to match): every head pile the same whole number in
      30 of 30 texts, every build's step 3000. matmul=int: the same whole numbers as float64 sums.
"""
import os, sys, json, math, hashlib, numpy as np
from model import HERE, tag
from snapio import scramble

A = dict(a.split("=") for a in sys.argv[1:])
d0, L0, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
ORDER, MATMUL = A.get("order", "plain"), A.get("matmul", "float64")
VARIANTS = A.get("plates", "finished,pushes,scr1,scr2,scr3").split(",")
TB = np.load(os.path.join(HERE, "machines", "tables.npz"))
SHARE, ARR, LOG2, EXP2 = TB["share"], TB["arrival"], TB["log2"], TB["exp2"]
ES, SC = 16, 10
EV = json.load(open(os.path.join(HERE, "evals", f"eval_{tag(d0, L0)}_{RECIPE}.json")))
TEXTS = np.array(EV["texts"]); LEAD = 8
STEPS = [int(s) for s in A["steps"].split(",")] if "steps" in A else sorted(int(s) for s in EV["snaps"])

def rshift(X, s):                                     # X x 2^-s, rounded to nearest: add half, shift (s may be 0 or negative)
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

_shift_py = np.frompyfunc(lambda x, s: (x << -s) if s <= 0 else ((x + (1 << (s - 1))) >> s), 2, 1)

def to16(X):                                          # each position on 16 places with its own power of two (value = x16 x 2^ex)
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)])
    sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), sh - ES

def product(x16, w, wmax):                            # whole-number meetings and sums; exact: every partial sum below 2^53
    assert float(np.abs(x16).sum(1).max()) * wmax < 2 ** 53
    if MATMUL == "int": return x16 @ w.astype(np.int64)
    with np.errstate(all="ignore"):
        p = x16.astype(np.float64) @ w
    assert np.isfinite(p).all() and (p == np.round(p)).all()
    return p.astype(np.int64)

def log2fix(v):                                       # log2 of a positive whole number: top 1's place + a table read
    v = int(v); t = v.bit_length() - 1
    m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
    return t * 65536 + int(LOG2[m])

def power(X, F):                                      # X x 2^(F / 65536): a table read of the fine part, then a shift
    F = np.broadcast_to(np.asarray(F, dtype=np.int64), X.shape); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    if np.abs(X).max(initial=0) < 2 ** 45: return rshift(X * EXP2[f], 16 - I)
    return _shift_py(X.astype(object) * EXP2[f].astype(object), (16 - I).astype(object)).astype(np.int64)

def race(vals):                                       # who arrives first: top digits first; at each digit, if anyone still in
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)   # has a 1 there, everyone with a 0 drops out
    alive = np.ones(len(u), dtype=bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])

class Machine:
    def __init__(self, t, variant):
        M = np.load(os.path.join(HERE, "machines", f"{tag(d0, L0)}_{RECIPE}_step{t}.npz"))
        self.d, self.L, self.NH, self.HS = (int(M[k]) for k in ("d", "L", "heads", "hs"))
        self.EPS40, self.HALF = int(M["eps40"]), int(M["half_log2_hs"])
        d = self.d
        perm = np.random.default_rng(12345).permutation(d) if ORDER == "shuffle" else np.arange(d)
        def plate(key, fname):                        # the variant's whole numbers on the finished plate's grid
            w, s = M[key + ".w"].astype(np.int64), M[key + ".s"].astype(np.int64)
            if variant == "finished": return w
            if variant == "pushes": return w - s
            if variant.startswith("scr"): return w - s + scramble(fname, s, int(variant[3:]))
            raise ValueError(variant)
        self.over16 = 0
        self.PL, self.NORM = {}, {}
        for l in range(self.L):
            for nm in ("ln1", "ln2"):
                k = f"{nm}.{l}"; self.NORM[k] = (M[k + ".g"][perm], int(M[k + ".ge"]), M[k + ".b"][perm])
            for nm in ("qkv", "proj", "fc", "mproj"):
                k = f"{nm}.{l}"; w = plate(k, k + ".w"); e, b = M[k + ".e"], M[k + ".b"]
                self.over16 += int((np.abs(w) > 32767).sum())
                if nm in ("qkv", "fc"): w = w[perm]                             # reads the stream: rows follow its lanes
                else: w, e, b = w[:, perm], e[perm], b[perm]                    # writes the stream: columns follow its lanes
                self.PL[k] = (w.astype(np.float64), e, b, float(np.abs(w).max()))
        self.NORM["lnf"] = (M["lnf.g"][perm], int(M["lnf.ge"]), M["lnf.b"][perm])
        w = plate("head", "head")[perm]; self.over16 += int((np.abs(w) > 32767).sum())
        self.PL["head"] = (w.astype(np.float64), M["head.e"], 0, float(np.abs(w).max()))
        self.WTE, self.WTE_E = plate("wte", "wte")[:, perm], M["wte.e"]
        self.WPE, self.WPE_E = plate("wpe", "wpe")[:, perm], M["wpe.e"]

    def gauge(self, R, name):                         # the water level, whole: centre, size by its halved log, gain, offset
        g, ge, b = self.NORM[name]; d = self.d
        x16, ex = to16(R)
        S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        c = d * x16 - S1[:, None]
        V = d * S2 - S1 * S1 + np.array([self.EPS40 >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else self.EPS40 << -(40 + 2 * int(e)) for e in ex])
        LS = np.array([log2fix(v) >> 1 for v in V])
        return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b

    def plain(self, X, name):                         # a plate: 16 places in, whole-number product, onto the stream's scale, bias
        w, e, b, wmax = self.PL[name]; x16, ex = to16(X)
        return rshift(product(x16, w, wmax), -(e[None, :] + ex[:, None] + ES)) + b

    def arrival(self, QKV):                           # GPT-2's order: q of every head, then k, then v
        P = len(QKV); d, HS = self.d, self.HS; out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
        for h in range(self.NH):
            q, k, v = QKV[:, h * HS:(h + 1) * HS], QKV[:, d + h * HS:d + (h + 1) * HS], QKV[:, 2 * d + h * HS:2 * d + (h + 1) * HS]
            q16, eq = to16(q); k16, ek = to16(k)
            sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - self.HALF)
            lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
            dd = lev[:, None] - sc
            a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
            assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
            out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
        return out

    def edge(self, H):                                # the bend: where the value arrives picks the share past the knife edge
        idx = rshift(H, ES - 12) + 32768
        share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, SHARE[np.clip(idx, 0, 65535)]))
        return rshift(H * share, 16)

    def forward(self, ids, last):
        ids = np.asarray(ids); pos = np.arange(len(ids))
        R = rshift(self.WTE[ids], -(self.WTE_E[ids][:, None] + ES)) + rshift(self.WPE[pos], -(self.WPE_E[pos][:, None] + ES))
        for l in range(self.L):
            R = R + self.plain(self.arrival(self.plain(self.gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
            R = R + self.plain(self.edge(self.plain(self.gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        y16, ey = to16(self.gauge(R[-last:], "lnf")); w, e, _, wmax = self.PL["head"]
        lg = rshift(product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
        return [race(row) for row in lg], lg

name = f"{tag(d0, L0)}_{RECIPE}"
tagx = "" if (ORDER, MATMUL) == ("plain", "float64") else "-" + "-".join(x for x in [ORDER if ORDER != "plain" else "", "matmulint" if MATMUL == "int" else ""] if x)
OUT = os.path.join(HERE, "words"); os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, f"words_{name}{tagx}.json")
res = json.load(open(dst)) if os.path.exists(dst) else dict(shape=tag(d0, L0), recipe=RECIPE, order=ORDER, matmul=MATMUL, snaps={})
print(f"machine {name}: order {ORDER}, matmul {MATMUL}", flush=True)
n_read = TEXTS.shape[1] - LEAD
for t in STEPS:
    ev = EV["snaps"][str(t)]; fin = np.array(ev["finished"]["top1"])
    rs = res["snaps"].setdefault(str(t), {})
    for var in VARIANTS:
        mc = Machine(t, var); fl = ev[var]; own = np.array(fl["top1"]); marg = np.array(fl["margin"])
        got, shas, top5, mm = [], [], [], []
        for i, ids in enumerate(TEXTS):
            g, lg = mc.forward(ids, n_read); got.append(g); shas.append(hashlib.sha256(lg.tobytes()).hexdigest())
            t2 = np.sort(lg, axis=1)[:, -2:]; mm.append(((t2[:, 1] - t2[:, 0]) / 2.0 ** ES).tolist())   # the machine's own lead
            top5.append([int(own[i, j]) in set(np.argsort(-lg[j], kind="stable")[:5].tolist()) for j in range(n_read)])
        got = np.array(got); bad = np.argwhere(got != own)
        rec = dict(agree_own=int((got == own).sum()), agree_finished=int((got == fin).sum()),
                   float_agree_finished=int(fl["agree_finished_top1"]), own_in_top5=int(np.sum(top5)),
                   texts_identical=int((got == own).all(1).sum()), control=int((got == np.roll(own, 1, 0)).sum()),
                   positions=int(got.size), values_past_16_places=mc.over16,
                   differ=[dict(text=int(i), pos=int(j), float_margin=float(marg[i, j])) for i, j in bad][:60],
                   max_float_margin_at_differ=float(max((marg[i, j] for i, j in bad), default=0.0)),
                   lead_moved_max=float(np.abs(np.array(mm) - marg).max()), float_lead_min=float(marg.min()),
                   words=got.tolist(), piles_sha256=shas)
        rs[var] = rec
        print(f"  step {t:5d} {var:9s} machine = float (same plates) {rec['agree_own']}/{rec['positions']}  texts {rec['texts_identical']}/30  "
              f"top5 {rec['own_in_top5']}  | = float finished {rec['agree_finished']} (float {rec['float_agree_finished']})  "
              f"control {rec['control']}  max float margin at a difference {rec['max_float_margin_at_differ']:.5f}  "
              f"lead moved at most {rec['lead_moved_max']:.5f} (float's smallest lead {rec['float_lead_min']:.5f})", flush=True)
        json.dump(res, open(dst, "w"))
base = os.path.join(OUT, f"words_{name}.json")
if tagx and os.path.exists(base):
    b = json.load(open(base))["snaps"]
    for t in STEPS:
        for var in VARIANTS:
            if str(t) in b and var in b[str(t)]:
                same = sum(x == y for x, y in zip(res["snaps"][str(t)][var]["piles_sha256"], b[str(t)][var]["piles_sha256"]))
                print(f"  {tagx[1:]} check step {t} {var}: every head pile the same whole number in {same} of 30 texts")
