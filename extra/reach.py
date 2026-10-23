"""REACH: how far past the first bend can the stored part be carried?  (GPT-2, the base-2 machine)

His picture: the plates never change ("we aren't training"); most of every pile cancels itself; a machine of the extra
only computes just what does not cancel. Everything entering layer 0 is stored: a word's row + a position's row. A neuron
far above its edge passes its pile straight through, so the stored part of its pile stays stored past the bend; far
below it passes ~nothing (the soft bend leaks a little); only neurons near the edge make genuinely new light.

Measured on the 30 refs texts, every position (648), every layer, whole numbers in every step:
  COUNTS  open:  share within 1/256 of 1 (share >= 65,280 of 65,536), the pile passes straight through
          shut:  output under 1 step (exactly 0 as a whole number at 2^16)
          between: the rest, split into below the edge (pile < 0, still sending: the leak) and above the edge
  LEAK    what below-edge neurons still send into neurons-out: photons in its 16-place window, meeting photons
          (light photons x plate-row photons), and size (sum of |output|), against all the neurons' output
  LOCAL   if the layer's whole pile were stored: the share of its neurons-out meeting that is the open neurons'
          straight pass (stored) against the rest (new)
  REACH   the stream split exactly, R = S + Nb + Nc at every step (asserted equal to the machine's own stream):
            S   stored: word + position rows, the fixed biases, every linear step of them with the machine's own
                coefficients (gauges, arrival shares); past a bend only through OPEN neurons, carried exactly
                (out = pile on the stored part)
            Nb  born at the bends: between neurons' whole output, open neurons' departure from straight, and
                everything downstream of it
            Nc  rounding carries: the machine rounds S + N as one; rshift(S) + rshift(N) is off by -1/0/+1
          At every bank the light that must still meet the plate is the window's light minus S's rounded part.
          Its meeting photons against the plain meeting = the share of the work storage cannot do. This is a LOWER
          bound on the extra's cost: reading S from storage adds its own carries and the combining of stored rows.
  CONTROLS (every value kept in distribution)
          edges shuffled: each layer's 3,072 neuron edges (fc biases) permuted once; counts, leak and local share on
                          the same light; and the whole machine re-run with shuffled edges for the reach
          constancy:      neurons open (or shut) at EVERY position, the only ones a fixed plate could hold once,
                          against edges shuffled and against each position's neurons permuted on their own
  ORDER   lines, neurons and parts poured in a random order: the split piles add to the machine's own, exactly.

PREDICTED BEFORE RUNNING (2026-10-04):
  P1 open 1-10% of neurons a layer; shut (exactly 0) under 5%; between over 85%, mostly below the edge.
  P2 the leak carries over half of the photons and over half of the meeting going into neurons-out, while its size
     (sum |output|) is under 20%: a photon count does not shrink with the value.
  P3 local: the open neurons' straight pass is under 20% of neurons-out's meeting at every layer.
  P4 reach: the new light's meeting at layer 1's first bank is over 70% of the plain meeting; from layer 2 on, over 90%.
  P5 edges shuffled: counts move by under 5 points; always-open neurons are more in the real machine than shuffled,
     but under 2% of neurons.
    python3 extra/reach.py [texts=30] [shuffled_machine=1]
Results (2026-10-04, 30 texts, 648 positions; 59 s, 2.2 GB): split exact at every layer (asserted), random order 720/720.
  open 0.00-0.11% a layer, shut 0.04-0.5%, between 99.5-99.9% (below the edge 73-88%); edges shuffled the same.
  leak: 73-87% of the photons into neurons-out, 56-73% of the size; its light into the stream 47-116% of the whole.
  local stored share of neurons-out 0.0-0.2%. reach: new light's meeting at layer 1 94-96%, layer 11 96-100%, top 99.8%.
  storage could carry 5.1% of the per-token plate meetings (66% of layer 0; 1.2% past the first bend); the edge-shuffled
  machine 4.4% (0.5% past the first bend). No neuron is open, or shut, at every position, in any layer.
  P1 open WRONG (0.0-0.1%, not 1-10%); P2 size WRONG (56-73%, not under 20%); P3, P4 right; P5 counts right, always-open 0 = 0.
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every product is bounded below 2^53
HERE = "/Users/abundancemachine/streaming-machine"
ARGS = dict(a.split("=") for a in sys.argv[1:]); sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
W, M, ES, EXP2, ARR, PHI = J.W, J.M, J.ES, J.EXP2, J.ARR, J.PHI
rshift, to16, gauge_log, log2fix, race = J.rshift, J.to16, J.gauge_log, J.log2fix, J.race
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS][:int(ARGS.get("texts", 30))]
_E = M["wte"]; ENTRY = [_E[np.array(t)].copy() for t in TEXTS]; del _E
WPE = M["wpe"][:32].copy(); t0 = time.time()

# ---------- photons ----------
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):                                         # photons (1s) per value, sign apart
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]
WROW = {}                                            # photons of each plate line (row), summed over the line's outputs
for name in BANKS:
    w = M[name + ".w"]; WROW[name] = np.concatenate([ones(w[i:i + 64]).sum(1) for i in range(0, w.shape[0], 64)])
print(f"plate photons read ({time.time() - t0:.0f}s)", flush=True)

# ---------- exact whole-number steps that stay exact when a part is wider than the machine's own light ----------
def meet(X, Y, by=None):                             # X @ Y exactly; split into 16-place pieces when a sum could pass 2^52
    X = np.asarray(X, np.int64); n = X.shape[-1]
    Yf = Y if Y.dtype == np.float64 else Y.astype(np.float64)
    bx = int(np.abs(X).max()) if X.size else 0; by = int(np.abs(Yf).max()) if by is None else by
    if bx * by * n < 2 ** 52: return (X.astype(np.float64) @ Yf).astype(np.int64)
    if bx >= by or by <= 1 << 16:
        a, b = meet(X >> 16, Yf, by), meet(X & 0xFFFF, Yf, by)
    else:
        Yi = Yf.astype(np.int64); a, b = meet(X, (Yi >> 16).astype(np.float64)), meet(X, (Yi & 0xFFFF).astype(np.float64))
    assert np.abs(a).max() < 2 ** 46; return (a << 16) + b
def srshift(X, s):                                   # rshift, falling back to Python whole numbers if a shift could overflow
    X = np.asarray(X, np.int64); s = np.broadcast_to(np.asarray(s, np.int64), np.broadcast_shapes(X.shape, np.shape(s)))
    up = int(np.maximum(-s, 0).max()) if s.size else 0
    if (int(np.abs(X).max()) if X.size else 0) < 2 ** (61 - up): return rshift(X, s)
    Xb = np.broadcast_to(X, s.shape).astype(object).ravel(); out = []
    for x, k in zip(Xb, s.ravel()):
        k = int(k); out.append(x << -k if k <= 0 else (x + (1 << (k - 1))) >> k)
    out = np.array(out, object).reshape(s.shape); assert all(abs(int(v)) < 2 ** 62 for v in out.ravel()); return out.astype(np.int64)
def spower(X, F):                                    # X x 2^(F/65536): table read of the fine part, then a shift
    F = np.asarray(F, np.int64); I, f = F >> 16, F & 0xFFFF; X = np.asarray(X, np.int64)
    if (int(np.abs(X).max()) if X.size else 0) < 2 ** 45: return rshift(X * EXP2[f], 16 - I)
    Xo = np.broadcast_to(X, np.broadcast_shapes(X.shape, F.shape)).astype(object) * np.broadcast_to(EXP2[f], np.broadcast_shapes(X.shape, F.shape)).astype(object)
    k = np.broadcast_to(16 - I, Xo.shape); out = np.array([x << -int(s) if s <= 0 else (x + (1 << (int(s) - 1))) >> int(s) for x, s in zip(Xo.ravel(), k.ravel())], object)
    return out.reshape(Xo.shape).astype(np.int64)
def share_of(H):                                     # the machine's edge: where the value arrives picks the share
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))

# ---------- counters ----------
class Acc(dict):
    def add(self, k, v): self[k] = self.get(k, 0) + v
A = {}                                               # A[(run, layer)] = Acc
def acc(run, l): return A.setdefault((run, l), Acc())
NEU = {}                                             # per-neuron counts over every position: (run, layer, kind) -> (3072,)
def nadd(run, l, kind, mask):
    k = (run, l, kind); NEU[k] = NEU.get(k, 0) + mask.sum(0).astype(np.int64)
RNG = np.random.default_rng(20261004)
EDGE_PERM = [RNG.permutation(3072) for _ in range(12)]          # the edge shuffle: one fixed permutation per layer
ORDER_OK = [0, 0]

def split16(parts, x16, sh):                         # the machine's rounded light split: S and Nb rounded alone, the carry joins Nc
    S16, B16 = rshift(parts[0], sh[:, None]), rshift(parts[1], sh[:, None]); return [S16, B16, x16 - S16 - B16]
def work(run, l, tag, x16, p16, name):               # meeting photons: plain, and the light storage cannot carry
    a = acc(run, l); wr = WROW[name]
    a.add(f"{tag} plain", int((ones(x16) @ wr).sum())); a.add(f"{tag} new", int((ones(x16 - p16[0]) @ wr).sum()))
    a.add(f"{tag} new Nb", int((ones(p16[1]) @ wr).sum())); a.add(f"{tag} new Nc", int((ones(p16[2]) @ wr).sum()))
    a.add(f"{tag} light", int(ones(x16).sum())); a.add(f"{tag} light S", int(ones(p16[0]).sum())); a.add(f"{tag} light new", int(ones(x16 - p16[0]).sum()))
def order_check(p16, name, pile):                    # lines and parts in a random order give the machine's own pile
    w = W[name][0]; perm = RNG.permutation(w.shape[0]); groups = np.array_split(perm, 12); tot = np.zeros_like(pile)
    for i in RNG.permutation(3):
        for g in RNG.permutation(len(groups)): tot = tot + meet(p16[i][:, groups[g]], w[groups[g]], 32767)
    ORDER_OK[0] += int(np.array_equal(tot, pile)); ORDER_OK[1] += 1

def gauged_split(parts, name, run, l, tag, b=None, check=False):
    R = parts[0] + parts[1] + parts[2]; w, we, b0 = W[name]; b = b0 if b is None else b
    x16, ex, LS = gauge_log(R); sh = ex + ES; p16 = split16(parts, x16, sh)
    piles = [meet(p, w, 32767) for p in p16]; pile = piles[0] + piles[1] + piles[2]
    if check: assert np.array_equal(pile, J.product(x16, w))
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    out = spower(pile, F) + b; oS = spower(piles[0], F) + b; oB = spower(piles[1], F)
    work(run, l, tag, x16, p16, name)
    if tag == "fc" and run == "real": order_check(p16, name, pile)
    return [oS, oB, out - oS - oB], out, pile
def plain_split(parts, name, run, l, tag):
    X = parts[0] + parts[1] + parts[2]; w, we, b = W[name]
    x16, ex = to16(X, -ES); sh = ex + ES; p16 = split16(parts, x16, sh)
    piles = [meet(p, w, 32767) for p in p16]; pile = piles[0] + piles[1] + piles[2]
    s = -(we[None, :] + ex[:, None] + ES)
    out = srshift(pile, s) + b; oS = srshift(piles[0], s) + b; oB = srshift(piles[1], s)
    work(run, l, tag, x16, p16, name)
    if tag == "mproj" and run == "real": order_check(p16, name, pile)
    return [oS, oB, out - oS - oB], out
def arrival_split(parts, QKV, run, l):               # J.attention, every head; the shares are the machine's own (coefficients)
    P = QKV.shape[0]; out = [np.zeros((P, 768), np.int64) for _ in range(3)]; a_ = acc(run, l)
    for h in range(12):
        q, k = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64]; sl = slice(1536 + h*64, 1536 + (h+1)*64)
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(J.product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + J.SC))
        Ash = np.zeros((P, P), np.int64); L = np.zeros(P, np.int64)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            Ash[p, :p+1] = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0); L[p] = log2fix([Ash[p, :p+1].sum()])[0]
        v = QKV[:, sl]; num = meet(Ash, v); full = spower(num, -L[:, None])
        oS = spower(meet(Ash, parts[0][:, sl]), -L[:, None]); oB = spower(meet(Ash, parts[1][:, sl]), -L[:, None])
        osl = slice(h*64, (h+1)*64); out[0][:, osl], out[1][:, osl], out[2][:, osl] = oS, oB, full - oS - oB
        ov = ones(v).sum(1); oa = ones(Ash)            # the arrival's own work, unchanged by storage (pilot's counting)
        a_.add("arrival scores", int(np.tril(ones(q16) @ ones(k16).T).sum())); a_.add("arrival mix", int((oa * ov[None, :]).sum()))
    return out, out[0] + out[1] + out[2]

def bend_split(Hp, H, run, l, b_fc, wm):             # the bend, the counts, the leak, the local share, the controls
    a = acc(run, l); share = share_of(H); out = rshift(H * share, 16)
    op, sh_ = share >= 65536 - 256, out == 0; mid = ~op & ~sh_; below = H < 0
    x16, ex = to16(out, -ES); sw = (ex + ES)[:, None]; on = ones(x16); wr = WROW[f"mproj.{l}"]; meetp = on * wr[None, :]
    P = H.shape[0]; a.add("positions", P)
    for k, m in (("open", op), ("shut", sh_), ("between", mid), ("between below", mid & below), ("between above", mid & ~below),
                 ("shut below", sh_ & below), ("far below", H < -int(4.4 * 65536))):
        a.add(f"n {k}", int(m.sum()))
    for k, m in (("open", op), ("between below", mid & below), ("between above", mid & ~below)):
        a.add(f"photons {k}", int(on[m].sum())); a.add(f"meeting {k}", int(meetp[m].sum())); a.add(f"size {k}", int(np.abs(out[m]).sum()))
    a.add("photons all", int(on.sum())); a.add("meeting all", int(meetp.sum())); a.add("size all", int(np.abs(out).sum()))
    for k, m in (("x<-4.4", H < -288358), ("-4.4..-2", (H >= -288358) & (H < -131072)), ("-2..0", (H >= -131072) & (H < 0)),
                 ("0..open", (H >= 0) & ~op), ("open", op & (share < 65536)), ("straight", share == 65536)):
        a.add(f"band {k}", int(m.sum()))
    leak16 = np.where(mid & below, x16, 0); a.add("leak into stream", int(np.abs(meet(leak16, wm, 32767)).sum())); a.add("all into stream", int(np.abs(meet(x16, wm, 32767)).sum()))
    a.add("open departure photons", int(ones(rshift(np.where(op, out - H, 0), sw)).sum()))
    S16 = rshift(np.where(op, H, 0), sw)               # LOCAL: the open neurons' straight pass, as if the pile were stored
    a.add("local stored meeting", int((ones(S16) * wr[None, :]).sum())); a.add("local new meeting", int((ones(x16 - S16) * wr[None, :]).sum()))
    if run == "real":
        nadd(run, l, "open", op); nadd(run, l, "shut", sh_)
        pp = np.array([RNG.permutation(3072) for _ in range(P)]); r_ = np.arange(P)[:, None]
        nadd("perpos", l, "open", op[r_, pp]); nadd("perpos", l, "shut", sh_[r_, pp])
        Hs = H - b_fc + b_fc[EDGE_PERM[l]]; ss = share_of(Hs); os_ = rshift(Hs * ss, 16)     # CONTROL: edges shuffled, same light
        ops, shs = ss >= 65536 - 256, os_ == 0; mids = ~ops & ~shs; bel = Hs < 0
        xs, es = to16(os_, -ES); ons = ones(xs); mps = ons * wr[None, :]; c = acc("edges", l); c.add("positions", P)
        for k, m in (("open", ops), ("shut", shs), ("between", mids), ("between below", mids & bel), ("between above", mids & ~bel)):
            c.add(f"n {k}", int(m.sum()))
        for k, m in (("open", ops), ("between below", mids & bel), ("between above", mids & ~bel)):
            c.add(f"photons {k}", int(ons[m].sum())); c.add(f"meeting {k}", int(mps[m].sum())); c.add(f"size {k}", int(np.abs(os_[m]).sum()))
        c.add("photons all", int(ons.sum())); c.add("meeting all", int(mps.sum())); c.add("size all", int(np.abs(os_).sum()))
        S16s = rshift(np.where(ops, Hs, 0), (es + ES)[:, None])
        c.add("local stored meeting", int((ones(S16s) * wr[None, :]).sum())); c.add("local new meeting", int((ones(xs - S16s) * wr[None, :]).sum()))
        nadd("edges", l, "open", ops); nadd("edges", l, "shut", shs); nadd("real", l, "below", below); nadd("edges", l, "below", bel)
        nadd("perpos", l, "below", below[r_, pp])
    outS = np.where(op, Hp[0], 0); outC = np.where(op, Hp[2], 0)
    return [outS, out - outS - outC, outC], out

def run_text(ids, entry, run, check):
    P = len(ids); z = np.zeros((P, 768), np.int64); R = [entry + WPE[:P], z, z.copy()]; Rm = entry + WPE[:P]
    for l in range(12):
        bfc = W[f"fc.{l}"][2] if run == "real" else W[f"fc.{l}"][2][EDGE_PERM[l]]
        qp, QKV, _ = gauged_split(R, f"qkv.{l}", run, l, "qkv", check=check)
        op_, o = arrival_split(qp, QKV, run, l)
        ap, att = plain_split(op_, f"proj.{l}", run, l, "proj"); R = [r + x for r, x in zip(R, ap)]
        Hp, H, _ = gauged_split(R, f"fc.{l}", run, l, "fc", b=bfc)
        bp_, out = bend_split(Hp, H, run, l, bfc, W[f"mproj.{l}"][0])
        mp, mo = plain_split(bp_, f"mproj.{l}", run, l, "mproj"); R = [r + x for r, x in zip(R, mp)]
        if check:                                    # the split adds to the machine's own stream, layer by layer
            Rm = Rm + J.plain_bank(J.attention(J.gauged_bank(Rm, f"qkv.{l}")), f"proj.{l}")
            Rm = Rm + J.plain_bank(J.edge(J.gauged_bank(Rm, f"fc.{l}")), f"mproj.{l}")
            assert np.array_equal(R[0] + R[1] + R[2], Rm), (l, "split does not add to the machine")
        s = acc(run, l)
        s.add("stream photons", int(ones(R[0] + R[1] + R[2]).sum())); s.add("stream photons S", int(ones(R[0]).sum()))
        s.add("stream photons N", int(ones(R[1] + R[2]).sum())); s.add("stream size S", int(np.abs(R[0]).sum())); s.add("stream size N", int(np.abs(R[1] + R[2]).sum()))
    gauged_split(R, "head", run, 12, "top")        # the read at the top, every position (the machine reads it once a token)

RUNS = ["real"] + (["shufmachine"] if ARGS.get("shuffled_machine", "1") == "1" else [])
for run in RUNS:
    t1 = time.time()
    for i, (ids, e) in enumerate(zip(TEXTS, ENTRY)):
        run_text(ids, e, run, check=(run == "real"))
        if i % 10 == 9: print(f"  {run}: {i + 1} texts ({time.time() - t1:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)
NP = sum(len(t) for t in TEXTS)
print(f"\nthe split S + Nb + Nc adds to the machine's own stream at every layer on all {len(TEXTS)} texts ({NP} positions): yes (asserted)")
print(f"random order of lines/neurons and of the three parts gives the machine's own fc and mproj piles: {ORDER_OK[0]}/{ORDER_OK[1]}")

def pct(a, b): return 100.0 * a / b if b else float("nan")
print("\nCOUNTS (share of neuron-positions; open = share within 1/256 of 1, shut = output exactly 0, between = the rest)")
print("  layer |  open   shut  between (below / above) | far below x<-4.4 | edges shuffled: open   shut  between")
for l in range(12):
    a, c = acc("real", l), acc("edges", l); n = a["positions"] * 3072
    print(f"  {l:5d} | {pct(a['n open'], n):5.1f}  {pct(a['n shut'], n):5.1f}  {pct(a['n between'], n):5.1f} ({pct(a['n between below'], n):5.1f} / {pct(a['n between above'], n):5.1f})"
          f" |      {pct(a['n far below'], n):5.1f}      |               {pct(c['n open'], n):5.1f}  {pct(c['n shut'], n):5.1f}  {pct(c['n between'], n):5.1f}")
print("\nLEAK: below-edge neurons that still send (between, pile < 0), against everything the neurons send into neurons-out")
print("  layer | photons  meeting   size  into stream | open's departure photons / all | edges shuffled: photons meeting size")
for l in range(12):
    a, c = acc("real", l), acc("edges", l)
    print(f"  {l:5d} | {pct(a['photons between below'], a['photons all']):6.1f}  {pct(a['meeting between below'], a['meeting all']):6.1f}  {pct(a['size between below'], a['size all']):6.1f}"
          f"  {pct(a['leak into stream'], a['all into stream']):6.1f}      |   {pct(a['open departure photons'], a['photons all']):6.2f}"
          f"                       |  {pct(c['photons between below'], c['photons all']):6.1f}  {pct(c['meeting between below'], c['meeting all']):6.1f}  {pct(c['size between below'], c['size all']):6.1f}")
print("\nLOCAL: if a layer's whole pile were stored, neurons-out's meeting photons that are a stored straight pass (open) vs new")
print("  layer | stored  new  (of plain)   meeting share by class: open  between-below  between-above | edges shuffled: stored  new")
for l in range(12):
    a, c = acc("real", l), acc("edges", l)
    print(f"  {l:5d} | {pct(a['local stored meeting'], a['meeting all']):5.1f}  {pct(a['local new meeting'], a['meeting all']):5.1f}"
          f"               {pct(a['meeting open'], a['meeting all']):5.1f}   {pct(a['meeting between below'], a['meeting all']):5.1f}   {pct(a['meeting between above'], a['meeting all']):5.1f}"
          f"          |  {pct(c['local stored meeting'], c['meeting all']):5.1f}  {pct(c['local new meeting'], c['meeting all']):5.1f}")
for run in RUNS:
    print(f"\nREACH ({'the real machine' if run == 'real' else 'CONTROL: the whole machine with each layer edges shuffled'}): meeting photons of the light storage cannot carry, as % of the plain meeting")
    print("  layer |   qkv    proj     fc   mproj | new = Nb + Nc at qkv (Nb / Nc photons as % of plain) | stream after the layer: photons S / N, size N")
    for l in range(13):
        a = acc(run, l)
        if l == 12:
            print(f"    top | {pct(a['top new'], a['top plain']):5.1f}"); continue
        print(f"  {l:5d} | " + "  ".join(f"{pct(a[f'{t} new'], a[f'{t} plain']):5.1f}" for t in ("qkv", "proj", "fc", "mproj"))
              + f" |   {pct(a['qkv new Nb'], a['qkv plain']):5.1f} / {pct(a['qkv new Nc'], a['qkv plain']):5.1f}"
              + f"                                 | {pct(a['stream photons S'], a['stream photons']):5.1f} / {pct(a['stream photons N'], a['stream photons']):5.1f}, {pct(a['stream size N'], a['stream size S'] + a['stream size N']):5.1f}")
    tot = {k: sum(acc(run, l).get(f"{t} {k}", 0) for l in range(13) for t in ("qkv", "proj", "fc", "mproj", "top")) for k in ("plain", "new")}
    l0 = {k: sum(acc(run, 0).get(f"{t} {k}", 0) for t in ("qkv", "proj", "fc", "mproj")) for k in ("plain", "new")}
    arr = sum(acc(run, l).get("arrival scores", 0) + acc(run, l).get("arrival mix", 0) for l in range(12))
    print(f"  per token: plate meetings {tot['plain'] / NP:,.0f} photons, of which storage could carry {(tot['plain'] - tot['new']) / NP:,.0f}"
          f" = {pct(tot['plain'] - tot['new'], tot['plain']):.1f}%  (layer 0's banks: {pct(l0['plain'] - l0['new'], l0['plain']):.1f}% of layer 0;"
          f" past the first bend, layers 1-11 + top: {pct((tot['plain'] - tot['new']) - (l0['plain'] - l0['new']), tot['plain'] - l0['plain']):.1f}%)")
    print(f"  with the arrival's scores and mixes ({arr / NP:,.0f}, unchanged): storage carries {pct(tot['plain'] - tot['new'], tot['plain'] + arr):.1f}% of meetings + arrival")
print("\nWHERE THE PILES SIT against the edge (x = pile in natural units; % of neuron-positions)")
print("  layer |  x<-4.4  -4.4..-2   -2..0   0..2.66  open<8  straight (share exactly 1)")
for l in range(12):
    a = acc("real", l); n = a["positions"] * 3072
    print(f"  {l:5d} | " + "  ".join(f"{pct(a[f'band {k}'], n):7.2f}" for k in ("x<-4.4", "-4.4..-2", "-2..0", "0..open", "open", "straight")))
print("\nCONSTANCY over all positions: neurons a fixed plate could hold once (always open = fold in; always shut = drop)")
print("  layer | ever open  always open  always shut  always below edge | edges shuffled: ever open  always open  always below | each position permuted: always below")
for l in range(12):
    r = {k: NEU[("real", l, k)] for k in ("open", "shut", "below")}; e = {k: NEU[("edges", l, k)] for k in ("open", "shut", "below")}; q = NEU[("perpos", l, "below")]
    print(f"  {l:5d} | {int((r['open'] > 0).sum()):9d}  {int((r['open'] == NP).sum()):11d}  {int((r['shut'] == NP).sum()):11d}  {int((r['below'] == NP).sum()):16d} |"
          f" {int((e['open'] > 0).sum()):24d}  {int((e['open'] == NP).sum()):11d}  {int((e['below'] == NP).sum()):12d} | {int((q == NP).sum()):35d}")
print(f"\nmax memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
