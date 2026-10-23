"""REACH, the same counts for BitNet b1.58 2B-4T (hard edge, squared above), on bitmedium_lean.py's whole-number
machine (run from its folder; nothing of it is changed: its one_depth is copied here with counters and checked equal;
beams kept packed and unpacked a depth at a time, the word table cut to the rows used, so it runs in about 1 GB).

A neuron's light: G (gate pile) meets the edge at 0; below it the bend sends EXACTLY nothing; above it the share on the
up pile U is relu(G)^2, then the sub-gauge and the 8-place field (q = -127..127) pour into the down beam.
  open:    share relu(G)^2 within 1/256 of 1 (natural units) -- U passes straight through at that token
  shut:    output under 1 step: q == 0 at the down beam (split: below the edge, exactly 0; above it but rounded to 0)
  between: the rest (above the edge, squared, pouring)
  leak:    below-edge neurons' output (asserted exactly 0)
  local:   share of the down meeting (photons of q x the down line's photons) from open neurons' straight pass
  CONTROLS every BitNet edge sits at 0 (no per-neuron offset), so shuffling the edges moves nothing: the counts are the
           piles' distribution by construction (checked: the counts with permuted edges are identical). Constancy
           (neurons below the edge at EVERY position, droppable) against each position's neurons permuted on their own.
Texts: the 30 refs texts (GPT-2 words decoded, BitNet's own words), every position.
PREDICTED BEFORE RUNNING: open (the 1/256 window) under 1%; below the edge 40-75% a depth; leak exactly 0; above the
edge every pouring neuron is new (squared); the 8-place field leaves 70-80% of the down lines dark.
    python3 extra/reach_bitnet.py [texts=30]
Results (2026-10-04, 30 texts, 638 positions; 48 s, 2.3 GB): the counted depth equals the machine's (asserted).
  open (1/256 window) 2,093 of 132,295,680 neuron-positions, every one of them under 1 step: 0 pour. below the edge
  38.6-72.8% a depth (56.7% all), exactly dark, leak 0; above but under 1 step 14-31%; between (pouring) 12.9-31.7%
  (22.3% all). Always below the edge at all 638 positions: 22-3,013 neurons a depth, against 0 with positions permuted.
"""
import os, sys, json, time, resource
ARGS = dict(a.split("=") for a in sys.argv[1:]); sys.argv = [sys.argv[0]]
HP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/hp"
SM = "/Users/abundancemachine/streaming-machine"
os.chdir(HP); sys.path.insert(0, HP)
src = open("bitmedium_lean.py").read(); src = src[:src.index("ids = tok(")]
from transformers import GPT2Tokenizer
g2 = GPT2Tokenizer.from_pretrained("gpt2")
REFS = json.load(open(f"{SM}/refs_many.json"))[:int(ARGS.get("texts", 30))]
STRS = [g2.decode(r["ids"] + r["ref"]) for r in REFS]
# lean load: the word table is read whole only to take its gauge (the same gE) and the rows these texts use
a, b = 'EMB = Dial(torch.cat([torch.round(emb[i:i + 8192].double() / gE).to(torch.int32) for i in range(0, emb.shape[0], 8192)]), gE); del emb', \
       'NEED = sorted({t for s in STRS for t in tok(s).input_ids}); ROW = {t: i for i, t in enumerate(NEED)}; EMB = Dial(torch.round(emb[torch.tensor(NEED)].double() / gE).to(torch.int32), gE); del emb'
assert a in src; src = src.replace(a, b)
# lean beams: each bank kept packed (2 places a photon, as stored) and unpacked only for the depth being poured; the
# unpacked photons are bitmedium_lean's own (the same unpack_weights), so the machine is unchanged
a2 = "    return Dial(unpack_weights(F.get_tensor(key + \".weight\"), dtype=torch.int8), float(F.get_tensor(key + \".weight_scale\").double()))"
b2 = "    return LazyBeam(key)"
assert a2 in src; src = src.replace(a2, b2)
a3 = src[src.index('print(f"  loaded in'):src.index("\n", src.index('print(f"  loaded in'))]
src = src.replace(a3, 'print(f"  loaded in {time.time() - t0:.0f}s (beams packed)", flush=True)')
CUR = [None]; _CACHE = {}
class LazyBeam:
    def __init__(self, key):
        self.key = key; self.z = int(key.split(".")[2]); self.packed = F.get_tensor(key + ".weight")
        self.g = float(F.get_tensor(key + ".weight_scale").double())
    @property
    def p(self):
        if self.z != CUR[0]: return unpack_weights(self.packed, dtype=torch.int8)
        if self.key not in _CACHE: _CACHE[self.key] = unpack_weights(self.packed, dtype=torch.int8)
        return _CACHE[self.key]
def at_depth(z): CUR[0] = z; _CACHE.clear()
exec(compile(src, "bitmedium_lean.py", "exec"))
_POP = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int64)
def ones8(q): return _POP[q.abs().clamp_max(255)]                     # photons of each 8-place value (|q| <= 127)
DOWN_LINE = {}                                      # photons of each down line (filled per depth)
C = {}
def add(z, k, v): C[(z, k)] = C.get((z, k), 0) + v
PERNEU = {}
def nadd(k, z, m): PERNEU[(k, z)] = PERNEU.get((k, z), 0) + m.sum(0).to(torch.int64)
gen = torch.Generator().manual_seed(20261004)

def one_depth_counted(X, z, n, rotz, LIM):           # bitmedium_lean's one_depth, line for line, with counters
    A = actq(normed(X, W[(z, "in")]))
    Q, KK, V = A.pile(W[(z, "q")]), A.pile(W[(z, "k")]), A.pile(W[(z, "v")])
    Q = Dial(Q.p.reshape(n, nq, d), Q.g); KK = Dial(KK.p.reshape(n, nk, d), KK.g)
    def spin(Dl, heads):
        r, a = to_polar(Dl.p[..., :half], Dl.p[..., half:]); a = a + rotz[:, None, :].expand(-1, heads, -1); u, v = to_flat(r, a)
        return Dial(torch.cat([u, v], dim=-1), Dl.g)
    Q, KK = spin(Q, nq), spin(KK, nk)
    KP = KK.p.repeat_interleave(nq // nk, dim=1); VP = V.p.reshape(n, nk, d).repeat_interleave(nq // nk, dim=1)
    SC = torch.einsum("qhd,khd->hqk", Q.p, KP); sg = Q.g * KK.g / math.sqrt(d)
    SP = torch.round(SC.double() * (sg / (math.log(2) / K))).to(torch.int64).masked_fill(LIM, torch.iinfo(torch.int64).min // 4)
    dd = (SP.max(-1, keepdim=True).values - SP).clamp_min(0); w = ATAB[dd % K] >> (dd // K).clamp_max(60)
    P = torch.einsum("hqk,khd->qhd", w, VP)
    O = Dial((P // w.sum(-1, keepdim=True).permute(1, 0, 2).clamp_min(1)).reshape(n, -1), V.g)
    X = X.plus(actq(normed(O, W[(z, "asub")])).pile(W[(z, "o")]))
    M = actq(normed(X, W[(z, "post")]))
    G, U = M.pile(W[(z, "gate")]), M.pile(W[(z, "up")])
    R = Dial(G.p.clamp_min(0), G.g); R2 = Dial(R.p * R.p, R.g * R.g).trim(WIDTH + 2)
    Hn = normed(Dial(R2.p * U.p, R2.g * U.g).trim(WIDTH + 2), W[(z, "fsub")])
    p = Hn.p; am = p.abs().amax(-1, keepdim=True).clamp_min(1); q = torch.sign(p) * ((p.abs() * 254 + am) // (2 * am))   # actq's own field
    H = actq(Hn); assert torch.equal(H.p, Dial(q * am, Hn.g / 127).trim(WIDTH).p)
    # ---- the counts ----
    below = G.p <= 0; assert int(q[below].abs().sum()) == 0                       # the hard edge: exactly nothing below
    lo, hi = 255 / 256 / G.g ** 2, 257 / 256 / G.g ** 2                           # the 1/256 window on relu(G)^2, as whole-number bounds on G^2
    g2_ = G.p.double() ** 2; op = ~below & (g2_ >= lo) & (g2_ <= hi)
    shut = q == 0; mid = ~op & ~shut
    oq = ones8(q); meet = oq * DOWN_LINE[z][None, :]
    add(z, "positions", n); add(z, "n open", int(op.sum())); add(z, "n shut", int(shut.sum())); add(z, "n between", int(mid.sum()))
    add(z, "n below", int(below.sum())); add(z, "n above dark", int((shut & ~below).sum())); add(z, "n open dark", int((op & shut).sum()))
    add(z, "photons all", int(oq.sum())); add(z, "meeting all", int(meet.sum())); add(z, "meeting open", int(meet[op].sum()))
    add(z, "leak photons", int(oq[below].sum())); add(z, "plain lines", n * q.shape[1]); add(z, "down lines pouring", int((q != 0).sum()))
    add(z, "down meeting if every line poured", int((torch.full_like(oq, 1) * DOWN_LINE[z][None, :]).sum()))
    # control: the edges shuffled (all at 0, so the same edge lands on every neuron: identical counts)
    perm = torch.randperm(G.p.shape[1], generator=gen); edge0 = torch.zeros(G.p.shape[1], dtype=torch.int64)
    assert torch.equal((G.p - edge0 + edge0[perm]) <= 0, below)
    nadd("below", z, below); nadd("open", z, op)
    pp = torch.stack([torch.randperm(G.p.shape[1], generator=gen) for _ in range(n)]); nadd("below perpos", z, torch.gather(below, 1, pp))
    return X.plus(H.pile(W[(z, "down")]))

t1 = time.time(); TX = []                            # depth by depth over all texts: each depth's beams unpacked once
for s_ in STRS:
    ids = tok(s_).input_ids; n = len(ids)
    rotz = torch.tensor([[round(m * t * ONE) for t in _th] for m in range(n)], dtype=torch.int64)
    LIM = (torch.arange(n)[None, :] > torch.arange(n)[:, None])[None]
    TX.append([Dial(EMB.p[torch.tensor([ROW[t] for t in ids])].to(torch.int64), EMB.g), n, rotz, LIM])
NP = sum(t[1] for t in TX)
for z in range(deep):
    at_depth(z); DOWN_LINE[z] = (W[(z, "down")].p != 0).sum(0).to(torch.int64)
    for i, t in enumerate(TX):
        X, n, rotz, LIM = t; Xn = one_depth_counted(X, z, n, rotz, LIM)
        if i == 0:                                   # the counted copy IS the machine's own depth (text 1, every depth)
            Xr = one_depth(X, z, n, rotz, LIM); assert torch.equal(Xn.p, Xr.p) and Xn.g == Xr.g
        t[0] = Xn
    if z % 5 == 4: print(f"  depth {z + 1} ({time.time() - t1:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)
print(f"\nBitNet b1.58 2B-4T, {len(STRS)} texts, {NP} positions, 6,912 neurons a depth; the counted depth equals bitmedium_lean's own (checked, text 1, every depth)")
print("leak below the edge: exactly 0 at every neuron-position (asserted); edges shuffled: identical counts (asserted: every edge is 0)")
def pct(a, b): return 100.0 * a / b if b else float("nan")
print("\nCOUNTS (% of neuron-positions) and the down beam")
print("  depth |  open  shut (below edge / above but under 1 step)  between | down lines pouring | open's share of the down meeting | always below edge  (each position permuted)")
for z in range(deep):
    g = lambda k: C[(z, k)]; nn = g("positions") * 6912
    print(f"  {z:5d} | {pct(g('n open'), nn):5.2f}  {pct(g('n shut'), nn):5.1f} ({pct(g('n below'), nn):5.1f} / {pct(g('n above dark'), nn):5.1f})"
          f"                   {pct(g('n between'), nn):5.1f}  |       {pct(g('down lines pouring'), g('plain lines')):5.1f}%       |"
          f"            {pct(g('meeting open'), g('meeting all')):6.3f}%            |  {int((PERNEU[('below', z)] == NP).sum()):6d}   ({int((PERNEU[('below perpos', z)] == NP).sum())})")
tot = {k: sum(C[(z, k)] for z in range(deep)) for k in ("n open", "n shut", "n between", "n below", "positions", "meeting all", "meeting open", "down lines pouring", "plain lines")}
nn = tot["positions"] * 6912
print(f"  all   | {pct(tot['n open'], nn):5.2f}  {pct(tot['n shut'], nn):5.1f} ({pct(tot['n below'], nn):5.1f} below)  between {pct(tot['n between'], nn):5.1f}"
      f" | pouring {pct(tot['down lines pouring'], tot['plain lines']):5.1f}% | open's share of the down meeting {pct(tot['meeting open'], tot['meeting all']):.3f}%")
print(f"  raw: open (1/256 window) {tot['n open']:,} of {nn:,} neuron-positions ({sum(C[(z, 'n open dark')] for z in range(deep)):,} of them under 1 step);"
      f" below the edge {tot['n below']:,}; pouring {tot['down lines pouring']:,}")
print(f"\nmax memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
