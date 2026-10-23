"""BitNet b1.58 2B-4T on circles (hp/bitmedium_lean.py's whole-number machine), watched. For 30 short passages, every
line's whole number after each of the 60 writes (two a depth: the look-back write through the o beam, then the neuron
write through the down beam). Recorded at every write, exactly as the add does it: the write's whole numbers, the two
shifts that line the gauges up, and the shift that sets the water level after (the add keeps the top 24 places of the
whole tensor; everything below falls off). Also at every depth: which of the 6,912 neurons end exactly dark (the gate
pile at or below the edge, where the bend sends exactly nothing).

Then the SAME machine again with everything taken in a random order (seed 4): the 2,560 lines, the 6,912 neurons of
every depth (a new order each depth), the 5 key heads and the 4 query heads riding each, the 64 pairs of places that
turn together in every q and k head (one order shared by q, k and the turn table, so the turns still meet), and the
128 places of every value head. Every recorded whole number must come back the same, only in the shuffled order.

Run from the hp folder (it execs medium.py's parts there):
    cd <scratchpad>/hp && python3 ~/streaming-machine/watch/bitnet_collect.py
Writes one file per passage to <scratchpad>/bwatch/text_NN.npz."""
import math, os, sys, glob, time, resource, torch, numpy as np
def peak(): return f"peak memory so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"
SCR = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
HP, OUT = SCR + "hp/", SCR + "bwatch/"
os.chdir(HP); os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], "22"]
src = open("medium.py").read()
exec(compile(src[:src.index('NAME = os.environ.get("MODEL"')], "medium.py", "exec"))     # Dial, by_size, normed, to_polar, to_flat, ATAB, K
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.integrations.bitnet import unpack_weights

TEXTS = [
    "The old lighthouse keeper climbed the stairs every evening to light the lamp before the fog rolled in.",
    "She poured the tea slowly, watching the steam curl up toward the cold kitchen window.",
    "Bees carry pollen from flower to flower, and without them many fruits would never grow.",
    "The train left the station twenty minutes late because a cow was standing on the tracks.",
    "In the morning the market fills with farmers selling bread, cheese, eggs and fresh green beans.",
    "My grandfather kept every letter he ever received in a wooden box under his bed.",
    "A thin layer of ice covered the pond, and the children were told not to walk on it.",
    "The museum opens at nine, but the line outside usually starts forming an hour earlier.",
    "When the power went out, we lit candles and played cards until the lights came back.",
    "Mountains in the north are covered with snow for most of the year, even in early June.",
    "He fixed the bicycle chain with a pair of pliers and a little bit of oil.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Rain fell all night on the tin roof, and nobody in the house could sleep.",
    "The small boat drifted past the harbor wall as the sun went down behind the hills.",
    "Most of the books in the library were donated by people who lived in the town.",
    "After the storm, the beach was covered with seaweed, shells and pieces of driftwood.",
    "The doctor listened to his heart, then asked him how long he had been feeling tired.",
    "Our neighbor plays the violin every Sunday afternoon with the windows wide open to the street.",
    "Copper wire carries electric current because its electrons move freely between the atoms.",
    "The cat sat on the warm stone step and watched the birds in the garden.",
    "Every spring the river floods the low fields, leaving behind a layer of rich dark mud.",
    "The teacher wrote the date on the board and asked the class to open their notebooks.",
    "A clock on the church tower struck twelve as the wedding guests walked out into the square.",
    "The bakery on the corner sells the best apple pie in the whole city.",
    "Sound travels faster through water than through air, which is why whales can hear each other far away.",
    "The farmer checked the fence after the wind knocked down two of the wooden posts.",
    "We followed the narrow path through the forest until we reached a small wooden bridge.",
    "Her first job was delivering newspapers on a bicycle before school each cold winter morning.",
    "The ship's captain studied the map carefully before choosing the route around the island.",
    "Salt dissolves in warm water much faster than it does in cold water.",
]

NAME = "microsoft/bitnet-b1.58-2B-4T"; tok = AutoTokenizer.from_pretrained(NAME); t0 = time.time()
IDS = [tok(t).input_ids for t in TEXTS]
for ids in IDS: assert 16 <= len(ids) <= 24, len(ids)
F = safe_open(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/model.safetensors"))[0], "pt")
deep, D, nq, nk = 30, 2560, 20, 5; d = D // nq; half = d // 2; base = 500000.0; EPS = 1e-5
_th = [base ** (-2 * i / d) for i in range(half)]

# the page table: the gauge from the whole table (as bitmedium_lean.py), the pages kept only for the words these passages use
sl = F.get_slice("model.embed_tokens.weight"); nrows = sl.get_shape()[0]          # read a slab at a time: the whole table never sits in memory
top = max(float(sl[i:i + 8192].abs().max().double()) for i in range(0, nrows, 8192)); gE = 2.0 ** math.ceil(math.log2(top / (1 << 20)))
USED = sorted(set(i for ids in IDS for i in ids)); ROW = {w: r for r, w in enumerate(USED)}
EMB = Dial(torch.round(torch.stack([sl[w:w + 1][0] for w in USED]).double() / gE).to(torch.int32), gE)

def beam(key):                                 # one photon up, one down, or dark, a byte each; one gauge for the bank
    return Dial(unpack_weights(F.get_tensor(key + ".weight"), dtype=torch.int8), float(F.get_tensor(key + ".weight_scale").double()))
def dial(w, bits=20, store=torch.int64):
    w = w.double(); gauge = 2.0 ** math.ceil(math.log2((float(w.abs().max()) or 1.0) / (1 << bits)))
    return Dial(torch.round(w / gauge).to(store), gauge)
_pile = Dial.pile
def pile(self, Wd):                            # the light poured through a beam's photons: whole numbers, exact (< 2^53)
    if Wd.p.dtype == torch.int8:               # poured 1,024 beam lines at a time (the same sums, less held at once)
        a = self.p.double(); out = torch.cat([(a @ Wd.p[i:i + 1024].double().T).to(torch.int64) for i in range(0, Wd.p.shape[0], 1024)], -1)
        return Dial(out, self.g * Wd.g).trim(WIDTH + 4)
    return _pile(self, Wd)
Dial.pile = pile
def actq(D):
    p = D.p; am = p.abs().amax(-1, keepdim=True).clamp_min(1); s = torch.sign(p)
    q = s * ((p.abs() * 254 + am) // (2 * am))
    return Dial(q * am, D.g / 127).trim(WIDTH)
W = {}
for z in range(deep):
    L = f"model.layers.{z}."
    for nm, key in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]:
        W[(z, nm)] = beam(L + key)
    for nm, key in [("in", "input_layernorm"), ("post", "post_attention_layernorm"), ("asub", "self_attn.attn_sub_norm"), ("fsub", "mlp.ffn_sub_norm")]:
        W[(z, nm)] = dial(F.get_tensor(L + key + ".weight"))
print(f"  loaded in {time.time() - t0:.0f}s: {sum(v.p.numel() for k, v in W.items() if v.p.dtype == torch.int8) / 1e9:.2f} billion photons, "
      f"{len(USED)} pages kept; {peak()}", flush=True)

# the add, watched: the machine's own add, plus the shifts it used, checked to rebuild its result exactly
_plus = Dial.plus
REC = None
def plus_watched(self, other, bits=WIDTH + 2):
    out = _plus(self, other, bits)
    lo = min(self.g, other.g)
    sa = min(40, max(0, round(math.log2(self.g / lo)))); sb = min(40, max(0, round(math.log2(other.g / lo))))
    sh = round(math.log2(out.g / lo))
    assert out.g == lo * float(1 << sh) and torch.equal(((self.p << sa) + (other.p << sb)) >> sh, out.p)
    if REC is not None: REC.append((other.p.clone(), sa, sb, sh, out.p.clone()))
    return out
Dial.plus = plus_watched

GATE = None
def one_depth(X, z, n, rotz, LIM):             # bitmedium_lean.py's depth, unchanged; GATE takes the gate pile
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
    if GATE is not None: GATE.append(G.p.clone())
    R = Dial(G.p.clamp_min(0), G.g); R2 = Dial(R.p * R.p, R.g * R.g).trim(WIDTH + 2)
    H = actq(normed(Dial(R2.p * U.p, R2.g * U.g).trim(WIDTH + 2), W[(z, "fsub")]))
    return X.plus(H.pile(W[(z, "down")]))

def run(ids, page=None, turnorder=None, depth_hook=None):
    global REC, GATE
    n = len(ids); rotz = torch.tensor([[round(m * t * ONE) for t in _th] for m in range(n)], dtype=torch.int64)
    if turnorder is not None: rotz = rotz[:, turnorder]
    LIM = (torch.arange(n)[None, :] > torch.arange(n)[:, None])[None]
    P = EMB.p if page is None else page
    X = Dial(P[torch.tensor([ROW[i] for i in ids])].to(torch.int64), EMB.g); X0 = X.p.clone()
    REC, GATE = [], []
    for z in range(deep):
        if depth_hook: saved = depth_hook(z)
        X = one_depth(X, z, n, rotz, LIM)
        if depth_hook:
            for key, v in saved.items(): W[key] = v
    out = (X0, REC, GATE); REC, GATE = None, None
    return out

# ---- the real order, recorded
t1 = time.time()
for i, ids in enumerate(IDS):
    X0, rec, gate = run(ids)
    states = np.stack([X0.numpy()] + [r[4].numpy() for r in rec]).astype(np.int32)           # 61 x n x 2560, each < 2^24
    writes = np.stack([r[0].numpy() for r in rec]); assert int(np.abs(writes).max()) < 2 ** 31
    shifts = np.array([[r[1], r[2], r[3]] for r in rec], dtype=np.int64)                     # sa, sb, sh per write
    dark = np.stack([(g <= 0).numpy() for g in gate]); below = np.stack([(g < 0).numpy() for g in gate])
    np.savez(OUT + f"text_{i:02d}.npz", ids=np.array(ids), states=states, writes=writes.astype(np.int32), shifts=shifts,
             dark=np.packbits(dark, axis=-1), below=np.packbits(below, axis=-1), n=len(ids))
    if i == 0: print(f"  first passage recorded ({len(ids)} positions) in {time.time() - t1:.1f}s; shifts sa/sb/sh seen: "
                     f"{sorted(set(map(int, shifts[:, 0])))} / {sorted(set(map(int, shifts[:, 1])))} / {sorted(set(map(int, shifts[:, 2])))}", flush=True)
print(f"  30 passages recorded in {time.time() - t1:.0f}s, {sum(len(x) for x in IDS)} positions; {peak()}", flush=True)

# ---- the same machine, everything in a random order
g = torch.Generator().manual_seed(4)
rho = torch.randperm(D, generator=g)                                   # the 2,560 lines
kap = torch.randperm(nk, generator=g)                                  # the 5 key heads
gam = [torch.randperm(nq // nk, generator=g) for _ in range(nk)]      # the 4 query heads riding each
phi = torch.randperm(half, generator=g)                                # the 64 turning pairs (shared by q, k and the turn table)
psi = [torch.randperm(d, generator=g) for _ in range(nk)]             # the 128 places of each value head
nu = [torch.randperm(6912, generator=g) for _ in range(deep)]         # the neurons, a new order every depth
phid = torch.cat([phi, phi + half])
qhead = [int(kap[h // 4]) * 4 + int(gam[h // 4][h % 4]) for h in range(nq)]                 # new query head h <- old
qrow = torch.cat([qhead[h] * d + phid for h in range(nq)])
krow = torch.cat([int(kap[m]) * d + phid for m in range(nk)])
vrow = torch.cat([int(kap[m]) * d + psi[m] for m in range(nk)])
ocol = torch.cat([qhead[h] * d + psi[h // 4] for h in range(nq)])
def permuted(z):
    saved = {(z, nm): W[(z, nm)] for nm in ["q", "k", "v", "o", "gate", "up", "down", "in", "post", "asub", "fsub"]}
    S = lambda nm: saved[(z, nm)]
    W[(z, "q")] = Dial(S("q").p[qrow][:, rho].contiguous(), S("q").g)
    W[(z, "k")] = Dial(S("k").p[krow][:, rho].contiguous(), S("k").g)
    W[(z, "v")] = Dial(S("v").p[vrow][:, rho].contiguous(), S("v").g)
    W[(z, "o")] = Dial(S("o").p[rho][:, ocol].contiguous(), S("o").g)
    W[(z, "gate")] = Dial(S("gate").p[nu[z]][:, rho].contiguous(), S("gate").g)
    W[(z, "up")] = Dial(S("up").p[nu[z]][:, rho].contiguous(), S("up").g)
    W[(z, "down")] = Dial(S("down").p[rho][:, nu[z]].contiguous(), S("down").g)
    for nm in ("in", "post"): W[(z, nm)] = Dial(S(nm).p[rho], S(nm).g)
    W[(z, "asub")] = Dial(S("asub").p[ocol], S("asub").g); W[(z, "fsub")] = Dial(S("fsub").p[nu[z]], S("fsub").g)
    return saved
t1 = time.time(); diff = tot = gdiff = gtot = shdiff = 0
PAGE_P = EMB.p[:, rho].contiguous()
for i, ids in enumerate(IDS):
    X0, rec, gate = run(ids, page=PAGE_P, turnorder=phi, depth_hook=permuted)
    f = np.load(OUT + f"text_{i:02d}.npz"); states, writes, shifts = f["states"], f["writes"], f["shifts"]; r = rho.numpy()
    dark = np.unpackbits(f["dark"], axis=-1, count=6912).astype(bool)
    st = np.stack([X0.numpy()] + [x[4].numpy() for x in rec]); wr = np.stack([x[0].numpy() for x in rec])
    diff += int(np.count_nonzero(st != states[:, :, r])) + int(np.count_nonzero(wr != writes[:, :, r])); tot += states.size + writes.size
    shdiff += int(np.count_nonzero(np.array([[x[1], x[2], x[3]] for x in rec]) != shifts))
    for z in range(deep):
        gdiff += int(np.count_nonzero((gate[z] <= 0).numpy() != dark[z][:, nu[z].numpy()])); gtot += dark[z].size
print(f"  everything in a random order (2,560 lines; 6,912 neurons, new order each depth; 5 key heads and the 4 query heads on each;\n"
      f"  64 turning pairs; 128 places of each value head): {diff} of {tot} recorded stream and write whole numbers differ,\n"
      f"  {shdiff} of {60 * 3 * len(IDS)} shifts differ, {gdiff} of {gtot} dark/lit neurons differ  ({time.time() - t1:.0f}s)", flush=True)
