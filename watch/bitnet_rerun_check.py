"""A checker's rerun of the BitNet whole-number machine, ONE DEPTH'S BEAMS IN MEMORY AT A TIME, against the records in
bwatch/text_NN.npz. The depth is bitmedium_lean.py's own one_depth (taken from that file's text, not from the copy in
bitnet_collect.py). Two passes per depth, all 30 passages each: the real order, and everything in a random order of the
checker's own (seed 99): the lines, the neurons (new order each depth), the key heads and the query heads on each, the
turning pairs (shared by q, k and the turn table), and the places of each value head.
Checked: every recorded stream and write whole number, every shift, the dark neurons, and the stream's gauge chain
(log2 of the gauge, from the machine's own float gauge) against the gauge the watch script rebuilt from the shifts.

    cd <scratchpad>/hp && python3 ~/streaming-machine/watch/bitnet_rerun_check.py"""
import math, os, sys, glob, time, resource, torch, numpy as np
SCR = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
os.chdir(SCR + "hp/"); sys.argv = [sys.argv[0], "22"]
src = open("medium.py").read()
exec(compile(src[:src.index('NAME = os.environ.get("MODEL"')], "medium.py", "exec"))
from safetensors import safe_open
from transformers.integrations.bitnet import unpack_weights
F = safe_open(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/model.safetensors"))[0], "pt")
deep, D, nq, nk = 30, 2560, 20, 5; d = D // nq; half = d // 2; base = 500000.0; EPS = 1e-5
_th = [base ** (-2 * i / d) for i in range(half)]
lean = open("bitmedium_lean.py").read()
exec(compile(lean[lean.index("def beam(key)"):lean.index("W, NW = {}, {}")], "bitmedium_lean.py", "exec"))   # beam, dial, pile, actq
exec(compile(lean[lean.index("CROSS = None"):lean.index("def run(ids)")], "bitmedium_lean.py", "exec"))      # one_depth, unchanged
W = {}
def peak(): return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"

LOG = []                                        # every add and every pile, as the machine makes them
_plus, _pile2 = Dial.plus, Dial.pile
def plus_seen(self, other, bits=WIDTH + 2):
    out = _plus(self, other, bits); LOG.append(("plus", self.g, other.g, other.p, out.p, out.g)); return out
def pile_seen(self, Wd):
    out = _pile2(self, Wd); LOG.append(("pile", out.p)); return out
Dial.plus, Dial.pile = plus_seen, pile_seen

MM = SCR + "brecheck/mm/"; os.makedirs(MM, exist_ok=True)
for i in range(30):                             # the records as plain arrays on disk, read through a memory map (little held)
    if not os.path.exists(MM + f"{i:02d}_dark.npy"):
        f = np.load(SCR + f"bwatch/text_{i:02d}.npz")
        for k in ("ids", "states", "writes", "shifts", "dark"): np.save(MM + f"{i:02d}_{k}.npy", f[k])
class Rec:
    def __init__(self, i): self.i = i
    def __getitem__(self, k):
        a = np.load(MM + f"{self.i:02d}_{k}.npy", mmap_mode="r")
        return Unpacked(a) if k == "dark" else a
class Unpacked:
    def __init__(self, a): self.a = a
    def __getitem__(self, z): return np.unpackbits(self.a[z], axis=-1, count=6912).astype(bool)
RD = [Rec(i) for i in range(30)]
sl = F.get_slice("model.embed_tokens.weight"); nrows = sl.get_shape()[0]
top = max(float(sl[i:i + 8192].abs().max().double()) for i in range(0, nrows, 8192)); gE = 2.0 ** math.ceil(math.log2(top / (1 << 20)))

g = torch.Generator().manual_seed(99)
rho = torch.randperm(D, generator=g); kap = torch.randperm(nk, generator=g); gam = [torch.randperm(nq // nk, generator=g) for _ in range(nk)]
phi = torch.randperm(half, generator=g); psi = [torch.randperm(d, generator=g) for _ in range(nk)]; nu = [torch.randperm(6912, generator=g) for _ in range(deep)]
pp = torch.cat([phi, phi + half])
oldq = [4 * int(kap[h // 4]) + int(gam[h // 4][h % 4]) for h in range(nq)]
qrow = torch.cat([oldq[h] * d + pp for h in range(nq)]); krow = torch.cat([int(kap[m]) * d + pp for m in range(nk)])
vrow = torch.cat([int(kap[m]) * d + psi[m] for m in range(nk)]); ocol = torch.cat([oldq[h] * d + psi[h // 4] for h in range(nq)])
R_ = rho.numpy()

def setup(ids, perm):
    n = len(ids); rotz = torch.tensor([[round(m * t * ONE) for t in _th] for m in range(n)], dtype=torch.int64)
    if perm: rotz = rotz[:, phi]
    LIM = (torch.arange(n)[None, :] > torch.arange(n)[:, None])[None]
    E = torch.round(torch.stack([sl[w:w + 1][0] for w in ids]).double() / gE).to(torch.int32).to(torch.int64)
    if perm: E = E[:, rho]
    return [Dial(E, gE), n, rotz, LIM, gE]

bad = {"stream": 0, "write": 0, "shift": 0, "dark": 0, "gauge": 0}; tot = {"stream": 0, "write": 0, "shift": 0, "dark": 0, "gauge": 0}
ST = {}; GQ = []
for perm in (False, True):
    for i in range(30):
        ST[(perm, i)] = setup([int(t) for t in RD[i]["ids"]], perm)
        s0 = RD[i]["states"][0][:, R_] if perm else RD[i]["states"][0]
        bad["stream"] += int(np.count_nonzero(ST[(perm, i)][0].p.numpy() != s0)); tot["stream"] += s0.size
t0 = time.time()
for z in range(deep):
    L = f"model.layers.{z}."
    for nm, key in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]:
        W[(z, nm)] = beam(L + key)
    for nm, key in [("in", "input_layernorm"), ("post", "post_attention_layernorm"), ("asub", "self_attn.attn_sub_norm"), ("fsub", "mlp.ffn_sub_norm")]:
        W[(z, nm)] = dial(F.get_tensor(L + key + ".weight"))
    real = {k: v for k, v in W.items()}; SS = lambda nm: real[(z, nm)]
    permd = {(z, "q"): Dial(SS("q").p[qrow][:, rho].contiguous(), SS("q").g), (z, "k"): Dial(SS("k").p[krow][:, rho].contiguous(), SS("k").g),
             (z, "v"): Dial(SS("v").p[vrow][:, rho].contiguous(), SS("v").g), (z, "o"): Dial(SS("o").p[rho][:, ocol].contiguous(), SS("o").g),
             (z, "gate"): Dial(SS("gate").p[nu[z]][:, rho].contiguous(), SS("gate").g), (z, "up"): Dial(SS("up").p[nu[z]][:, rho].contiguous(), SS("up").g),
             (z, "down"): Dial(SS("down").p[rho][:, nu[z]].contiguous(), SS("down").g), (z, "in"): Dial(SS("in").p[rho], SS("in").g),
             (z, "post"): Dial(SS("post").p[rho], SS("post").g), (z, "asub"): Dial(SS("asub").p[ocol], SS("asub").g), (z, "fsub"): Dial(SS("fsub").p[nu[z]], SS("fsub").g)}
    for perm in (False, True):
        W.clear(); W.update(permd if perm else real)
        for i in range(30):
            st = ST[(perm, i)]; X, n, rotz, LIM, g0 = st; LOG.clear()
            X = one_depth(X, z, n, rotz, LIM); st[0] = X
            plus = [r for r in LOG if r[0] == "plus"]; piles = [r for r in LOG if r[0] == "pile"]
            assert len(plus) == 2 and len(piles) == 7
            rec = RD[i]
            for j, (_, ga, gb, wp, op, og) in enumerate(plus):
                w = 2 * z + j; lo = min(ga, gb)
                sh = [min(40, max(0, round(math.log2(ga / lo)))), min(40, max(0, round(math.log2(gb / lo)))), round(math.log2(og / lo))]
                ws, ss = (rec["writes"][w][:, R_], rec["states"][w + 1][:, R_]) if perm else (rec["writes"][w], rec["states"][w + 1])
                bad["write"] += int(np.count_nonzero(wp.numpy() != ws)); tot["write"] += ws.size
                bad["stream"] += int(np.count_nonzero(op.numpy() != ss)); tot["stream"] += ss.size
                bad["shift"] += int(np.count_nonzero(np.array(sh) != rec["shifts"][w])); tot["shift"] += 3
            e = 0                               # the watch script's gauge for after write 2z+1, from the recorded shifts
            for w in range(2 * z + 2): e += int(rec["shifts"][w][2]) - int(rec["shifts"][w][0])
            q = math.log2(X.g / (g0 * 2.0 ** e)); GQ.append(q); bad["gauge"] += int(abs(q) >= 0.5); tot["gauge"] += 1
            dk = (piles[4][1] <= 0).numpy(); dr = rec["dark"][z][:, nu[z].numpy()] if perm else rec["dark"][z]
            bad["dark"] += int(np.count_nonzero(dk != dr)); tot["dark"] += dr.size
    W.clear(); del real, permd
    print(f"  depth {z:2d} done ({time.time() - t0:.0f}s, peak {peak()}): differ so far " + ", ".join(f"{k} {bad[k]}/{tot[k]}" for k in bad), flush=True)
print(f"  gauge: machine float gauge vs gauge rebuilt from shifts, log2 ratio min {min(GQ):.3f} max {max(GQ):.3f}, exactly 0 on {sum(q == 0 for q in GQ)}/{len(GQ)}")
print("  FINAL (real order and checker's random order, seed 99): " + ", ".join(f"{k} {bad[k]} of {tot[k]} differ" for k in bad))
