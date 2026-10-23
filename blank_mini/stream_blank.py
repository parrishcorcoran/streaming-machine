"""The streaming machine. GPT-2, every layer working every step on what the layer below passed up the step
before. Nothing is ever reset: when every layer has gone still, the word at the top is added as the next token,
and everything already there stays standing.

  values between layers : positions on two circles (phase 4096 x laps 4096), one gauge per token.
                          An edge fires only where a position moved, so the state between layers is visible.
  weights               : positions on circles of --base (4096 x 2, 8 x 8 or 2 x 24 digits, all 24 bits), one gauge
                          per column. A product is each digit poured, shifted by its place. Whole numbers.
  the bend              : the lap's edge. The share that gets past is looked up from a table, not computed.
  layer norm, attention : GPT-2's own (float).

  --order shuffle   layers fire in a random order each step, each reading what is below it now
  --init noise      every layer starts with junk on it instead of zeros
  --perm SEED       the layers are wired in a random depth order (which feeds which)

    python3 streaming_machine.py [--base 4096|8|2] [--words 12] [--prompt "..."] [--order shuffle] [--init noise] [--perm 0]
"""
import argparse, math
import torch, torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--words", type=int, default=12)
ap.add_argument("--order", default="fixed", choices=["fixed", "shuffle"])
ap.add_argument("--init", default="zero", choices=["zero", "noise"])
ap.add_argument("--perm", type=int, default=-1)
ap.add_argument("--show", type=int, default=2, help="print every step for this many words")
ap.add_argument("--base", type=int, default=4096, choices=[2, 8, 16, 64, 4096], help="circle size the weights are written on")
a = ap.parse_args()

PHASE = LAPS = 4096
HALF = PHASE * LAPS // 2 - 1                        # 8,388,607: the far side of the dial is the negatives
BASE = a.base; DIGITS = math.ceil(24 / math.log2(BASE))   # 24 bits: 4096 x 2, 8 x 8, 2 x 24


def positions(v):
    """a value as a position on the two circles; one gauge for the whole token"""
    g = v.abs().amax(-1, keepdim=True).clamp_min(1e-30) / HALF
    return torch.round(v / g).to(torch.int64), g


class CircleBank(nn.Module):
    """a weight bank on circles of BASE positions: each weight is its digits, written once. A product is each digit
    poured (up to BASE-1 steps), shifted by its place, the places landing at once. Whole numbers throughout; held in
    float64, which carries every one of these sums exactly (all stay below 2**53)."""
    def __init__(s, conv):
        super().__init__()
        W = conv.weight.detach().double()
        s.gw = W.abs().amax(0).clamp_min(1e-30) / HALF                  # one gauge per column
        q = torch.round(W / s.gw).to(torch.int64)
        sign, mag = torch.sign(q), q.abs()
        s.digits = [(sign * ((mag // BASE ** j) % BASE)).double() for j in range(DIGITS)]   # the digit patterns, once
        s.b = conv.bias.detach().double()

    def forward(s, x):
        xi, gx = positions(x)
        xi = xi.double()
        pile = sum((xi @ d) * float(BASE ** j) for j, d in enumerate(s.digits))   # every place poured, shifted, added
        return pile * gx * s.gw + s.b


EDGE_N, EDGE_SPAN = 1 << 20, 16.0                   # places across the edge's profile, and how far it reaches


class EdgeBend(nn.Module):
    """the bend as the lap's edge: what share of a value gets past, looked up by where it arrives"""
    def __init__(s):
        super().__init__()
        p = torch.linspace(-EDGE_SPAN, EDGE_SPAN, EDGE_N, dtype=torch.float64)
        s.share = 0.5 * (1 + torch.erf(p / math.sqrt(2)))               # the profile, written once

    def forward(s, x):
        i = torch.round((x + EDGE_SPAN) / (2 * EDGE_SPAN) * (EDGE_N - 1)).long()
        share = torch.where(i < 0, torch.zeros_like(x),
                torch.where(i >= EDGE_N, torch.ones_like(x), s.share[i.clamp(0, EDGE_N - 1)]))
        return x * share


tok = GPT2Tokenizer.from_pretrained("gpt2")
ref = GPT2LMHeadModel.from_pretrained(a.model).eval()
m = GPT2LMHeadModel.from_pretrained(a.model).double().eval()
tr = m.transformer
for blk in tr.h:
    blk.attn.c_attn, blk.attn.c_proj = CircleBank(blk.attn.c_attn), CircleBank(blk.attn.c_proj)
    blk.mlp.c_fc, blk.mlp.c_proj = CircleBank(blk.mlp.c_fc), CircleBank(blk.mlp.c_proj)
    blk.mlp.act = EdgeBend()
L = len(tr.h)
wiring = list(range(L))
g = torch.Generator().manual_seed(max(a.perm, 0))
if a.perm >= 0:
    wiring = torch.randperm(L, generator=g).tolist()

ids = tok(a.prompt, return_tensors="pt").input_ids
greedy = ref.generate(ids, max_new_tokens=a.words, do_sample=False, pad_token_id=50256)[0, ids.shape[1]:].tolist()


def run(layer_idx, h):
    out = tr.h[layer_idx](h)
    return out[0] if isinstance(out, tuple) else out


def strike(seq):
    return tr.wte(seq) + tr.wpe(torch.arange(seq.shape[1]))[None]


seq = ids.clone(); S = seq.shape[1]
x0 = strike(seq)
emitted = [torch.zeros_like(x0) if a.init == "zero" else torch.randn(x0.shape, generator=g, dtype=x0.dtype) * x0.abs().mean() * 4
           for _ in range(L + 1)]
prev = [None] * (L + 1)
gen, turn, total_edges = [], 0, 0

print(f"{a.prompt!r}  order {a.order}, start {a.init}, wiring {'as trained' if a.perm < 0 else wiring}")
print(f"weights on circles of {BASE} x {DIGITS}: a product takes up to {DIGITS * (BASE - 1):,} steps to pour")
print(f"turn | edges fired between layers 0..{L} (positions that moved)                          | top")
while len(gen) < a.words:
    struck, show = turn, len(gen) < a.show
    while True:
        turn += 1
        emitted[0] = strike(seq)
        if a.order == "fixed":
            new = [emitted[0]] + [run(wiring[k], emitted[k]) for k in range(L)]
        else:
            new = list(emitted)
            for k in torch.randperm(L, generator=g).tolist():
                new[k + 1] = run(wiring[k], new[k])
        fired = []
        for k in range(L + 1):
            q, gk = positions(new[k])
            if prev[k] is None: f = q.numel()
            elif prev[k].shape != q.shape: f = int((q[:, :-1] != prev[k]).sum()) + q.shape[-1]
            else: f = int((q != prev[k]).sum())
            fired.append(f); prev[k] = q
            new[k] = q.double() * gk                                     # what the next layer reads is what crossed
        emitted = new; total_edges += sum(fired)
        lg = tr.ln_f(emitted[L])[0, -1] @ tr.wte.weight.T
        top2 = lg.topk(2)
        word = int(top2.indices[0])
        if show:
            print(f"{turn:4d} | " + " ".join(f"{f:5d}" for f in fired) + f" | {tok.decode([word])!r:10s} {float(top2.values[0] - top2.values[1]):+.3f}")
        if sum(fired) == 0 or turn - struck > 60:
            break
    gen.append(word)
    note = "" if len(gen) <= len(greedy) and word == greedy[len(gen) - 1] else f"   (model says {tok.decode([greedy[len(gen) - 1]])!r})"
    print(f"  word {len(gen):2d}: {tok.decode([word])!r:12s} settled after {turn - struck} steps{note}")
    seq = torch.cat([seq, torch.tensor([[word]])], 1)
    for k in range(L + 1):
        emitted[k] = torch.cat([emitted[k], torch.zeros(1, 1, emitted[k].shape[-1], dtype=emitted[k].dtype)], 1)

same = gen == greedy[:len(gen)]
print(f"\nstreaming : {tok.decode(gen)!r}")
print(f"the model : {tok.decode(greedy)!r}")
print(f"{'IDENTICAL' if same else 'DIFFERENT'}; {turn} steps for {len(gen)} words; edges fired {total_edges:,}")
