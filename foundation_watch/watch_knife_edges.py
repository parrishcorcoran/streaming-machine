"""WATCH THE KNIFE EDGES IN THE STREAMING MACHINE. stream_sentence.py's machine (all 12 depths live every turn, depth k
reads what depth k-1 emitted last turn, values between depths on a dial, nothing ever cleared). At every depth the 3,072
middle neurons' piles meet the Gaussian knife edge (share = Phi(pile)). Watched at the newest position:
  per turn: how many edges change side (dark <-> bright) at each depth as the wave climbs;
  when the column stands: where the piles sit on the edge, and how many sit right on it;
  token after token: how many edges land on the same side as the token before's.
    python3 watch_knife_edges.py [n_tokens] [prompt]"""
import sys, math, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 6
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
tok = GPT2Tokenizer.from_pretrained("gpt2"); m = GPT2LMHeadModel.from_pretrained("gpt2").float().eval()
tr, blocks = m.transformer, list(m.transformer.h); L = len(blocks)
WH, H = 2 ** 23 - 1, 4096 ** 2 // 2 - 1
for blk in blocks:
    for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
        W = mod.weight.data; gq = W.abs().amax(0).clamp(min=1e-12) / WH; mod.weight.data = torch.round(W / gq) * gq
PILE = [None] * L                                                     # each depth's knife-edge piles, this turn
for k, blk in enumerate(blocks):
    blk.mlp.c_fc.register_forward_hook(lambda mod, inp, out, k=k: PILE.__setitem__(k, out[0].clone()))
def dial(v): s = v.abs().amax(-1, keepdim=True).clamp(min=1e-30) / H; return torch.round(v / s), s
def chord(ids): return tr.wte(ids) + tr.wpe(torch.arange(ids.shape[1]))[None]
def region(p):
    return [int((p < -4).sum()), int(((p >= -4) & (p < -2)).sum()), int(((p >= -2) & (p < 0)).sum()), int(((p >= 0) & (p < 2.66)).sum()), int((p >= 2.66).sum())]
seq = tok(prompt, return_tensors="pt").input_ids; S = seq.shape[1]
emitted = [torch.zeros(1, S, 768) for _ in range(L + 1)]; prevq = [None] * (L + 1)
cols, words = [], []
print(f"{prompt!r}: the knife edges at the newest position, depth 1..12 (3,072 edges each)\n")
for t in range(NTOK):
    turn, lastside = 0, [None] * L
    if t < 2: print(f"token {t + 1}: edges changing side this turn, depth 1..12        | head")
    while True:
        turn += 1; new = [chord(seq)] + [None] * L
        for k in range(1, L + 1):
            out = blocks[k - 1](emitted[k - 1]); new[k] = out[0] if isinstance(out, tuple) else out
        moved = 0; flips = []
        for k in range(L + 1):
            q, s = dial(new[k])
            moved += (q.numel() if prevq[k] is None or prevq[k].shape != q.shape else int((q != prevq[k]).sum()))
            prevq[k] = q; new[k] = q * s
        for k in range(L):
            side = PILE[k][-1] > 0
            flips.append(0 if lastside[k] is None else int((side != lastside[k]).sum())); lastside[k] = side
        emitted = new
        res = tr.ln_f(emitted[L])[0, -1] @ tr.wte.weight.T; top = res.argmax().item()
        if t < 2: print(f"  {turn:3d} | " + " ".join(f"{f:4d}" for f in flips) + f" | {tok.decode([top])!r}")
        if moved == 0 or turn > 40: break
    piles = [PILE[k][-1].clone() for k in range(L)]
    cols.append(piles); words.append(top)
    if t == 0:
        print("\n  where the piles sit on the edge when the column stands (newest position), per depth:")
        print("  depth   far dark <-4 | -4..-2 | -2..0 | 0..2.66 | open >2.66 |  on the knife |x|<0.25 | pass-through share (mean)")
        for k in range(L):
            p = piles[k]; r = region(p); phi = 0.5 * (1 + torch.erf(p / math.sqrt(2)))
            print(f"   {k + 1:2d}     {r[0]:5d}       | {r[1]:5d}  | {r[2]:5d} |  {r[3]:5d}  |   {r[4]:4d}     |     {int((p.abs() < 0.25).sum()):5d}          |  {float(phi.mean()):.3f}")
    if t >= 1:
        same = [100.0 * float(((cols[-1][k] > 0) == (cols[-2][k] > 0)).float().mean()) for k in range(L)]
        print(f"token {t + 1:2d} {tok.decode([words[-1]])!r:12s}: edges on the same side as the token before's, depth 1..12: " + " ".join(f"{x:3.0f}" for x in same))
    seq = torch.cat([seq, torch.tensor([[top]])], 1)
    for k in range(L + 1): emitted[k] = torch.cat([emitted[k], torch.zeros(1, 1, 768)], 1)
# control: two unrelated texts' columns
o = tok(" Water boils at a temperature of", return_tensors="pt").input_ids
_ = m(o); ctl = [PILE[k][-1].clone() for k in range(L)]
same = [100.0 * float(((cols[-1][k] > 0) == (ctl[k] > 0)).float().mean()) for k in range(L)]
print(f"\ncontrol: last column against an unrelated text's last column, same side, depth 1..12: " + " ".join(f"{x:3.0f}" for x in same))
print(f"\nwords: {tok.decode(words)!r}")
