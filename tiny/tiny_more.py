"""MORE TEACHING with snapshots at 3k, 6k, 12k, 24k steps (same settings as tiny.py).
A tiny model made here from seed 0: make it blank, try it, teach it on text, then remove the blank exactly.
Shape: 128 wide, 4 layers, 4 heads, 128 places, GPT-2's 50,257 words (word table shared by input and output).
Teaching: AdamW, step size 1e-3, no decay (so the blank is not shrunk), fp32 on the CPU.
Remove: X = W - S, every weight exact (S remade from the seed). Every weight rebuilt as S + X and checked bit for bit."""
import sys, math, time, json, numpy as np, torch
from fractions import Fraction
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
torch.set_num_threads(8)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
SNAP = {3000, 6000, 12000, 24000, STEPS}
SEED, B, T = 0, 8, 128
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=T, vocab_size=50257)

def blank():
    torch.manual_seed(SEED)
    return GPT2LMHeadModel(cfg)

def params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}   # the shared word table appears once

tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = torch.tensor(tok(open("shakespeare.txt").read()).input_ids)
cut = int(0.9 * len(ids)); tr, ho = ids[:cut], ids[cut:]
print(f"text: {len(ids):,} words, teach on {len(tr):,}, held out {len(ho):,}")

# held-out windows, fixed
g = torch.Generator().manual_seed(1234)
HO = torch.stack([ho[i:i+T+1] for i in torch.randint(0, len(ho) - T - 1, (64,), generator=g).tolist()])

@torch.no_grad()
def try_it(m, label):
    m.eval()
    ls, hs = [], []
    for c in range(0, len(HO), 8):                      # 8 windows at a time (memory)
        lg = m(HO[c:c+8, :T]).logits
        ls.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), HO[c:c+8, 1:].reshape(-1)).item())
        hs.append((lg.argmax(-1) == HO[c:c+8, 1:]).float().mean().item()); del lg
    loss, hit = float(np.mean(ls)), float(np.mean(hs))
    p = "ROMEO:\nO"
    pl = m(torch.tensor([tok(p).input_ids])).logits[0, -1].softmax(-1)
    top = pl.topk(5)
    words = ", ".join(f"{tok.decode([int(i)])!r} {v:.3f}" for v, i in zip(top.values, top.indices))
    print(f"[{label}] held-out loss {loss:.3f} (blank guess ln 50257 = {math.log(50257):.3f}); next word right {100*hit:.1f}%")
    print(f"[{label}] after {p!r}: {words}")
    return dict(loss=loss, right=hit)

# 1. make it blank, and make it again from the seed: bit for bit the same
m = blank(); S = params(m); S2 = params(blank())
same = all(torch.equal(S[n], S2[n]) for n in S)
nS = sum(v.numel() for v in S.values())
print(f"blank made from seed {SEED}: {nS:,} weights; made again from the seed, bit for bit the same: {same}")
assert all(torch.equal(S[n], torch.load('start.pt')[n]) for n in S)

# 2. try it blank
r_blank = try_it(m, "blank")

# 3. teach it
m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
g = torch.Generator().manual_seed(99); t0 = time.time()
for step in range(1, STEPS + 1):
    st = torch.randint(0, len(tr) - T - 1, (B,), generator=g)
    xb = torch.stack([tr[i:i+T+1] for i in st.tolist()])
    loss = m(xb[:, :T], labels=xb[:, :T]).loss   # the model shifts by one itself
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 1000 == 0 or step == STEPS:
        print(f"  step {step}: loss {loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if step in SNAP:
        W = params(m); torch.save(W, f"taught_{step}.pt")
        if step == 3000:
            old = torch.load("taught.pt"); print("  step 3000 equals the first run's taught.pt bit for bit:", all(torch.equal(W[n], old[n]) for n in W))
        try_it(m, f"step {step}"); m.train()

