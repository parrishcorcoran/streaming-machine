"""The tiny model taught in whole pieces. Every weight is always exactly  blank + n * q,  n a whole number, q = 2^-K
one piece shared by the whole model. Each step: the teaching's wished move (AdamW, same settings as tiny.py) plus the
carried leftover becomes a whole number of pieces added to n; the leftover under half a piece carries to the next step
(used only while teaching, never stored). float64 holds blank + n*q exactly; every step is checked exact (two-sum).
The stored model is the seed and the whole numbers n. Same text, order and 3,000 steps as the fp32 run."""
import sys, math, time, json, numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
torch.set_num_threads(8)
K = int(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
q = 2.0 ** -K
SEED, B, T = 0, 8, 128
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=T, vocab_size=50257)
torch.manual_seed(SEED); m = GPT2LMHeadModel(cfg)
assert all(torch.equal(p.detach(), torch.load("start.pt")[n]) for n, p in m.named_parameters())
m = m.double()                                             # fp32 blank -> float64, exact
P = dict(m.named_parameters())
S = {n: p.detach().clone() for n, p in P.items()}          # the blank, float64 (its values are the fp32 ones)
N = {n: torch.zeros_like(p, dtype=torch.int64) for n, p in P.items()}
C = {n: torch.zeros_like(p) for n, p in P.items()}         # carried leftover, teaching only

tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = torch.tensor(tok(open("shakespeare.txt").read()).input_ids)
cut = int(0.9 * len(ids)); tr, ho = ids[:cut], ids[cut:]
g = torch.Generator().manual_seed(1234)
HO = torch.stack([ho[i:i+T+1] for i in torch.randint(0, len(ho) - T - 1, (64,), generator=g).tolist()])

@torch.no_grad()
def try_it(label):
    m.eval(); ls, hs = [], []
    for c in range(0, len(HO), 8):
        lg = m(HO[c:c+8, :T]).logits
        ls.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), HO[c:c+8, 1:].reshape(-1)).item())
        hs.append((lg.argmax(-1) == HO[c:c+8, 1:]).float().mean().item()); del lg
    p = "ROMEO:\nO"; pl = m(torch.tensor([tok(p).input_ids])).logits[0, -1].softmax(-1); top = pl.topk(5)
    words = ", ".join(f"{tok.decode([int(i)])!r} {v:.3f}" for v, i in zip(top.values, top.indices))
    loss, hit = float(np.mean(ls)), float(np.mean(hs))
    print(f"[{label}] held-out loss {loss:.3f}; next word right {100*hit:.1f}%; after {p!r}: {words}", flush=True)
    m.train(); return loss, hit

ROUNDED = {}
@torch.no_grad()
def set_exact(n):
    """P[n] = blank + N*q added in float64 (a fixed rule, so seed + n rebuild it bit for bit).
    Two-sum counts the weights whose add dropped some of a tiny blank's lowest places."""
    a, b = S[n], N[n].double() * q
    assert torch.all((b / q) == N[n].double())                 # n*q itself exact
    s = a + b; bb = s - a; err = (a - (s - bb)) + (b - bb)
    ROUNDED[n] = int((err != 0).sum())
    P[n].copy_(s)

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
g = torch.Generator().manual_seed(99); t0 = time.time(); m.train()
for step in range(1, STEPS + 1):
    st = torch.randint(0, len(tr) - T - 1, (B,), generator=g)
    xb = torch.stack([tr[i:i+T+1] for i in st.tolist()])
    loss = m(xb[:, :T], labels=xb[:, :T]).loss
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    before = {n: p.detach().clone() for n, p in P.items()}
    opt.step()
    with torch.no_grad():
        for n, p in P.items():
            wish = (p - before[n]) + C[n]
            k = torch.round(wish / q).to(torch.int64)
            C[n] = wish - k.double() * q
            N[n] += k
            set_exact(n)
    if step % 500 == 0 or step == STEPS:
        moved = sum(int((v != 0).sum()) for v in N.values()); tot = sum(v.numel() for v in N.values())
        print(f"  step {step}: loss {loss.item():.3f}  weights moved so far {100*moved/tot:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

loss_ho, right = try_it(f"pieces 2^-{K}")
# rebuild from the seed and the whole numbers alone, in a fresh model, and compare bit for bit
torch.manual_seed(SEED); fresh = GPT2LMHeadModel(cfg).double()
wrong = sum(int((fp.detach() + N[n].double() * q != P[n].detach()).sum()) for n, fp in fresh.named_parameters())
rounded = sum(ROUNDED.values())
print(f"rebuilt from seed {SEED} + whole numbers: wrong {wrong}; weights whose add dropped tiny blank places: {rounded}")

def H(x):
    _, c = np.unique(x, return_counts=True); c = c.astype(np.float64); p = c / c.sum()
    return float(-(c * np.log2(p)).sum())
tot = 0; bits = 0.0; zero = 0; rows = {}
for n, v in N.items():
    a = v.numpy().ravel(); b = H(a); tot += a.size; bits += b; zero += int((a == 0).sum())
    rows[n] = dict(n=a.size, zero=int((a == 0).sum()), bits=b / a.size, max_abs=int(np.abs(a).max()))
big = sorted(rows.items(), key=lambda kv: -kv[1]["n"])[:4]
for n, r in big:
    print(f"  {n:32s} {r['n']:>9,}  n exactly 0 {100*r['zero']/r['n']:5.1f}%  bits a weight {r['bits']:5.2f}  largest |n| {r['max_abs']}")
print(f"PIECE 2^-{K}: {tot:,} weights | n exactly 0 {100*zero/tot:.2f}% | exact bits a weight (packed whole numbers) "
      f"{bits/tot:.3f} | held-out loss {loss_ho:.3f}, right {100*right:.1f}% | blank: seed (0 bits) | rebuilt wrong {wrong}")
torch.save({n: v.to(torch.int32) for n, v in N.items()}, f"pieces_n_{K}.pt")
json.dump(dict(rebuilt_wrong=wrong, rounded=rounded, K=K, steps=STEPS, weights=tot, zero=zero, bits_per_weight=bits / tot, loss=loss_ho, right=right, rows=rows),
          open(f"pieces_{K}.json", "w"), indent=1)
