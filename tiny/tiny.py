"""A tiny model made here from seed 0: make it blank, try it, teach it on text, then remove the blank exactly.
Shape: 128 wide, 4 layers, 4 heads, 128 places, GPT-2's 50,257 words (word table shared by input and output).
Teaching: AdamW, step size 1e-3, no decay (so the blank is not shrunk), fp32 on the CPU.
Remove: X = W - S, every weight exact (S remade from the seed). Every weight rebuilt as S + X and checked bit for bit."""
import sys, math, time, json, numpy as np, torch
from fractions import Fraction
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
torch.set_num_threads(8)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
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
torch.save(S, "start.pt")

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
    if step % 250 == 0 or step == STEPS:
        print(f"  step {step}: loss {loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
W = params(m); torch.save(W, "taught.pt")
r_taught = try_it(m, "taught")

# 4. remove the blank exactly: X = W - S
def exact_diff(w, s):
    """w - s for fp32 arrays, exact. float64 holds it whenever the two powers of two are within 28 places
    (the span is then at most 53 places); the rest go through exact fractions."""
    w64, s64 = w.astype(np.float64), s.astype(np.float64)
    x = w64 - s64
    ew = np.frexp(w64)[1]; es = np.frexp(s64)[1]
    far = (w64 != 0) & (s64 != 0) & (np.abs(ew - es) > 28)
    out = x.astype(object); n_far = int(far.sum())
    for i in np.flatnonzero(far):
        out[i] = Fraction(float(w64[i])) - Fraction(float(s64[i]))
    return out, n_far

def H(keys):
    _, c = np.unique(keys, return_counts=True); c = c.astype(np.float64); p = c / c.sum()
    return float(-(c * np.log2(p)).sum())

def exact_size(a):
    """bits to write every value of a float64 array exactly: one table per tensor for (sign, top place, length)
    packed, then the places below the top written plainly. Zero is its own symbol."""
    nz = a != 0
    mant, e = np.frexp(np.abs(a[nz])); M = (mant * 2.0 ** 53).astype(np.int64)
    assert np.all(M.astype(np.float64) == mant * 2.0 ** 53)
    tz = np.log2((M & -M).astype(np.float64)).astype(np.int64)
    top = e - 1; ln = 52 - tz
    key = np.zeros(a.size, np.int64)                     # 0 = the zero symbol
    key[nz] = 1 + ((np.sign(a[nz]).astype(np.int64) + 1) // 2) + 2 * ((top + 400) * 64 + ln)
    return H(key) + float(ln.sum()), H(key), float(ln.sum())

rows = {}; tot = dict(n=0, zero=0, far=0, bad=0, W=0.0, X=0.0, Wpack=0.0, Xpack=0.0)
for n in S:
    w = W[n].numpy().ravel(); s = S[n].numpy().ravel()
    x, n_far = exact_diff(w, s)
    # rebuild: S + X must give W bit for bit
    back = np.array([float(Fraction(float(si)) + (xi if isinstance(xi, Fraction) else Fraction(float(xi)))) for si, xi in zip(s, x)], dtype=np.float64) \
        if n_far else (s.astype(np.float64) + x.astype(np.float64))
    bad = int(np.sum(back.astype(np.float32) != w)) + int(np.sum(back != w.astype(np.float64)))
    assert n_far == 0, f"{n}: {n_far} pairs need fractions; size count would be inexact"
    xf = x.astype(np.float64)
    bx, px, rx = exact_size(xf); bw, pw, rw = exact_size(w.astype(np.float64))
    zero = int(np.sum(xf == 0))
    Ls, Lx = float(np.linalg.norm(s)), float(np.linalg.norm(xf))
    rows[n] = dict(n=w.size, zero=zero, far=n_far, bad=bad, W_bits=bw / w.size, X_bits=bx / w.size,
                   X_packed_part=px / w.size, X_raw_part=rx / w.size, S_len=Ls, X_len=Lx)
    tot["n"] += w.size; tot["zero"] += zero; tot["far"] += n_far; tot["bad"] += bad; tot["W"] += bw; tot["X"] += bx
    tot["Wpack"] += pw; tot["Xpack"] += px
    print(f"{n:32s} {w.size:>9,}  X exactly 0: {100*zero/w.size:5.1f}%   bits a weight exact: W {bw/w.size:5.2f}  "
          f"X {bx/w.size:5.2f} (outer {px/w.size:4.2f} + inner {rx/w.size:5.2f})   length X/S {(f'{Lx/Ls:7.3f}' if Ls else 'start 0'):>7s}  "
          f"rebuilt wrong {bad}", flush=True)

N = tot["n"]
print(f"\nALL {N:,} weights: rebuilt S + X wrong {tot['bad']} | X exactly 0: {100*tot['zero']/N:.2f}% | "
      f"exact bits a weight: W {tot['W']/N:.2f}, X {tot['X']/N:.2f} (outer {tot['Xpack']/N:.2f}), S 0 (the seed) | "
      f"far pairs via fractions {tot['far']}")
json.dump(dict(blank=r_blank, taught=r_taught, steps=STEPS, rows=rows,
               total=dict(weights=N, wrong=tot["bad"], X_zero=tot["zero"], W_bits=tot["W"]/N, X_bits=tot["X"]/N)),
          open(f"result_{STEPS}.json", "w"), indent=1)
