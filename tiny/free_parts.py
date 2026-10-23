"""Take the free parts out of the exact whole-piece model (q = 2^-6, pieces_n_6.pt; full score).
Model: W = S + n*q exactly. Predict n from free / tiny stored parts, store only r = n - pred (whole numbers).
  pred = round(((c-1)*S + shift_g) / q)   c: one per plate; shift_g: word table only, one 128-vector per group
  groups: word rows never in the text / in it (1 flag per row stored).
Rebuild n = pred + r exactly. Control: the same with another blank (seed 1) in place of S for the prediction."""
import numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
q = 2.0 ** -6
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=128, vocab_size=50257)
torch.manual_seed(0); S = {n: p.detach().double().numpy() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
torch.manual_seed(1); SB = {n: p.detach().double().numpy() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
N = {n: v.numpy().astype(np.int64) for n, v in torch.load("pieces_n_6.pt").items()}
tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = np.array(tok(open("shakespeare.txt").read()).input_ids); tr = ids[:int(0.9 * len(ids))]
seen = np.zeros(50257, bool); seen[np.unique(tr)] = True

def H(x):
    _, c = np.unique(x, return_counts=True); c = c.astype(np.float64); p = c / c.sum(); return float(-(c * np.log2(p)).sum())

def run(Sp, label):
    tot = dict(n=0, base=0.0, left=0.0, side=0.0, wrong=0); parts = {}
    for name, n in N.items():
        s = Sp[name]; Wv = S[name] + n * q                       # the real model's weights (true blank)
        side = 0.0
        if s.ndim == 2 and np.any(s != 0):
            c = float(np.sum(Wv * s) / np.sum(s * s)); side += 64        # one c per plate
            g = (c - 1) * s
        else:
            g = np.zeros_like(s)
        if "wte" in name:
            sh = np.zeros_like(s)
            for msk in (~seen, seen):
                sh[msk] = (n[msk] * q - g[msk]).mean(0)                   # one shift per group
            side += 2 * 128 * 64 + 50257                                  # two shifts + one flag per row
            g = g + sh
        pred = np.round(g / q).astype(np.int64)
        r = n - pred
        wrong = int(np.sum(pred + r != n))
        b0, b1 = H(n.ravel()), H(r.ravel()) + side        # H gives total bits
        k = "word table" if "wte" in name else ("place table" if "wpe" in name else "body")
        a = parts.setdefault(k, [0, 0.0, 0.0, 0]); a[0] += n.size; a[1] += b0; a[2] += b1; a[3] += int(np.sum(r == 0))
        if "wte" in name:
            for lab, msk in (("never in the text", ~seen), ("in the text", seen)):
                print(f"   [{label}] word rows {lab:18s}: leftovers exactly 0 {100*np.mean(r[msk]==0):5.1f}%, "
                      f"bits a weight {H(n[msk].ravel())/msk.sum()/128:.2f} -> {H(r[msk].ravel())/msk.sum()/128:.2f}")
        tot["n"] += n.size; tot["base"] += b0; tot["left"] += b1; tot["wrong"] += wrong
    for k, (cnt, b0, b1, z) in parts.items():
        print(f"   [{label}] {k:12s} {cnt:>9,} weights: bits a weight {b0/cnt:.3f} -> {b1/cnt:.3f}  (leftovers exactly 0: {100*z/cnt:.1f}%)")
    print(f"   [{label}] WHOLE MODEL {tot['n']:,} weights: {tot['base']/tot['n']:.3f} -> {tot['left']/tot['n']:.3f} bits a weight "
          f"(stored parts counted) | rebuilt wrong {tot['wrong']}")

# the model itself rebuilt from seed + n, bit for bit (as in the run), once more here
torch.manual_seed(0); fresh = {n: p.detach().double() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
print("blank remade from seed 0 matches:", all(np.array_equal(fresh[n].numpy(), S[n]) for n in S))
run(S, "real blank"); run(SB, "control: another blank")
