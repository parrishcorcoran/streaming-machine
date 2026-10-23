"""Where is the blank after more teaching? For each snapshot (3k, 6k, 12k, 24k steps):
WORD TABLE, split into words never in the teaching text and words in it:
  - the shared move: one vector for the whole group, its share of the group's move;
  - the blank after the shared move is taken off: cos(row - shared move, its own blank), and the same against
    another blank (seed 1, never used) as the control.
BODY PLATES: the blank at its own place (cos), after the shared row-move and column-move are taken off (cos),
  and in moved form: the overlap pattern of neurons G = W W^T and of lines H = W^T W against the blank's
  (off-diagonal corr), each against the control blank. Own place is taken off each row/column first for G/H."""
import sys, numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=128, vocab_size=50257)
S = {n: v.double().numpy() for n, v in torch.load("start.pt").items()}
torch.manual_seed(1); SB = {n: p.detach().double().numpy() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = np.array(tok(open("shakespeare.txt").read()).input_ids); tr = ids[:int(0.9 * len(ids))]
seen = np.zeros(50257, bool); seen[np.unique(tr)] = True
steps = [int(a) for a in sys.argv[1:]] or [3000, 6000, 12000, 24000]

def rowcos(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
def cos(A, B): return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))
def offcorr(P, Q):
    i, j = np.triu_indices(P.shape[0], 1); return float(np.corrcoef(P[i, j], Q[i, j])[0, 1])
def rem_rows(W, B):     # take the blank's own place off each row: w_i - a_i b_i
    a = np.sum(W * B, 1) / np.sum(B * B, 1); return W - a[:, None] * B

for st in steps:
    W = {n: v.double().numpy() for n, v in torch.load(f"taught_{st}.pt").items()}
    print(f"\n===== step {st} =====")
    Wt, St, Bt = W["transformer.wte.weight"], S["transformer.wte.weight"], SB["transformer.wte.weight"]
    for lab, msk in (("word rows never in the text", ~seen), ("word rows in the text", seen)):
        X = Wt[msk] - St[msk]; u = X.mean(0); R = X - u; left = Wt[msk] - u
        print(f"  {lab:28s} {msk.sum():6,} | move/blank length {np.mean(np.linalg.norm(X,axis=1)/np.linalg.norm(St[msk],axis=1)):6.2f} | "
              f"one shared move = {100*(1-np.sum(R*R)/np.sum(X*X)):5.1f}% of the move | shared move off: cos with own blank "
              f"{rowcos(left, St[msk]).mean():.3f}, with another blank {rowcos(left, Bt[msk]).mean():+.3f}")
    print(f"  {'plate':34s} {'own place':>9s} {'(ctrl)':>7s} {'shared moves off':>16s} {'(ctrl)':>7s} {'G neurons':>9s} {'(ctrl)':>7s} {'H lines':>8s} {'(ctrl)':>7s}")
    for n in W:
        if W[n].ndim != 2 or "wte" in n: continue
        w, s, b = W[n], S[n], SB[n]
        if "wpe" not in n: w, s, b = w.T, s.T, b.T          # GPT-2 plates are (in, out): rows = neurons
        own, ownc = cos(w, s), cos(w, b)
        wl = w - w.mean(0, keepdims=True); wl = wl - wl.mean(1, keepdims=True)
        sh, shc = cos(wl, s), cos(wl, b)
        gw = rem_rows(w, s); gwc = rem_rows(w, b)
        G, Gc = offcorr(gw @ gw.T, s @ s.T), offcorr(gwc @ gwc.T, b @ b.T)
        hw = rem_rows(w.T, s.T); hwc = rem_rows(w.T, b.T)
        Hh, Hc = offcorr(hw @ hw.T, s.T @ s), offcorr(hwc @ hwc.T, b.T @ b)
        print(f"  {n:34s} {own:9.3f} {ownc:+7.3f} {sh:16.3f} {shc:+7.3f} {G:9.3f} {Gc:+7.3f} {Hh:8.3f} {Hc:+7.3f}")
