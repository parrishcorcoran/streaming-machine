"""Did the body's move come in on the light's blank? Exact whole-piece model W = S + n*q (q = 2^-6).
Three runs over the same text: A = the model; B = the word table's real blank swapped for blank #1 (moves kept);
C = blank #1 swapped for blank #2. dReal = light(A) - light(B): the real blank as carried by the light to each plate.
dCtrl = light(B) - light(C): a same-sized swap of two blanks the model never had.
For each body plate with move X (in x out):
  lined-up ratio = [sum |d @ X|^2 / sum |d|^2] / [|X|^2 / in]   (1 = the light change is not along the move)
  pattern corr  = off-diagonal corr of the move's line overlaps X X^T with the light change's line overlaps d^T d."""
import numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
torch.set_num_threads(8)
q = 2.0 ** -6; T = 128
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=T, vocab_size=50257)
def blank(seed):
    torch.manual_seed(seed); return {n: p.detach().double() for n, p in GPT2LMHeadModel(cfg).named_parameters()}
S, B1, B2 = blank(0), blank(1), blank(2)
N = torch.load("pieces_n_6.pt")
W = {n: S[n] + N[n].double() * q for n in S}
tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = torch.tensor(tok(open("shakespeare.txt").read()).input_ids); tr = ids[:int(0.9 * len(ids))]
g = torch.Generator().manual_seed(7)
X0 = torch.stack([tr[i:i+T] for i in torch.randint(0, len(tr) - T, (32,), generator=g).tolist()])   # 4,096 places

m = GPT2LMHeadModel(cfg).double().eval()
plates = [n[:-len(".weight")] for n in W if W[n].ndim == 2 and ("attn.c_" in n or "mlp.c_" in n)]
mods = dict(m.named_modules())
def lights(wte):
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(W[n] if n != "transformer.wte.weight" else wte)
        got = {}
        hs = [mods[pl].register_forward_hook(lambda mod, inp, out, pl=pl: got.__setitem__(pl, inp[0].reshape(-1, inp[0].shape[-1]).clone()))
              for pl in plates]
        m(X0); [h.remove() for h in hs]
    return got
wte = W["transformer.wte.weight"]
LA = lights(wte); LB = lights(wte - S["transformer.wte.weight"] + B1["transformer.wte.weight"])
LC = lights(wte - S["transformer.wte.weight"] + B2["transformer.wte.weight"])

def offcorr(P, Q):
    i, j = np.triu_indices(P.shape[0], 1); return float(np.corrcoef(P[i, j], Q[i, j])[0, 1])
print(f"{'plate':24s} {'light change size':>17s} {'lined up: real':>15s} {'control':>8s} {'pattern: real':>14s} {'control':>8s}")
for pl in plates:
    Xm = (N[pl + ".weight"].double() * q).numpy()                  # the plate's move, (in, out)
    dR = (LA[pl] - LB[pl]).numpy(); dC = (LB[pl] - LC[pl]).numpy()
    base = np.sum(Xm * Xm) / Xm.shape[0]
    rat = lambda d: (np.sum((d @ Xm) ** 2) / np.sum(d * d)) / base
    size = np.linalg.norm(dR) / np.linalg.norm(LA[pl].numpy())
    print(f"{pl:24s} {size:17.3f} {rat(dR):15.3f} {rat(dC):8.3f} {offcorr(Xm @ Xm.T, dR.T @ dR):14.3f} {offcorr(Xm @ Xm.T, dC.T @ dC):8.3f}")
