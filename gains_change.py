"""What the collapsed plate is made of, piece to piece: every bend's share (GELU(x)/x) and every gauge (1/sigma),
at every depth, for each piece against the piece before it, and against a far piece (control). fp32 GPT-2.
    python3 gains_change.py
"""
import json, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); refs = json.load(open("refs_many.json"))
cap = {}
for k, blk in enumerate(m.transformer.h):
    blk.mlp.act.register_forward_hook(lambda mod, i, o, k=k: cap.__setitem__(("bend", k), (o / torch.where(i[0] != 0, i[0], torch.ones_like(i[0])))[0]))
    for nm in ("ln_1", "ln_2"):
        getattr(blk, nm).register_forward_hook(lambda mod, i, o, key=(nm, k): cap.__setitem__(key, (i[0].var(-1, unbiased=False) + mod.eps).rsqrt()[0]))
nb, fb, ng, fg = [], [], [], []
for r in refs:
    ids = r["ids"] + r["ref"][:16]; m(torch.tensor([ids])); S = len(r["ids"]); P = len(ids)
    for t in range(S, P - 1):
        far = 1 if t > 6 else P - 1
        for k in range(12):
            b = cap[("bend", k)]; nb.append((b[t + 1] - b[t]).abs()); fb.append((b[far] - b[t]).abs())
            for nm in ("ln_1", "ln_2"):
                gg = cap[(nm, k)]; ng.append(((gg[t + 1] - gg[t]).abs() / gg[t]).item()); fg.append(((gg[far] - gg[t]).abs() / gg[t]).item())
nb, fb = torch.cat(nb), torch.cat(fb); ng, fg = torch.tensor(ng), torch.tensor(fg)
for th in (0.001, 0.01, 0.1):
    print(f"bend shares that change by more than {th:5}: next piece {100 * (nb > th).float().mean():5.1f}%   far piece {100 * (fb > th).float().mean():5.1f}%")
print(f"bend shares, typical change: next piece {nb.median():.3f}   far piece {fb.median():.3f}   (a share runs 0 to 1)")
print(f"gauges, typical relative change: next piece {100 * ng.median():.1f}%   far piece {100 * fg.median():.1f}%")
print(f"gauges that change by more than 1%: next piece {100 * (ng > 0.01).float().mean():.0f}%   far piece {100 * (fg > 0.01).float().mean():.0f}%")
