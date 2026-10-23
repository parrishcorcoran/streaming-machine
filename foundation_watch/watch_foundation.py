"""WATCH THE FOUNDATION. The streaming machine (basics/code/stream_sentence.py's machine: all 12 blocks live every turn,
block k reads what block k-1 emitted last turn, values between blocks on a dial, nothing ever cleared).
  1. The weights alone: one position with NO light at all (the chord is zero). Watch it stand. That is the foundation.
  2. Strike one token onto the standing foundation and watch: per turn, notes moved; once it stands again, per depth,
     how many notes are still exactly where the foundation had them.
  3. The same for a few different tokens: what each changes, and whether they change the same notes.
    python3 watch_foundation.py"""
import sys, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
tok = GPT2Tokenizer.from_pretrained("gpt2"); m = GPT2LMHeadModel.from_pretrained("gpt2").float().eval()
tr, blocks = m.transformer, list(m.transformer.h); L = len(blocks)
WH, H = 2 ** 23 - 1, 4096 ** 2 // 2 - 1
for blk in blocks:
    for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
        W = mod.weight.data; gq = W.abs().amax(0).clamp(min=1e-12) / WH; mod.weight.data = torch.round(W / gq) * gq
def dial(v): s = v.abs().amax(-1, keepdim=True).clamp(min=1e-30) / H; return torch.round(v / s), s
def stand(chord, emitted, prev, show=True, label=""):
    turn = 0
    while True:
        turn += 1; new = [chord] + [None] * L
        for k in range(1, L + 1):
            out = blocks[k - 1](emitted[k - 1]); new[k] = out[0] if isinstance(out, tuple) else out
        moved = []
        for k in range(L + 1):
            q, s = dial(new[k]); moved.append(768 if prev[k] is None else int((q != prev[k]).sum())); prev[k] = q; new[k] = q * s
        emitted = new
        res = tr.ln_f(emitted[L])[0, -1] @ tr.wte.weight.T; top = res.argmax().item()
        if show: print(f"  {label}{turn:3d} | " + " ".join(f"{f:4d}" for f in moved) + f" | {tok.decode([top])!r}")
        if sum(moved) == 0 or turn > 40: return emitted, prev, res
zero = torch.zeros(1, 1, 768)
print("1. THE WEIGHTS ALONE: one position, no light at all (word row 0, position row 0)")
print("  turn | notes moved at depth 0..12 (of 768 each)                                  | the weights alone say")
F, Fq, Fres = stand(zero, [zero.clone() for _ in range(L + 1)], [None] * (L + 1))
print("  the weights' own five:", [tok.decode([i]) for i in Fres.topk(5).indices.tolist()])
print("  the foundation's size at each depth (largest |value|):", " ".join(f"{float(F[k].abs().max()):.2f}" for k in range(L + 1)))
cols = {}
for word in [" France", " the", " cat", " quantum", "."]:
    i = tok.encode(word)[0]; chord = (tr.wte.weight[i] + tr.wpe.weight[0])[None, None]
    print(f"\n2. STRIKE {word!r} ONTO THE STANDING FOUNDATION (position 0)")
    E, Eq, res = stand(chord, [f.clone() for f in F], [q.clone() for q in Fq], show=(word == " France"), label="")
    same = [int((Eq[k] == Fq[k]).sum()) for k in range(L + 1)]
    rel = [float((E[k] - F[k]).norm() / E[k].norm()) for k in range(L + 1)]
    print(f"  stands. notes still exactly where the foundation had them, depth 0..12: " + " ".join(f"{s:3d}" for s in same))
    print(f"  size of (token column - foundation) / token column, depth 0..12:      " + " ".join(f"{r:.2f}" for r in rel))
    print(f"  says: {[tok.decode([j]) for j in res.topk(5).indices.tolist()]}")
    cols[word] = E
print("\n3. WHAT DIFFERENT TOKENS CHANGE: (column A - foundation) against (column B - foundation), at depths 1, 6, 12")
ws = list(cols)
for d in (1, 6, 12):
    row = []
    for a in range(len(ws)):
        for b in range(a + 1, len(ws)):
            da, db = (cols[ws[a]][d] - F[d]).flatten(), (cols[ws[b]][d] - F[d]).flatten()
            row.append(f"{ws[a].strip()}/{ws[b].strip()} same sign {float(((da > 0) == (db > 0)).float().mean()) * 100:.0f}%")
    print(f"  depth {d:2d}: " + "  ".join(row[:5])); print("            " + "  ".join(row[5:]))
