"""Depth 12 is depth 0: feed the top straight back in at the bottom instead of collapsing to one word and expanding
its row. normal: the word's row | state: the top's state (gauged, sized like a row) | waveform: sum of rows weighted by chance.
The word is still read at each step, only to show the text. fp32 GPT-2.
    python3 loop.py
"""
import json, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); E = m.transformer.wte.weight; refs = json.load(open("refs_many.json"))
rowsize = E.norm(dim=-1).mean()
for mode in ["normal", "waveform", "state"]:
    right = ident = 0
    for r in refs:
        emb = E[torch.tensor(r["ids"])]; got = []
        for k in range(16):
            out = m(inputs_embeds=emb[None], output_hidden_states=True)
            lg = out.logits[0, -1]; w = int(lg.argmax()); got.append(w)
            if mode == "normal": nxt = E[w]
            elif mode == "waveform": nxt = torch.softmax(lg, -1) @ E
            else: h = m.transformer.ln_f(out.hidden_states[-1][0, -1]); nxt = h / h.norm() * rowsize
            emb = torch.cat([emb, nxt[None]], 0)
        want = r["ref"][:16]; k = next((i for i in range(16) if got[i] != want[i]), 16); right += k; ident += k == 16
    print(f"  {mode:9s}: texts identical {ident}/30, words right before the first wrong one {right}/480")
