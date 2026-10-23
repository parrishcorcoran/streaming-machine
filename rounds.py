"""All depths at once, in rounds. The new piece's gains (gauges, arrival weights, bend shares) are guessed all at once
(from the previous piece); the light goes through the whole depth as ONE collapsed operation with those gains frozen;
the gains it actually produces are read off; repeat until nothing changes. Count rounds. fp32 GPT-2, earlier pieces settled.
    python3 rounds.py
"""
import json, math, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); refs = json.load(open("refs_many.json")); sd = {k.replace("transformer.", ""): v for k, v in m.state_dict().items()}
def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def run(ids, frozen=None):
    """frozen: the new (last) piece's gains per depth; None = the real network. Returns its gains, its top read."""
    P = len(ids); x = sd["wte.weight"][ids] + sd["wpe.weight"][:P]; got = []
    for l in range(12):
        g = lambda k: sd[f"h.{l}.{k}"]
        c = x - x.mean(-1, keepdim=True); s1 = (c.var(-1, unbiased=False) + 1e-5).rsqrt()
        if frozen: s1 = torch.cat([s1[:-1], frozen[l]["s1"][None]])
        h = (c * s1[:, None]) * g("ln_1.weight") + g("ln_1.bias")
        q, k, v = (h @ g("attn.c_attn.weight") + g("attn.c_attn.bias")).split(768, -1)
        q, k, v = [t.view(P, 12, 64).transpose(0, 1) for t in (q, k, v)]
        a = torch.softmax((q @ k.transpose(1, 2) / 8).masked_fill(torch.triu(torch.ones(P, P, dtype=torch.bool), 1), -1e30), -1)
        alast = a[:, -1, :].clone()
        if frozen: a = a.clone(); a[:, -1, :] = frozen[l]["a"]
        x = x + (a @ v).transpose(0, 1).reshape(P, 768) @ g("attn.c_proj.weight") + g("attn.c_proj.bias")
        c = x - x.mean(-1, keepdim=True); s2 = (c.var(-1, unbiased=False) + 1e-5).rsqrt()
        if frozen: s2 = torch.cat([s2[:-1], frozen[l]["s2"][None]])
        f = ((c * s2[:, None]) * g("ln_2.weight") + g("ln_2.bias")) @ g("mlp.c_fc.weight") + g("mlp.c_fc.bias")
        share = gelu(f) / torch.where(f != 0, f, torch.ones_like(f))
        if frozen: share = torch.cat([share[:-1], frozen[l]["share"][None]])
        x = x + (f * share) @ g("mlp.c_proj.weight") + g("mlp.c_proj.bias")
        got.append(dict(s1=s1[-1], a=alast, s2=s2[-1], share=gelu(f[-1]) / torch.where(f[-1] != 0, f[-1], torch.ones_like(f[-1]))))
    c = x - x.mean(-1, keepdim=True); sf = (c.var(-1, unbiased=False) + 1e-5).rsqrt()
    word = int((((c * sf[:, None]) * sd["ln_f.weight"] + sd["ln_f.bias"])[-1] @ sd["wte.weight"].T).argmax())
    return got, word
def diff(G, H): return max(max((G[l][k] - H[l][k]).abs().max().item() for k in G[l]) for l in range(12))
word_round, settle_round = [], []
for r in refs:
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"])
    for t in range(S, len(ids)):                       # the piece at position t is the new one
        seq = ids[:t + 1]; real, true_word = run(seq)
        G, _ = run(ids[:t])                            # guess: the previous piece's gains, all at once
        G = [dict(s1=G[l]["s1"], a=torch.cat([G[l]["a"], torch.zeros(12, 1)], 1), s2=G[l]["s2"], share=G[l]["share"]) for l in range(12)]
        wr = None; sr = None
        for rnd in range(1, 41):
            newG, w = run(seq, frozen=G)               # one all-at-once round through the whole depth
            if w == true_word and wr is None: wr = rnd
            if w != true_word: wr = None
            if diff(newG, real) < 1e-5: sr = rnd; break
            G = newG
        word_round.append(wr or 41); settle_round.append(sr or 41)
wr, sr = torch.tensor(word_round).float(), torch.tensor(settle_round).float()
print(f"{len(wr)} new pieces, each starting from the previous piece's gains")
print(f"  rounds until the word is right and stays right: median {int(wr.median())}, mean {wr.mean():.1f}, worst {int(wr.max())}")
print(f"  rounds until every gain has settled (to 1e-5): median {int(sr.median())}, mean {sr.mean():.1f}, worst {int(sr.max())}   (41 = not settled in 40)")
for k in (1, 2, 3, 4, 6, 8, 12): print(f"  word right by round {k:2d}: {100 * (wr <= k).float().mean():5.1f}%")
