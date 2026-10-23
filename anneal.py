"""All sites at once, annealed toward the ground state of the joint energy
    H(w1..wN) = sum_t E_t,  E_t = -log( e^{-E_w} / Z ) at site t given the sites before it.
Every wave: one pass gives every site's energies for the text it holds now; every site redraws its word from
e^{-E/T}/Z at the current temperature, all at once. T falls from 1 to 0 over HOT waves, then COLD waves at zero.
Keeps the lowest joint energy seen. Compared with site-by-site (each site takes its lowest-energy word in turn).
    python3 anneal.py
"""
import json, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False); torch.manual_seed(0)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); refs = json.load(open("refs_many.json"))
N, HOT, COLD = 16, 40, 16
def energies(seq, S):                     # -log probability of every word, at each of the N sites
    lp = torch.log_softmax(m(torch.tensor([seq])).logits[0], -1)
    return -lp[S - 1:S - 1 + N]           # site t's word is read at the position before it
def H(E, words): return float(E[torch.arange(N), torch.tensor(words)].sum())
lower = same = 0; dH = []; firsts = []
for r in refs:
    ids, greedy = r["ids"], r["ref"][:N]; S = len(ids)
    Hg = H(energies(ids + greedy, S), greedy)
    cur = [ids[-1]] * N; best = (1e9, None, 0)
    for wave in range(HOT + COLD):
        E = energies(ids + cur, S)
        h = H(E, cur)
        if h < best[0] - 1e-6: best = (h, list(cur), wave)
        T = max(0.0, 1.0 - wave / HOT)
        cur = (torch.distributions.Categorical(logits=-E / T).sample() if T > 0 else E.argmin(-1)).tolist()
    E = energies(ids + cur, S); h = H(E, cur)
    if h < best[0] - 1e-6: best = (h, list(cur), HOT + COLD)
    d = best[0] - Hg; dH.append(d); firsts.append(best[2])
    if d < -1e-4: lower += 1
    if best[1] == greedy: same += 1
    print(f"  {r['prompt'][:34]:34s} site-by-site H {Hg:7.2f}   annealed best H {best[0]:7.2f} ({d:+6.2f}) at wave {best[2]:2d}   {'same text' if best[1] == greedy else 'different text'}")
print(f"lower joint energy than site-by-site: {lower}/30; same text: {same}/30; mean change {sum(dH)/30:+.2f}; best found at wave {sorted(firsts)[15]} (median) of {HOT + COLD}")
