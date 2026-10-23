"""The top read in 16,384 RANDOM groups, made once: each group a centre (its words' average row) and a radius (its
farthest word from the centre). Per piece: read the 16,384 centres; the most any word in a group can score is
centre score + size of the light x radius; read the best-looking group exactly for a real score; drop every group
that can't reach it; read the survivors' words exactly. Same winner, guaranteed.
    python3 head_groups.py
"""
import json, torch
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False); torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval(); refs = json.load(open("refs_qwen.json"))
E = m.lm_head.weight.double(); Vn, G = E.shape[0], 16384
lab = torch.randperm(Vn) % G                                                           # random groups, made once
C = torch.zeros(G, E.shape[1], dtype=torch.float64).index_add_(0, lab, E) / torch.bincount(lab, minlength=G).double()[:, None]
rad = torch.zeros(G, dtype=torch.float64).scatter_reduce_(0, lab, (E - C[lab]).norm(dim=1), "amax")
order = torch.argsort(lab); starts = torch.searchsorted(lab[order], torch.arange(G + 1))
members = lambda g: order[starts[g]:starts[g + 1]]
rows, same, alive_n = [], 0, []
for r in refs:
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"])
    X = m(torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][0, S - 1:-1].double()   # the top state, final gauge applied
    for x in X:
        cs = C @ x; up = cs + x.norm() * rad + 1e-6 * cs.abs().max()
        g0 = int(cs.argmax()); best = (E[members(g0)] @ x).max()
        alive = torch.nonzero(up >= best).flatten()
        cand = torch.cat([members(int(g)) for g in alive]); win = int(cand[int((E[cand] @ x).argmax())])
        same += win == int((E @ x).argmax()); rows.append(G + len(members(g0)) + len(cand)); alive_n.append(len(alive))
rows = torch.tensor(rows).double(); alive_n = torch.tensor(alive_n).double()
print(f"random groups: groups that survive per piece median {int(alive_n.median()):,} of {G:,}")
print(f"rows read per piece: median {int(rows.median()):,}, mean {int(rows.mean()):,}  (full read: {Vn:,})   same winner: {same}/{len(rows)}")
