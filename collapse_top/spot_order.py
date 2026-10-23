"""ORDER-FREE CHECK of the spot bound. spot_top.py run with order=1 (the words and the lines, with the light, the word
table and the lines' groups together, put in another order before anything) on the control positions, against the
main run in the machine's order on the same positions: survivors and words entering at every stage, meetings, every
part of the bound's own work, words read to the end, their divides, the race winner, the words kept (mapped back to
the machine's word numbers) and their exact scores. Where a top 5 differs, the scores there are printed (a tie between
two words is put in order by where the words sit, so it can change with the order; nothing else may).
    python3 collapse_top/spot_order.py [write=signed]
"""
import sys, numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
ARG = dict(a.split("=") for a in sys.argv[1:]); write = ARG.get("write", "signed")
a, b = np.load(f"{D}/spot_{write}_real_s1.npz"), np.load(f"{D}/spot_{write}_real_s1_order_sub.npz")
idx = b["sel"]; print(f"{write}: {len(idx)} positions, both orders")
for k in ("plain916", "plainw", "pldiv", "plst", "lightph", "win_pl"):
    print(f"  {k}: equal {np.array_equal(a[k][idx], b[k])}")
print("  checks (order run): " + ", ".join(f"{k} {v}" for k, v in zip(b["bad_keys"].tolist(), b["bad"].tolist())))
def kept(Z, G, s, q): return Z[f"kept_{G}_{s}"][Z[f"koff_{G}_{s}"][q]:Z[f"koff_{G}_{s}"][q + 1]]
def score(Z, q, ids):
    u = Z["uids"][Z["uoff"][q]:Z["uoff"][q + 1]]; sc = dict(zip(u.tolist(), Z["usc"][Z["uoff"][q]:Z["uoff"][q + 1]].tolist()))
    return [sc[w] for w in ids.tolist()]
for G in b["gs"]:
    for s in ("diag", "light"):
        G = int(G); same = {k: bool(np.array_equal(a[f"{k}_{G}_{s}"][idx], b[f"{k}_{G}_{s}"]))
                            for k in ("surv", "enter", "meet", "parts", "bad", "final", "findiv", "finst", "win")}
        kk = all(np.array_equal(kept(a, G, s, int(q)), kept(b, G, s, j)) for j, q in enumerate(idx))
        ks = all(score(a, int(q), kept(a, G, s, int(q))) == score(b, j, kept(b, G, s, j)) for j, q in enumerate(idx))
        d = np.flatnonzero(~(a[f"top_{G}_{s}"][idx] == b[f"top_{G}_{s}"]).all(1))
        print(f"  G={G:2d} {s:5s}: equal in both orders: " + ", ".join(f"{k} {v}" for k, v in same.items()) +
              f", words kept {kk}, their exact scores {ks}; top 5 differs at {len(d)} of {len(idx)} positions")
        for j in d:
            q = int(idx[j]); t1, t2 = a[f"top_{G}_{s}"][q], b[f"top_{G}_{s}"][j]
            print(f"     position {q}: machine order {t1.tolist()} scores {score(a, q, t1)}; other order {t2.tolist()} scores {score(b, j, t2)}")
