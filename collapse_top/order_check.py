"""ORDER-FREE CHECK. staged_top.py run with order=1 (the words and the lines, light and stamp together, put in another
order before anything) against the same run in the machine's order: survivors at every stage, words entering every
stage, work at every stage, the plain work, the saved final words and their exact scores (mapped back to the machine's
word numbers), the top 5 and the race winner. Where a top 5 differs, the scores there are printed (a tie between two
words is put in order by where the words sit, so it can change with the order; nothing else may).
    python3 collapse_top/order_check.py
"""
import numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
for sc in ("stamp", "light", "diag"):
    a, b = np.load(f"{D}/staged_{sc}_real.npz"), np.load(f"{D}/staged_{sc}_real_order.npz")
    same = {k: bool(np.array_equal(a[k], b[k])) for k in ("sel", "surv", "enter", "work", "plain", "win_st", "win_pl", "fin_off", "bad")}
    fin_same = bool(np.array_equal(a["fin_sc"], b["fin_sc"]))
    fin_ids_same = bool(np.array_equal(a["fin_idx"], b["fin_idx"]))
    d = np.flatnonzero(~(a["top_st"] == b["top_st"]).all(1))
    print(f"{sc}: equal in both orders: " + ", ".join(f"{k} {v}" for k, v in same.items()) +
          f", final words {fin_ids_same}, their exact scores {fin_same}; top 5 differs at {len(d)} of {len(a['top_st'])} positions")
    off = a["fin_off"]
    for q in d:
        ids, sc_ = a["fin_idx"][off[q]:off[q + 1]], a["fin_sc"][off[q]:off[q + 1]]
        score = dict(zip(ids.tolist(), sc_.tolist()))
        print(f"   position {q}: machine order {a['top_st'][q].tolist()} scores {[score[w] for w in a['top_st'][q]]}")
        print(f"   {' ' * len(str(q))}           other order   {b['top_st'][q].tolist()} scores {[score[w] for w in b['top_st'][q]]}"
              f"   (plain read, machine order: {a['top_pl'][q].tolist()})")
