"""ORDER-FREE CHECK of the fine-grid staged top read. fine_staged_top.py run with order=1 (the words and the lines, light
and stamp together, put in one shared random order before anything) against the same run in the machine's order, the
diagonal schedule on real light, both bound rows: survivors and words entering at every stage, meetings at every stage,
the bound's photons at every stage, the plain work, the saved final words and their exact scores (mapped back to the
machine's word numbers), the check counts, the race winners and the top 5. Where a top 5 differs, the scores there are
printed (two words with the same score are put in order by where the words sit, so a tie can change with the order;
nothing else may).
    python3 collapse_top/fine_order_check.py
"""
import numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
a, b = np.load(f"{D}/fine_diag_real.npz"), np.load(f"{D}/fine_diag_real_order.npz")
print("fine grid, diagonal schedule, real light: the machine's order against one shared random order of words and lines")
for k in ("sel", "ks", "plain", "win_pl", "bad"):
    print(f"  {k}: equal {bool(np.array_equal(a[k], b[k]))}")
for row, sx in (("plain bound", ""), ("signed bound", "_s")):
    same = {k: bool(np.array_equal(a[k + sx], b[k + sx])) for k in ("surv", "enter", "work", "bw", "finw", "win_st", "fin_off", "fin_idx", "fin_sc")}
    d = np.flatnonzero(~(a["top_st" + sx] == b["top_st" + sx]).all(1))
    dpl = np.flatnonzero(~(a["top_pl"] == b["top_pl"]).all(1))
    print(f"  [{row}] equal in both orders: " + ", ".join(f"{k} {v}" for k, v in same.items()) +
          f"; top 5 differs at {len(d)} of {len(a['sel'])} positions (the plain read's own top 5 differs at {len(dpl)})")
    off = a["fin_off" + sx]
    for q in d:
        ids, sc = a["fin_idx" + sx][off[q]:off[q + 1]], a["fin_sc" + sx][off[q]:off[q + 1]]
        score = dict(zip(ids.tolist(), sc.tolist()))
        print(f"    position {q}: machine order {a['top_st' + sx][q].tolist()} scores {[score[w] for w in a['top_st' + sx][q]]}")
        print(f"    {' ' * len(str(q))}           other order   {b['top_st' + sx][q].tolist()} scores {[score[w] for w in b['top_st' + sx][q]]}"
              f"   (plain read: machine order {a['top_pl'][q].tolist()}, other order {b['top_pl'][q].tolist()})")
