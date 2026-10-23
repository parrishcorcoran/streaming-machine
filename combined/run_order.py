"""ORDER-FREE CHECK (refs texts 0, 14 and 29). Every bank's lines are put in a random order first: plate rows, and the
light with them at run time. The line-ups are then built on that order, and every pile's parts (stored tips and what
meets the plate) are added in a random order. Must be identical to the machine's order:
  - every pile and value of all 12 layers, the stream and the light at the top (against the plain machine);
  - every count (against the same texts in combined/refs_counts.json from run_refs.py).
Ties between equal weights can change places; their gap is 0, so nothing they meet changes.
(The top read's order-free check was done with collapse_top/order_check.py.)
    python3 combined/run_order.py
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
def log(s): print(s, flush=True)
import machine as K
SAVED = json.load(open(f"{K.HERE}/combined/refs_counts.json"))["per_text"]
K.build_lineups(perm_seed=77)
for t in (0, 14, 29):
    ids = K.TEXTS[t]; ref = K.plain_run(ids, keep_top=False); rec = {}; C = K.Count()
    K.forward(ids, C, rec, parts_rng=np.random.default_rng(500 + t))
    bad = K.same_as_plain(rec, ref); saved = SAVED[t]
    diff = sorted(k for k in set(C) | set(saved) if C.get(k) != saved.get(k))
    log(f"  text {t}: every pile and value identical: {not bad}{'' if not bad else ' -- ' + ', '.join(bad[:6])}; "
        f"counts identical to the machine's order: {len(C) - len([k for k in diff if k in C])}/{len(saved)} "
        f"{'(all)' if not diff else '-- differ: ' + '; '.join(f'{k}: {saved.get(k)} vs {C.get(k)}' for k in diff[:6])}")
log(f"({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
