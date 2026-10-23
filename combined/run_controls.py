"""CONTROLS for the line-up's count (6 refs texts: 0, 5, 10, 15, 20, 25; every position; every bank).
  A. LIGHT SHUFFLED: at every line-up, each position's light is shuffled across its lines (every value kept, only which
     line it meets changes). The running sums and meetings are counted again. The machine itself runs on the real
     light.
  B. NO RANKING: each neuron's lines are put in a random order of their own instead of in order of size. The gaps are
     then differences between random neighbours. Abel's sum is exact in any order, so the piles must still be identical;
     only the lit pixels change.
Predicted (written before the run): A within a few percent of the real light's count, because the gaps are the plate's
own and the running sums' size is set by the light's values. B far above plain (gaps as wide as the weights, plus the
running sums), which would show that the saving is the ranking.
    python3 combined/run_controls.py
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
def log(s): print(s, flush=True)
import machine as K
K.ALT_COUNT = True
SEL = [0, 5, 10, 15, 20, 25]
res = {}
for how in ("rank", "random"):
    K.build_lineups(how); C = K.Count(); bad = 0
    for t in SEL:
        ids = K.TEXTS[t]; ref = K.plain_run(ids, keep_top=False); rec = {}
        K.forward(ids, C, rec, side=False, shuffled=np.random.default_rng(t) if how == "rank" else None)
        bad += bool(K.same_as_plain(rec, ref)); del ref, rec
    res[how] = C
    log(f"{'ranked by size' if how == 'rank' else 'CONTROL B, random order per neuron'}: every pile and value identical to "
        f"the plain machine on {len(SEL) - bad}/{len(SEL)} texts ({time.time() - t0:.0f}s, {K.rss()} MB)")
R, Q = res["rank"], res["random"]
log(f"\nLIT-PIXEL MEETINGS of the line-ups, as a share of the plain meetings of the same light ({len(SEL)} texts, every position)")
log(f"  {'bank':8s} {'plain':>16s} {'line-up':>16s} {'A: light shuffled':>18s} {'B: no ranking':>16s}")
for kind in ("qkv", "proj", "fc", "mproj"):                  # the 45 banks (the entry's leftovers meet less than full light)
    part = "banks"
    pl = R[f"plain|{part}|{kind} meetings, plain"]; lu = R[f"comb|{part}|{kind} meetings, line-up"]
    sh = R[f"ctl|{part}|{kind} meetings, line-up, light shuffled across its lines"]; rn = Q[f"comb|{part}|{kind} meetings, line-up"]
    log(f"  {kind:8s} {pl:>16,} {K.pct(lu, pl):>15}% {K.pct(sh, pl):>17}% {K.pct(rn, pl):>15}%")
for k in ("comb|extra|line-up: running sums written in single waves", "comb|extra|line-up: light poured into each neuron's running sum"):
    log(f"  {k.split('|')[2]}: ranked {R[k]:,}, no ranking {Q[k]:,}")
pl = sum(v for k, v in R.items() if k.startswith("plain|banks|")); lu = sum(v for k, v in R.items() if k.startswith("comb|banks|"))
al = sum(v for k, v in R.items() if k.startswith("alt|banks|"))
wr = R["comb|extra|line-up: running sums written in single waves"]; po = R["comb|extra|line-up: light poured into each neuron's running sum"]
log(f"\nTHE RUNNING SUMS' WRITING (the 45 banks; writing and pours here include proj.0's and fc.0's line-ups, as in run_refs.py):")
log(f"  gaps in single waves x running sums in single waves: meetings {K.pct(lu, pl)}% of plain, + writing the running sums "
    f"{K.pct(wr, pl)}%, + pours {K.pct(po, pl)}% = {K.pct(lu + wr + po, pl)}%")
log(f"  gaps in single waves x running sums as they are:     meetings {K.pct(al, pl)}% of plain, + pours {K.pct(po, pl)}% = {K.pct(al + po, pl)}%")
log(f"({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
