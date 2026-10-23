"""COMBINED v2 -- the tables' exposure moved with the plates held at round one's v12 A=8 (the whole machine).
v1 negative, read: every one-kind coarsening of the plates by a whole place lost 1-2 words, always among the same
texts (12, 23, then 7, 15, 16). combined_look_margins.py: text 12's right word leads the next by 0.0074 of a unit at
the race, text 16 by 0.035, text 7 by 0.084; the other 27 lead by 0.15 or more. So 30/30 is decided by text 12 alone:
any grain at the race of more than ~0.01 makes it a coin toss. 30/30 is a coin, not a level; the levels are 29 and 28.
The tables are the cheaper lever: one place of c is 1 bit on each of 38.6M values (~4.8 MB), and c stops at 36 because
the entry's cap (no line coarser than 2^12) holds most lines from there on; past it, only a coarser cap moves.
Picture for the cap: E is read twice -- at the entry (5 words a text, every line, its own light) and at the top through
the lens (all 50,257 words race). The cap is the entry's grain: 2^12 at 2^16 is a 1/16 step on values of ~0.1-0.3.
Predictions before running (plates v12 A=8 throughout):
  c=35: 29-30   c=36: 29   c=37 (cap 12): 28-29   c=37 cap 13: 27-28   c=38 cap 14: 24-26"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
RUNS = [(35, 12, "29-30"), (36, 12, "29"), (37, 12, "28-29"), (37, 13, "27-28"), (38, 14, "24-26")]
sel = [int(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(len(RUNS))
for i in sel:
    c, cap, pred = RUNS[i]
    judge(f"v2 plates v12 A=8 + tables c={c} cap {cap}", cfg_all("v12", 8), c, cap,
          note="tables moved, plates held", predicted=pred)
