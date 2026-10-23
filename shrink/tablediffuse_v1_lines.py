"""TABLEDIFFUSE v1 -- carry each value's leftover grain to the NEXT LINE OF THE SAME WORD, natural line order, first
order (classic error diffusion along a scan line; the chain starts fresh at every word).
The leftover is measured in fractions of each line's own step (whole numbers at 2^16): the next line repays it in its
own step. For the lines the lens governs (e_i = c - place(gain_i)) a step is the same top light on every line, so this
IS the carry in the top's units; the capped faint-gain lines (step 2^12) are not asked to repay more than ~a step.
Two other frames at c=32 only: carry in the top's units (2^place(gain)) -- the faint-gain lines then owe the top light
that only a huge change of E can repay; carry in E's own units -- the coarse capped lines dump 2^11-sized leftovers
onto lines exposed at 2^8.

The relation, before running: the developed value is v + (u_prev - u), u the bounded leftover. Its grain is the
difference of two neighbouring leftovers: the pile over a word's lines telescopes (cancels to < 1 step), and every
value's grain power doubles (|1 - z^-1|^2 averaged = 2): 1/12 -> 2/12 step^2. The cancellation lands in one direction
only: a word's white level (equal light on every line). The gauge removes exactly that direction from every read
(each gauged plate's column adds to 0 over its 768 lines; T's rows add to 0), so the entry never saw it; the top sees
the word's grain against x - mean(x), whose line-to-line pattern has nothing to do with the line order. So: no gain,
and the doubled grain acts like half a place coarser exposure.
Predicted: c=32 28-30, c=33 27-29, c=34 27-29, c=35 25-28, c=37 20-27 (against nearest + screen phases: c=32 29-30,
c=34 29-30, c=37 23-28). Stored size within +0.3% of nearest at c=32-34, +0.5-2% at c=37.
Frames at c=32: top units 20-28; E units 24-29.
    python3 shrink/tablediffuse_v1_lines.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v1 carry to next line, natural order, c={c}", c=c, how="lines") for c in (32, 33, 34, 35, 37)]
       + [dict(name="v1b carry to next line in the top's units, c=32", c=32, how="lines", unit="top"),
          dict(name="v1c carry to next line in E's units, c=32", c=32, how="lines", unit="entry")])
