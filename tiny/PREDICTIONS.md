# Predictions, written before the run (2026-10-08)

Tiny model, seed 0, 128 wide, 4 layers, word table shared in and out; taught on Shakespeare, AdamW 1e-3, no decay, fp32.

1. Made again from the seed: bit for bit the same. Costs nothing to store.
2. Blank: held-out loss about ln 50257 = 10.82; next word right about 0%.
3. Taught (3,000 steps): held-out loss about 5 to 6.
4. Remove: S + X rebuilds every weight with 0 wrong.
5. X exactly 0: close to 0% of weights. Every weight gets moved every step (the shared word table gets pushed by the
   output side for every word, seen or not).
6. Exact bits a weight: W about 26 to 28; X about 20 to 24. Taking the blank out saves only the top few places of
   each weight, because the low places of every taught weight are the leftover of thousands of fp32 steps, and exact
   keeps all of them. Under 1 bit a weight: NOT reached this way.

# Whole pieces (pieces.py), written before the run (2026-10-08)

Same as the fp32 run, but every weight = blank + n * q exactly, n whole, one q for the whole model.
- Every step exact (two-sum 0), blank + n*q rebuilds every weight.
- q = 2^-10: learns about like fp32 (loss about 4.8-5.0); n exactly 0 about 1-3%; about 8-9 bits a weight.
- q = 2^-8: loss about 4.9-5.1; n exactly 0 about 5%; about 6-7 bits a weight.
- q = 2^-6: loss about 5.0-5.5; n exactly 0 about 10-20%; about 4-5 bits a weight.
- Under 1 bit a weight: not at these pieces. AdamW pushes every weight a similar amount every step, so nearly every
  weight moves; under 1 bit needs most weights never to move.

# Bigger pieces, written before the run (2026-10-09)
- q = 2^-4: exact, 0 wrong; about 2.8-3.2 bits a weight; never moved about 8-15%; score about the same (loss 4.8-4.9).
- q = 2^-2: exact, 0 wrong; about 0.8-1.5 bits a weight; never moved about 40-70% (a weight needs a steady push of 0.125
  to move once); score worse (loss about 5.0-6.0).

# Mixed pieces: body 2^-2, word/place tables 2^-6, written before the run (2026-10-09)
- exact, 0 wrong.
- body about 0.8 bits a weight (about 85% never moved), tables about 4.9 bits.
- score back near 2^-6's: about 27.5-28.5% right, loss about 4.82-4.88. If so the slip at 2^-2 came from the word table.

# Hold: piece 2^-6, a weight moves only when its push reaches 8 pieces (0.125), written before the run (2026-10-09)
- exact, 0 wrong.
- never moved about like the 2^-2 run: whole model about 30-40%, body about 80-88%.
- bits: body about 1.2-1.6 (moved weights land finely, so each costs more than at 2^-2), whole model about 2.5-3.
- score: between 2^-2 (24.5%) and 2^-6 (28.4%), about 26%. Moving late hurts as well as landing coarse.

# Take the free parts out of the exact whole-piece model (2^-6), written before the run (2026-10-09)
Free: the blank (seed). Stored once: c per plate (the blank's growth (c-1)*S), the word table's shared shift for
words never in the text and for words in it (2 x 128 numbers), 1 flag per word row (in the text or not).
What is left: whole numbers r = n - round(((c-1)*S + shift)/q); rebuild n = that + r, exact.
- rebuilt wrong 0.
- word table: 4.95 -> about 1.5-2.0 bits a weight (the never-seen rows' leftovers almost all 0).
- body: 3.83 -> about 3.5-3.7 (the blank's growth is a small part of the move).
- whole model: 4.82 -> about 2.0-2.8.
- control (another blank, seed 1, in place of the real one): no saving from the growth part.

# The body's move against the blank carried in on the light, written before the run (2026-10-09)
Model: whole pieces 2^-6. Light change from swapping the word table's real blank for another (real), and from
swapping two foreign blanks (control). Lined-up ratio: how strongly each plate's move picks up that light change,
against an average direction (1 = not lined up). Also the overlap pattern of the move's lines against the light change's.
- layer 0 plates: real ratio about 1.2-2, control about 1.0; pattern corr real clearly above control.
- deeper layers: weaker, real nearer control.
