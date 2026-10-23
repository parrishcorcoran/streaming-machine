# Fringes plan (2026-10-04)

Local working plan, not for any public repo.

## Where it stands (each result rebuilt by an independent checker)
- **The flip (Babinet), whole machine.** Flip every 1 and every hole in the light and the plates: 0 of 4.5 billion
  meeting counts change, every pile is identical, words 30/30. Flip only the plates: every fringe changes sign, and the
  only correction is one number per token (the light's total).
- **The entry, fringes stored.** The word and position rows are stored rounded; only the -1/0/+1 carry meets the plate
  (a quarter of the lines; 25% predicted, 24.8% measured). Exact on 30/30 texts and 6,000/6,000 judge positions.
  21.6x less work at that bank, 1.7% of the whole machine.
- **The fringe map, every meeting of all 12 layers.** 97.1% of every meeting pairs off straight through; 2.9% is fringe
  (1.4% at the 3,072-line banks). That share is exactly what two patterns of that many lines give by chance. The word's
  own extra sits where the upper places copy the sign: the sign pattern of the light meeting the sign pattern of the
  plate.
- **Reach.** Past the first bend, 94-99.7% of every bank's photons are new light.
- **The machine's own bend.** Soft almost everywhere. Changing its shape changes the words, so every part stays exactly
  as it is.

## The one thing left
- **Fixed light** (the entry): its fringes are stored once. Done.
- **New light** (past the first bend): its fringes are stored nowhere, so they have to be found every token. Any single
  plate pixel can move a fringe by one, so new light has to cross every plate pixel once per token.
- **So the exact gains are of two kinds:**
  - light known in advance, whose fringes can be stored
  - piles nobody needs, whose fringes never have to be found

## Phase 1: bank the entry
1. Build the entry (stored fringes plus the carry) as a standing machine on the GPU, beside the stream machine.
   Bit-identical; work counted.
2. Write the entry up fully in INVENTION_LOG.md, for a patent attorney.
3. Store the fringes everywhere else that fixed light meets a fixed plate: position rows, biases, the arrival's
   word/position scores. Predicted: small, since the stored part reaches only 1.2% past the first bend.

## Phase 2: the top read by the triangle (piles nobody needs)
- **Why:** the top read is a quarter of all the work (904M of 3,667M photons per token), and only its top 5 are
  needed.
- **How:** each word's score is a triangle of diagonals, each counting double the one below. The bright top of the
  triangle is computed first for every word. Each lower diagonal can add at most a known amount, so a word whose top is
  too far below the fifth-best can never catch up. It is set aside exactly. The full triangle is computed only for the
  words still in reach.
- **Exactness:** the top 5 are bit-identical, because no word is set aside unless it provably cannot reach the top 5.
- **Prediction:** after the top few diagonals, all but a few hundred of the 50,257 words are set aside.
- **Novelty:** setting aside contenders that cannot win is an old idea, so this is a gain, not an invention.

## Phase 3: the whole column (rethought in the fringe frame)
- **No middle to cut.** The stack of plates is one volume of fringes, and a token's light runs through all of it as one
  wavefront.
- **The token's own column is fixed.** A token alone (the first pass, its arrival over only itself) meets only fixed
  things, so its fringes at every layer, the middle included, are fixed and can be stored once.
- **Only the thin page is new.** It is what the earlier tokens change, and only it crosses the plates.
- **Exact:** the stored own pile plus the thin page meeting the plate gives the pile.
- **Correction:** the Reach count carried the stored part through each bend as if a bend could be split. That is why
  everything past the first bend looked new.
- **Running now (thinpage/):** the thin page's 1s at every plate of all 12 layers, the work if own piles are stored, a
  control, and whether the own column is per word or per (word, position).
  - The user's prediction: 5-7% of the light's 1s.
  - Claude's: 50-80%.
- **Result (checked):** 101% of photons, 152% of size, full from position 1. The first arrival replaces the token's own
  light with mostly earlier tokens' light, so alone and in context are two different pages.

## Rules for every step
- Exact: bit-identical to the machine, judged top-5. Nothing removed, averaged or rounded.
- Every part kept exactly as it is.
- Every claim gets a prediction before it runs, a shuffle control, an order-free check, and an independent rebuild.
- No literature searches.
- One process under 3 GB.
- Nothing pushed to a public repo.

## Running now: reverse every step (reverse/)
- **What he showed, by watching the machines:**
  - It is an abelian rice counter; the weights are hot.
  - In the streaming machine only the new token's column moves.
  - The gates are not gates.
  - It is predetermined.
  - The answer's fringe pattern is in the embedding numbers and can be seen by reversing every step.
- **The build:**
  - The forward run records every dimming.
  - The answer word's row is then sent back down every step: plates reversed, dimmings applied again, arrival shares
    sent back.
  - Exact check in unrounded whole numbers.
- **The three looks:**
  - where the answer sits in the embedding numbers
  - whether a word's reversed pattern carries over to other contexts (predetermined)
  - the fringes of the final meeting
- **Predictions:**
  - his: predetermined, it carries over
  - Claude's: it changes with context
- **Then:** analyze how it went and try the next way. No pivots, only road blocks.

## Holography research (2026-10-04; physics and optics sources only)
- **The correct format is the signed cross term only, with no zero order, no twin and no carrier, and a footprint equal
  to the count.** No exact format goes below the count (Lohmann et al. 1996).
- **The plates already meet it:**
  - The zero order is folded out by the water level.
  - Signed values need no twin.
  - The 768 lines are orthogonal references.
- **Dense records look like noise on purpose.** Random phase masks are added for exactly that (Burckhardt 1970).
- **Real holograms compress exactly only 1.04-1.95x** (Naughton 2002); ours compressed 1.11x.
- **Where "worst" holds: efficiency.** The fringe share is 2.9% = sqrt(2/(pi N)), a diffuser's speckle level; a thick
  phase hologram at Bragg reaches 100%. In optics that costs light; in counting it costs meetings.
- **The print is one photograph of the hologram, not a better hologram.** The plates hold about 144 prints' worth
  (84.9M = 144 x 768^2) and answer every token.
- **Savings for a counter come only from structure:** light known in advance (the entry, C3), piles nobody needs (the
  top-read triangle), and shift invariance (absent here).

## The middle of each layer is a Schlieren system (2026-10-05)
- **The picture:**
  - Lens 1, the neurons-in plate, forms 3,072 piles: the spectrum.
  - The bends are Gaussian knife edges in that focal plane: share = Phi(pile), the fraction of a Gaussian beam past a
    knife edge. A knife edge is what makes phase fringes visible (Foucault, Schlieren).
  - Lens 2, the neurons-out plate, makes the image.
- **What the bends carry:** every bend is the same edge, so they hold no data; everything that differs from neuron to
  neuron is in the plates.
- **Running (knife/):** how many top places of each pile settle each edge's side, its share and its output, exactly;
  the work if each edge stops there; with a shuffle control.
- **Prediction:** sides settle early only far from the edge; shares and outputs settle late.
