# Recording plan: record only what's necessary (2026-10-04)

Local working plan. Nothing here goes to a public repo.

## The idea (his words, and what holography adds)
- About 97% of every meeting pairs off and contributes exactly nothing; the pile is made only of the fringes.
- The 50% exact cut proved the format matters. Signed digits and the line-up left fewer lit pixels, the same
  information, and every pile identical.
- Holography gives the two moves:
  1. Record only the signed cross term in the efficient form (0/pi phase, no zero order, no twin, no carrier).
  2. Match the reference beam to the record (Bragg). Read off-angle, a thick hologram scatters like a diffuser, which
     is our 2.9% fringe share. Read matched, the light goes where it belongs.
- In our machine:
  - Move 1 is the signed-digit, line-up format.
  - Move 2 is turning the 768-line stream once, folded into the plates, so that the light the plates see lands on
    few lines. Bends are not turned: each acts on its own neuron in the wide middle.

## Rules for every step
- **Exact:**
  - Every turn is exactly reversible.
  - The water level is unchanged by the turn: the turn keeps every length and keeps the all-ones direction fixed.
  - Every pile equals the unturned machine's pile in exact numbers.
  - Judged bit for bit where rounding allows, and by the top-5 judge (judge5.py) everywhere.
- **The turn is never searched.** It is set by a direction we already have:
  - the weights-alone foundation, the pattern the weights stand in with no light (as watched)
  - the first-place stone that every later arrival pours from
- **Counted:** lit-pixel meetings, which means light photons x plate photons on each line, with the plates counted
  after the turn too (a turn can lengthen the plates' numbers). Storage is counted at every step.
- **Controls and checks:** a shuffle control and an order-free check on every measurement, and an independent rebuild
  of every claim.
- No float in the machine path, no decomposition, no literature search.
- **Novelty:** turning a stream basis and folding the turn into the weights is known in itself. Ours would be the exact
  whole-number turn, set by the foundation, and counted in lit pixels.

## Phases
- **A. Measure before building.** Record the light at every plate input (30 refs texts). Then:
  - **Foundation directions:**
    - F_l, the weights-alone column at each layer (position 0, word row and position row both zero)
    - S_l, the first-place stone (the position-0 column of a real word)
  - **What to measure:** how much of each later token's light lies along F_l and S_l at every plate, against a random
    direction and shuffled lines.
  - **The turn:** put that direction on one line with an exact rational reflection, then count at the machine's own
    16-place water level:
    - the turned light's lit pixels
    - the turned plates' lit pixels
    - the total lit-pixel meetings
  - **Prediction:** a modest share along the stone, growing with depth. Lit pixels fall by less than the energy share,
    because lit pixels go with the log of a value's size.
- **B. Build the turned machine,** only if A shows a gain. Exact rational or integer-lifting turns folded into every
  plate, water levels exact. Every pile identical in exact numbers, judge5 6000/6000, work and storage counted.
- **C. Record in the full format.** The turned plates written in signed digits with the line-up. That's the saved
  recording, and it's measured three ways:
  - storage
  - lit-pixel meetings per token
  - exactness
- **D. More directions.** Only after C. If one direction pays, add the next one set by a relation:
  - the top read's shared reversed pattern
  - the arrival's stone direction per layer
  Measure again each time.

## Build plan (2026-10-05): things on disk that run, no more side tests until these exist
1. **Precomputed layer 0 for the whole vocabulary** (entry/, base2_entry.py): IN PROGRESS.
   - Every word and every position's meeting with the first plate is stored; per token, only the carry meets the plate.
   - Done when it is bit-identical (30 texts, judge 6000/6000, 2,000 random word-position pairs over the whole
     vocabulary), with size and real time measured.
2. **The fringe recorder:** each token's collapsed print written to disk during a run, plus a reader that gives the exact
   answer from one print in one meeting.
3. **The base plate's own fringes:** the weights meeting the weights, computed once with no token, saved, and measured
   for how much pairs off.
4. **Real speed:** the stored layer 0 and the collapsing top read ported into the Swift/GPU stream machine. Words per
   second against 226 (stream machine) and 120 (PyTorch), with identical words.

## The target (his prediction, 2026-10-05)
- **His prediction:** the fringes-only machine needs **50,000-500,000 operations per token**, against the plain
  machine's 85 million multiply-adds.
- **Counted in analog operations,** where one gate's push onto its 768 lines is one operation:
  - about 37,000 gate pushes
  - 144 arrival shares per earlier token
  - a few dozen turns and water levels
  - the collapsing top read
  - Together that is about 40,000-60,000 for short texts, rising with text length.
- **It holds if two things hold:**
  - fan-out is free (true in light and analog; on a digital chip each push is 768 adds)
  - the glow at every gate can be carried in fringes-only form
- **The build plan measures the glow** layer by layer, starting with step 2 at layer 0.

## Build progress (2026-10-05)
- **Step 1, BUILT and confirmed:** precomputed layer 0 qkv.0 for every word and position (entry/, base2_entry.py).
  23.2x fewer meetings at that plate.
- **Step 2, BUILT (checker running):** layer 0 complete from storage (layer0/, base2_layer0.py).
  - Layer 0 is at 68.5% of plain.
  - The glow is on every edge, as bright as the straight half.
  - Word-and-position tables save only 1-3% past layer 0.
- **Step 3, RUNNING:**
  - the side-paired glow at every layer's gates
  - base2_fringes.py: one exact machine with every exact cut (stored layer 0, signed digits everywhere, knife-edge
    stops by diagonal with each edge meeting lens 2 as it settles, the collapsing top read on fine diagonals with the
    bound at pile scale)
  - Predicted: 50-55% of plain, every word identical.
- **Step 4:** the Swift/GPU port, for real speed.
