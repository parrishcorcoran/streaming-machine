# Invention log

Dated, factual record kept for a possible provisional patent application. It notes what was tried, what was measured
(each claim checked by an independent rebuild), and what prior art was found. Nothing here is a claim of novelty until
a working method exists and has been checked against the literature.

## 2026-10-04

### Goal (the user's words)
Keep all the resolution and decrease the compute. The plates and the 50,257 entry rows never change ("we aren't
training"). Most of every pile cancels itself (measured 93-98%), so a model of the extra only would compute just what
does not cancel. Everything is made of adds and rotations, so it can be reversed: "Adds are time... subtraction is what
actually solves it."

### Measured and checked (base-2 GPT-2)
- **Exact machines on the GPU, words identical to the plain machine:**
  - the streaming machine: 226 words/s
  - the up/down machine (every place +1 or -1, piles from XOR counts): 96 words/s
- **The 1s' motion (S1, eye level, soup):** no travelling shape beyond shuffle controls.
  - The stream's 1s move like a coin pusher, which is the spread of values.
  - One thing does survive the shuffle: a write points the same way as the light it lands on more often than chance.
- **The soup:** between waves, the change is exactly zero behind the front and as thick as the light at and ahead of it.
  Pouring only the change saves 3% over the full soup and cannot go below the one-word machine.
- **Storage:** about 80 MB at 30/30, against 335 MB. That work used known methods (quantization family) and is not
  part of any invention.

### Prior art found
- Parallel fixed-point decoding: Song et al. 2021; Santilli et al. 2023; CLLMs 2024.
- Delta and event-driven networks: DeltaCNN 2022; Eventful Transformers 2023; DeltaDEQ 2024.
- Reversible residual networks: RevNet 2017; Reformer 2020.
- Every compression technique tried overnight was already known: fax, JBIG, NAF, dithering, error diffusion, E8, bias
  correction.

### Open (the user's idea, being built)
A reversible base-2 machine: every step an add or a rotation with its input kept, run forward with adds and backward
with subtracts. The fixed parts (the plates and the entry rows) are used to fold work backward into what is stored, so
that only "the extra" is computed per token.

Not yet shown to save any compute. The reversible machine is being built and checked first, bit for bit.

### Prior art found 2026-10-04 (layer-0 extras)
- **Graef, "Transformer tricks: Precomputing the first layer"** (arXiv 2402.13388, 2024).
  - For rotary-position models, the first layer's norm and Q, K, V are precomputed per vocabulary token and read back.
  - It states that this cannot be done for models with absolute position embeddings followed by pre-norm (GPT-2 style).
- **Kamoda et al., "Weight-based Analysis of Detokenization..."** (NAACL Findings 2025, arXiv 2501.15754).
  - For GPT-2, the first layer's attention score splits exactly into token, position and per-row-scale terms.
  - This is analysis only: no stored tables and no inference.
- **Golden, "Equivalent Linear Mappings of LLMs"** (2025). Each input is frozen into an exactly linear system: analysis
  per input, a full forward pass still needed, no tables.
- **Not found:** GPT-2's first layer at inference as stored word and position tables plus one number per row, exactly.
  Also not found: carrying stored parts through bends in their exactly-open or exactly-blocked regions, so that only
  near-edge units are computed fresh.
- **Status:** the layer-0 exact pilot is still running (extra/extra_layer0.py). Nothing is claimed until it is checked.

### 2026-10-04: layer-0 pilot (base-2 GPT-2): exact (confirmed by independent rebuild)
- **The entry bank (qkv.0):** the word's row and the position's row are stored, rounded the way the water level rounds
  them. Only the carry meets the plate:
  - the carry is -1 / 0 / +1 on a quarter of the lines (peaks and troughs): measured 24.8%, predicted 25%
  - meetings fall 21.6x (61.8M -> 2.9M photons per token)
- **Exactness:** every pile and the stream after layer 0 are bit-identical on 30/30 texts and 300/300 judge windows.
  judge5: 6000/6000 in the top 5, 6000/6000 same top word.
- **Past the arrival:** the machine rounds 41-47 places off the arrival's output before the proj meeting, so stored tips
  mixed by the shares are not identical. Next step: move that rounding after the meeting, which needs wider whole
  numbers.
- **Net exact saving:** 28.3% of layer 0 = 1.7% of the whole machine per token.
- **Script:** extra/extra_layer0.py (local only; not to be pushed to any public repo).

### 2026-10-04: how far the stored part reaches (confirmed by independent rebuild)
- The stream splits exactly into stored part (word + position rows carried through) + new light. Checked equal to the
  machine's stream at every layer on 30/30 texts.
- From layer 1 on, the new light is 94-99.7% of every bank's meeting photons. Storage carries 5.1% of the whole machine's
  meetings: 66% of layer 0, 1.2% past the first bend.
- The machine's own bend: 99.5%+ of neuron-positions sit in its soft part. Exactly open 0.0-0.14%, exactly dark 0.0-0.5%.
  Below the edge, the faint light is 73-87% of the photons entering the out-bank.
- BitNet's bend: exactly dark below the edge. 77.7% of its down lines are dark (56.7% below the edge, 21.0% round to 0).

### 2026-10-04: thin page (own column vs in context) (confirmed by independent rebuild)
- A token alone (its arrival over only itself) has a fixed column. Stored own pile + thin page meeting the plate =
  the pile, exactly: 1470/1470 banks.
- The thin page is not thin:
  - 101% of the light's photons and 152% of its size, from position 1 on
  - almost no lines exactly 0
  - with own piles stored, work is 102.5% of plain
  - control (another word's own column): 104%
- The own column depends on position: 72-80% of photons change when the same word moves.
- Previous token's light as the page: 96% of photons.
- Predictions: the user 5-7%, Claude 50-80%. Both wrong.

### 2026-10-04: finished fringes compared (thinpage/finished.py; not yet independently rebuilt)
- Every tip table rebuilt from its tips equals the machine's pile, at every bank of every text.
- Tips (light place x plate place, per neuron) exactly equal, positions 1 on, at qkv.6 / proj.6 / fc.6 / mproj.6:
  - the token vs itself alone: 2.1 / 2.0 / 2.0 / 1.2%
  - vs the token before: 2.1 / 2.5 / 2.1 / 1.3%
  - vs an unrelated token at the same position: 2.0 / 2.1 / 2.0 / 1.2%
- Piles keeping their bright/dark side:
  - vs alone: from 100% at qkv.0 down to about 57%, near the control
  - vs the token before, above the control at the arrival's out-plate (proj): 73-79% against 53-71%
- Top 5 shared: alone 3.2%, the token before 8.2%, control 4.9%.

### 2026-10-04: reverse every step (reverse/; checked by an independent rebuild)
- **The forward run records every dimming:** water levels, bend shares, arrival shares, and every rounding as an exact
  whole number over a power of two.
- **The reverse:** the answer word's row is sent back down every step: plates transposed, dimmings applied again,
  arrival shares sent back.
- **Exact:** embedding numbers met with R_w, plus c_w, equals the machine's score digit for digit, 15/15, in
  unrounded whole numbers. With 16 places kept per step, the top 5 by reversed patterns equal the machine's top-5 set
  30/30.
- **The reversed pattern belongs to the text's dimmings, not to the word:** the answer's R and a random word's R share
  their sign on 95% of entries.
- **It is not carried across texts:** the same word's R from another text matches at 54% (control 54%). Through
  another text's dimmings, the answer is in the top 5 on 3/30 (coincidence 1/30).
- **Which dimmings stand,** taking dimmings from another text (the rest are the text's own):
  - water levels only: answer kept 30/30
  - arrival shares only: 22/30
  - bends at earlier positions only: 18/30
  - newest column's bends: from another text 1/30, from the column before 2/30, from the token alone 7/30
- **Checker's correction:** the single-layer bend swap held all other layers' shares frozen. With the other bends set
  by the light, swapping in another text's bend at layers 0-4 alone drops the answer to 8-15/30.
- **The roadblock:** the newest column's bends (36,864 shares) are written by the newest light.

### 2026-10-04: the first token is the foundation (base2 arrival shares; Claude's own run, not yet independently rebuilt)
- Over 30 refs texts, positions 1 on, 88,992 arrivals: 57% of all arrival light pours from the first token. If spread
  evenly over the earlier tokens it would be 13%.
- By layer: 18% at layer 0, 48% at layer 3, 69% at layer 5, 77-80% at layers 7-10.
- The first token is the biggest source in 77% of arrivals, and in 90-96% from layer 5 on.
- Running now (foundation/): each token standing on the first token as the base, and the page the other tokens add.

### 2026-10-04: the newest film (reverse/film_*; checked by an independent rebuild)
- **The split is exact.** The answer's lead over the 2nd word splits over every newest-column bend neuron at each layer,
  digit for digit, in unrounded whole numbers (36/36 layer checks).
- **It is a balance, not a few grains.** The parts pushing toward the answer and toward the 2nd word are each about 50
  times their difference; the lead is what is left over.
  - Half of sum |part| sits in 13% of the neurons, which is the shuffle's value.
  - The top-100 grains are shared across texts at about the shuffle's level, except at layer 0.
- **Shut grains matter as much as bright ones,** because their being shut is set by the light.
- **Developing only some grains, the rest from the token-alone column (not exact; changes values):**
  - half the grains developed: 15/30 answers in the top 5
  - 99% developed: the top-5 set kept 21/30 (random grains), 6/30 (largest grains)

### 2026-10-04: standing on the first token (foundation/; checked by an independent rebuild)
- Identity exact at 5,880/5,880 banks: in-context pile = foundation pile + page meeting the plate.
- **The foundation is laid by the first place, not the first word.** A foreign first word gives nearly the same
  foundation: page 94.3% of photons, 86.7% of size, against the real 93.7% / 84.4%. At position 0, changing the word
  changes the column by only 14.7% of its size.
- **The page on the foundation, positions 2 on:** 94% of the light's photons, 84% of its size; work 94% of plain.
  - It is thinnest just above the bottom (fc.0 51%, mproj.0 39%, qkv.1 43% of size), thicker through the layers (about
    90% by layer 5), and thicker with more tokens stacked.
  - The old alone base was 152% of size.
- **Second level (first token + the token just before):** page 62% of size, 89% of photons; the answer stays in the top
  5 at 73% of positions.
- **Predictions:** the user's "thin page" did not hold; Claude's held for photons, not for size.
- **Rerun of collapse.py (the top read as a collapsing waveform, 2026-10-03):** 480/480 same word; survivors after the
  coarse 8-place read median 398 of 50,257.

### 2026-10-04: the top read as a collapsing waveform, exact top 5 (collapse_top/; checker pending)
- Staged reads down the top places with exact bounds. Every read is exact: top-5 set and order identical at all 6,480
  positions (480 refs + judge5 6000/6000 in top 5, 6000/6000 same top word).
- Survivors of 50,257, median, at three depths of the read:
  - after the top 4 places: all 50,257
  - after the top 8 places: 48,524 reading the stamps' places first, 42,808 reading the light's places first
  - after the top 12 places: 12 (stamps first), 10 (light first)
- Work, as a share of the plain top read:
  - stamps' places first: 66%
  - light's places first: 60%, which saves 10% of the whole machine
  - diagonal by diagonal: no saving
- Prediction (30-50%) wrong.
- Control: shuffled light collapses more (52%).
- The real light's unread places carry the same direction as its read places, filling 13% of the bound against 4% for
  shuffled light. The sign rides on every place in sign-and-photons writing.

### 2026-10-05: the knife edge is the same backwards and forwards (Claude's own check; base2.py tables)
- **The table is exactly symmetric.** PHI[32768+k] + PHI[32768-k] = 65536 for all 32,767 k, and PHI[32768] = 32768.
  The Gaussian knife edge is exactly Babinet-symmetric.
- **The machine reads its table one step at a time** (idx = rshift(H, 4)), so mirrored piles can land one step apart:
  bend(H) - bend(-H) = H exactly for 98.2% of random H, and off by 1-5 units otherwise.
- **So every bend splits into two halves:** bend(H) = H/2 + e(H).
  - The straight half passes like half-silvered glass. Per layer it folds into one fixed 768x768 turn, the same for every
    token.
  - The symmetric half e sees only the pile's size. In exact numbers e(x) = x(Phi(x) - 1/2) >= 0, which is square-law
    near the edge.
- **Running (turns/):** record the 12 fixed turns, and split each token's answer lead exactly into what came through
  each half.

### 2026-10-05: the combined exact machine (combined/; confirmed by an independent rebuild)
- **What it combines:** the entry (stored rows plus the -1/0/+1 carry), the line-up on every bank, and the collapsing
  top read.
- **Exact everywhere:** every pile identical on all 648 refs positions and 300 judge windows; judge5 6000/6000 and
  6000/6000.
- **Work against plain (3,666,596,259 photon meetings per token):**
  - every add counted: 70%
  - running sums met as they are: 62%
  - light meeting plate only: 41%
- **The cost found here:** rewriting each neuron's running sum in single waves costs 25% of the plain bank work. The
  earlier "about half" count had left it out.

### 2026-10-05: when each knife edge is decided (knife/; checked: core measurement holds, text corrections)
- **The method:** each knife edge's pile is counted diagonal by diagonal from the top, with exact bounds on what is
  not yet counted. Nothing in the machine changes.
- **The side (which side of the edge) settles early for nearly every edge:**
  - by the top 12-16 of 29 diagonals
  - 23% of lens 1's (fc's) meetings
  - median D14 even for the crowd at -2..0
- **The share and the output (what lens 2 meets) settle only at D24-D28.**
  - Stopping each edge there saves 8.2% of fc, which is 2% of the whole machine.
  - The control is the same over all layers, but differs at layer 0.
- **The road block:** lens 2 meets brightness, not just side.
- **The machine drops most of lens 2's low places when its pile joins the stream** (median 11-13 places).
- **Running (knife/stream_*):** stop lens 1 when what joins the stream is provably fixed.

### 2026-10-05: phase C recording (recording/C/; confirmed by an independent rebuild, wording corrected)
- **The recording:** every plate and the top table written as signed-digit up/down sheets in line-up order (198
  files). The machine runs from the files alone.
- **Exact everywhere:** all 648 refs positions identical; judge5 6000/6000 and 6000/6000.
- **Lit-pixel meetings:** banks 58.65% of plain, whole machine 69.13%.
- **Storage:**
  - Sheets in each plate's own line order pack to 153.0 MB, the floor.
  - In line-up order the sheets pack to 134.8 MB, but the order costs at least 93 MB, so the line-up adds about 75 MB.
- **The private repo is updated** (parrishcorcoran/fringes, commit 619c592).

### 2026-10-05: straight and symmetric halves (turns/; checked; measured numbers hold, one summary refuted)
- **The 12 fixed turns are recorded exactly** (turns/turns.npz, 33 MB). The identity holds at 138,240 of 138,240
  entries.
- **Lead split per layer:** the straight and symmetric halves are of equal size and opposite sign (343 of 360). The
  newest bends' part is the 8-9% they leave between them.
- **Correction by the checker to "a 1-in-1e5 remainder of the fixed turns against the symmetric halves":** with every
  bend's straight half folded into the turns, several large parts cancel together, not two:
  - the entry numbers through the turns: |.| about 1e5 leads
  - the symmetric halves: about 1.6e5
  - fc's bias carried through the turns: about 3.5e5
  - the arrival constants: about 5.7e4

  Added in that order, what is left is 6.1e4, then 4.0e3, then 20, then exactly the lead. The symmetric halves are
  larger than the entry part by a median factor of about 4.
- **Write split:** the symmetric half is 0.5-11% larger than the straight half at layers 7-11, and equal within 0.9% at
  layers 0-6.

### 2026-10-05: what moves the knife edges (edges/; checked: measurements hold, prose ranges corrected)
- **The method:** every edge's pile is split exactly into pushes from each source (word row, position row, every lower
  arrival write A_l and middle write M_l, biases) with the water level's scale held. The split adds back 360/360 and
  is order-free.
- **The edges stand mostly dark:** 83-87% from depth 2 on.
- **The fixed part (biases + position row + line 447) is the standing pattern:**
  - It puts 81-90% of edges on their true side (73% at depth 1).
  - It is equally right for an unrelated token (within 0.25 points), so it carries nothing of the token.
  - It almost only says "dark".
  - It lands within one table step of the true share on only 0.03-0.07% of edges.
- **The biases push hardest at most depths.** The older writes push more by kind at depths 4, 10 and 11.
- **Line 447 (the stone) pushes 2.7-5.1%:** the fc plates are dark there. The stone reaches the edges through the
  water level.
- **The word row's own push collapses after depth 0:** 16.6% at depth 0, then 1-3%.
- **The first arrival write A0 is the largest single write at every depth:** 23% falling to 8%.
- **The token decides about 14-19% of the sides (from depth 3),** through the whole stack of writes, no single one
  deciding; the older writes carry the most.
- **The arriving wave flips 5-8% of the edges at each depth from depth 2.**

### 2026-10-05: the collapsing top read down the diagonals, redone (collapse_top/fine_*; confirmed by an independent rebuild)
- **Exact top 5 at 6,480/6,480** (one exact tie for 5th place in the order run). judge5 6000/6000 and 6000/6000.
  Bound broken 0.
- **Diagonal schedule:** meetings 176.4M per position, 19.3% of the plain top read (915.6M). Predicted 25-35%.
  - The survivors collapse at the top 14-18 diagonals: median 42,535, then 418, then 18 words.
- **The bound's own work in photons is 186.1M (20.3%), mostly divides (108M).** Meetings plus bound come to 39.6% of
  the plain top read.
- **Estimated whole combined machine with this top read:** 63.7% of plain (from 69.8%). This is an estimate; it is
  not yet run on all 648 positions.
- **Next:** make the set-aside test at the pile's own scale, which removes the divides (predicted 28-30% of the top
  read), and carry each word's bound down between stages.

### 2026-10-05: stop lens 1 when what joins the stream is fixed (knife/stream_*; measured once, checker pending)
- **Exact everywhere:** every fixed value equals the machine's own.
- **Nothing is fixed early.** The value joining the stream is fixed only at the full pile: 0.2% of lines at D28, none
  earlier. The next light is fixed for 84% of lines at D28.
- **No position's 768 lines are all fixed before the end:** 10 of 7,776.
- **The interval widths at lens 2 add up** from 3,072 edges (16-31 stream units at D28), so lens 1 saves 0.0002%.
  Trying pushes costs 25-253% more.
- **The control is the same,** so this is set by the plate sizes and the edges still unsure, not by where the light
  falls.
- **The road block:** lens 2 sums 3,072 uncertain edges, and their uncertainty adds up faster than the stream's rounding
  drops it.

### 2026-10-05: the spot bound for the top read (collapse_top/spot_*; measured, checker below)
- **The idea (his):** a word's row need not be finished if its big values are not in the spots where the light is big.
- **The method:** an exact bound from lit counts per random line group, with G = 1, 8, 16, 32.
- **Exact everywhere:** 6,480/6,480 top 5, and judge5 6000/6000.
- **What it changes:** no word is set aside earlier than the top 14 diagonals for any G. At the deciding stages it
  removes about half the remaining words in signed digits, and 10-25% in plain photons.
- **The cost:** the bound's own work grows with G (table values added for every group), so the totals are 29.2% of
  the plain read at G=1 (plain photons, diagonals, the best) but 73-549% at G>1.
- **The best exact top read measured:** plain photons, one group, fine diagonals: 29.2% (41.6% counting every divide).

### 2026-10-05: BUILT, precomputed layer 0 for the whole vocabulary (entry/, base2_entry.py; confirmed by an independent check)
- **What is stored:** qkv.0's meeting with every word row and every position row at every shift the water level can
  use (an exhaustive survey of all 51,463,168 pairs found s = 0, 1, 2, 4), 1.96 GB.
- **At runtime:** base2_entry.py reads two stored rows and meets only the -1/0/+1 carry. It never meets the full light
  with the qkv.0 plate.
- **Exact everywhere:**
  - 30/30 texts (every pile and stream)
  - judge5 6000/6000 and 6000/6000
  - all 170,315 stored rows used and checked
  - 20,000 independent random (word, position) pairs over the whole vocabulary
  - 1,000 one-word and 1,000 two-word random sequences
- **Meetings at qkv.0:** 61.8M down to 2.66M photons per token (23.2x fewer). This is 1.6% of the whole machine.
- **Real time in numpy on the M4:** 1.0-1.3x at qkv.0, no change on the whole forward. The forward's time is 75% the
  Python race loop inside the arrival. Real speed needs a compiled pour (Swift/GPU), which is step 4.

### 2026-10-05: BUILT, layer 0 complete from storage (layer0/, base2_layer0.py; confirmed by an independent check)
- **Exact everywhere:** 30/30 texts, 300/300 windows, judge5 6000/6000 and 6000/6000. Every stored row checked; random
  sequences checked.
- **Layer 0 per token:** 219.2M down to 150.1M photon meetings (68.5%). Precomputed: 31.5% of layer 0, which is 1.9%
  of the whole machine.
  - qkv.0: 4.4% of plain is left
  - fc.0: 91% left (its fixed part saves 4.8% at positions 1 on)
  - proj.0: 92% left
  - mproj.0: 96% left
- **Stored:** 3.4 GB across steps 1 and 2.
- **The knife edges:**
  - The word and position fixed part pushes 57-60% and alone puts 89-91% of the edges on their final side.
  - The KV pushes 40-43% and moves 9-11% of the edges.
  - The glow is on every edge.
  - Checker's correction: on the dark edges the glow cancels about 83% of the straight half's size, not all of it.

### 2026-10-05: BUILT, the fringes machine (base2_fringes.py, fringes_machine/; confirmed by an independent check)
- **Every exact cut in one machine:** layer 0 from storage, signed digits at all 48 banks, knife-edge stops by diagonal,
  and the top read down fine diagonals with the set-aside done at pile scale and the bound carried down.
- **Exact everywhere:** 648/648 refs positions and 300/300 windows identical (96 arrays each); judge5 6000/6000 and
  6000/6000.
- **Work: 1,745,178,133 photon meetings and adds per token = 47.60% of plain (3,666,596,259), every add counted.**
- **What each cut saved, as a share of plain:**
  - signed digits: 30.87%
  - layer 0 from storage: 1.07%
  - knife stops: 1.02%
  - top read: 19.45% (the top read itself is now 21.13% of its plain cost)
- **The gates paired by side (glow measurement; confirmed):** the glow is on 99.84% of edges and writes about 0.89 of
  each layer's whole write. Pairing by side adds 4% of the machine, so the gates do not reduce to side bits.
- **Checker's caveats:** the OWN tables cover only the test texts (worth 0.009% of plain), and fc.0's stored rows are
  counted as one row (+36,762 a token, still 47.60%).

### 2026-10-05: Phase V, principled removal of digits (reach/; checked, see the checker's note)
- **Bit-exact reach: 0 of 548,667,924 lit signed digits can go.** 75,066 of 82,944 bank neurons, and every word, have
  some position where the pile sits exactly on its bucket's edge.
- **Removing single digits for real:** all 98,000 random single digits changed a kept value.
- **Answer-exact margin: proven removal is 0 on every plate.** The smallest top-5 gap at a judge position is 2 units,
  and 3,518 of the 6,000 positions have a gap below 4,096.
- **Removing just one plate's place-0 digits for real changes the top 5** at 15-221 of the 6,000 positions (median
  45).
- **The exact bounds are loose with depth** (2.45x at the top table, 80-4,460x at layer 11, unbounded below). But even
  an exact bound would remove nothing, because the top-5 near-ties are a few units apart.
- **Conclusion:** both principled tiers remove nothing on GPT-2; every digit reaches a kept value or the top 5
  somewhere.

### 2026-10-05: the line-up across the neurons on each line, "way 2" (lineup2/; checked, main numbers hold)
- **The method:** each line's weights are sorted across its neurons once. Per token the light on a line meets only the
  tiny steps between neighbouring weights, and a running product walks down the line.
- **Exact everywhere:** 30/30 texts, 300/300 windows, judge5 6000/6000 and 6000/6000.
- **The banks: 56.17% of plain.**
  - gap meetings: 24.08%
  - pours (each neuron's exposure landing in its pile): 32.08%
- **The top table: 38.07% of the plain read.** The gap meetings are only 1.22% (93% of the steps are zero); the rest
  is pours.
- **The walk's cost falls with the number of neurons on a line; the hand-off stays about 10 photons a pair.** Photons a
  pair for the walk:
  - 768 neurons a line: 9.8
  - 2,304: 7.0
  - 3,072: 6.4
  - 50,257 (the top table): 0.28
- **Storage: 155.16 MB** (gaps 57.3 MB + orders at their floor 97.2 MB). The orders cost what the gaps save.
- **Next:**
  - pour the top table band by band, only into the words still in the race (predicted 12-17% of the read)
  - measure on wide plates (Qwen3-0.6B, then 14B), where the walk heads toward nothing and the banks toward about a
    third of plain

### 2026-10-05: BUILT, fringes machine v2 (base2_fringes2.py, fringes_machine2/)
- **Additions to base2_fringes.py, each verified exact:**
  - the top read's bound starts where a light-only relation says it can bite: 47.20%
  - the top read in signed digits with the one-sided bound: 46.34%
  - the arrival's products in signed digits: 46.21%
  - one shared line order per plate: passes on 0 of 48 plates (its gaps are 1.035-1.060x the weights' lit digits),
    so not used
- **Total: 1,694,214,555 photon meetings and adds per token = 46.21% of plain, every add counted.**
- **Exact everywhere:** 30/30 texts, 300/300 windows, judge5 6000/6000 and 6000/6000.
- **The banks are now 90.7% of the remaining work.** The top read is 16.05% of its plain cost, and the arrival is
  0.5% of the work.

### 2026-10-05: the word table walked and poured band by band (topband/)
- **The method:** each line of the word table is walked once (way 2, 1.21% of the plain read). Each word's exposure is
  then poured in 8 bands from its top places down, only into words still in the race, with the collapsing read's exact
  set-aside after each band.
- **Exact:** top 5 and race winner at all 6,648 positions; judge5 6000/6000 and 6000/6000.
- **The top read is 10.88% of the plain read** (the collapsing read was 21.13%).
- **Words still in:** all 50,257 through places 18 and up, then a median of 70 after places 14-17, 6 after places
  10-13, and 5 after that.
- **Whole machine with this read in place of base2_fringes.py's: 45.07% of plain** (it was 47.60%).
- **Not yet combined with v2's signed-digit arrival and bound start;** combined it would be about 44.7%.

### 2026-10-05: Phase VI-a, the cuts carried to Qwen3-0.6B (qwen06b/; checked, no number fails)
- **Every cut checked exact** on the existing whole-number Qwen machine (qwen/qwen_base2.py), on all 196 plates, the
  word table and all 613 positions.
- **Signed digits:** 59.48% of the plates' plain work (GPT-2: 58.65%).
- **Way 2 (the line-up across neurons): 49.14% of the plates' plain work**, and 44.4% at gate and up. It does better
  than on GPT-2 because Qwen's 8-place (bf16) weights tie often: 30-59% of the gaps are zero.
- **The top read:** 15.84% of the plain read. Exact top 5 at 613/613.
- **Estimated whole machine** (not yet built as one): way 2 + the top read = 40.2% of plain.
- **Predicted for a 14B model** (from the weights alone, not measured): 37.8-40.4% of plain. There the walk nearly
  vanishes, and the pours (each neuron's exposure landing in its pile) are the road block, at about 33-40% of plain.

### 2026-10-05: BUILT, the combined exact Qwen3-0.6B machine (qwen_fringes.py, qwen_fringes/)
- **Exact everywhere:** every pile, arrival output and stream matches qwen_base2.py at all 613 positions of the 30
  texts. The top 5 and race winner match 613/613, and the same 16 greedy words come out on 30/30 texts.
- **37.55% of Qwen's plain work** (16.40G photon meetings per token), every add counted. Each addition, cumulative:
  - way 2 at every plate: 63.26% (the plates go to 49.14% of their own plain work)
  - the band top read: 37.72% (the read goes to 6.74% of its plain cost; a median of 8 words is left after band 3)
  - the arrival in signed digits: 37.55%
- **Pours written in signed digits do not pay at 0.6B's widths:** writing them is a second walk, so the total rises to
  42.69%. Not used.
- **The pours are now 66% of what is left:** each neuron's product landing in its pile.

### 2026-10-05: BUILT, Qwen machine v2 (qwen_fringes2.py)
- **Two changes:**
  - the arrival's products as whole-number int8/int64 products (no float in the machine path; a second float in the
    gauge's log2 also removed)
  - one line-up per line across the whole q+k+v slab (4,096 neurons) and across gate+up together (6,144)
- **Exact everywhere:** 613/613 positions, the 16 greedy words, order-free.
- **The walk per pair:** q+k+v 2.82 (it was 4.5-6.4 per plate); gate+up 2.07 (it was 3.5). Zero gaps: 64% and 74%.
- **Whole machine: 5,602M photon meetings per token = 34.16% of Qwen's plain work** (qwen_fringes.py: 37.55%).
- **The pours (each neuron's exposure landing in its pile) are now 73% of the machine.**

### 2026-10-06: Phase IV, the exact cuts on the M4's GPU (base2_stream_fr.swift, gpu_fr/)
- **The collapsing top read on the GPU, exact:** the word table is stored as a high byte plus two low nibbles.
  - All words are read from the high byte; a median of 38,740 words is left.
  - The middle nibble is read only for those, leaving a median of 12; the low nibble only for those.
  - Top 5 and the race winner match 480/480; the same words on 30/30 prompts.
  - The table's bytes read per token fall to 68% of plain.
- **Real speed, from the checker's 10 alternating rounds:** +3.9% words/s after the prompt from the top read.
  - The stored layer 0 is exact, but its speed gain is not reproducible (-0.2 to +0.1%).
  - Against PyTorch on the same Mac: about 2.5x MPS fp16 and 2.1x CPU fp32.
- **The road block, in plain terms:** the GPU pays per byte read, not per lit photon.
  - The chip meets all 16 places of a weight in one multiply, so signed digits, the line-up and the knife stops
    (the photon cuts) do not become GPU time.
  - The 12 layers read their 170 MB of plates whole every token, and that is 80% of the token's time.
  - The photon cuts become time on hardware that pays per photon (analog, optical, bit-serial).

### 2026-10-06: Phase IV-b, the plates read at the memory's speed (base2_stream_fr2.swift, gpu_fr2/; checked)
- **The wide plate read:** each lane reads 8 neighbouring piles of a line (16 bytes). The pile is formed once, with
  whole-number adds, where it is used.
- **Joined steps:** the water level and gauge are made by the step that makes the light, so a layer goes from 18 GPU
  steps to 10.
- **Exact:** 0 of 37,324,800 piles differ, the hashes of pile, light and water level match 21,600/21,600 (including
  the order-free check), and the same words on 30/30. The checker's own CPU rebuild matches too.
- **Plates read at 89 GB/s** (it was 69), the memory's working speed.
- **Speed:** 297 words/s after the prompt and 274 including it. That is +21% over base2_stream_fr, +24.5% over
  base2_stream.swift, and about 2.5-3x PyTorch on this Mac.
- **Dark lines** (light exactly 0) are 0.09% of the plate bytes, and the test costs more than it saves, so they are not
  used.
- **What blocks:** the GPU now reads the plates as fast as its memory allows, so from here it gets faster only with
  fewer bytes. The exact floor is about 14.4 bits a weight against the 16 stored, so at most about 10% more.

### 2026-10-06: his idea, the foundation is the blank starting model (his exact words)
- "The blank untrained model with random bell curve wieghts! That's the plate and the embedding table! That's the
  "foundation" we have been looking for."
- "It should be 100% removable after training. We could even test on a tiny model."
- "The random weights have random values and everything. That is trained is built on top. It creates the topographical
  map. That's IT! And the entire thing pre-training could be removed"
- **Session's restatement (not his words):** each finished weight = the foundation (its random starting draw, worn down
  by the building) + what was built on top. The foundation can be regrown from its seed, so it costs no storage. The
  test bed is Pythia, whose starting weights were saved at step 0.

### 2026-10-06: his foundation idea tested on Pythia (foundation0/; independent rebuild, not refuted)
- **The machine:** a whole-number Pythia machine (pythia_base2.py) at 14m, 70m and 160m. It gives fp32's words at
  480/480 on every size, is order-free, and its int64 products are identical.
- **Most of the blank start was worn away during the building:**
  - the input and output tables kept 1.4-4% of their start
  - the plates kept 9-28% (70m) and 7-67% (160m), far more than the shave alone leaves (8-730x)
  - so the plates' built relief follows the foundation's own random relief, with the same sign
  - the start is only about 0.5-5% of each finished plate's energy
- **241 input-table rows were never built on** (padding and runs of spaces). Each is exactly one number x its random
  start, the same number for all 241 rows: 100% foundation, 100% removable.
- **Run with the surviving foundation removed:**
  - input table only: 70m keeps every word (480/480 greedy, 30/30 texts); 14m and 160m keep 97-98%; top 5 at 100%
  - output table only: top 5 kept 99.8-100%, top 1 86-99%
  - plates: about half the words are lost at 70m and 160m. The knife edges hold (6-13% change side), but the water
    level is lost in the late layers, where the text's first word stops holding the level.
  - At 160m, removing the real foundation costs more than removing a scrambled copy of it: the light leans on the
    real one.
  - Removing the whole unworn start (W - W0) loses every word, because it digs a new random relief into the plates.
- **Storage:** stored as finished = regrown foundation + what was built, exact; it saves under 0.07 bits a weight,
  because the foundation left is small.

### 2026-10-06: his question "The base blank slate should be equal in ups and downs right? In base 2?" (checked)
- **Yes.** At Pythia's blank start (step 0), every place of every plate has ups equal to downs, up to the coin-toss
  leftover (about the square root of the count). The same holds per plate, per neuron and per line, at 70m and 160m, in
  sign-and-size and in signed digits, the same as a fresh bell-curve draw.
  - Example: one place of the 70m output table has 327,121 ups against 326,548 downs.
- **The finished weights depart from that balance in two ways, both built:**
  - **Uneven far beyond the coin toss, along chosen lanes:**
    - MLP in: 40-69% of its input lanes; whole plates off by up to 11 coin-toss widths.
    - Word tables: 80-100% of their lanes.
    - Output table: off by 85-168 widths at a place.
  - **More even than the blank slate, along the stream:** each word row, attention-out line and MLP-out line adds a
    level across the stream that is a quarter to a half of the blank slate's leftover. It is a new, smaller level, not
    the start's level kept.

### 2026-10-06: his sentence "If the blank slate were saved somewhere you would be able to remove EXACTLY each blank weight exactly"
- **The blank slate is saved** (Pythia step0). Removing what is left of each blank weight after the building's shave
  (c x blank, with c read from the 241 never-built rows) changes each weight by a few grains (14m, 70m) or about 39
  grains (160m).
- **Words kept:**
  - 70m: every word (top-1 100%, 30/30 texts)
  - 14m: 99% (6 near ties lost)
  - 160m: 94% top-1 and 100% top-5
- **Removing the whole blank** (not shaved) loses every word.
- **A scrambled blank of the same size** does the same at 14m and 70m. At 160m the real remainder moves the leads
  30-50% more: the light runs partly along the blank's grooves.
- **Storage saved: about 0.01 bits a weight**, all of it from the 241 dark rows, which become exactly 0. The building had
  already worn the blank down to almost nothing; what is worth storing is what was built.

### 2026-10-06: his sentence "It's a zero sum game. If one weight goes up another goes down." (zerosum/; checked)
- **Inside each write line, across the stream's lanes, what was built is a zero-sum game.** Each word row and each
  attention-out and MLP-out line nets to 12-34% of a coin toss's leftover.
  - It holds at 14m, 70m and 160m, and over every saved span of the building.
  - The write biases, which start at 0, also net to nearly 0.
- **Not zero-sum** in one lane summed over all lines, nor place by place, and mixed for the neurons that read the
  stream.
- **The "10 becomes 12" picture** (a change within 20% of the blank) fits only about 5% of weights, the same as chance.
  A typical change is about as large as the finished weight.
- **Removing each write line's level** (the water level pours it off before every read):
  - the words stay: GPT-2 judge5 6000/6000 in the top 5, 5995/6000 the same top word
  - a same-size change that goes up on half the lanes and down on the other half changes 345 top words
  - it is not digit-for-digit exact in the machine as built: each read cuts to 16 places before pouring off the level,
    and GPT-2's read plates carry rounding leftovers in their column sums
  - way around: pour the level off before the cut. Then the biases, word rows and position rows are stored level-free
    with every read identical, on all four machines. The weight rows remain a road block.
  - work saved: none (the level's work is 0.0027% of plain)

### 2026-10-06: his correction, where the blank goes (his exact words)
- "You just can't be looking at it right. Maybe you are looking at the coin pusher aspect. It is zero sum because it
  must be. It doens have to be in the same spot. If you run the blank model ALL that compute is the opposite sides
  canceling that we are trying to remove."
- "Maybe try a couple small model, train a little bit and then remove blank slates or see wheee they go. It's an exact
  thing."
- "because the embedding table is the opposite as the ENTIRE all neuron rows, a tiny change in embedding is a bigger
  change on other side, so if it's not balanced perfectly then the blank is stuck inside the embedding table. Either way
  ITS THERE because that's the foundation we found. It HAS to be because the model has no way to remove it"
- **Test launched (blankgo/):** tiny models built from known blanks, with every push kept exactly in whole numbers, with
  the shave off and on. It tracks where the blank goes in the work: taken out in place, cancelled by an opposite part in
  what was built, or steered around by the light from the word table.

### 2026-10-06: which plate holds the late water level when the surviving blank is removed (foundation0/late/; checked)
- **Each layer's own qkv plate carries that layer's level.** Removing the surviving blank from qkv.9-11 alone moves
  42-61% of their own levels, while every proj, fc and mproj plate moves them under 4%.
  - Why such a small relief moves such tall levels: the late keys are 97-99% the plate's fixed bias light, and each
    meeting is 400-600x the row's height, so a sliver of change swings the level.
- **Keeping the blank only on the late qkv plates** (removed from 92% of plate weights and both tables):
  - the late level holds
  - the words don't: top-1 59% (160m) and 61% (70m), top-5 84-90%
- **Road block:** the light meets the blank's relief at every plate. Lifting it shifts each plate's light by 1-17%
  (most at fc). The knife edges keep their sides, but the early levels move and the close races at the end flip.
- At 160m, removing the real blank costs more than removing a scrambled one of the same size, once a whole plate kind
  is removed.

### 2026-10-06: his idea, the blank's work is cancelled in the word table (his exact words)
- "The blank work must be in the embedding table equal in math to the MLP. It has to cancel. So it's either tiny bit of
  all the 150k tokens embeddings vs MLP, we must not be the only one to realize this . Our work of canceling might be
  doing some of it to begin with. Lining up and cancelling might do most. Can you line up the blanks and see if it all
  cancels?"
- **Test launched (lineblanks/):**
  - the blank and finished Pythia plates lined up (way 2), with ups paired against downs by size
  - every write line in the whole model summed lane by lane
  - whether the word rows carry the opposite of what the blank adds to the stream through attention-out and MLP-out,
    position by position and spread thin over all words

### 2026-10-06: where the blank goes, tiny models built exactly from known blanks (blankgo/; checked, one range corrected)
- **12 tiny models:** GPT-2 shape, width 64/128, 2/4 layers, 3 recipes, 3,000 pushes each. Every weight is a whole
  number, and finished = blank + every push held exactly at every step.
- **With the shave off, the blank stays in every weight exactly. The model has no way to remove it** (his words hold).
  The shave on shrinks it to 85% after 3,000 steps.
- **The blank's work is still there in the finished machine:** 27-32% of all the meeting size (scaled push), 37-63%
  (plain push). Inside it, 84-94% still cancels, up against down, like coin tosses, the same as in the blank machine.
- **Nothing inside the machine cancels it. What was built runs ALONG it:** the finished piles carry 1.4-1.8x the
  blank's own pattern. The light leans into the blank (up to 1.57x chance at the down plates). The word rows also lean
  into the blank first plate (1.07-1.20x chance); the word table is pushed along the blank, not balanced against it.
- **An opposite stands only at the end:** the output word table cancels about 55% of the blank's word scores.
- **Removing the blank costs words** (about half the top words kept). Swapping in a scrambled blank costs more than
  removing it: what was built is fitted to this exact blank, line by line.
- **Checker correction:** at the one-word stage of plain push, removing the blank costs -0.09 to +0.26 nats, not the
  -0.09 to +0.10 first stated.

### 2026-10-06: the blanks lined up (lineblanks/; checked)
- **Our line-up (way 2) takes the same share of work off the blank plates as off the finished ones:**
  - 70m: 40.45% against 39.13%
  - 160m: 44.86% against 44.31%
  - a fresh bell draw gives the blank's number to within 0.03 points
  - So the line-up cancels the neighbours any bell curve has; it does not remove a blank part.
- **Ups paired with downs by size:** a blank line never pairs off to zero; it leaves a bell's own leftover (5.7% of its
  size at 512 neurons, 0.57% at 50,304).
- **Every write added lane by lane:** the blank's writes add to the coin toss and no further. The word table and the MLP
  are not partners (r +0.03 / -0.01, inside the random-order range).
- **The word row does not carry the opposite of the blank's push** into the stream, neither word by word (cos +0.02)
  nor spread thin over all words (cos +0.05-0.06, inside the random range).
- **Where cancelling does happen:** in the finished machine, between its own MLP writes. One middle MLP write (about 22%
  of everything written) is taken back by the late MLP writes, and the 160m's writes cancel 26% of their summed
  squares. The zero-sum is real, not in the same spot, but between the machine's own writes, not the blank against
  the word table.
- **Not predicted:** almost all of the finished output word table is one shared column. In the median lane, 99.9% of
  the 50,304 words have the same sign, and the mean word column is 0.96 of a typical column's size. That shared column
  adds the same amount to every word's score, so it never decides the race. The blank has none of it (0.004).

### 2026-10-06: the output table as one standing pattern plus each word's fringe (sharedcol/; checked)
- **The shared column's share of the top table's summed squares:**
  - Pythia 70m: 91.7%; 160m: 89.7%
  - GPT-2's T: 82.7%, held in a few lanes (7 lanes hold 75% of it)
  - Qwen3-0.6B's tied table: 21.1%
  - the blank and fresh-draw controls: about 0
- **The exact cut** (store one standing column + each word's fringe; the light meets the column once per position,
  which adds the same amount to every word, so the race and the top 5 come from the fringes alone), Pythia 70m and
  160m:
  - top 5 and race winner 622/622, words 480/480, 0 piles differ
  - the top read falls 20.5% / 18.0% (plain 1s), 16.7% / 15.1% (signed digits)
  - the whole Pythia machine falls to 86.75% / 93.09% of plain; in line-up (way 2) form, 39.13% -> 37.24% and
    44.31% -> 43.39%
  - footprint +0.1% / +0.8%
- **Road block:** a value's lit places follow how many places it spans, not its size. Taking away 90% of the squares
  shrinks a value about 3.6x (about 2 places), which removes one lit place in five.
- **Not cut, by the rule fixed before running (fringes at least 5% fewer lit places):** GPT-2 (2.2%), Qwen (0.65%),
  Pythia 14m (1.2%).

### 2026-10-06: his idea, the tips are what the building made (his exact words)
- "I really feel like the tips are the only thing the model learned and needs and it's because all the non tips are
  the blank slate somehowz"
- **Test launched (tips_blank/):** per plate and per diagonal, how much of the work pairs off (the non-tips) in the blank,
  the finished plate, a fresh bell draw and the built part alone, on the same light.
  - test beds: Pythia 70m/160m, the tiny models over their building, GPT-2 and Qwen
  - plus whether the finished tips carry what the blank lacks, against a light-shuffle control

### 2026-10-06/07: group piles across light lines, the one thing the search said we missed (groups/; checked, not refuted)
- **How it works:** take a random group of g light lines and build every subset's pile once per token. Each neuron, at
  each plate place, then adds one ready pile instead of meeting its lit pixels one at a time. Nothing extra is stored.
- **Exact:** every pile digit for digit at every plate and position (GPT-2: 48 banks x 648 positions; Qwen3-0.6B:
  196 plates x 30 texts).
- **Work, table included, as a share of plain photon meetings:**

| | signed digits | line-up | groups g=8 | groups g=12 | up/hole form g=12 (not yet confirmed) |
|---|---|---|---|---|---|
| GPT-2, 48 banks | 58.65% | 56.17% | 35.44% | 31.03% | 21.04% |
| Qwen3-0.6B, 196 plates | 59.48% | 49.14% (slab 44.45%) | 37.44% | 32.02% | 24.01% |

- **Controls:** a second random grouping and shuffled light give the same within 0.03 points; the order-free check
  gives identical piles. The saving is a law of the plate's digits, the group size and the light's sizes.
- **Up/hole form:** every place is either a photon up or a hole, so by the flip (Babinet) a subset's complement is
  exactly its opposite pile, and half the table comes free.
- **Road block:** the reads fall only as 1/g, while the table grows as 2^g and is shared only by the neurons that read
  the same light, so narrow plates cap g at about 12. Way around: the top tables (50,257 / 151,936 words share one
  light).
- **14B estimate (labelled, not run):** 30.3% (signed digits) and 20.3% (up/hole) at g=16.

### 2026-10-07: the tips and the blank (tips_blank/; second count agrees; checker corrected several summary sentences)
- **"the tips are the only thing the model ... needs": holds exactly.** The tips alone rebuilt every pile again, at
  every plate of every model (Pythia 70m/160m, the tiny models at every snapshot, GPT-2, Qwen): 0 piles differed.
- **"the only thing the model learned": half holds.**
  - Everything the building made shows in the tips.
  - Most of the tips' size is coin-toss speckle that any plate leaves (GPT-2 and Qwen: about 88% by size, 99% counting
    place pairs alike, is what the same plate leaves with the light shuffled).
  - What was built is the extra at the top 5-7 diagonals, where the light's sign places meet the plate's. Pythia's
    word table adds a standing pattern: its top diagonal pairs off only 4%.
  - At the plates that write into the stream, the building made the top tips SMALLER than chance.
- **"all the non tips are the blank slate somehow": not literally.**
  - Pythia-70m keeps 0.06% of its blank, and the blank's pattern takes no part in the finished pairing.
  - Below the top diagonals the non-tips match any bell draw of the same spreads, so they look blank because every
    bell-curve plate pairs off the same way.
  - Checker's corrections: at the MLP-out plates the building reaches 4-8 diagonals deeper (tiny models), and a few
    Qwen plates differ from the fresh draw by their stored format.

### 2026-10-07: BUILT, the group machines run end to end (base2_groups.py, qwen_groups.py, groups2/; checked, not refuted)
- **GPT-2 (base2_groups.py): 711,431,068 photon meetings a token = 19.40% of plain** (base2_fringes2 was 46.21%).
  - Exact: every pile, stream and light identical on the 30 refs texts and all 300 judge5 windows.
  - judge5: 6000/6000 in the top 5, 6000/6000 the same top word, top 5 equal in set and order 6000/6000.
  - By part, share of plain: qkv 19.52%, proj 25.28%, fc 21.15%, mproj 25.35%; the top read is 8.93% of the plain
    read.
- **Qwen3-0.6B (qwen_groups.py): 3,105,280,101 a token = 18.94% of plain** (qwen_fringes2 was 34.16%).
  - Exact: every pile, arrival output and stream; top 5 613/613; the 16 greedy words on 30 texts.
  - The 196 plates are at 23.88% of their plain work; the top read is at 5.39% of the plain read.
- **How:**
  - every plate by group piles at g = 12, with each place read in up/hole form or signed digits by a rule fixed from
    the plate alone
  - the top read by a group table of 16 lines read top place down, setting words aside after each place
- **Confirmed along the way:** the up/hole form, rebuilt independently, equal number for number (GPT-2 plates 21.04%,
  Qwen 24.01%). The per-place choice gives 20.26% and 22.11%.
- **Writing the ready piles** costs 2.10 points (GPT-2) and 1.35 points (Qwen), counted in the totals. Without it:
  17.30% and 17.59%.
- **Not yet confirmed under rule 10:** the checker checked samples, not the full rebuild.

### 2026-10-07: the group machines CONFIRMED (confirm_groups/)
- **Separate code** rebuilt both machines from scratch.
- **GPT-2:**
  - every pile and stream on all 648 positions and all 300 judge5 windows
  - judge5 6000/6000 three ways
  - 711,431,068 a token = 19.40% of plain, every part and all 523 layer counts equal
  - its own grouping gives 19.41%
- **Qwen3-0.6B:**
  - every pile, arrival output and stream on all 613 positions, top 5 613/613, greedy words 480/480
  - 18.9376%, equal but for the top read's bound count (1,988 a token, convention unstated)
- **Counting audit:** the delivered shares price group reads in signed digits and plain in binary 1s.
  - With plain also rewritten in signed digits: 33.67% (GPT-2) and 32.71% (Qwen).
  - With nothing rewritten on either side: 23.40% and 23.80%.
  - Against the binary plain machine (the unit of every earlier result), about 20% with the small corrections.

### 2026-10-07: his base, the word table lined up, and the base carried down the layers (basedown/; checked)
- **His words:**
  - "It's removable as in convert them to base 2 and line them all up, any shared 1s in a line cancel, any shared zeros
    cancel"
  - "Everything these cancel with ARE the base. So every single weight they touch we can extract those also, and
    alllllll the way down all The layers"
- **GPT-2's word table lined up line by line (50,257 words a line):** 89% of the lit places cancel against the
  neighbour (5.01 -> 0.55 a value). Storing it needs the order, 13.42 bits a value, against the exact floor of 14.95.
- **Every plate's light lined up across the tokens (exact, every pile identical):**
  - share that cancels: 70% with all 648 positions in one soup, 30% within a text, against the shuffle's 67% and 26%,
    so the cancelling is set mostly by how many tokens share a line
  - the walk on the gaps: 21.6% / 43.7% of plain
  - the landings (each product into its own token's pile): 32.1%, the same in any order or soup
  - whole: 53.7% / 75.8%, against the group machine's banks at 22.8%
- **Landing the gaps through group piles:** +2.1 / +3.1 points worse than the group machine.
- **Road block in his picture:** a product belongs to one token's pile at one neuron, and nothing else can use it, so
  only the light can be shared across piles. That is what the group piles already do.

### 2026-10-08: removing the flat on both sides (flat_design/, flat_build/; checked) and backward from the token (backward/; checked)
- **His words:** "THE FLAT IS THE CANCELLING." "it has to be BOTH SIDES" "Start on the next token and work backwards"
- **The extra (the tips), exactly, over the whole GPT-2 machine:** 6.24% of plain counted as uncancelled meetings
  (plates 3.3-6.8%, top read 8.5%). Written out as photons in up/hole form it is about 11-23% of plain by plate.
- **Four independent designs, judged:** all reach the same machine change. Built exact (pick 1): GPT-2 18.86% of
  plain (691.5M a token, from 19.40%), judge5 6000/6000 three ways, by sizing each light table by the gates that read
  it (g = 11 at proj and mproj).
  - The plate's sign run on top (pick 2): banks -0.63 points, exact; the whole machine about 18.4% (projected).
- **Backward from the token, every stop proved:** 76% of plain (12x the extra). An exact stop has to assume the
  unpoured grains do not cancel, so the bound is the flat counted as if it did not cancel.
- **What is left, in his picture:** inside each gate, the ready piles from different groups cancel against each other
  (about 98% of what is left). A counting machine has to add each of those. An interference plate (optical or analog)
  would cancel them on the detector for free and see only the extra.
