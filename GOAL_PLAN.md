# The goal plan (2026-10-05)

**Goal:** the smallest saved model, the smallest footprint, and the least work per token, by removing every cancelling
section. Built from pieces already measured.

Baselines:
- plain base-2 machine: 335 MB stored, 3,667M photon meetings a token
- GPT-2 itself: 498 MB fp32, 124M multiply-adds a token

## Two tiers of "exact" (his call)
- **Tier A, bit-exact:** every pile identical. Its storage floor is about 226 MB (the coin-flip grain).
- **Tier B, answer-exact:** the top 5 identical at all 6,000 judge positions, and only digits that never change any of
  them are removed. Not yet measured on the full judge. An earlier 30-word judge reached 30/30 at 80 MB.

## Phase I: the exact machine, every exact cut in one (base2_fringes.py, building)
| piece | what it removes | measured |
|---|---|---|
| layer 0 from storage (steps 1-2) | the table's cancelling at layer 0, done once | layer 0 at 68.5%; costs 3.4 GB of tables |
| signed digits on every plate and light | 1s that cancel inside each number | banks at 58.7% |
| knife-edge stops by diagonal | low diagonals that cannot change an edge's output | fc at 91.8% |
| collapsing top read, fine diagonals, bound at pile scale | words that cannot reach the top 5 | read at 19% (meetings), about 30% with the bound |
| line-up with ONE shared order per plate, or per-line neuron order (to build) | ups and downs paired off in bulk with one running sum | meetings 37% if the sums are shared |

- **Target:** about 50% of plain work, every pile identical.

## Phase II: the smallest exact footprint
- **Plates:** signed-digit sheets in plate order, packed at the floor, 153.1 MB. Line-up order adds about 75 MB, so it
  is used only if Phase I shows it pays in work.
- **The word table E:** packed, 72.1 MB.
- **The top table T:** NOT stored. It is E through the last water level's lens, so 0.34 MB given E.
- **Small parts** (biases, the knife-edge table, the arrival and log tables): a few MB.
- **Target:** about 226 MB exact, against 335 MB.
- **The trade:**
  - The layer-0 tables (3.4 GB) buy 1.9% of the work. For the smallest footprint they are dropped; for the least work
    they are kept.
  - The first word's whole column per word is another 3.7 GB, and buys one token's work per text.

## Phase III: fringes past layer 0 (the decision point)
Three results come in now:
- the side-paired glow at every layer's gates (Phase I run)
- the 50,257 patterns + KV test (running)
- the joint-recording holography pass (running)

What they decide:
- **If the glow is small, or patterns + KV holds:** design the fringes-only middle (stored own patterns plus the KV's
  turns, glow corrected) and add it to the machine.
- **If not:** the gates are the essential per-token compute, and the machine keeps them, cut exactly by the knife-edge
  stops.

## Phase IV: real speed
Port the final exact machine to Swift/GPU:
- signed-digit sheets
- the knife-edge stops
- the collapsing top read
- the stored layer 0 if kept

Measure words/s against 226 (stream machine) and 120 (PyTorch), with identical words.

## Phase V (only if he chooses Tier B): the answer-exact footprint
- **What gets measured:** for every plate place, whether any of its digits ever changes any top 5 on all 6,000 judge
  positions. Digits that never do are removed, plate by plate, with fresh-coin-flip and rounding controls.
- **Target:** 60-80 MB. The work drops with the lit places per weight (about 1.7 instead of 4.7).

## Every step
- exact as its tier says
- every part's work counted
- footprint in bytes
- an independent checker
- no literature search on machine learning
- nothing public

## His decisions (2026-10-05)
- **Both tiers. Least work is what matters,** because the cuts are meant for big models (14B, Kimi), where there is far
  more to remove.
- **GPT-2 stays the exact testbed.** Every cut is measured as a share of the work, so it carries.

## Phase VI: carry it to big models
- First Qwen3-0.6B, which runs on this Mac.
- Then 14B, streamed one layer at a time from disk on the Mac, or run on the Strix Halo through GitHub.
- Each cut is measured as a share, and the same exact checks are run.

## Phase V, made principled (no search)
1. **Bit-exact reach.** Remove the weight digits that provably cannot reach any kept value at any of the 6,480
   positions.
2. **Answer-exact.** Remove the digits whose largest possible effect on any word's score is below that position's
   top-5 margin, at every one of the 6,000 judge positions.
