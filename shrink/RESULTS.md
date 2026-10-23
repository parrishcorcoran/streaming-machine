# Shrinking the stored machine: results

Base-2 GPT-2. Stored as written it is about 335 MB:
- the 48 plates at 16 bits: 169.9 MB
- the entry table E at 18 bits: 86.8 MB
- the top table T at 16 bits: 77.2 MB
- small pieces

Every number here comes from a real file that was written and then decoded back. Systems that change values are judged
only by the right next word out of 30. Each family's best result was rechecked by an independent checker.

## Round one (night of 2026-10-04): six families

### Exact (nothing changed)

| what | stored | bits a value |
|---|---|---|
| E, best coder (flat field + photon planes, adaptive contexts) | 72.1 MB (was 86.8) | 14.95 (was 18) |
| T on its own | 54.8 MB (was 77.2) | 11.35 (was 16) |
| **T given E** (T is E through the last water level's lens, 768 whole-number gains) | **0.34 MB, about 54 bits a word** | 0.07 |
| E + T together, exact | 72.9 MB (was 164.0) | |
| all 48 plates | 153.1 MB (was 169.9) | 14.4 |

**Why exact storage stops there.** Below about the 3rd place under each value's top photon, the places are coin flips: lit 49-50% of the time, with no pattern in any context tried. A coin flip costs 1 bit, so about 11 of E's 15 bits are grain.

**"Store only the 1s" on full values is worse.** E 17.9 bits, T 14.2. Saying where each photon sits costs as much as writing every place, because the dark places are as unpredictable as the lit ones.

### Developed (photons dropped), judged by the words

Tables. Only E is stored. It's exposed for the lens and rounded to nearest, and T is printed from it:

| tables stored | E + T | right next word |
|---|---|---|
| c = 32 | 34.6 MB | 30 / 30 |
| c = 34 | 25.0 MB | 29 / 30 |
| c = 37 | 15.0 MB | 28 / 30 |

Plates, with each line and each neuron given its own exposure:

| plates stored | bits per weight | photons per weight | right next word |
|---|---|---|---|
| v12, A = 8: 56.7 MB | 5.34 | 1.69 | 30 / 30 |
| v14, A = 6: 45.9 MB | 4.32 | 1.34 | 29 / 30 |

### The whole machine together (shrink/combined_check.py)

| plates | tables | right next word | stored |
|---|---|---|---|
| v12 A=8 | c=34 | **30 / 30** | **81.7 MB** (4.1x smaller than 335 MB) |
| v12 A=8 | c=32 | 30 / 30 | 91.4 MB |
| v14 A=6 | c=34 | 28 / 30 | 70.8 MB |

### The extreme: a word's 768 values in tens of bits

It fails. The best attempts:
- sign of each value only (768 bits a word): 15/30
- the 16 loudest values only: 0-4/30
- the 2 loudest values only: 0-6/30

What carries the word is the pattern across the 768 lines, at a few places of depth each.

## Round two: error diffusion, the frontier, photons only

Each family's best was rechecked by an independent checker and confirmed.

- **Error diffusion on the plates (halftoning): no gain.** Carrying each weight's rounding error to the next line turns
  its error into e_i - e_(i-1). The light on neighbouring lines is unrelated, so the pile sees twice the error power
  (second order: 6x). Words fall in the order of that power. Plain rounding stays best.
- **Error diffusion on the tables: a small, shaky gain.** A lattice parity screen (E8) on the lowest kept places gives
  31.5 MB at 30/30, against 34.6 MB. But 2 of 5 grain patterns give 29.
- **Storing only the photons of developed values: works, at about 3 bits per photon.** It saves 0.3 MB on the plates
  and 0.7 MB on each table setting. Whole machine from the photon files: **80.7 MB, 30/30**. Once a value is developed
  to 1-2 photons, the file really is "only the 1s": 1.69 photons per weight at about 3.1 bits each.
- **The combined frontier (exposure per depth and per bank kind, whole machine judged at once):**

| right next word | stored | how |
|---|---|---|
| 30 / 30 | 72.5 MB | layers 4-7 plates coarser (A=5), the rest A=8, tables c=35 |
| 29 / 30 | 66.0 MB | qkv and mproj at A=5 everywhere, proj and fc at A=5 in layers 4-7 |
| 28 / 30 | 65.2 MB | |

  **Caveat 1: the 30/30 at 72.5 MB is a lucky draw.** Re-laying the grain gives 25-29 words, 27.6 on average; only
  1 of 13 grain patterns gives 30. The words that tip are spread over several close texts. A dependable 30/30 today is
  the 80.7 MB point.
- **Caveat 2: the judge is small.** It is 30 texts of 5 tokens, and only 109 of the 50,257 entry rows are ever read
  directly. Every other row of the developed entry table is tested only through the printed top table at the read.

## Where it stands

| right next word | stored | vs 335 MB as written |
|---|---|---|
| 30 / 30, dependable | 80.7 MB | 4.2x smaller |
| 30 / 30, one lucky draw | 72.5 MB | 4.6x |
| 29 / 30 | 66.0 MB | 5.1x |

A word in tens of bits: no (0-15 / 30). Exact storage: about 228 MB, held at that floor by the coin-flip grain under
each value's top three places.
