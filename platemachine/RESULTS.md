# The plate machine, Phase 1-3 (2026-10-08; spec approved by Parrish, his corrections: no old repos, no quantization comparisons)

## Phase 1: GPT-2 as plates and light, simulated exactly (plate_machine.py)
- Every pile identical to base2.py's own product: 86,314,248 on the 30 refs texts, 898,738,800 on the 300 judge
  windows, 0 differing.
- Words: 30 refs texts 479/480 (base2.py's own score); judge5 6000/6000 in the reference top 5, 6000/6000 same top word.
- Work a token (the 648 refs positions):
  - photons sent 265,405
  - detections 31,829,601 (about 241 a pile; 133,201 piles a token: 82,944 in the banks, 50,257 at the top read)
  - adds 31,829,601 (one a detection)
  - arrival and the rest 12,578,930 (counted as base2_fringes2.py counts them: 8,020,119 + 4,558,811)
  - TOTAL 76,503,536 = 2.09% of plain (3,666,596,259); counting machine 691,505,279 (18.86%); the extra 6.24%
  - by part, detections: qkv 6.67M, proj 2.22M, fc 8.97M, mproj 2.24M, top read 11.73M
  - the top read's race: 2,167,243 compares a token, kept apart
- Plain rebuilt: qkv, proj, mproj equal to the earlier unit exactly; fc and the top read differ by their non-meeting
  parts (divides, biases, bends), which the earlier unit includes.
- Prediction (written first): about 75M a token, about 2% of plain, 200-250 detections a pile. Held.
- In the simulation the meetings are still made by the computer; the count is what a plate does, where the meetings
  cancel by interference.

## Phase 3: the stored size (stored_size.py)
- The plates' sheets, up/hole/dark at each of 16 places: as is 25.36 bits a weight (392 MB); at their floor 16.76 bits
  a weight (259 MB). The 16-place whole numbers themselves: 16 bits (247 MB). Ternary: 1.58 (24 MB).
- The plate form cuts the work, not the store.

## Phase 2: what each detection needs (written, nothing run, no old repos)
1. The sign. A detector reads brightness, the square of the summed wave, so the sign is lost. Reading +3 apart from -3
   needs a fixed reference wave added at the detector; the brightness then carries the net in a straight line, and
   the hole (the flipped wave) reads as -1.
2. Exact whole numbers. Each net lies between -768 and +768. The read has to land on the right one of 1,537 values
   every time: the noise well under half a step, at 32M detections a token.
3. Each detector only counts. The place scale 2^(a+b) is applied after the read, as adds (counted), so no detector
   needs more than 1,537 levels.
4. Size: all detectors of a bank at once: fc about 746,000 (3,072 piles x about 243), the top read about 11.7M.

## Rule slips
- refs run peaked at 4.9 GB and stored_size.py at 3.7 GB, over the 3 GB line (each the only heavy process). Fixed for
  the judge run (2.89 GB) by holding the plates as their 16-place whole numbers and building the sheets in slices.
  No number changed.
