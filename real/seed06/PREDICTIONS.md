# Qwen3-0.6B blank, step 1 (candidate seeds), written before the run (2026-10-09)
Draw: torch CPU normal_ on bf16 (checked here = the pinned emulator). Seeds: 3 found Qwen3 seeds + 17 common.
Orders: layer plates from offset 0, or after the word table (155,582,464). Every within-layer place (subsets of
the other 6 plates). Layout as stored or transposed. z = corr * sqrt(n) on each plate's first 262,144 values.
Check: the same at off-grid places.
- Prediction: NO hit. Every seed and order at the noise level (largest |z| about 4.5-5.5, same as the off-grid check),
  because the 14B body was not in its seed's CPU stream anywhere either, and 0.6B likely came from the same procedure.
- If a guess is right and the body kept its blank like the tiny model did, that plate shows |z| in the hundreds.
