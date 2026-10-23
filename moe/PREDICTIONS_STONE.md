# The stone across the 32 sibling experts: predictions, written before the run (2026-10-09)

His go: "Ready to go" (after the offer of the look across the 32 sibling experts).

**The picture (the session's words).**
- In each layer, the 32 expert plates have the same shape and were drawn the same way, each from its own stone.
- Each expert was built only by the tokens the router sent it.
- Separate stones share nothing with each other. So whatever the siblings share was built, and the stone sits in the
  part each one has alone.
- A stone on its own is featureless:
  - every stream place equally bright, within sqrt(2/k): 4.42% on gate|up (k = 1,024 hidden entries a place) and 6.25%
    on down (k = 512);
  - no two places related: their cosine across the k entries averages 0, with a spread of exactly 1/k, so
    R = k x (that spread) = 1.

Only the stream-place side of each plate is shared between siblings. A hidden neuron j in one expert has nothing to do
with neuron j in another, so the stream places are the only side the siblings can be compared on.

## A. The never-lit neurons: what the stone left behind where no light ever came

- 22 gate|up columns are all exactly zero: 20 in layer 0 and 2 in layer 1.
- A neuron the light never reached is like Pythia's word rows the light never reached: whatever it holds is the stone
  worn down to c.
- **Prediction:**
  - they are 11 neurons, each with gate and up both zero;
  - their down lines are exactly zero too, so c = 0 there: the stone is fully worn away.
  - If the down lines hold a small bell instead, that bell is the stone, and its spread is c x the draw's spread.

## B. Stream-place brightness, and the floor across siblings

- Every expert in a layer was drawn with the same spread and shaved by the same c, so the stone puts the same floor
  under all 32. That gives a relation: the stone's share of expert e is at most (the quietest sibling's mean square) /
  (e's mean square).
- **Prediction:** the loudest-to-quietest mean-square ratio is 1.3-3 in each layer. The bound on the stone's share of
  the median expert is then a weak 40-80%.
- Brightness spread across stream places: 10-60% for each expert, well above the stone's 4.42% / 6.25%.
- Siblings' brightness profiles related at 0.3-0.9. Control (places shuffled in each expert): 0.00 +- 0.03.

## C. Relations between stream places

- R for each expert: 2-20, against a stone's 1. Control (every weight of the plate shuffled, every value kept): R =
  1.00 +- 0.01.
- Siblings' relations patterns related at 0.1-0.6. Control (places shuffled in each expert): 0.000 +- 0.003.
- **Reference:** 32 fresh bell plates of the same shapes (seed 2026) through the same steps: R = 1, brightness spread
  4.42% / 6.25%, siblings 0. That is the stone's own look.

## What would show his 40-90%

- If most of each plate were stone, every expert would sit near the reference: R near 1, brightness spread near 4.4%,
  siblings near 0.
- The session's expectation:
  - A shows c = 0;
  - B and C show built structure that the siblings share;
  - the stone, if any is left, is the featureless remainder, and B bounds it from above.

---

# Results (added after the run; stone_look.log, stone_A.log, stone_look.json)

**A. The dark neurons.**
- The earlier claim was wrong. The 22 columns are not "all exactly zero": 0 of their weights are exactly zero.
- They are 11 neurons (10 in layer 0, 1 in layer 1). In each one, gate, up and its down line all shrank together, to
  1e-21 to 1e-31 against a plate rms of 0.017.
- Only 1 of the 11 is bell-shaped (kurtosis 3.3); the others have kurtosis 6 to 696.
- So what they hold is the neuron as it was when it went dark, shrunk, not a clean stone.
- Prediction: 11 neurons, gate and up both: **held**. Down lines exactly zero: **wrong** (shrunk with them, not zero).

**B.**
- rms 0.0155-0.018 for every expert (the draw's setting is 0.02).
- Loud/quiet 1.14-2.53 (predicted 1.3-3: held, some lower).
- Bound on the stone's share of the median expert: 67-95% (predicted 40-80%: weaker than predicted).
- Brightness spread 10-38% against the stone's 4.42/6.25% (predicted 10-60%: **held**).
- Siblings' brightness: down 0.49-0.83 (**held**); gate|up 0.04-0.45 (predicted 0.3-0.9: **wrong**, lower).
- Controls: 0.000 +- 0.004.

**C.**
- R: gate|up 1.46-4.86 (median of layers about 1.8); down 1.13-1.74. Predicted 2-20: **wrong**, much closer to a
  stone.
- Down plates in layers 0-2 have experts with R = 0.70-0.77: their places are related less than a stone's.
- R with every weight shuffled: 1.000 (control **held**).
- Siblings' relations: 0.025-0.14. Predicted 0.1-0.6: **wrong**, low. Control: 0.0000 +- 0.0002 (**held**).
- Reference (fresh bell plates): R 1.00, brightness spread 4.4/6.2%, siblings 0.000. As derived.
