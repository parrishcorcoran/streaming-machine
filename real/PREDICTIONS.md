# Pythia-70m, written before the run (2026-10-09)
- blank: loss about 10.8 (pure guess), right about 0-2%. finished: loss about 3.5-4.5 on the Shakespeare, right 25-35%.
- rebuilt wrong 0.
- W exact about 13-14 bits a weight (fp16: side, power of two, 10 inner places).
- W - S (fp32 blank): more than W, about 18-22, because it keeps the blank's fine places.
- W - fp16(S): about 12.5-13.5, saving under 1 bit against W.
- c (blank left inside): about 0.3-0.5 in the body (decay shrinks it); the blank about 20-50% of each piece's length.
- unchanged weights: tiny, except unused word rows if any.
- under 1 bit a weight: not reached.
