"""DEVELOP family, version 9: three things the earlier negatives asked for.
(1) WHY DID DITHERING COLLAPSE T?  v2: T top-2 by chance 7-8/30 (nearest 25), top-1 by chance 0 (nearest 21); v4: word cut
b=8 by chance 23 (nearest 28). Reading: the race at the top only feels DIFFERENCES between words. A fixed rounding
gives two words with the same value on a line the same error -- it cancels in the race. Independent chance per value
gives every word its own error even where all words carry the same light -- nothing cancels. A phone's tremor shifts the
WHOLE frame at once, not each pixel separately. Test: one random offset per line, shared by all 50,257 words
(u = r_i x q >> 16, r_i a 16-bit whole number per line): still exact on average over lines, but common to all words.
Predicted: T top-2 shared 21-26 (near nearest 25, far above 7-8); top-1 shared 15-21 (not 0); cut b=8 shared 26-28.
If these stay near the independent-chance numbers, the reading is wrong.
(2) The one-negative ladder (E exposed for the lens, cap 12, T printed), more rungs: c = 32, 34, 37, 39.
Predicted: c=32 29-30, c=34 27-29, c=37 22-27, c=39 15-24 (v8b: c=31 30, c=33 28, c=35 28).
(3) The whole machine at the extreme, both tables from one negative: E sign only (768 bits a word) + T printed:
3-12; E white plate + 2 loudest photons (~50 bits a word) + T printed: the floor, 0-6.
    python3 develop_v9_shared_tremor.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [
    dict(name="v9 T top-2 shared tremor", T=dict(how="topk", k=2, seed=902, shared=True)),
    dict(name="v9 T top-1 shared tremor", T=dict(how="topk", k=1, seed=901, shared=True)),
    dict(name="v9 T cut b=8 shared tremor", T=dict(how="cut", b=8, seed=908, shared=True)),
]
SPECS += [dict(name=f"v9 E lens exposure c={c} cap 12, T printed", E=dict(how="lines", c=c, cap=12), T=dict(how="print"))
          for c in (32, 34, 37, 39)]
SPECS += [
    dict(name="v9 whole machine: E sign only, T printed", E=dict(how="sign"), T=dict(how="print")),
    dict(name="v9 whole machine: E plate + loudest 2, T printed", E=dict(how="dark", inner=dict(how="loud", n=2)), T=dict(how="print")),
]
if __name__ == "__main__": run(SPECS, sys.argv[1])
