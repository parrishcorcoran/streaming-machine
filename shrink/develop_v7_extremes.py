"""DEVELOP family, version 7: the extreme the user hopes for -- a word's 768 values in very few bits.
  sign            : only the sign of each value (1 bit a value, 768 bits a word) at the word's mean loudness
  sign CONTROL    : the same signs put on random lines within each word (same count up and down)
  plate + sign    : the white plate kept (768 numbers for ALL words), each word only the sign of its departure
  sign by chance  : 1 bit a value, +L or -L with chance (L+v)/2L (L = the word's loudest): exact on average
  loud n          : only the n loudest values of each word, each one photon (+-2^place), all else dark
                    (n = 16 raw; on the departure from the plate n = 64, 16, 2; CONTROL n = 16 on random lines)
  n = 2 is "a few tens of bits a word": 2 x (which of 768 lines ~9.6 bits + up/hole + place ~3 bits) ~ 25 bits.
Information in hand before running: naming one of 50,257 words takes 15.6 bits -- that is the word's index, which the
machine already has; the table's job is the word's light on 768 lines. Measured so far: one photon a value (~3.4 bits)
gives E 29, T 21; the controls (each word's own values on random lines) give E 2, T 0.
Predicted before running:
  E: sign 18-28; sign CONTROL 0-4; plate+sign 24-30; sign by chance 5-20; loud-16 raw 0-5; plate loud-64 3-15,
     loud-16 1-8, loud-2 0-4 (the control floor); loud-16 random lines 0-4.
  T: sign 3-15; sign CONTROL 0; plate+sign 8-20; sign by chance 0-2 (each word its own error: v2 showed the race cannot
     take it); loud-16 raw 0 (T's loudest values sit on the same few lines for every word, they name nothing);
     plate loud-64 0-6, loud-16 0-3, loud-2 0; loud-16 random lines 0.
    python3 develop_v7_extremes.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = []
for tab, s0 in (("E", 700), ("T", 800)):
    SPECS += [
        dict(name=f"v7 sign only {tab}", **{tab: dict(how="sign")}),
        dict(name=f"v7 CONTROL sign only, signs on random lines {tab}", **{tab: dict(how="sign", shuffle=s0 + 1)}),
        dict(name=f"v7 plate + sign of departure {tab}", **{tab: dict(how="dark", inner=dict(how="sign"))}),
        dict(name=f"v7 sign by chance {tab}", **{tab: dict(how="signdither", seed=s0 + 2)}),
        dict(name=f"v7 loudest 16 raw {tab}", **{tab: dict(how="loud", n=16)}),
        dict(name=f"v7 plate + loudest 64 {tab}", **{tab: dict(how="dark", inner=dict(how="loud", n=64))}),
        dict(name=f"v7 plate + loudest 16 {tab}", **{tab: dict(how="dark", inner=dict(how="loud", n=16))}),
        dict(name=f"v7 plate + loudest 2 {tab}", **{tab: dict(how="dark", inner=dict(how="loud", n=2))}),
        dict(name=f"v7 CONTROL plate + 16 random lines {tab}", **{tab: dict(how="dark", inner=dict(how="loud", n=16, random_lines=s0 + 3))}),
    ]
if __name__ == "__main__": run(SPECS, sys.argv[1])
