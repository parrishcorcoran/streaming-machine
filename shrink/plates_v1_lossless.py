"""PLATES v1 -- lossless. Every weight of all 48 plates written as its photons (single-wave writing: up +1, hole -1),
fax-style runs, read against the line meter and the neuron meter. Real bytes from the integer range coder, decoded
back and checked. Side information: each neuron's power of two (1 byte) and a 4-byte length per plate.
Control: the same coder on each plate with its values shuffled at random (fixed seed) -- keeps the distribution,
destroys any line / neuron structure.
Prediction before running: ~14.4 bits/value (places 0-8 are lit half the time with no pattern: ~9 bits of pure
noise per weight), ~153 MB; shuffle control ~0.1-0.2 bits worse (the line brightness from the folded gain)."""
from plates_lib import *
rec = run_version("v1 lossless photons + meters", lambda n, q, e: (q.astype(np.int32), 0), side_per_col=1,
                  note="identity; words() run on the decoded plates as a plumbing check")
rng = np.random.default_rng(7); tot = 0
for n in NAMES:
    q = plate(n).astype(np.int32); qs = rng.permutation(q.ravel()).reshape(q.shape)
    tot += store(qs, check=False)[0]
ctl = {"version": "v1 control: values shuffled inside each plate", "coded_bytes": tot,
       "bits_per_value": round(8 * (tot + side_bytes(1)) / NVAL, 4), "total_MB": round((tot + side_bytes(1)) / 1e6, 3)}
log(ctl); print(ctl)
