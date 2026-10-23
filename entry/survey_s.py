"""WHICH WATER-LEVEL SHIFTS s CAN OCCUR AT qkv.0, OVER EVERY (word, position) PAIR.

At qkv.0 the light of a token is its word's entry row + its position's row, R = wte[w] + wpe[p] (base2.py forward).
The water level (base2.to16) puts each position on 16 bits: top = bit length of the OR of the 768 magnitudes (the
same as the bit length of the largest one), s = max(top - 15, 0), x16 = rshift(R, s). So s is fixed by the pair
alone. This script finds s for all 50,257 x 1,024 = 51,463,168 pairs, whole numbers only: s = how many of the
powers 2^15, 2^16, 2^17, ... the largest magnitude reaches (that is bit length - 15, floored at 0).

PREDICTION (written before the run, from the row sizes: position 0's largest line 297,410; positions 1..1022 between
51,992 and 106,246; position 1023 only 950; words between 20,672 and 116,992):
  - s takes the values 0, 1, 2, 3, 4.
  - Position 0 gives only s = 3 or 4 (its sum's largest line is between 297,410 - 116,992 = 180,418 and 414,402).
  - s = 4 occurs only at position 0.
  - s = 0 occurs at position 1023 for the words whose largest line is below 32,768 - 950 (about 18,000 words), and
    rarely or never elsewhere (it would need the word to cancel the position's large lines).
  - s = 3 at positions >= 1 is rare (a word's large line and the position's large line, same sign, same line).
  - s = 1 and 2 hold almost all pairs at positions 1..1022.
Writes entry/survey_s.npz: count[s, p] (words at position p with shift s), wmask[w], pmask[p] (bit s set when the
word / position takes shift s with some partner), smax.
    python3 entry/survey_s.py
"""
import os, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
t0 = time.time()
M = np.load(f"{HERE}/base2_gpt2.npz")
wte = M["wte"]; assert np.abs(wte).max() < 2 ** 30; wte = wte.astype(np.int32)       # whole numbers, all below 2^17
wpe = M["wpe"]; assert np.abs(wpe).max() < 2 ** 30; wpe = wpe.astype(np.int32)       # below 2^19; sums below 2^20
NW, NP = wte.shape[0], wpe.shape[0]
SMAX = 8                                                     # room for s = 0..7; the largest possible here is 4 (sum < 2^20)
count = np.zeros((SMAX, NP), np.int64); wmask = np.zeros(NW, np.uint8); pmask = np.zeros(NP, np.uint8)
CH = 32
for i in range(0, NW, CH):
    a = wte[i:i + CH]
    m = np.abs(a[:, None, :] + wpe[None, :, :]).max(2)       # (words, positions): the largest magnitude of the sum
    s = np.zeros(m.shape, np.int64)
    for k in range(15, 15 + SMAX): s += (m >= (1 << k))     # s = bit length - 15, floored at 0
    assert s.max() < SMAX - 1
    for v in np.unique(s):
        hit = s == v
        count[v] += hit.sum(0)
        wmask[i:i + CH] |= np.where(hit.any(1), np.uint8(1 << int(v)), np.uint8(0))
        pmask |= np.where(hit.any(0), np.uint8(1 << int(v)), np.uint8(0))
    if i % (CH * 400) == 0: print(f"  words {i:6d} ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)
assert count.sum() == NW * NP
np.savez(f"{HERE}/entry/survey_s.npz", count=count, wmask=wmask, pmask=pmask)
print(f"\nall {NW * NP:,} pairs ({time.time() - t0:.0f}s)")
for v in range(SMAX):
    if count[v].sum() == 0: continue
    ps = np.flatnonzero(count[v]); nw = int(((wmask >> v) & 1).sum())
    where = f"positions {ps.tolist()}" if len(ps) <= 12 else f"{len(ps)} positions ({ps.min()}..{ps.max()})"
    print(f"  s = {v}: {int(count[v].sum()):>12,} pairs; {nw:6,} words take it at some position; {where}")
print(f"  position 0: s values {sorted(set(np.flatnonzero(count[:, 0]).tolist()))}  counts {count[:, 0][count[:, 0] > 0].tolist()}")
print(f"  position 1023: s values {sorted(set(np.flatnonzero(count[:, 1023]).tolist()))}  counts {count[:, 1023][count[:, 1023] > 0].tolist()}")
nset = np.array([bin(int(x)).count('1') for x in wmask])
print(f"  shifts a word can take (count of distinct s per word): {dict(zip(*[x.tolist() for x in np.unique(nset, return_counts=True)]))}")
print(f"  word masks (bit s set): {dict(zip(*[x.tolist() for x in np.unique(wmask, return_counts=True)]))}")
print(f"  position masks: {dict(zip(*[x.tolist() for x in np.unique(pmask, return_counts=True)]))}")
print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
