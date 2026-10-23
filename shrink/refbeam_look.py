"""REFERENCE BEAM family, a first look (no coding): how big is the light every word shares (per line), and how big is
each word's own offset, compared with the word itself. Numbers only, whole-number reads of the stored tables.
    python3 refbeam_look.py
"""
import numpy as np
D = '/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/rb/'
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
def naf(A): A = np.abs(np.asarray(A).astype(np.int64)); return ones(A ^ (3 * A))
for name in ("E", "T"):
    A = np.load(D + name + ".npy").astype(np.int64)
    N, L = A.shape
    med = np.median(A, axis=0).astype(np.int64)            # per-line middle value (whole: odd N)
    mean = A.sum(0) // N
    rowmean = A.sum(1) // L
    sd_line = A.std(0); sd_all = A.std()
    print(f"== {name}: {N} words x {L} lines, values {A.min()}..{A.max()}, spread (all) {sd_all:.0f}")
    print(f"   per-line middle value: |med| mean {np.abs(med).mean():.0f}, max {np.abs(med).max()} at line {np.abs(med).argmax()}")
    print(f"   per-line spread: min {sd_line.min():.0f} median {np.median(sd_line):.0f} max {sd_line.max():.0f} (line {sd_line.argmax()})")
    big = np.argsort(-np.abs(mean))[:8]
    print("   8 biggest line means:", [(int(j), int(mean[j]), int(sd_line[j])) for j in big])
    # energy of the common part vs the whole
    e_all = (A.astype(np.float64) ** 2).sum(); e_ref = N * (mean.astype(np.float64) ** 2).sum()
    print(f"   energy in the shared part (per-line mean): {e_ref / e_all * 100:.2f}% of all light")
    print(f"   per-word offset (row mean): |.| mean {np.abs(rowmean).mean():.1f}, spread {rowmean.std():.1f}; "
          f"expected from chance ~ {sd_all / np.sqrt(L):.1f}")
    R = A - med[None, :]
    print(f"   photons/value: word {ones(A).mean():.3f} (single wave {naf(A).mean():.3f}); minus per-line middle "
          f"{ones(R).mean():.3f} ({naf(R).mean():.3f})")
    # how many lines does the reference beam change much
    gain = 0.5 * np.log2(((A - 0) ** 2).mean(0) / np.maximum(((A - mean[None]) ** 2).mean(0), 1))
    print(f"   Gaussian bits saved per value by removing line mean: mean {gain.mean():.4f}; lines saving >0.1 bit: {(gain > 0.1).sum()}, top: {np.round(np.sort(gain)[-6:], 3)}")
    del A, R
