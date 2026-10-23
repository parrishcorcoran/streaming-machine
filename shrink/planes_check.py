"""PLANES family, end-to-end check of the best lossless system (version 7b) against the machine itself.
    python3 planes_check.py store    -> code E and T with version 7b, decode from the bytes + stored dark levels alone,
                                        save the decoded tables to the scratchpad (small process, no machine)
    python3 planes_check.py machine  -> load table_bits (the harness), check the decoded tables equal its E and T
                                        exactly, and ask the machine: right next word out of 30
"""
import sys, os, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
OUT = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
if sys.argv[1] == "store":
    from planes_lib import *
    MODE = M_HIGH | M_LINE | M_SIGNC | M_WEXP | M_REFLINE
    for tb in ["E", "T"]:
        A = load(tb); R, C = A.shape
        off = ((A[:32768].astype(np.int64).sum(0) + (1 << 14)) >> 15).astype(np.int32)
        n, mark, B, nb, data = code(A - off[None, :], MODE); del A, B
        side = 2 * 32 + 5 + 16 + C * (int(np.abs(off).max()).bit_length() + 1)
        D = np.zeros((R, C), np.int32); m2 = np.zeros(nb + 1, np.int64)       # the reader: bytes + dark levels only
        lib().planes(1, D.ctypes.data, R, C, nb, MODE, None, data.ctypes.data, len(data), m2.ctypes.data)
        D += off[None, :]
        np.save(f"{OUT}/planes_{tb}_decoded.npy", D)
        print(f"{tb}: {n:,} bytes + {side:,} bits side = {(n * 8 + side) / D.size:.4f} bits/value = {(n * 8 + side) / 8e6:.3f} MB")
else:
    sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); os.chdir("/Users/abundancemachine/streaming-machine")
    from table_bits import *
    Ed = np.load(f"{OUT}/planes_E_decoded.npy"); Td = np.load(f"{OUT}/planes_T_decoded.npy")
    print("decoded entry table equals harness E:", bool(np.array_equal(Ed.astype(np.int64), E)))
    print("decoded top table equals harness T:", bool(np.array_equal(Td.astype(np.int64), T.astype(np.int64))))
    print("right next word out of 30, decoded tables:", words(entry=Ed.astype(np.int64), top=Td.astype(np.int16)))
