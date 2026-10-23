"""FAX family, end-to-end check of the best system: the entry table stored by version 8a (flat field + JBIG contexts),
the top table stored by version 7 (its difference from the entry table seen through the gain glass), both decoded from
their bytes alone, then handed to the machine.
    python3 fax_check.py store     -> codes, decodes, writes the decoded tables to the scratchpad (small process)
    python3 fax_check.py machine   -> loads table_bits (the harness), checks the decoded tables equal its E and T,
                                      and asks the machine: right next word out of 30
"""
import sys, os, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
OUT = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"

if sys.argv[1] == "store":
    from fax_common import *
    import fax_v6_jbig as v6, fax_v7_top_from_entry as v7
    E = load("E"); n2 = 1 << (E.shape[0].bit_length() - 1); sh = n2.bit_length() - 1
    off = ((E[:n2].astype(np.int64).sum(0) + (n2 >> 1)) >> sh).astype(np.int32)
    bufs, nb = v6.code(E - off[None, :])
    eb = sum(len(b) for b in bufs); del E
    Ed = v6.decode(bufs, nb, (50257, 768)) + off[None, :]; del bufs           # the reader has only bytes + off
    T = load("T")
    P, dc, side = v7.predict(Ed, v7.G)                                         # reference made from the DECODED entry
    b1, b2, rounds, mant = v7.code(T, P, dc, side); del T
    tb = len(b1) + len(b2) + len(mant) + sum(len(b) for b in rounds)
    Td = v7.decode(b1, b2, rounds, mant, P, dc, side).astype(np.int32); del P, dc, side
    np.save(f"{OUT}/fax_E_decoded.npy", Ed); np.save(f"{OUT}/fax_T_decoded.npy", Td)
    side_e = 2 * 32 + 5 + 64 * (nb + 1) + 768 * (int(np.abs(off).max()).bit_length() + 1)
    side_t = 768 * 29 + (4 + len(rounds)) * 32 + 5
    print(f"entry table: {eb:,} bytes + {side_e:,} bits side = {(eb * 8 + side_e) / Ed.size:.3f} bits/value")
    print(f"top table given entry: {tb:,} bytes + {side_t:,} bits side = {(tb * 8 + side_t) / Td.size:.4f} bits/value")
    print(f"both tables: {(eb * 8 + side_e + tb * 8 + side_t) / 8e6:.2f} MB  (stored as they are: 86.84 + 77.19 = 164.03 MB)")
else:
    sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); os.chdir("/Users/abundancemachine/streaming-machine")
    from table_bits import *
    Ed = np.load(f"{OUT}/fax_E_decoded.npy"); Td = np.load(f"{OUT}/fax_T_decoded.npy")
    print("decoded entry table equals harness E:", bool(np.array_equal(Ed.astype(np.int64), E)))
    print("decoded top table equals harness T:", bool(np.array_equal(Td.astype(np.int64), T.astype(np.int64))))
    print("right next word out of 30, decoded tables:", words(entry=Ed, top=Td.astype(np.int16)))
