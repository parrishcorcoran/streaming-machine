"""PHOTONS family: the whole machine read from the photon files alone, and asked for the right next word out of 30.
The stored numbers decode back exactly (photons_store.py checks np.array_equal), so the words must be round one's:
v12 A=8 + c=32 30/30, v12 A=8 + c=34 30/30, v14 A=6 + c=34 28/30. This run confirms it from the files.
    python3 shrink/photons_words.py
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
os.chdir("/Users/abundancemachine/streaming-machine")
import table_bits as tb
from photons_store import iter_plates, read_tables, SP, log
assert tb.words() == 30
tb.E = None; tb.T = None; tb._WTE0 = tb._WTE0.astype(np.int32)   # same whole numbers in 32 bits (memory)
import gc; gc.collect()

def put_plates(path):
    for n, (v, s, e) in iter_plates(path):          # one plate at a time
        assert np.array_equal(e, tb.M[n + ".e"].astype(np.int64))
        w = np.clip(np.left_shift(v.astype(np.int64), s.astype(np.int64)), -32768, 32767).astype(np.int16)
        tb.W[n] = (w.astype(np.float64), tb.W[n][1], tb.W[n][2]); del v, s, w

sizes = {}
for pf in ("photons_plates_v12_A8.php", "photons_plates_v14_A6.php"):
    t0 = time.time(); put_plates(SP + pf); sizes[pf] = os.path.getsize(SP + pf)
    w_alone = int(tb.words())
    for c in (32, 34):
        tf = f"photons_tables_c{c}.pht"; D, Tp, cc = read_tables(SP + tf); sizes[tf] = os.path.getsize(SP + tf)
        w = int(tb.words(entry=D, top=Tp)); del D, Tp
        tot = sizes[pf] + sizes[tf]
        log({"version": f"WORDS {pf} + {tf}", "words": w, "plates_alone_words": w_alone, "total_bytes": tot,
             "total_MB": round(tot / 1e6, 3), "decoded_from_file_only": True, "secs": round(time.time() - t0, 1)})
    for n in [k for k in tb.W if k != "head"]: tb.W[n] = (tb.M[n + ".w"].astype(np.float64), tb.W[n][1], tb.W[n][2])   # the machine's own plates back
