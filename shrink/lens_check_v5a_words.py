"""CHECKER for LENS v5a, part 2: right next word out of 30 with E^ and T both rebuilt from the stored bytes only
(written by lens_check_v5a_bytes.py), plus the shuffle control (inverse gains on randomly permuted lines, seed 0)."""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
from table_bits import E, T, words
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/lenscheck"
t0 = time.time(); r = {}
r["unchanged"] = int(words()); print(r, f"[{time.time()-t0:.0f}s]", flush=True)
Tb = np.load(f"{SP}/Tb.npy"); r["Tb_equals_harness_T"] = bool(np.array_equal(Tb, T))
Eh = np.load(f"{SP}/Eh.npy")
r["v5a_entry_Eh_top_Tb"] = int(words(entry=Eh, top=Tb)); print(r, f"[{time.time()-t0:.0f}s]", flush=True)
del Eh
Ehs = np.load(f"{SP}/Ehs.npy")
r["control_shuffled_inverse_gains"] = int(words(entry=Ehs, top=Tb)); print(r, f"[{time.time()-t0:.0f}s]", flush=True)
del Ehs
print(json.dumps(r))
json.dump(r, open(f"{SP}/part2.json", "w"), indent=1)
