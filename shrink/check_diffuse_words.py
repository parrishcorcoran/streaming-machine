"""CHECK of the diffuse_plates family -- step 2, the words. Loads the plates (and constants) decoded in step 1 from the
files alone (scratchpad/check_dec_<tag>.npz), puts ALL 48 in the machine at once with the neuron powers of two read
from the file (not the machine's own), adds the stored constants to the biases for the dark-frame file, and asks for
the right next word out of 30. Also reports how many weights differ from the original plates (after decoding)."""
import sys, os, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import table_bits as tb
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
assert tb.words() == 30                                   # truth from the untouched machine
orig = {n: tb.W[n] for n in NAMES}
for tag in (sys.argv[1:] or ["v0", "v3", "v4", "v9", "r1v14"]):
    z = np.load(os.path.join(SP, f"check_dec_{tag}.npz"))
    e_same = all(np.array_equal(z[n + ".e"], np.asarray(orig[n][1], np.int64)) for n in NAMES)
    dark = (NAMES[0] + ".c") in z.files
    changed = 0; nval = 0
    try:
        for n in NAMES:
            w = z[n]; changed += int((w != tb.plate(n)).sum()); nval += w.size
            b = orig[n][2] + (z[n + ".c"] if dark else 0)
            tb.W[n] = (w.astype(np.float64), z[n + ".e"].astype(orig[n][1].dtype), b)
        wd = int(tb.words())
    finally:
        for n in NAMES: tb.W[n] = orig[n]
    print(tag, json.dumps({"words": wd, "dark_constants": dark, "file_powers_equal_machine": e_same,
                           "changed_share": round(changed / nval, 4)}), flush=True)
    del z
assert tb.words() == 30                                   # machine restored
