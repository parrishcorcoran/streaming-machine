"""DIFFUSE -- each version's best written to its own real file in the scratchpad, read back by the decoder alone
(read_file / read_file_dark see only the file), checked equal to what the encoder developed, and judged by words.
Files: diffuse_<name>.bin.  The decoder for v0/v3/v4 is round one's (same format, plates_best_store.read_into_machine
reads them too); v9 files add one coded constant per neuron after each plate."""
from diffuse_dark import *
import diffuse_dark as dd
jobs = [("v0_plain_v12_A8", "v12", 8, "plain", None),
        ("v3_carry_across_neurons_v12_A8", "v12", 8, "across", None),
        ("v4_carry_in_steps_v14_A6", "v14", 6, "steps", None),
        ("v9_chained_dark_frame_v12_A6", "v12", 6, "plain", "chained")]
dev = {"plain": lambda n, q, s: develop_plain(q, s),
       "across": lambda n, q, s: develop_carry(q, s, axis=1, kind=1, unit=False),
       "steps": lambda n, q, s: develop_carry(q, s, axis=0, kind=1, unit=True)}
for name, rule, A, d, dark in jobs:
    path = os.path.join(SP, f"diffuse_{name}.bin")
    if dark:
        dd.POS = 1; run_dark(f"STORED {name}", rule, A, dev[d], windows(), chained=True, path=path, keep=True)
    else:
        run(f"STORED {name}", rule, A, dev[d], path=path, keep=True)
